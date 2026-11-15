"""构建产物的可运行性校验与修复。

为什么需要这个脚本
------------------
PyInstaller 按 PATH 搜索 DLL 依赖。conda 会把 **base** 环境的 `Library\\bin`
放进 PATH，于是构建时有些 DLL 被从 base 环境收集，而不是从当前构建环境：

    D:\\miniforge\\Library\\bin\\libexpat.dll              ← 错误来源
    D:\\miniforge\\envs\\excel_assistant\\Library\\bin\\libexpat.dll  ← 正确来源

两个版本导出表不同。`pyexpat.pyd` 按 **ordinal** 引用 `libexpat.dll` 的导出，
base 版本对不上，加载时报：

    ImportError: DLL load failed while importing pyexpat:
    操作系统无法运行 %1

这个 `%1` 是 Windows 错误 182（ERROR_INVALID_ORDINAL）的消息模板——「按序号
导入的函数不存在」。同一类问题还会波及 `libssl` / `libcrypto` / `liblzma` /
`sqlite3`，表现为后续某个模块导入失败或 SSL 行为异常。

本脚本做三件事
--------------
1. **修复**：凡是从「非构建环境」收集的 DLL，用构建环境的同名文件覆盖。
2. **冒烟测试**：用 ctypes 逐个加载对 DLL 敏感的标准库扩展，确认能加载。
3. **产物校验**：`llm_config.json` 声明的服务与模型文件必须在发行目录内。

用法（通常在 build_windows.ps1 末尾自动调用）
    python packaging/verify_dist.py dist/ExcelAssistant
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

# 对 DLL 版本敏感的标准库扩展。selector/ssl/sqlite3/lzma/bz2 都依赖 conda 提供
# 的 DLL，是「来源错误」最容易暴露的地方。
SENSITIVE_EXTENSIONS = (
    "pyexpat.pyd",
    "_elementtree.pyd",
    "_ssl.pyd",
    "_hashlib.pyd",
    "_sqlite3.pyd",
    "_lzma.pyd",
    "_bz2.pyd",
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_bins() -> list[Path]:
    """找出可能污染构建的 Library\\bin 目录。

    第一个是当前环境（正确来源），其余是上层环境（错误来源）。
    """
    own = Path(sys.prefix) / "Library" / "bin"
    others = []
    parent = Path(sys.prefix).parent
    for _ in range(3):  # 最多向上三层，覆盖 conda 的 base 布局
        candidate = parent / "Library" / "bin"
        if candidate.is_dir() and candidate != own:
            others.append(candidate)
        parent = parent.parent
    return [own] + others


def repair_dlls(internal: Path) -> list[str]:
    """把来自错误环境的 DLL 换成构建环境的版本，返回修复清单。"""
    own_bin, *other_bins = _candidate_bins()
    if not own_bin.is_dir():
        print(f"  跳过 DLL 修复：找不到 {own_bin}")
        return []

    own_map = {item.name.lower(): item for item in own_bin.glob("*.dll")}
    other_maps = [
        {item.name.lower(): item for item in folder.glob("*.dll")}
        for folder in other_bins
    ]

    repaired: list[str] = []
    for bundled in sorted(internal.glob("*.dll")):
        key = bundled.name.lower()
        if key not in own_map:
            continue
        bundled_hash = _digest(bundled)
        if bundled_hash == _digest(own_map[key]):
            continue  # 已经是构建环境的版本
        # 只有当它能被证明来自其他环境时才替换，避免动到第三方 DLL
        from_other = any(
            key in mapping and _digest(mapping[key]) == bundled_hash
            for mapping in other_maps
        )
        if not from_other:
            continue
        bundled.write_bytes(own_map[key].read_bytes())
        repaired.append(bundled.name)
    return repaired


def smoke_test(internal: Path) -> list[str]:
    """用 ctypes 加载对 DLL 敏感的标准库扩展，返回失败清单。"""
    failures: list[str] = []
    handle = os.add_dll_directory(str(internal))
    try:
        for name in SENSITIVE_EXTENSIONS:
            path = internal / name
            if not path.is_file():
                continue
            try:
                ctypes.WinDLL(str(path), use_last_error=True)
            except OSError as error:
                code = getattr(error, "winerror", "?")
                failures.append(f"{name}（WinError={code}：{error}）")
    finally:
        handle.close()
    return failures


def verify_payload(dist: Path) -> list[str]:
    """`llm_config.json` 里声明的路径必须真的存在。"""
    problems: list[str] = []
    config_path = dist / "llm_config.json"
    if not config_path.is_file():
        return [f"缺少 {config_path.name}"]

    raw = json.loads(config_path.read_text(encoding="utf-8"))

    def check(key: str, *, directory: bool = False) -> None:
        value = str(raw.get(key) or "").strip()
        if not value:
            return
        target = dist / value.replace("/", "\\")
        ok = target.is_dir() if directory else target.exists()
        if not ok:
            kind = "目录" if directory else "文件"
            problems.append(f"{key} 指向的{kind}不存在：{value}")

    backend = str(raw.get("backend") or "llama_server").strip().lower()
    if backend == "transformers":
        # 进程内后端：只需要模型目录，没有子进程也没有下载阶段。
        if not str(raw.get("model_dir") or "").strip():
            problems.append("backend=transformers 但缺少 model_dir")
        check("model_dir", directory=True)
        model_dir = str(raw.get("model_dir") or "").strip()
        if model_dir:
            target = dist / model_dir.replace("/", "\\")
            if target.is_dir() and not (target / "config.json").is_file():
                problems.append(f"模型目录缺少 config.json：{model_dir}")
        return problems

    has_download = bool(str(raw.get("model_url") or "").strip())
    check("local_server")
    if not has_download:
        # 没有下载地址时模型必须随包，否则首次启动必然失败
        check("local_model_path")
    return problems


def verify_embedded_deps(dist: Path) -> list[str]:
    """transformers 后端必须在发行目录里带齐 torch / transformers。

    这个检查针对一个真实风险：`--exclude-module torch` 是 llama_server 后端的
    优化，如果构建时忘了按后端切换，就会产出一个「能启动但一提问就 ImportError」
    的包。PyInstaller 不会自己发现这种矛盾。
    """
    internal = dist / "_internal"
    config_path = dist / "llm_config.json"
    if not config_path.is_file():
        return []
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if str(raw.get("backend") or "").strip().lower() != "transformers":
        return []

    problems: list[str] = []
    # PyInstaller 可能把包放进 _internal 目录，也可能收进 PYZ 归档，
    # 所以先看目录；目录里没有时用 .pyz 里的名字兜底。
    for package in ("torch", "transformers"):
        if (internal / package).is_dir():
            continue
        if _pyz_contains(internal, package):
            continue
        problems.append(
            f"缺少 {package}：backend=transformers 需要它，"
            "构建时不要排除 torch/transformers"
        )
    return problems


def _pyz_contains(internal: Path, package: str) -> bool:
    """检查 PYZ 归档里是否有某个顶层包。"""
    try:
        from PyInstaller.archive.readers import ZlibArchiveReader
    except ImportError:
        return False
    for archive in internal.glob("*.pyz"):
        try:
            reader = ZlibArchiveReader(str(archive))
        except Exception:  # noqa: BLE001 - 归档损坏时交给其他检查报告
            continue
        names = getattr(reader, "toc", None) or []
        if any(str(name).split(".")[0] == package for name in names):
            return True
    return False


def main() -> int:
    dist = Path(sys.argv[1] if len(sys.argv) > 1 else "dist/ExcelAssistant").resolve()
    if not dist.is_dir():
        print(f"发行目录不存在：{dist}")
        return 1
    internal = dist / "_internal"
    if not internal.is_dir():
        print(f"找不到 _internal：{internal}")
        return 1

    print(f"校验发行目录：{dist}")

    repaired = repair_dlls(internal)
    if repaired:
        print(f"  修复 {len(repaired)} 个来自错误环境的 DLL：")
        for name in repaired:
            print("    -", name)
    else:
        print("  DLL 来源正确")

    failures = smoke_test(internal)
    if failures:
        print(f"  导入冒烟测试失败 {len(failures)} 项：")
        for item in failures:
            print("    -", item)
    else:
        print("  导入冒烟测试通过")

    payload_problems = verify_payload(dist)
    if payload_problems:
        print("  产物校验失败：")
        for item in payload_problems:
            print("    -", item)
    else:
        print("  产物校验通过")

    deps_problems = verify_embedded_deps(dist)
    if deps_problems:
        print("  嵌入式依赖校验失败：")
        for item in deps_problems:
            print("    -", item)
    else:
        print("  嵌入式依赖校验通过")

    if failures or payload_problems or deps_problems:
        print("结果：不可交付")
        return 1
    print("结果：可以交付")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
