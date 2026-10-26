"""数据集目录的日期归档。

每次构建都会产生一整套表格、样本与结果文件；写进同一个目录会互相覆盖。
这里按天建目录，同一天多次构建再加序号，读命令则自动解析到最新一份。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import re

DATE_DIR_PATTERN = re.compile(r"^(?P<day>\d{4}-\d{2}-\d{2})(?:-(?P<index>\d+))?$")

# 判定"这个目录里已经有数据集"的标志文件。
DATASET_MARKER = "samples.jsonl"


def is_dataset_dir(path: Path) -> bool:
    return (path / DATASET_MARKER).is_file()


def _is_empty_dir(path: Path) -> bool:
    try:
        return not any(path.iterdir())
    except OSError:
        return False


def dated_run_dir(base: str | Path, *, day: date | None = None) -> Path:
    """为本次构建分配一个不冲突的目录：``<base>/<YYYY-MM-DD>[-N]``。

    同一天重复构建时序号递增，避免覆盖既有数据。
    """
    root = Path(base)
    stamp = (day or date.today()).isoformat()
    candidate = root / stamp
    if not candidate.exists() or _is_empty_dir(candidate):
        return candidate

    index = 2
    while True:
        candidate = root / f"{stamp}-{index}"
        if not candidate.exists() or _is_empty_dir(candidate):
            return candidate
        index += 1


def _sort_key(name: str) -> tuple[str, int] | None:
    match = DATE_DIR_PATTERN.match(name)
    if not match:
        return None
    return match.group("day"), int(match.group("index") or 1)


def list_dataset_dirs(base: str | Path) -> list[Path]:
    """列出 ``base`` 下所有日期目录，按时间从旧到新排序。"""
    root = Path(base)
    if not root.is_dir():
        return []
    found: list[tuple[tuple[str, int], Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        key = _sort_key(child.name)
        if key is not None:
            found.append((key, child))
    found.sort(key=lambda item: item[0])
    return [path for _, path in found]


def latest_dataset_dir(base: str | Path) -> Path | None:
    """返回最新一份数据集目录；找不到时返回 None。"""
    candidates = [
        path for path in list_dataset_dirs(base) if is_dataset_dir(path)
    ]
    if candidates:
        return candidates[-1]
    # 兼容早期"直接放在 datasets/"下的布局。
    root = Path(base)
    return root if is_dataset_dir(root) else None


def resolve_dataset_dir(base: str | Path) -> Path:
    """解析读命令要用的数据集目录。

    - 传入的路径本身就是数据集目录（含 samples.jsonl）时原样使用；
    - 否则在日期目录里取最新一份；
    - 都没有时抛出带指引的错误（写命令应改用 :func:`dated_run_dir`）。
    """
    root = Path(base)
    if is_dataset_dir(root):
        return root

    latest = latest_dataset_dir(root)
    if latest is not None:
        return latest

    raise FileNotFoundError(
        f"{root} 下没有找到数据集（缺少 {DATASET_MARKER}）。"
        "请先运行 build 生成，或用 --data 指定具体日期目录。"
    )
