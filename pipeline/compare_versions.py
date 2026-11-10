"""对比多个微调版本在同一留出集上的表现，定位每处改动带来的收益。

用法：
    python pipeline/compare_versions.py --versions v2,v3,v4
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable

GROUPS = {"a": "A 组（新数据值·训练表结构）", "b": "B 组（新表结构·新业务域）"}


def _failed(result: dict) -> bool:
    return not result.get("passed")


def _expected_last(result: dict, tool: str) -> bool:
    tools = result.get("expected_tools") or []
    return result.get("kind") == "execute" and bool(tools) and tools[-1] == tool


def _missing_step(result: dict) -> bool:
    """失败的执行类样本里，实际步骤少于期望。"""
    return (
        result.get("kind") == "execute"
        and _failed(result)
        and len(result.get("actual_tools") or []) < len(result.get("expected_tools") or [])
    )


def _self_reported_field_error(result: dict) -> bool:
    """失败的执行类样本里，模型自述「字段找不到」。"""
    answer = result.get("answer") or ""
    markers = ("没找到", "找不到", "没查到", "没查找到", "漏看", "漏了")
    return result.get("kind") == "execute" and _failed(result) and any(m in answer for m in markers)


def _phantom_chart(result: dict) -> bool:
    """期望画图、实际没调 create_chart，却声称已画好。"""
    answer = result.get("answer") or ""
    return (
        result.get("kind") == "execute"
        and "create_chart" in (result.get("expected_tools") or [])
        and "create_chart" not in (result.get("actual_tools") or [])
        and any(w in answer for w in ("画好", "已生成", "图表已", "画完", "生成了"))
    )


def _clarify(result: dict) -> bool:
    return result.get("kind") == "clarify"


# (名称, 判定函数, 指标类型)。rate = 子集内通过率；count = 命中条数
FOCUS: list[tuple[str, Callable[[dict], bool], str]] = [
    ("以 create_chart 结尾（通过率）", lambda r: _expected_last(r, "create_chart"), "rate"),
    ("以 describe_data 结尾（通过率）", lambda r: _expected_last(r, "describe_data"), "rate"),
    ("澄清类（通过率）", _clarify, "rate"),
    ("执行类失败（条数）", lambda r: r.get("kind") == "execute" and _failed(r), "count"),
    ("其中缺少步骤（条数）", _missing_step, "count"),
    ("其中自述字段找不到（条数）", _self_reported_field_error, "count"),
    ("其中声称画好未调用（条数）", _phantom_chart, "count"),
]


def load(directory: Path, group: str, version: str) -> list[dict] | None:
    path = directory / f"exp-{group}-{version}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def rate(results: list[dict]) -> tuple[int, int, float]:
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)
    return passed, total, (passed / total * 100 if total else 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="对比多个微调版本")
    parser.add_argument("--dir", default="build")
    parser.add_argument("--versions", default="v2,v3,v4")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    directory = Path(args.dir)
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]

    available: dict[tuple[str, str], list[dict]] = {}
    for group in GROUPS:
        for version in versions:
            results = load(directory, group, version)
            if results is not None:
                available[(group, version)] = results

    if not available:
        print("没有可用的结果文件")
        return 1

    lines: list[str] = ["## 总体通过率\n"]
    lines.append("| 组 | 样本数 | " + " | ".join(versions) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in versions) + " |")
    for group in GROUPS:
        sizes = {len(available[(group, v)]) for v in versions if (group, v) in available}
        if not sizes:
            continue
        cells = []
        for version in versions:
            results = available.get((group, version))
            if results is None:
                cells.append("—")
                continue
            passed, total, pct = rate(results)
            cells.append(f"**{pct:.1f}%**（{passed}/{total}）")
        lines.append(
            f"| {GROUPS[group]} | {'/'.join(map(str, sorted(sizes)))} | "
            + " | ".join(cells)
            + " |"
        )
    lines.append("")

    lines.append("## 分类型\n")
    lines.append("| 组 | 类型 | " + " | ".join(versions) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in versions) + " |")
    for group in GROUPS:
        for kind, label in (("execute", "执行类"), ("clarify", "澄清类")):
            cells = []
            for version in versions:
                results = available.get((group, version))
                if results is None:
                    cells.append("—")
                    continue
                subset = [r for r in results if r.get("kind") == kind]
                if not subset:
                    cells.append("无样本")
                    continue
                passed, total, pct = rate(subset)
                cells.append(f"{pct:.1f}%（{passed}/{total}）")
            lines.append(f"| {GROUPS[group]} | {label} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## 关注指标\n")
    for name, predicate, kind in FOCUS:
        lines.append(f"### {name}\n")
        lines.append("| 组 | " + " | ".join(versions) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in versions) + " |")
        for group in GROUPS:
            cells = []
            for version in versions:
                results = available.get((group, version))
                if results is None:
                    cells.append("—")
                    continue
                subset = [r for r in results if predicate(r)]
                if kind == "rate":
                    if not subset:
                        cells.append("无样本")
                    else:
                        passed, total, pct = rate(subset)
                        cells.append(f"{pct:.1f}%（{passed}/{total}）")
                else:
                    cells.append(f"{len(subset)} 条")
            lines.append(f"| {GROUPS[group]} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## 失败原因分布\n")
    for group in GROUPS:
        if not any((group, v) in available for v in versions):
            continue
        lines.append(f"### {GROUPS[group]}\n")
        for version in versions:
            results = available.get((group, version))
            if results is None:
                continue
            counter = Counter()
            for result in results:
                if result.get("passed"):
                    continue
                failure = result.get("failure") or "未知"
                counter[failure.split("：")[0].strip() or "未知"] += 1
            detail = "，".join(f"{k} {v}" for k, v in counter.most_common()) or "无失败"
            lines.append(f"- **{version}**：{detail}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
