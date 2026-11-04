"""汇总三模型在留出评测集上的结果，输出 Markdown 报告片段。

约定输入文件名为 `build/exp-{组}-{模型}.json`：
    组   a / b（对应 datasets-holdout 的 sft-a / sft-b）
    模型 strong / base / v2

用法：
    python pipeline/compare_models.py --dir build --out build/exp-report.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

MODELS = {
    "strong": "强模型 deepseek-flash",
    "base": "未微调基座 Qwen2.5-1.5B",
    "v2": "微调 v2（QLoRA）",
}
GROUPS = {
    "a": "A 组（新数据值 · 训练表结构）",
    "b": "B 组（新表结构 · 3 个新业务域）",
}


def load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(items: list[dict]) -> tuple[int, int, float]:
    passed = sum(1 for item in items if item.get("passed"))
    total = len(items)
    return passed, total, (passed / total * 100 if total else 0.0)


def summarize(data: dict) -> dict:
    results = data.get("results", [])
    execute = [r for r in results if r.get("kind") == "execute"]
    clarify = [r for r in results if r.get("kind") == "clarify"]

    reasons = Counter()
    for result in results:
        if result.get("passed"):
            continue
        failure = result.get("failure") or "未知"
        reasons[failure.split("：")[0].strip() or "未知"] += 1

    equivalent = sum(1 for r in results if r.get("pass_reason"))
    # 只统计执行类的空调用：澄清类本就不该调用工具，算进去会虚高。
    empty_calls = sum(1 for r in execute if not r.get("actual_tools"))

    return {
        "overall": _rate(results),
        "execute": _rate(execute),
        "clarify": _rate(clarify),
        "reasons": reasons,
        "equivalent": equivalent,
        "empty_calls": empty_calls,
        "total": len(results),
        "by_len": _by_expected_len(execute),
    }


def _by_expected_len(execute: list[dict]) -> dict[int, tuple[int, int]]:
    """按期望步数分桶，定位「几步之后开始出错」。"""
    buckets: dict[int, list[dict]] = {}
    for result in execute:
        buckets.setdefault(len(result.get("expected_tools") or []), []).append(result)
    return {steps: (sum(1 for r in items if r.get("passed")), len(items)) for steps, items in sorted(buckets.items())}


def render(data: dict[tuple[str, str], dict], dir_path: Path) -> str:
    lines: list[str] = []

    lines.append("## 总体结果\n")
    lines.append("| 组 | 模型 | 通过 | 通过率 | 其中结果等价 | 执行类空调用 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for group in GROUPS:
        for model in MODELS:
            stats = data.get((group, model))
            if stats is None:
                lines.append(f"| {GROUPS[group]} | {MODELS[model]} | — | 缺数据 | — | — |")
                continue
            passed, total, rate = stats["overall"]
            lines.append(
                f"| {GROUPS[group]} | {MODELS[model]} | {passed}/{total} | "
                f"**{rate:.1f}%** | {stats['equivalent']} | {stats['empty_calls']} |"
            )
        lines.append("")

    lines.append("## 分会话类型\n")
    lines.append("| 组 | 模型 | 执行类 | 澄清类 |")
    lines.append("| --- | --- | --- | --- |")
    for group in GROUPS:
        for model in MODELS:
            stats = data.get((group, model))
            if stats is None:
                continue
            ep, et, er = stats["execute"]
            cp, ct, cr = stats["clarify"]
            execute_text = f"{ep}/{et} = {er:.1f}%" if et else "无样本"
            clarify_text = f"{cp}/{ct} = {cr:.1f}%" if ct else "无样本"
            lines.append(f"| {GROUPS[group]} | {MODELS[model]} | {execute_text} | {clarify_text} |")
        lines.append("")

    lines.append("## 按期望步数拆分（执行类）\n")
    lines.append("| 组 | 模型 | " + " | ".join(f"{n} 步" for n in (1, 2, 3)) + " |")
    lines.append("| --- | --- | " + " | ".join("---" for _ in (1, 2, 3)) + " |")
    for group in GROUPS:
        for model in MODELS:
            stats = data.get((group, model))
            if stats is None:
                continue
            cells = []
            for steps in (1, 2, 3):
                bucket = stats["by_len"].get(steps)
                cells.append(f"{bucket[0]}/{bucket[1]} = {bucket[0] / bucket[1] * 100:.1f}%" if bucket else "—")
            lines.append(f"| {GROUPS[group]} | {MODELS[model]} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## 失败原因分布\n")
    for group in GROUPS:
        lines.append(f"### {GROUPS[group]}\n")
        for model in MODELS:
            stats = data.get((group, model))
            if stats is None:
                continue
            if not stats["reasons"]:
                lines.append(f"- {MODELS[model]}：无失败")
                continue
            detail = "，".join(f"{name} {count}" for name, count in stats["reasons"].most_common())
            lines.append(f"- {MODELS[model]}：{detail}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="汇总三模型留出集评测结果")
    parser.add_argument("--dir", default="build", help="评测结果目录")
    parser.add_argument("--prefix", default="exp", help="结果文件名前缀")
    parser.add_argument("--out", default="", help="把 Markdown 写到该文件；留空则打印")
    args = parser.parse_args()

    dir_path = Path(args.dir)
    data: dict[tuple[str, str], dict] = {}
    for group in GROUPS:
        for model in MODELS:
            payload = load(dir_path / f"{args.prefix}-{group}-{model}.json")
            if payload is not None:
                data[(group, model)] = summarize(payload)

    if not data:
        print(f"在 {dir_path} 下没找到 {args.prefix}-{{a,b}}-{{strong,base,v2}}.json")
        return 1

    text = render(data, dir_path)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"已写入 {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
