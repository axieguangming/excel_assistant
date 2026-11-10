"""在同一批样本上配对比较两个版本，用 McNemar 精确检验判断差异是否显著。

为什么需要它
------------
`compare_versions.py` 只报「通过率 +N 点」。而在同一批留出样本上评两个模型时，
两个通过率**并不独立**：两个模型会在同一批简单题上同时通过、同一批难题上同时
失败。用两比例检验（或正态近似）会严重低估检出力。

实测差异有多大（396 条留出集，v4 → DPO）：

    非配对两比例检验   +2.2 点   p = 0.39（看起来完全没效果）
    McNemar 配对检验   +2.2 点   p = 0.023（显著）

同一份数据、同一个差异，结论从「无效」变成「有效」。差别全在检验方法：
真正携带信息的是**不一致的样本**（一个通过、一个失败），本例只有 13 条
（v4 独有通过 2 条，DPO 独有通过 11 条）。

用法
----
    python pipeline/compare_paired.py --baseline v4 --candidate dpo
    python pipeline/compare_paired.py --baseline v4 --candidate dpo --kinds execute
    python pipeline/compare_paired.py --baseline v4 --candidate dpo2 --prefix exp2

输出包含每组的 2×2 列联表、精确 p 值与不一致样本清单（便于逐条看改动）。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path, *, kind: str | None = None) -> dict[str, bool]:
    """读取评测结果，返回 {sample_id: 是否通过}。"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    outcome: dict[str, bool] = {}
    for result in payload["results"]:
        if kind and result.get("kind") != kind:
            continue
        outcome[result["sample_id"]] = bool(result.get("passed"))
    return outcome


def mcnemar_exact(only_baseline: int, only_candidate: int) -> float:
    """McNemar 精确检验（双侧）。

    只在不一致样本上做二项检验：若两个模型等价，那么
    「只有 A 通过」和「只有 B 通过」应当各占一半（p = 0.5）。
    """
    discordant = only_baseline + only_candidate
    if discordant == 0:
        return 1.0
    tail = min(only_baseline, only_candidate)
    probability = sum(
        math.comb(discordant, index) for index in range(tail + 1)
    ) / (2**discordant)
    return min(1.0, probability * 2)


def wilson_interval(passed: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 置信区间——小样本比例比正态近似稳。"""
    if total == 0:
        return (0.0, 0.0)
    phat = passed / total
    denominator = 1 + z * z / total
    centre = (phat + z * z / (2 * total)) / denominator
    margin = (
        z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total)) / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def compare(
    baseline: dict[str, bool],
    candidate: dict[str, bool],
    *,
    label: str,
) -> dict:
    """配对比较，返回统计量与不一致样本清单。"""
    shared = [key for key in baseline if key in candidate]
    both = [key for key in shared if baseline[key] and candidate[key]]
    gained = [key for key in shared if candidate[key] and not baseline[key]]
    lost = [key for key in shared if baseline[key] and not candidate[key]]
    neither = [key for key in shared if not baseline[key] and not candidate[key]]

    total = len(shared)
    baseline_pass = len(both) + len(lost)
    candidate_pass = len(both) + len(gained)

    return {
        "label": label,
        "n": total,
        "both": len(both),
        "neither": len(neither),
        "gained": len(gained),
        "lost": len(lost),
        "baseline_pass": baseline_pass,
        "candidate_pass": candidate_pass,
        "baseline_rate": baseline_pass / total if total else 0.0,
        "candidate_rate": candidate_pass / total if total else 0.0,
        "delta": (candidate_pass - baseline_pass) / total if total else 0.0,
        "p_value": mcnemar_exact(len(lost), len(gained)),
        "gained_ids": sorted(gained),
        "lost_ids": sorted(lost),
    }


def print_block(stats: dict) -> None:
    base_lo, base_hi = wilson_interval(stats["baseline_pass"], stats["n"])
    cand_lo, cand_hi = wilson_interval(stats["candidate_pass"], stats["n"])
    discordant = stats["gained"] + stats["lost"]
    print(f"\n{stats['label']}（{stats['n']} 条）")
    print(
        f"  基线 {stats['baseline_pass']}/{stats['n']} = {stats['baseline_rate'] * 100:.1f}%"
        f"（95% CI {base_lo * 100:.1f}–{base_hi * 100:.1f}%）"
    )
    print(
        f"  对比 {stats['candidate_pass']}/{stats['n']} = {stats['candidate_rate'] * 100:.1f}%"
        f"（95% CI {cand_lo * 100:.1f}–{cand_hi * 100:.1f}%）"
    )
    print(f"  差值 {stats['delta'] * 100:+.1f} 点")
    print(
        f"  配对表：都对 {stats['both']}，都错 {stats['neither']}，"
        f"仅基线对 {stats['lost']}，仅对比对 {stats['gained']}"
    )
    verdict = "显著" if stats["p_value"] < 0.05 else "不显著"
    print(
        f"  McNemar 精确检验：不一致 {discordant} 条，"
        f"p = {stats['p_value']:.4f}（{verdict}，α=0.05）"
    )
    if discordant and discordant <= 20:
        if stats["gained_ids"]:
            print(f"    新通过：{', '.join(stats['gained_ids'])}")
        if stats["lost_ids"]:
            print(f"    回退：{', '.join(stats['lost_ids'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description="配对比较两个评测版本（McNemar）")
    parser.add_argument("--baseline", required=True, help="作为基线的版本名，如 v4")
    parser.add_argument("--candidate", required=True, help="要比较的版本名，如 dpo")
    parser.add_argument("--directory", default="build", help="评测结果目录")
    parser.add_argument("--prefix", default="exp", help="文件名前缀，如 exp / exp2")
    parser.add_argument(
        "--groups", default="a,b", help="要比较的组（逗号分隔）；留空表示全部"
    )
    parser.add_argument(
        "--kinds", default="", help="只看某类样本（execute / clarify）；留空表示全部"
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    kind = args.kinds.strip() or None

    baseline_all: dict[str, bool] = {}
    candidate_all: dict[str, bool] = {}
    stats_all: list[dict] = []

    for group in groups:
        base_path = directory / f"{args.prefix}-{group}-{args.baseline}.json"
        cand_path = directory / f"{args.prefix}-{group}-{args.candidate}.json"
        if not base_path.is_file() or not cand_path.is_file():
            print(f"跳过 {group} 组：缺少 {base_path.name} 或 {cand_path.name}")
            continue
        baseline = load(base_path, kind=kind)
        candidate = load(cand_path, kind=kind)
        stats_all.append(
            compare(baseline, candidate, label=f"{group.upper()} 组")
        )
        baseline_all.update(baseline)
        candidate_all.update(candidate)

    if not stats_all:
        print("没有可比较的评测结果")
        return 2

    scope = f"（仅 {kind} 类）" if kind else ""
    print(f"=== {args.baseline} → {args.candidate} 配对比较{scope} ===")
    for stats in stats_all:
        print_block(stats)

    if len(stats_all) > 1:
        print_block(compare(baseline_all, candidate_all, label="合计"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
