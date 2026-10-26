"""按工作簿分组划分训练/验证/测试集，避免同一表格泄漏到多个分区。"""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Iterable, Mapping

SPLIT_NAMES = ("train", "val", "test")


def _primary_categories(samples: Iterable[Mapping]) -> set[str]:
    categories: set[str] = set()
    for sample in samples:
        categories.update(str(item) for item in (sample.get("task_categories") or []))
    return categories


def _pick_for_coverage(
    pool: list[str],
    groups: Mapping[str, list[dict]],
    category_map: Mapping[str, set[str]],
    count: int,
    rng: random.Random,
) -> list[str]:
    """贪心挑选工作簿，优先让分区覆盖更多任务类别。

    覆盖率相同时优先样本更多的工作簿，避免 eval 分区过小。
    """
    chosen: list[str] = []
    covered: set[str] = set()
    remaining = list(pool)
    rng.shuffle(remaining)

    while remaining and len(chosen) < count:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                len(category_map[remaining[index]] - covered),
                len(groups[remaining[index]]),
            ),
        )
        winner = remaining.pop(best_index)
        chosen.append(winner)
        covered |= category_map[winner]
    return chosen


def stratified_split(
    samples: Iterable[Mapping],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    min_eval_workbooks: int = 2,
) -> dict[str, list[dict]]:
    """按工作簿分组 **且** 按任务类别分层划分。

    相比纯随机分组，这里先为 eval 分区挑选能覆盖最多任务类别的工作簿，
    避免出现"测试集只有一个工作簿、全是图表任务"这种无法解释的分数。

    - 同一工作簿仍整体进入同一分区，不产生泄漏。
    - val/test 至少各 min_eval_workbooks 个工作簿（工作簿不足时自动减少）。
    """
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios 必须是三个非负数。")
    total_ratio = sum(ratios)
    if total_ratio <= 0:
        raise ValueError("ratios 之和必须大于 0。")

    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[str(sample["workbook_id"])].append(dict(sample))
    if not groups:
        return {name: [] for name in SPLIT_NAMES}

    category_map = {
        workbook_id: _primary_categories(items)
        for workbook_id, items in groups.items()
    }
    workbook_ids = sorted(groups)
    rng = random.Random(seed)

    total_workbooks = len(workbook_ids)
    if total_workbooks < 3:
        # 工作簿太少时无法同时保证三个分区，全部用于训练并在报告里体现。
        return {
            "train": [
                sample for workbook_id in workbook_ids for sample in groups[workbook_id]
            ],
            "val": [],
            "test": [],
        }

    # 预留至少一个训练工作簿，否则模型没有可用数据。
    budget = total_workbooks - 1
    wanted_test = max(min_eval_workbooks, round(total_workbooks * ratios[2] / total_ratio))
    wanted_val = max(min_eval_workbooks, round(total_workbooks * ratios[1] / total_ratio))
    # 工作簿很少时，min_eval_workbooks 必须让位于"三个分区都非空"。
    cap = max(1, budget // 2)
    wanted_test = min(wanted_test, cap)
    wanted_val = min(wanted_val, cap)
    if wanted_test + wanted_val > budget:
        # 按比例压缩，优先保证 test（最终评测集）。
        wanted_test = min(wanted_test, budget)
        wanted_val = min(wanted_val, max(0, budget - wanted_test))
    # 保证 val/test 在可行时都不为空。
    if wanted_test == 0 and budget >= 2:
        wanted_test = 1
    if wanted_val == 0 and budget - wanted_test >= 1:
        wanted_val = 1
    # 极端情况下 test 仍为空，把 val 的名额让给 test。
    if wanted_test == 0 and budget >= 1:
        wanted_test = 1
        wanted_val = 0

    test_ids = _pick_for_coverage(workbook_ids, groups, category_map, wanted_test, rng)
    remaining = [workbook_id for workbook_id in workbook_ids if workbook_id not in test_ids]
    val_ids = _pick_for_coverage(remaining, groups, category_map, wanted_val, rng)
    train_ids = [
        workbook_id
        for workbook_id in workbook_ids
        if workbook_id not in test_ids and workbook_id not in val_ids
    ]

    splits = {
        "train": [sample for workbook_id in train_ids for sample in groups[workbook_id]],
        "val": [sample for workbook_id in val_ids for sample in groups[workbook_id]],
        "test": [sample for workbook_id in test_ids for sample in groups[workbook_id]],
    }
    for name in SPLIT_NAMES:
        splits[name].sort(key=lambda sample: sample["sample_id"])
    return splits


def coverage_summary(splits: Mapping[str, Iterable[Mapping]]) -> dict[str, dict]:
    """汇总每个分区的工作簿数、样本数与任务类别，便于核对分层效果。"""
    summary: dict[str, dict] = {}
    for name, items in splits.items():
        items = list(items)
        summary[name] = {
            "samples": len(items),
            "workbooks": len({str(item["workbook_id"]) for item in items}),
            "categories": sorted(_primary_categories(items)),
            "charts": sum(
                1 for item in items if "chart" in (item.get("task_categories") or [])
            ),
        }
    return summary


def find_leaks(splits: Mapping[str, Iterable[Mapping]]) -> list[str]:
    """返回同时出现在多个分区的工作簿 ID。"""
    owners: dict[str, set[str]] = defaultdict(set)
    for split_name, samples in splits.items():
        for sample in samples:
            owners[str(sample["workbook_id"])].add(split_name)
    return sorted(
        workbook_id for workbook_id, names in owners.items() if len(names) > 1
    )
