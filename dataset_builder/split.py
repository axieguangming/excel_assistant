"""按工作簿分组划分训练/验证/测试集，避免同一表格泄漏到多个分区。"""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Iterable, Mapping

SPLIT_NAMES = ("train", "val", "test")


def grouped_split(
    samples: Iterable[Mapping],
    *,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> dict[str, list[dict]]:
    """按 workbook_id 整组分配，返回三个分区。"""
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios 必须是三个非负数。")
    total_ratio = sum(ratios)
    if total_ratio <= 0:
        raise ValueError("ratios 之和必须大于 0。")

    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        groups[str(sample["workbook_id"])].append(dict(sample))

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)

    total = sum(len(group) for group in groups.values())
    targets = {
        name: total * ratio / total_ratio
        for name, ratio in zip(SPLIT_NAMES, ratios)
    }
    counts = {name: 0 for name in SPLIT_NAMES}
    splits: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}

    remaining = group_ids
    # 组数足够时先保证每个分区非空，否则小数据集会把 val/test 挤成空集。
    if len(group_ids) >= len(SPLIT_NAMES):
        for name, group_id in zip(SPLIT_NAMES, group_ids[: len(SPLIT_NAMES)]):
            splits[name].extend(groups[group_id])
            counts[name] += len(groups[group_id])
        remaining = group_ids[len(SPLIT_NAMES) :]

    for group_id in remaining:
        # 把剩余组分配给当前缺口最大的分区，缺口相同时优先样本较少的分区。
        target = max(
            SPLIT_NAMES,
            key=lambda name: (targets[name] - counts[name], -counts[name]),
        )
        splits[target].extend(groups[group_id])
        counts[target] += len(groups[group_id])

    for name in ("train", "val", "test"):
        splits[name].sort(key=lambda sample: sample["sample_id"])
    return splits


def find_leaks(splits: Mapping[str, Iterable[Mapping]]) -> list[str]:
    """返回同时出现在多个分区的工作簿 ID。"""
    owners: dict[str, set[str]] = defaultdict(set)
    for split_name, samples in splits.items():
        for sample in samples:
            owners[str(sample["workbook_id"])].add(split_name)
    return sorted(
        workbook_id for workbook_id, names in owners.items() if len(names) > 1
    )
