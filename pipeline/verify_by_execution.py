"""用「执行结果是否一致」作为忠实度的最终判据。

参数级比对只能近似判断语义；真正的判据是**执行后数据是否相同**。
典型情况：链里是 ``nunique(订单号)``，模型解析成 ``count(订单号)``。
当订单号在该表中唯一时，两者数值完全相同，模型的答案就是对的。

本脚本对已保存的解析结果做离线复算：
1. 在表上执行期望链，得到期望数据
2. 执行解析链，得到实际数据
3. 忽略列名、只比数值/文本内容，逐单元格比较
4. 相同即视为忠实（记录为 result_equivalent）

用法：
    python pipeline/verify_by_execution.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from dataset_builder.executor import execute_chain
from dataset_builder.fidelity import is_semantically_equivalent
from dataset_builder.tables import load_tables


def values_equal(left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, str]:
    """只比较内容，忽略列名；形状必须一致。"""
    if left.shape != right.shape:
        return False, f"形状不同 {left.shape} vs {right.shape}"
    left_values = left.reset_index(drop=True).astype(str)
    right_values = right.reset_index(drop=True).astype(str)
    # 逐列比较（列顺序有意义，列名无意义）
    if left_values.shape[1] != right_values.shape[1]:
        return False, "列数不同"
    for index in range(left_values.shape[1]):
        if not left_values.iloc[:, index].equals(right_values.iloc[:, index]):
            return False, f"第 {index + 1} 列内容不同"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="按执行结果判断忠实度")
    parser.add_argument("--variants", default="build/variants.json")
    parser.add_argument("--fidelity", default="build/variants-fidelity.json")
    parser.add_argument("--data", default="datasets/2026-10-29")
    parser.add_argument("--out", default="build/variants-faithful-v2.json")
    args = parser.parse_args()

    variants = json.loads(Path(args.variants).read_text(encoding="utf-8"))
    fidelity = json.loads(Path(args.fidelity).read_text(encoding="utf-8"))
    tables = {
        t.table_id: t for t in load_tables(Path(args.data) / "tables")
    }

    # 收集"参数级不一致"的变体键
    suspects: dict[tuple[str, str], dict] = {}
    for item in fidelity["mismatches"]:
        key = (str(item["workbook_id"]), item["question"])
        suspects[key] = item

    # 逐条用执行结果裁决
    verdict: dict[tuple[str, str], tuple[bool, str]] = {}
    stats = Counter()
    for (workbook, question), item in suspects.items():
        table = tables.get(workbook)
        if table is None:
            verdict[(workbook, question)] = (False, "找不到表格")
            continue
        expected_steps = item.get("expected") or []
        parsed_steps = item.get("parsed") or []
        if not parsed_steps:
            verdict[(workbook, question)] = (False, "解析为空")
            continue
        if is_semantically_equivalent(expected_steps, parsed_steps):
            verdict[(workbook, question)] = (True, "参数等价")
            stats["参数等价"] += 1
            continue

        left = execute_chain(table.frame, expected_steps)
        right = execute_chain(table.frame, parsed_steps)
        if not left.ok or not right.ok:
            verdict[(workbook, question)] = (
                False,
                f"执行失败: {left.error or right.error}",
            )
            stats["执行失败"] += 1
            continue

        equal, reason = values_equal(left.frame, right.frame)
        verdict[(workbook, question)] = (equal, reason)
        stats["结果相同" if equal else "结果不同"] += 1

    # 组装最终忠实集
    chains_out = []
    kept = dropped = 0
    for chain in variants["chains"]:
        workbook = str(chain["workbook_id"])
        kept_variants = []
        for variant in chain.get("variants", []):
            key = (workbook, variant["question"])
            if key in verdict and not verdict[key][0]:
                dropped += 1
                continue
            kept_variants.append(variant)
            kept += 1
        if not kept_variants:
            continue
        chains_out.append(
            {
                "workbook_id": workbook,
                "steps": chain["steps"],
                "origin": chain.get("origin", "llm"),
                "rows": chain["rows"],
                "columns": chain["columns"],
                "templates": [],
                "variants": kept_variants,
                "rejected": [],
            }
        )

    total = kept + dropped
    payload = {
        "stats": {
            "chains": len(chains_out),
            "kept_variants": kept,
            "dropped_variants": dropped,
            "fidelity_rate": round(kept / max(1, total), 4),
            "verdict_breakdown": dict(stats),
        },
        "chains": chains_out,
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    print("=== 用执行结果裁决参数不一致的变体 ===")
    for reason, count in stats.most_common():
        print(f"  {count:5d}  {reason}")
    print()
    print(f"保留链        {len(chains_out)} / {len(variants['chains'])}")
    print(f"保留变体      {kept} / {total}")
    print(f"丢弃变体      {dropped}")
    print(f"忠实率        {payload['stats']['fidelity_rate'] * 100:.1f}%")
    print(f"输出          {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
