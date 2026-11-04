"""小样试跑：对单张工作表的操作链生成口语化变体，便于人工审阅质量。

用法：
    python pipeline/trial_variants.py survey-20251078 --count 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_builder.chain import canonical_chain, normalize_chain
from dataset_builder.quality import chain_quality_warnings
from dataset_builder.tables import load_tables
from dataset_builder.variants import ColloquialVariantWriter


def main() -> int:
    parser = argparse.ArgumentParser(description="口语化变体小样试跑")
    parser.add_argument("workbook", help="工作簿 ID，例如 survey-20251078")
    parser.add_argument("--count", type=int, default=4, help="每条链要几条变体")
    parser.add_argument("--data", default="datasets/2026-10-29")
    parser.add_argument("--out", default="build/trial-variants.json")
    args = parser.parse_args()

    root = Path(args.data)
    samples = [
        json.loads(line)
        for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tables = {table.table_id: table for table in load_tables(root / "tables")}
    table = tables.get(args.workbook)
    if table is None:
        print(f"找不到工作簿 {args.workbook}")
        return 2

    chains: dict[str, dict] = {}
    for sample in samples:
        if sample["workbook_id"] != args.workbook:
            continue
        steps = normalize_chain(sample["expected_actions"], max_steps=32)
        chains.setdefault(canonical_chain(steps), sample)

    writer = ColloquialVariantWriter()
    print(f"工作簿 {args.workbook}: {len(chains)} 条链 x {args.count} 变体")
    print(f"列: {[str(c) for c in table.frame.columns]}\n")

    output: list[dict] = []
    accepted = rejected = 0
    started = time.time()

    for index, sample in enumerate(chains.values(), start=1):
        steps = normalize_chain(sample["expected_actions"], max_steps=32)
        result = sample["expected_result"]
        warnings = chain_quality_warnings(
            steps,
            total_rows=int(len(table.frame)),
            columns=[str(name) for name in table.frame.columns],
        )
        if warnings:
            print(f"[{index}] {sample['sample_id']} 链质量问题：")
            for warning in warnings:
                print(f"    ! {warning}")
        try:
            report = writer.write(
                table,
                steps,
                count=args.count,
                rows=result.get("row_count"),
                columns=result.get("columns"),
                existing_questions=[sample["user_request"]],
            )
        except Exception as error:  # noqa: BLE001 - 试跑阶段需要看到失败原因
            print(f"[{index}] {sample['sample_id']} 调用失败: {type(error).__name__}: {error}")
            output.append(
                {
                    "sample_id": sample["sample_id"],
                    "steps": steps,
                    "error": f"{type(error).__name__}: {error}",
                    "variants": [],
                    "rejected": [],
                }
            )
            continue

        accepted += len(report.accepted)
        rejected += len(report.rejected)
        print(f"[{index}/{len(chains)}] {sample['sample_id']} "
              f"通过 {len(report.accepted)} / 拒绝 {len(report.rejected)}")
        for item in report.accepted:
            print(f"    ({item.style}) {item.question}")
            print(f"        -> {item.answer}")
        for item in report.rejected:
            print(f"    [拒绝] {item['reason']}: {item.get('question', '')[:70]}")
        print()

        output.append(
            {
                "sample_id": sample["sample_id"],
                "steps": steps,
                "quality_warnings": warnings,
                "original_question": sample["user_request"],
                "original_answer": sample["expected_response"],
                "variants": [item.as_dict() for item in report.accepted],
                "rejected": report.rejected,
            }
        )

    elapsed = time.time() - started
    total = accepted + rejected
    rate = (accepted / total * 100) if total else 0.0
    print("=" * 70)
    print(f"耗时 {elapsed:.1f}s（平均 {elapsed / max(1, len(chains)):.1f}s/链）")
    print(f"通过 {accepted} 条，拒绝 {rejected} 条，通过率 {rate:.1f}%")

    out_path = Path(args.out)
    out_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"结果: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
