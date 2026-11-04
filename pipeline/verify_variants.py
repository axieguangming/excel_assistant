"""用反向解析校验口语化变体的忠实度。

对每条变体问题调用 `LLMQuestionParser`，把它解析回操作链，再与期望链比较。
这是变体能否进入训练集的真正门禁：措辞再自然，只要解析出的链不同就说明
问题改变了语义。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_builder.chain import chains_equivalent, normalize_chain
from dataset_builder.fidelity import is_semantically_equivalent
from dataset_builder.llm import LLMQuestionParser
from dataset_builder.tables import load_tables

DEFAULT_INPUT = "build/trial-variants.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="变体忠实度校验")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--data", default="datasets/2026-10-29")
    args = parser.parse_args()

    output = json.loads(Path(args.input).read_text(encoding="utf-8"))
    tables = {
        table.table_id: table for table in load_tables(Path(args.data) / "tables")
    }

    # 变体记录里没有 workbook_id，用 sample_id 前缀反推。
    sample_to_workbook = {}
    for path in (Path(args.data) / "splits").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sample = json.loads(line)
                sample_to_workbook[sample["sample_id"]] = sample["workbook_id"]

    reader = LLMQuestionParser()
    total = 0
    matched = 0
    strict_matched = 0
    equivalent_only = 0
    mismatches: list[dict] = []

    for entry in output:
        sample_id = entry["sample_id"]
        workbook = sample_to_workbook.get(sample_id)
        table = tables.get(workbook)
        if table is None:
            print(f"[跳过] {sample_id}: 找不到工作簿 {workbook}")
            continue
        expected = normalize_chain(entry["steps"], max_steps=32)

        for variant in entry["variants"]:
            total += 1
            question = variant["question"]
            try:
                payload = reader.parse(table, question)
                steps = payload.get("steps") if isinstance(payload, dict) else None
                parsed = normalize_chain(steps, max_steps=32)
            except Exception as error:  # noqa: BLE001
                mismatches.append(
                    {
                        "sample_id": sample_id,
                        "question": question,
                        "reason": f"解析失败: {type(error).__name__}: {error}",
                    }
                )
                continue

            if chains_equivalent(parsed, expected):
                matched += 1
                strict_matched += 1
            elif is_semantically_equivalent(expected, parsed):
                # 只在列名/冗余排序上有差异，结果值正确。
                matched += 1
                equivalent_only += 1
            else:
                mismatches.append(
                    {
                        "sample_id": sample_id,
                        "question": question,
                        "reason": "解析出的链不一致",
                        "expected": expected,
                        "parsed": parsed,
                    }
                )

    rate = matched / total * 100 if total else 0.0
    print(f"变体总数 {total}，忠实 {matched}，不一致 {len(mismatches)}，忠实率 {rate:.1f}%")
    print(f"  其中严格一致 {strict_matched}，仅在列名/冗余排序上等价 {equivalent_only}\n")

    for item in mismatches:
        print(f"[{item['sample_id']}] {item['question']}")
        print(f"  原因: {item['reason']}")
        if "expected" in item:
            print(f"  期望: {json.dumps(item['expected'], ensure_ascii=False)}")
            print(f"  解析: {json.dumps(item['parsed'], ensure_ascii=False)}")
        print()

    Path("build/trial-variants-fidelity.json").write_text(
        json.dumps(
            {
                "total": total,
                "matched": matched,
                "strict_matched": strict_matched,
                "equivalent_only": equivalent_only,
                "mismatches": mismatches,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
