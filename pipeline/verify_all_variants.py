"""对全量变体做忠实度校验（反向解析 + 语义比对）。

5675 条变体串行校验约需 2.4 小时，因此用线程池并发。LLM 调用是网络 IO 密集，
并发 8 不会打满本地 CPU，也不会触发常见的速率限制。

用法：
    python pipeline/verify_all_variants.py --workers 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_builder.chain import chains_equivalent, normalize_chain
from dataset_builder.fidelity import is_semantically_equivalent
from dataset_builder.llm import LLMQuestionParser
from dataset_builder.tables import load_tables


def verify_one(parser, table, expected, variant) -> dict:
    question = variant["question"]
    record = {
        "workbook_id": None,
        "question": question,
        "style": variant.get("style", ""),
        "ok": False,
        "reason": "",
        "expected": expected,
        "parsed": None,
    }
    try:
        payload = parser.parse(table, question)
        steps = payload.get("steps") if isinstance(payload, dict) else None
        parsed = normalize_chain(steps, max_steps=32)
    except Exception as error:  # noqa: BLE001 - 单条失败不影响整批
        record["reason"] = f"解析失败: {type(error).__name__}: {error}"
        return record

    record["parsed"] = parsed
    if chains_equivalent(parsed, expected):
        record["ok"] = True
        record["strict"] = True
    elif is_semantically_equivalent(expected, parsed):
        record["ok"] = True
        record["strict"] = False
    else:
        record["reason"] = "解析出的链不一致"
    return record


def main() -> int:
    parser_args = argparse.ArgumentParser(description="全量变体忠实度校验")
    parser_args.add_argument("--variants", default="build/variants.json")
    parser_args.add_argument("--data", default="datasets/2026-10-29")
    parser_args.add_argument("--workers", type=int, default=8)
    parser_args.add_argument("--out", default="build/variants-fidelity.json")
    parser_args.add_argument("--limit", type=int, default=0, help="只校验前 N 条链")
    args = parser_args.parse_args()

    payload = json.loads(Path(args.variants).read_text(encoding="utf-8"))
    chains = payload["chains"]
    if args.limit:
        chains = chains[: args.limit]

    tables = {
        table.table_id: table for table in load_tables(Path(args.data) / "tables")
    }

    jobs: list[tuple] = []
    for entry in chains:
        table = tables.get(entry["workbook_id"])
        if table is None:
            continue
        expected = normalize_chain(entry["steps"], max_steps=32)
        for variant in entry["variants"]:
            jobs.append((entry["workbook_id"], table, expected, variant))

    print(f"待校验变体 {len(jobs)} 条，并发 {args.workers}", flush=True)

    parser = LLMQuestionParser()
    lock = threading.Lock()
    results: list[dict] = []
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(verify_one, parser, table, expected, variant): workbook
            for workbook, table, expected, variant in jobs
        }
        for index, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            record["workbook_id"] = futures[future]
            with lock:
                results.append(record)
            if index % 200 == 0:
                elapsed = time.time() - started
                rate = index / elapsed
                print(
                    f"  {index}/{len(jobs)}  已用 {elapsed:.0f}s  "
                    f"速度 {rate:.1f} 条/s  剩余 {(len(jobs) - index) / max(rate, 0.01):.0f}s",
                    flush=True,
                )

    ok = [item for item in results if item["ok"]]
    strict = [item for item in ok if item.get("strict")]
    mismatches = [item for item in results if not item["ok"]]
    rate = len(ok) / max(1, len(results))

    print("\n=== 忠实度 ===")
    print(f"  总数        {len(results)}")
    print(f"  忠实        {len(ok)}（{rate * 100:.1f}%）")
    print(f"    严格一致  {len(strict)}")
    print(f"    语义等价  {len(ok) - len(strict)}")
    print(f"  不一致      {len(mismatches)}")

    reason_counts: dict[str, int] = {}
    for item in mismatches:
        key = item["reason"].split(":")[0]
        reason_counts[key] = reason_counts.get(key, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:24s} {count}")

    print("\n不一致样例（最多 10 条）:")
    for item in mismatches[:10]:
        print(f"  [{item['workbook_id']}] {item['question']}")
        print(f"      {item['reason']}")

    Path(args.out).write_text(
        json.dumps(
            {
                "total": len(results),
                "ok": len(ok),
                "strict": len(strict),
                "equivalent_only": len(ok) - len(strict),
                "rate": round(rate, 4),
                "elapsed_seconds": round(time.time() - started, 1),
                "mismatches": mismatches,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n报告: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
