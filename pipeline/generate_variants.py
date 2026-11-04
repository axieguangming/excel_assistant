"""对链池批量生成多样化问法。

输入：`pipeline/prepare_chains.py` 产出的 `chains-pool.json`
输出：`build/variants.json`（每条链一条记录，含原规则问法与 LLM 口语变体）

设计要点：

- **规则链保留原模板问法**：完全放弃模板会让模型丢失规范化表达能力，
  但把模板占比从 100% 降到 1/6 左右，避免退化成句式记忆。
- **LLM 链只有口语问法**：这些链本身由 LLM 提出，没有对应的模板问法。
- **断点续跑**：每处理完一条链就写回缓存，中断后重跑不会重复付费。
- **答案按实际执行结果生成**：行数与列名来自真实执行，避免答复与数据不符。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_builder.chain import canonical_chain, normalize_chain
from dataset_builder.executor import execute_chain
from dataset_builder.tables import load_tables
from dataset_builder.variants import ColloquialVariantWriter


def cache_key(chain: dict, count: int) -> str:
    return (
        f"{chain['workbook_id']}||"
        f"{canonical_chain(normalize_chain(chain['steps'], max_steps=32))}||"
        f"{count}"
    )


def load_cache(path: Path) -> dict[str, dict]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="批量生成口语化问法")
    parser.add_argument("--chains", default="build/chains-pool.json")
    parser.add_argument("--data", default="datasets/2026-10-29")
    parser.add_argument("--count", type=int, default=5, help="每条链生成几条口语变体")
    parser.add_argument("--out", default="build/variants.json")
    parser.add_argument("--cache", default="build/variants-cache.json")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 条链（小样）")
    args = parser.parse_args()

    payload = json.loads(Path(args.chains).read_text(encoding="utf-8"))
    chains = payload["chains"]
    if args.limit:
        chains = chains[: args.limit]

    tables = {
        table.table_id: table for table in load_tables(Path(args.data) / "tables")
    }
    # 原规则问法需要从样本里取，用于保留 1 条模板表述。
    original: dict[str, dict] = {}
    for path in (Path(args.data) / "splits").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sample = json.loads(line)
                original[sample["sample_id"]] = sample

    cache = load_cache(Path(args.cache))
    writer = ColloquialVariantWriter()

    results: list[dict] = []
    total_variants = 0
    total_rejected = 0
    processed = reused = 0
    started = time.time()

    for index, chain in enumerate(chains, start=1):
        workbook = chain["workbook_id"]
        table = tables.get(workbook)
        if table is None:
            continue
        steps = normalize_chain(chain["steps"], max_steps=32)
        outcome = execute_chain(table.frame, steps)
        if not outcome.ok or outcome.frame is None:
            continue
        rows = int(len(outcome.frame))
        columns = [str(column) for column in outcome.frame.columns]

        key = cache_key(chain, args.count)
        if key in cache:
            entry = cache[key]
            reused += 1
        else:
            processed += 1
            existing_questions: list[str] = []
            source_sample = chain.get("source_sample")
            if source_sample and source_sample in original:
                existing_questions.append(
                    str(original[source_sample]["user_request"])
                )
            try:
                report = writer.write(
                    table,
                    steps,
                    count=args.count,
                    rows=rows,
                    columns=columns,
                    existing_questions=existing_questions,
                )
            except Exception as error:  # noqa: BLE001 - 单链失败继续
                entry = {
                    "workbook_id": workbook,
                    "steps": steps,
                    "rows": rows,
                    "columns": columns,
                    "variants": [],
                    "rejected": [{"reason": f"{type(error).__name__}: {error}", "question": ""}],
                }
            else:
                entry = {
                    "workbook_id": workbook,
                    "steps": steps,
                    "rows": rows,
                    "columns": columns,
                    "variants": [item.as_dict() for item in report.accepted],
                    "rejected": report.rejected,
                }
            cache[key] = entry

            if processed % 20 == 0:
                Path(args.cache).write_text(
                    json.dumps(cache, ensure_ascii=False), encoding="utf-8"
                )
                elapsed = time.time() - started
                print(
                    f"  进度 {index}/{len(chains)}  新生成 {processed}  "
                    f"耗时 {elapsed:.0f}s  预计剩余 {elapsed / processed * (len(chains) - index):.0f}s",
                    flush=True,
                )

        variants = entry.get("variants") or []
        rejected = entry.get("rejected") or []
        total_variants += len(variants)
        total_rejected += len(rejected)

        # 规则链保留 1 条原模板问法，保证模型不丢失规范化表达能力。
        templates: list[dict] = []
        source_sample = chain.get("source_sample")
        if source_sample and source_sample in original:
            sample = original[source_sample]
            templates.append(
                {
                    "question": sample["user_request"],
                    "answer": sample["expected_response"],
                    "style": "template",
                    "note": "规则模板问法（保留少量）",
                }
            )

        results.append(
            {
                "workbook_id": workbook,
                "steps": steps,
                "origin": chain.get("origin", "rule"),
                "rows": rows,
                "columns": columns,
                "templates": templates,
                "variants": variants,
                "rejected": rejected,
            }
        )

    Path(args.cache).write_text(
        json.dumps(cache, ensure_ascii=False), encoding="utf-8"
    )

    styles = Counter(
        item["style"] for entry in results for item in entry["variants"]
    )
    out = {
        "stats": {
            "chains": len(results),
            "new_generated": processed,
            "reused_from_cache": reused,
            "colloquial_variants": total_variants,
            "template_variants": sum(len(e["templates"]) for e in results),
            "rejected": total_rejected,
            "accept_rate": round(
                total_variants / max(1, total_variants + total_rejected), 4
            ),
            "styles": dict(styles),
            "elapsed_seconds": round(time.time() - started, 1),
        },
        "chains": results,
    }
    Path(args.out).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    stats = out["stats"]
    print("\n=== 生成结果 ===")
    print(f"  链数              {stats['chains']}")
    print(f"  口语变体          {stats['colloquial_variants']}")
    print(f"  模板问法          {stats['template_variants']}")
    print(f"  被过滤            {stats['rejected']}（通过率 {stats['accept_rate'] * 100:.1f}%）")
    print(f"  风格分布          {stats['styles']}")
    print(f"  耗时              {stats['elapsed_seconds']}s")
    print(f"  输出              {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
