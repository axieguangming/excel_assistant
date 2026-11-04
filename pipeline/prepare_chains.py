"""准备训练用操作链池：规则链质量过滤 + LLM 生成新链，合并去重。

流程：
    1. 收集规则生成器产出的链（每张表去重），按 `quality.py` 剔除语义可疑的链。
    2. 用 LLM 为每张表补生成候选链，逐条做结构校验、质量检查、执行校验、
       与已有链去重；LLM 原始输出缓存到磁盘，重跑不会重复付费。
    3. 合并两批链，输出 `build/chains-pool.json` 供问法生成阶段使用。

用法：
    python pipeline/prepare_chains.py --llm-target 500
    python pipeline/prepare_chains.py --skip-llm        # 只用规则链
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_builder.chain import (
    ChainValidationError,
    canonical_chain,
    normalize_chain,
)
from dataset_builder.executor import execute_chain
from dataset_builder.quality import chain_quality_warnings
from dataset_builder.tables import SyntheticTable, load_tables


def collect_rule_chains(
    root: Path, tables: dict[str, SyntheticTable]
) -> tuple[list[dict], list[dict]]:
    """收集规则链并按质量过滤，返回 (保留, 剔除)。"""
    samples = [
        json.loads(line)
        for line in (root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    kept: list[dict] = []
    dropped: list[dict] = []
    seen: set[str] = set()

    for sample in samples:
        workbook = sample["workbook_id"]
        table = tables.get(workbook)
        if table is None:
            continue
        try:
            steps = normalize_chain(sample["expected_actions"], max_steps=32)
        except (ChainValidationError, ValueError):
            continue
        key = f"{workbook}||{canonical_chain(steps)}"
        if key in seen:
            continue
        seen.add(key)

        warnings = chain_quality_warnings(
            steps, total_rows=int(len(table.frame))
        )
        record = {
            "workbook_id": workbook,
            "steps": steps,
            "origin": "rule",
            "source_sample": sample["sample_id"],
        }
        if warnings:
            dropped.append({**record, "warnings": warnings})
        else:
            kept.append(record)

    return kept, dropped


def select_rule_chains(
    kept: list[dict],
    limit: int,
    *,
    tables: dict[str, SyntheticTable],
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """把规则链裁到上限，优先保证「每张表至少一条」且类别分布均衡。

    规则链的业务质量明显低于 LLM 链，因此总量需要限流。但直接截断前 N 条会让
    靠后的工作簿完全没有规则样本，所以按「轮转抽样」：每轮从每张表取一条，
    在同一张表内部优先取覆盖面不同的操作链。
    """
    if limit <= 0 or len(kept) <= limit:
        return kept, []

    by_table: dict[str, list[dict]] = defaultdict(list)
    for item in kept:
        by_table[item["workbook_id"]].append(item)

    rng = random.Random(seed)
    table_ids = sorted(by_table)
    for table_id in table_ids:
        # 打乱表内顺序，让抽样覆盖不同操作链而不是固定取前几条。
        rng.shuffle(by_table[table_id])

    selected: list[dict] = []
    round_index = 0
    while len(selected) < limit:
        progressed = False
        for table_id in table_ids:
            items = by_table[table_id]
            if round_index < len(items):
                selected.append(items[round_index])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        round_index += 1

    chosen = {id(item) for item in selected}
    dropped = [item for item in kept if id(item) not in chosen]
    return selected, dropped


def llm_candidates(
    tables: dict[str, SyntheticTable],
    *,
    per_table: int,
    cache_path: Path,
    max_steps: int,
    reuse: bool = True,
    label: str = "",
) -> dict[str, list]:
    """按表调用 LLM 生成候选链，结果缓存到磁盘。

    每轮使用独立的缓存文件，因此「再生成一轮」不会覆盖上一轮的候选。
    ``reuse`` 为假时忽略缓存内容并重新请求。
    """
    cache: dict[str, list] = {}
    if reuse and cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"{label}复用缓存 {cache_path.name}：{len(cache)} 张表")

    from dataset_builder.llm import LLMChainProposer

    proposer = LLMChainProposer()
    pending = [wid for wid in tables if wid not in cache]
    print(
        f"{label}需要调用 LLM 的表：{len(pending)} 张"
        f"（每张请求 {per_table} 条）"
    )

    for index, workbook in enumerate(pending, start=1):
        table = tables[workbook]
        try:
            chains = proposer.propose(table, per_table, max_steps=max_steps)
        except Exception as error:  # noqa: BLE001 - 单表失败不应中断整批
            print(f"  [{index}/{len(pending)}] {workbook} 调用失败：{error}")
            cache[workbook] = []
            continue
        cache[workbook] = chains if isinstance(chains, list) else []
        print(
            f"  [{index}/{len(pending)}] {workbook} 返回 {len(cache[workbook])} 条候选"
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False), encoding="utf-8"
        )
    return cache


def vet_llm_chains(
    cache: dict[str, list],
    tables: dict[str, SyntheticTable],
    existing: set[str],
    *,
    max_steps: int,
) -> tuple[list[dict], list[dict]]:
    """逐条校验 LLM 候选链：结构 → 质量 → 执行 → 去重。"""
    kept: list[dict] = []
    dropped: list[dict] = []
    seen = set(existing)

    for workbook, chains in cache.items():
        table = tables.get(workbook)
        if table is None:
            continue
        for raw in chains:
            try:
                steps = normalize_chain(raw, max_steps=max_steps)
            except (ChainValidationError, ValueError) as error:
                dropped.append(
                    {
                        "workbook_id": workbook,
                        "steps": raw,
                        "origin": "llm",
                        "reasons": [f"invalid_chain: {error}"],
                    }
                )
                continue

            key = f"{workbook}||{canonical_chain(steps)}"
            if key in seen:
                dropped.append(
                    {
                        "workbook_id": workbook,
                        "steps": steps,
                        "origin": "llm",
                        "reasons": ["duplicate"],
                    }
                )
                continue

            warnings = chain_quality_warnings(
                steps, total_rows=int(len(table.frame))
            )
            if warnings:
                dropped.append(
                    {
                        "workbook_id": workbook,
                        "steps": steps,
                        "origin": "llm",
                        "reasons": warnings,
                    }
                )
                continue

            outcome = execute_chain(table.frame, steps)
            if not outcome.ok or outcome.frame is None:
                dropped.append(
                    {
                        "workbook_id": workbook,
                        "steps": steps,
                        "origin": "llm",
                        "reasons": [f"execution_failed: {outcome.error}"],
                    }
                )
                continue
            if len(outcome.frame) == 0:
                dropped.append(
                    {
                        "workbook_id": workbook,
                        "steps": steps,
                        "origin": "llm",
                        "reasons": ["empty_result"],
                    }
                )
                continue

            seen.add(key)
            kept.append(
                {
                    "workbook_id": workbook,
                    "steps": steps,
                    "origin": "llm",
                    "result_rows": int(len(outcome.frame)),
                    "result_columns": [str(c) for c in outcome.frame.columns],
                }
            )

    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description="构建训练用操作链池")
    parser.add_argument("--data", default="datasets/2026-10-29")
    parser.add_argument(
        "--no-rule",
        action="store_true",
        help="完全不用规则链，全部由 LLM 生成",
    )
    parser.add_argument(
        "--rule-target", type=int, default=200,
        help="规则链保留上限（--no-rule 时忽略）；0=不限制",
    )
    parser.add_argument("--llm-target", type=int, default=1000, help="LLM 链目标数量")
    parser.add_argument(
        "--llm-rounds", type=int, default=2,
        help="LLM 生成轮数；每轮每张表请求 --llm-per-table 条，各轮缓存独立",
    )
    parser.add_argument(
        "--llm-per-table", type=int, default=11,
        help="每轮每张表请求的候选条数",
    )
    parser.add_argument("--llm-max-steps", type=int, default=3)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument(
        "--no-reuse-llm-cache", dest="reuse_llm_cache",
        action="store_false", default=True,
        help="忽略已有候选缓存，强制重新请求（会重复付费）",
    )
    parser.add_argument("--seed", type=int, default=20251030)
    parser.add_argument("--limit-tables", type=int, default=0, help="只处理前 N 张表（小样验证）")
    parser.add_argument("--out", default="build/chains-pool.json")
    args = parser.parse_args()

    root = Path(args.data)
    tables = {table.table_id: table for table in load_tables(root / "tables")}
    if args.limit_tables:
        tables = dict(list(tables.items())[: args.limit_tables])
    print(f"工作簿 {len(tables)} 张")

    started = time.time()

    # ---------- 阶段 1：规则链（可选） ----------
    rule_kept: list[dict] = []
    rule_dropped: list[dict] = []
    if args.no_rule:
        print("\n[规则链] --no-rule：完全跳过（链全部由 LLM 生成）")
    else:
        rule_kept, rule_dropped = collect_rule_chains(root, tables)
        print(
            f"\n[规则链] 保留 {len(rule_kept)} 条，剔除 {len(rule_dropped)} 条 "
            f"（剔除率 {len(rule_dropped) / max(1, len(rule_kept) + len(rule_dropped)) * 100:.1f}%）"
        )
        reason_counter: Counter[str] = Counter()
        for item in rule_dropped:
            for warning in item["warnings"]:
                reason_counter[warning.split(":")[0]] += 1
        for reason, count in reason_counter.most_common():
            print(f"    {reason:32s} {count}")

    # ---------- 阶段 2：LLM 多轮生成 ----------
    llm_kept: list[dict] = []
    llm_dropped: list[dict] = []
    if args.skip_llm:
        print("\n[LLM 链] 已跳过")
    else:
        existing = {
            f"{item['workbook_id']}||{canonical_chain(item['steps'])}"
            for item in rule_kept
        }

        for round_index in range(1, max(1, args.llm_rounds) + 1):
            # 第一轮沿用此前的缓存文件名，后续轮次各用独立文件，
            # 这样「再生成一轮」不会覆盖上一轮的候选。
            cache_name = (
                "llm-chains-raw.json"
                if round_index == 1
                else f"llm-chains-raw-round{round_index}.json"
            )
            label = f"[LLM 第 {round_index} 轮] "
            print(f"\n{label}缓存 {cache_name}")
            cache = llm_candidates(
                tables,
                per_table=args.llm_per_table,
                cache_path=Path("build") / cache_name,
                max_steps=args.llm_max_steps,
                reuse=args.reuse_llm_cache,
                label=label,
            )
            kept, dropped = vet_llm_chains(
                cache, tables, existing, max_steps=args.llm_max_steps
            )
            print(f"  通过 {len(kept)} 条，剔除 {len(dropped)} 条")
            counter: Counter[str] = Counter()
            for item in dropped:
                for reason in item["reasons"]:
                    counter[reason.split(":")[0]] += 1
            for reason, count in counter.most_common():
                print(f"    {reason:32s} {count}")

            existing.update(
                f"{item['workbook_id']}||{canonical_chain(item['steps'])}"
                for item in kept
            )
            llm_kept.extend(kept)
            llm_dropped.extend(dropped)

            print(f"  累计 LLM 链：{len(llm_kept)} 条")
            if len(llm_kept) >= args.llm_target:
                print(f"  已达到目标 {args.llm_target} 条，停止追加轮次")
                break

        if len(llm_kept) > args.llm_target:
            # 超出目标时轮转裁剪，避免某些表被砍光。
            llm_kept, surplus = select_rule_chains(
                llm_kept,
                args.llm_target,
                tables=tables,
                seed=args.seed,
            )
            llm_dropped.extend(
                {**item, "reasons": ["over_target"]} for item in surplus
            )
            print(f"\n[LLM 链] 超出目标，轮转裁掉 {len(surplus)} 条")

    # ---------- 阶段 3：规则链限流 ----------
    rule_before = len(rule_kept)
    if args.rule_target and len(rule_kept) > args.rule_target:
        rule_kept, rule_surplus = select_rule_chains(
            rule_kept,
            args.rule_target,
            tables=tables,
            seed=args.seed,
        )
        rule_dropped.extend(
            {**item, "warnings": ["over_target: 规则链超出配比上限"]}
            for item in rule_surplus
        )
        print(f"\n[规则链] 按配比限流 {rule_before} -> {len(rule_kept)} 条")

    # ---------- 阶段 4：合并输出 ----------
    pool = [*rule_kept, *llm_kept]
    by_workbook: dict[str, int] = defaultdict(int)
    for item in pool:
        by_workbook[item["workbook_id"]] += 1

    steps_hist = Counter(len(item["steps"]) for item in pool)
    origin_hist = Counter(item["origin"] for item in pool)

    payload = {
        "data": str(root),
        "stats": {
            "total": len(pool),
            "by_origin": dict(origin_hist),
            "by_steps": dict(sorted(steps_hist.items())),
            "rule_kept": len(rule_kept),
            "rule_dropped": len(rule_dropped),
            "llm_kept": len(llm_kept),
            "llm_dropped": len(llm_dropped),
            "workbooks_with_chains": len(by_workbook),
            "avg_chains_per_workbook": round(len(pool) / max(1, len(by_workbook)), 1),
        },
        "chains": pool,
        "rejected": [*rule_dropped, *llm_dropped],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 合并结果 ===")
    print(f"  总链数            {len(pool)}")
    print(f"  规则链 / LLM 链   {len(rule_kept)} / {len(llm_kept)}")
    print(f"  覆盖工作簿        {len(by_workbook)}")
    print(f"  每表平均链数      {payload['stats']['avg_chains_per_workbook']}")
    print(f"  步数分布          {dict(sorted(steps_hist.items()))}")
    print(f"  耗时              {time.time() - started:.0f}s")
    print(f"  输出              {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
