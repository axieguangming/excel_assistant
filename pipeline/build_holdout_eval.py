"""构建留出评测集：两批独立于训练数据的表与任务。

两批的强度不同：

- **A 组（新数据值，沿用训练表结构）**：用训练域 + 全新随机种子。
  数据值、空值位置、重复行全部不同，但列名与类型一致。
  能回答「78.8% 里有多少来自对训练数据的拟合」。

- **B 组（新表结构）**：用 `HOLDOUT_DOMAINS` 里三个从未出现在训练中的业务域，
  字段词汇刻意避开训练集。能回答「遇到没见过的业务表还能不能用」。

每组产出两类样本：

- **执行类**：LLM 生成操作链 → 质量校验 → 真实执行 → 生成口语化问法 → 忠实度校验
- **澄清类**：对同一批表检测可验证歧义 → 生成「应当先反问」的问法

两批都输出为 SFT 轨迹格式，因此可直接用 `finetune/eval_sft.py` 评测。

产物布局：

    datasets-holdout/
    ├── tables-a/                 A 组表格 xlsx + index.json
    ├── tables-b/                 B 组表格 xlsx + index.json
    ├── sft-a/{train,test}.jsonl  A 组轨迹
    ├── sft-b/{train,test}.jsonl  B 组轨迹
    └── report.json               构建报告

用法：
    python pipeline/build_holdout_eval.py --chains-per-table 8 --clarify-per-table 2
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# 复用澄清问法生成逻辑（pipeline 不是包，按目录加入路径）。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_builder.ambiguity import detect  # noqa: E402
from dataset_builder.chain import (  # noqa: E402
    ChainValidationError,
    canonical_chain,
    normalize_chain,
)
from dataset_builder.executor import execute_chain  # noqa: E402
from dataset_builder.fidelity import is_semantically_equivalent  # noqa: E402
from dataset_builder.llm import (  # noqa: E402
    DatasetLLM,
    LLMChainProposer,
    LLMQuestionParser,
)
from dataset_builder.quality import chain_quality_warnings  # noqa: E402
from dataset_builder.tables import (  # noqa: E402
    DOMAINS,
    HOLDOUT_DOMAINS,
    SyntheticTable,
    load_tables,
    save_tables,
)
from dataset_builder.tables import generate_tables  # noqa: E402
from dataset_builder.variants import ColloquialVariantWriter  # noqa: E402
from finetune.dataset import (  # noqa: E402
    sample_to_messages,
    tool_definitions,
    write_jsonl,
)
from make_ambiguity_dataset import ask_questions, is_usable  # noqa: E402

# A 组种子：训练数据用 20251029–20251078，这里取 2026 年，确保不重叠。
SEED_GROUP_A = 20260301
SEED_GROUP_B = 20260501


def build_clarify_records(
    tables: list[SyntheticTable],
    *,
    llm: DatasetLLM,
    per_table: int,
    per_kind: int,
    cache_path: Path,
) -> tuple[list[dict], Counter]:
    """为留出表生成「应当先澄清」的样本。

    只使用可验证的歧义（`dataset_builder.ambiguity.detect`）：每个歧义都有明确的
    两个及以上选项，因此标准答复可以直接由模板渲染，不依赖 LLM 判断对错。
    """
    rejects: Counter = Counter()
    cache: dict[str, list[str]] = {}
    if cache_path.is_file():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    jobs: list[tuple[SyntheticTable, object]] = []
    for table in tables:
        found = detect(table)
        # 每张表最多取 per_table 个不同种类的歧义，避免单一表占满样本。
        chosen, seen_kinds = [], set()
        for ambiguity in found:
            if ambiguity.kind in seen_kinds:
                continue
            seen_kinds.add(ambiguity.kind)
            chosen.append(ambiguity)
            if len(chosen) >= per_table:
                break
        jobs.extend((table, ambiguity) for ambiguity in chosen)

    records: list[dict] = []
    for index, (table, ambiguity) in enumerate(jobs, start=1):
        key = f"{table.table_id}||{ambiguity.kind}"
        questions = cache.get(key)
        if questions is None:
            try:
                questions = ask_questions(llm, table, ambiguity, count=per_kind)
            except Exception as error:  # noqa: BLE001 - 单个失败不中断
                print(f"    澄清问法失败 {table.table_id}/{ambiguity.kind}：{error}")
                rejects["ask_failed"] += 1
                continue
            cache[key] = questions
            if index % 20 == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(
                    json.dumps(cache, ensure_ascii=False), encoding="utf-8"
                )
        if not questions:
            rejects["empty_questions"] += 1
            continue

        for offset, question in enumerate(questions):
            usable, reason = is_usable(ambiguity, question)
            if not usable:
                rejects[f"unusable:{reason}"] += 1
                continue
            sample = {
                "sample_id": f"{table.table_id}-c{len(records) + 1:03d}",
                "workbook_id": table.table_id,
                "user_request": question,
                "expected_response": ambiguity.render(offset),
                "expected_actions": [],
                "task_categories": ["clarify"],
            }
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "tools": tool_definitions(),
                    "messages": sample_to_messages(sample, table.frame),
                }
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return records, rejects


def build_group(
    *,
    name: str,
    domains: tuple[str, ...],
    seed: int,
    tables_per_domain: int,
    chains_per_table: int,
    max_steps: int,
    root: Path,
    proposer: LLMChainProposer,
    writer: ColloquialVariantWriter,
    parser: LLMQuestionParser,
    clarify_per_table: int,
    clarify_per_kind: int,
) -> dict:
    """生成一组表 + 链 + 问题，返回统计信息。"""
    print(f"\n{'=' * 70}\n{name}: {len(domains)} 个域 × {tables_per_domain} 张表")
    tables = generate_tables(
        tables_per_domain * len(domains),
        seed=seed,
        domains=domains,
    )
    # 每组独立目录，各自带 index.json，避免两组互相覆盖。
    tables_dir = root / f"tables-{name}"
    save_tables(tables, tables_dir, table_format="xlsx")

    stats = {
        "tables": len(tables),
        "chains_proposed": 0,
        "chains_accepted": 0,
        "questions_accepted": 0,
        "questions_faithful": 0,
        "clarify_records": 0,
        "rejects": {},
    }
    rejects: dict[str, int] = stats["rejects"]

    records: list[dict] = []
    for index, table in enumerate(tables, start=1):
        label = f"  [{index}/{len(tables)}] {table.table_id}"
        try:
            candidates = proposer.propose(
                table, chains_per_table, max_steps=max_steps
            )
        except Exception as error:  # noqa: BLE001 - 单表失败不中断
            print(f"{label} 提议失败：{error}")
            rejects["proposer_error"] = rejects.get("proposer_error", 0) + 1
            continue
        stats["chains_proposed"] += len(candidates)

        accepted = 0
        seen: set[str] = set()
        for raw in candidates:
            try:
                steps = normalize_chain(raw, max_steps=max_steps)
            except (ChainValidationError, ValueError):
                rejects["invalid_chain"] = rejects.get("invalid_chain", 0) + 1
                continue
            key = canonical_chain(steps)
            if key in seen:
                rejects["duplicate"] = rejects.get("duplicate", 0) + 1
                continue

            warnings = chain_quality_warnings(
                steps, total_rows=len(table.frame)
            )
            if warnings:
                rejects["quality"] = rejects.get("quality", 0) + 1
                continue

            outcome = execute_chain(table.frame, steps)
            if not outcome.ok or outcome.frame is None:
                rejects["execution_failed"] = rejects.get("execution_failed", 0) + 1
                continue
            if len(outcome.frame) == 0:
                rejects["empty_result"] = rejects.get("empty_result", 0) + 1
                continue

            # 生成一条口语化问题
            rows = int(len(outcome.frame))
            columns = [str(column) for column in outcome.frame.columns]
            try:
                report = writer.write(
                    table, steps, count=1, rows=rows, columns=columns
                )
            except Exception:  # noqa: BLE001
                rejects["question_error"] = rejects.get("question_error", 0) + 1
                continue
            if not report.accepted:
                rejects["question_rejected"] = rejects.get("question_rejected", 0) + 1
                continue

            variant = report.accepted[0]
            stats["questions_accepted"] += 1

            # 忠实度：把问题解析回链，执行后比结果
            try:
                payload = parser.parse(table, variant.question)
                parsed = normalize_chain(
                    payload.get("steps") if isinstance(payload, dict) else None,
                    max_steps=max_steps,
                )
            except Exception:  # noqa: BLE001
                rejects["parse_error"] = rejects.get("parse_error", 0) + 1
                continue

            equivalent = is_semantically_equivalent(steps, parsed)
            if not equivalent:
                replayed = execute_chain(table.frame, parsed)
                if not (replayed.ok and replayed.frame is not None):
                    rejects["replay_failed"] = rejects.get("replay_failed", 0) + 1
                    continue
                left = outcome.frame.reset_index(drop=True).astype(str)
                right = replayed.frame.reset_index(drop=True).astype(str)
                if left.shape != right.shape or not all(
                    left.iloc[:, i].equals(right.iloc[:, i])
                    for i in range(left.shape[1])
                ):
                    rejects["not_faithful"] = rejects.get("not_faithful", 0) + 1
                    continue

            stats["questions_faithful"] += 1
            stats["chains_accepted"] += 1
            accepted += 1
            seen.add(key)

            sample = {
                "sample_id": f"{table.table_id}-{accepted:03d}",
                "workbook_id": table.table_id,
                "user_request": variant.question,
                "expected_response": variant.answer,
                "expected_actions": steps,
                "task_categories": [],
            }
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "tools": tool_definitions(),
                    "messages": sample_to_messages(sample, table.frame),
                }
            )

        print(f"{label} 候选 {len(candidates)} → 采用 {accepted}")

    # ---- 澄清类样本 ----
    clarify_records: list[dict] = []
    if clarify_per_table > 0:
        print(f"  生成澄清样本（每表最多 {clarify_per_table} 个歧义点）")
        clarify_records, clarify_rejects = build_clarify_records(
            tables,
            llm=DatasetLLM(),
            per_table=clarify_per_table,
            per_kind=clarify_per_kind,
            cache_path=root / "cache" / f"clarify-{name}.json",
        )
        records.extend(clarify_records)
        stats["clarify_records"] = len(clarify_records)
        for reason, count in clarify_rejects.items():
            rejects[f"clarify:{reason}"] = rejects.get(f"clarify:{reason}", 0) + count
        print(f"  澄清样本 {len(clarify_records)} 条")

    out_dir = root / f"sft-{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(records, out_dir / "train.jsonl")
    # 评测脚本按 split 读文件；这里把同一份轨迹也作为 test 分区，便于直接评测。
    write_jsonl(records, out_dir / "test.jsonl")
    stats["records"] = len(records)
    print(f"  {name} 产出 {len(records)} 条轨迹 → {out_dir}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="构建留出评测集")
    parser.add_argument("--out", default="datasets-holdout")
    parser.add_argument("--tables-per-domain", type=int, default=3)
    parser.add_argument("--chains-per-table", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument(
        "--groups",
        default="a,b",
        help="要构建的组：a=新数据值（训练结构），b=新表结构",
    )
    parser.add_argument(
        "--only-domains",
        default="",
        help="只处理指定域（逗号分隔），用于小样验证",
    )
    parser.add_argument(
        "--clarify-per-table",
        type=int,
        default=2,
        help="每张表生成多少个歧义点的澄清样本；0 表示不生成",
    )
    parser.add_argument(
        "--clarify-per-kind",
        type=int,
        default=1,
        help="每个歧义点生成几条问法",
    )
    args = parser.parse_args()

    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    groups = {item.strip().lower() for item in args.groups.split(",") if item.strip()}
    only = {item.strip() for item in args.only_domains.split(",") if item.strip()}

    proposer = LLMChainProposer()
    writer = ColloquialVariantWriter()
    qparser = LLMQuestionParser()

    started = time.time()
    report: dict = {"groups": {}}

    if "a" in groups:
        domains = tuple(d for d in DOMAINS if not only or d in only)
        if domains:
            report["groups"]["a"] = build_group(
                name="a",
                domains=domains,
                seed=SEED_GROUP_A,
                tables_per_domain=args.tables_per_domain,
                chains_per_table=args.chains_per_table,
                max_steps=args.max_steps,
                root=root,
                proposer=proposer,
                writer=writer,
                parser=qparser,
                clarify_per_table=args.clarify_per_table,
                clarify_per_kind=args.clarify_per_kind,
            )

    if "b" in groups:
        domains = tuple(d for d in HOLDOUT_DOMAINS if not only or d in only)
        if domains:
            report["groups"]["b"] = build_group(
                name="b",
                domains=domains,
                seed=SEED_GROUP_B,
                tables_per_domain=args.tables_per_domain,
                chains_per_table=args.chains_per_table,
                max_steps=args.max_steps,
                root=root,
                proposer=proposer,
                writer=writer,
                parser=qparser,
                clarify_per_table=args.clarify_per_table,
                clarify_per_kind=args.clarify_per_kind,
            )

    report["elapsed_seconds"] = round(time.time() - started, 1)
    (root / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 70}\n=== 汇总 ===")
    for name, stats in report["groups"].items():
        print(
            f"  {name} 组：表 {stats['tables']}，链 {stats['chains_accepted']}，"
            f"轨迹 {stats['records']}，剔除 {stats['rejects']}"
        )
    print(f"  耗时 {report['elapsed_seconds']}s")
    print(f"  输出 {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
