"""命令行入口：生成数据集与回放校验。

示例：
    python -m dataset_builder.cli build --out datasets --proposer template
    python -m dataset_builder.cli replay --samples datasets/samples.jsonl \
        --tables-dir datasets/tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from dataset_builder.pipeline import (
    BuildConfig,
    build_dataset,
    read_jsonl,
    replay_samples,
    write_build_result,
)
from dataset_builder.tables import (
    DOMAINS,
    generate_tables,
    load_tables,
    save_tables,
)


def _components(name: str):
    if name == "template":
        from dataset_builder.templates import (
            TemplateChainProposer,
            TemplateQuestionWriter,
        )

        return TemplateChainProposer(), TemplateQuestionWriter(), None

    if name == "llm":
        from dataset_builder.llm import (
            LLMChainProposer,
            LLMQuestionParser,
            LLMQuestionWriter,
        )

        parser = LLMQuestionParser()
        return (
            LLMChainProposer(),
            LLMQuestionWriter(),
            lambda table, question, feedback="": parser.parse(
                table, question, feedback
            ),
        )

    raise ValueError(f"未知的生成器：{name}")


def _cmd_build(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    if args.tables_dir:
        tables = load_tables(args.tables_dir)
    else:
        tables = generate_tables(
            args.table_count,
            seed=args.seed,
            domains=args.domains or None,
        )
        index_path = save_tables(tables, out_dir / "tables")
        print(f"表格索引：{index_path}")

    proposer, writer, parser = _components(args.proposer)
    result = build_dataset(
        tables,
        proposer=proposer,
        writer=writer,
        parser=parser,
        config=BuildConfig(
            samples_per_table=args.per_table,
            max_steps=args.max_steps,
            max_chain_attempts=args.max_attempts,
            render_charts=args.render_charts,
            verify_questions=not args.no_verify_questions,
            split_seed=args.seed,
        ),
        out_dir=out_dir,
    )
    paths = write_build_result(result, out_dir)

    print(json.dumps(result.report, ensure_ascii=False, indent=2))
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0 if result.samples else 1


def _cmd_replay(args: argparse.Namespace) -> int:
    samples = read_jsonl(args.samples)
    tables = load_tables(args.tables_dir)
    report = replay_samples(samples, {table.table_id: table for table in tables})

    print(json.dumps(
        {
            "checked": report["checked"],
            "mismatches": len(report["mismatches"]),
        },
        ensure_ascii=False,
        indent=2,
    ))
    for mismatch in report["mismatches"][:10]:
        print(json.dumps(mismatch, ensure_ascii=False))
    return 0 if not report["mismatches"] else 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset_builder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="生成表格并构建数据集")
    build.add_argument("--out", default="datasets", help="输出目录")
    build.add_argument("--table-count", type=int, default=6, help="仿真表格数量")
    build.add_argument(
        "--domains",
        nargs="*",
        choices=list(DOMAINS),
        help="限定仿真场景；默认轮转全部场景",
    )
    build.add_argument("--tables-dir", help="复用已有表格目录而不是重新生成")
    build.add_argument("--per-table", type=int, default=6, help="每张表的目标样本数")
    build.add_argument("--max-steps", type=int, default=4, help="操作链最大步数")
    build.add_argument("--max-attempts", type=int, default=12, help="每张表最多尝试的候选链")
    build.add_argument("--seed", type=int, default=20251023)
    build.add_argument(
        "--proposer",
        choices=("template", "llm"),
        default="template",
        help="template 为离线模板；llm 调用已配置的远程模型（会产生费用）",
    )
    build.add_argument("--render-charts", action="store_true", help="同时渲染图表 PNG")
    build.add_argument(
        "--no-verify-questions",
        action="store_true",
        help="跳过问题反向解析校验（样本会标记为 needs_review）",
    )
    build.set_defaults(func=_cmd_build)

    replay = subparsers.add_parser("replay", help="重新执行样本并核对结果")
    replay.add_argument("--samples", default="datasets/samples.jsonl")
    replay.add_argument("--tables-dir", default="datasets/tables")
    replay.set_defaults(func=_cmd_replay)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
