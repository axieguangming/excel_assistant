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
        generated = generate_tables(
            args.table_count,
            seed=args.seed,
            domains=args.domains or None,
        )
        index_path = save_tables(
            generated,
            out_dir / "tables",
            table_format=args.table_format,
        )
        print(f"表格索引：{index_path}")
        # 用写盘后读回的表格建样本，真值与用户上传该文件后看到的数据完全一致。
        tables = load_tables(out_dir / "tables")

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


def _cmd_resplit(args: argparse.Namespace) -> int:
    """重新划分已有样本，不重新调用模型（省时省钱）。"""
    from dataset_builder.split import coverage_summary, find_leaks, stratified_split

    root = Path(args.data)
    samples_path = root / "samples.jsonl"
    if not samples_path.is_file():
        print(f"找不到样本文件：{samples_path}", file=sys.stderr)
        return 2

    samples = read_jsonl(samples_path)
    if not samples:
        print(f"{samples_path} 中没有样本。", file=sys.stderr)
        return 2

    splits = stratified_split(
        samples,
        seed=args.seed,
        min_eval_workbooks=args.min_eval_workbooks,
    )
    leaks = find_leaks(splits)
    if leaks:
        print(f"检测到工作簿泄漏：{leaks}", file=sys.stderr)
        return 1

    for name, items in splits.items():
        path = root / "splits" / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items),
            encoding="utf-8",
        )

    summary = coverage_summary(splits)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


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
    build.add_argument(
        "--table-format",
        choices=("xlsx", "csv"),
        default="xlsx",
        help="生成表格的文件格式；默认 xlsx，与真实用户上传的 Excel 一致",
    )
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

    evaluate = subparsers.add_parser(
        "eval", help="在冻结分区上评测模型（驱动真实 Agent）"
    )
    evaluate.add_argument("--data", default="datasets", help="数据集根目录")
    evaluate.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="要评测的分区；默认 test",
    )
    evaluate.add_argument("--limit", type=int, help="只评测前 N 条，用于快速冒烟")
    evaluate.add_argument("--model", help="模型名，例如 qwen2.5-1.5b-instruct")
    evaluate.add_argument(
        "--base-url",
        help="OpenAI 兼容服务地址，例如 http://127.0.0.1:8080/v1；不填则用项目当前配置",
    )
    evaluate.add_argument("--api-key", help="本地服务通常随便填；默认 none")
    evaluate.add_argument("--temperature", type=float, default=0.0)
    evaluate.add_argument("--out", default="datasets/eval-report.json")
    evaluate.set_defaults(func=_cmd_eval)

    resplit = subparsers.add_parser(
        "resplit", help="用分层抽样重新划分已有样本（不调用模型）"
    )
    resplit.add_argument("--data", default="datasets", help="数据集根目录")
    resplit.add_argument("--seed", type=int, default=0)
    resplit.add_argument(
        "--min-eval-workbooks",
        type=int,
        default=2,
        help="val/test 至少包含的工作簿数，用于保证类别覆盖",
    )
    resplit.set_defaults(func=_cmd_resplit)

    return parser


def _cmd_eval(args: argparse.Namespace) -> int:
    from dataset_builder.eval import (
        EvalConfig,
        evaluate,
        failure_breakdown,
        format_summary,
        make_chat_model,
        write_report,
    )

    config = EvalConfig(
        split=args.split,
        limit=args.limit,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
    )
    try:
        llm = make_chat_model(config)
    except Exception as error:
        print(f"无法初始化模型：{error}", file=sys.stderr)
        return 2

    report = evaluate(data_dir=args.data, llm=llm, config=config)
    path = write_report(report, args.out)

    print(format_summary(report))
    breakdown = failure_breakdown(report)
    if breakdown:
        print("\n失败原因：")
        for reason, count in breakdown:
            print(f"  {count:>3}  {reason}")
    print(f"\n报告：{path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
