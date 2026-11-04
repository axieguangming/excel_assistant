"""把变体链池转换成 LoRA 微调轨迹。

输入：`pipeline/generate_variants.py` 产出的 `variants.json`
输出：`build/sft-expanded/{train,val,test}.jsonl` + `tools.json`

与 `finetune/convert.py` 的区别：数据源不是规则样本，而是「链 × 问法」的展开。
轨迹构造仍复用 `finetune.dataset.sample_to_messages`，保证 system prompt、
工具 schema、工具观察渲染与线上完全一致。

`--reuse-split-from` 可以沿用已有数据集的划分（按工作簿），使新旧模型在同一
测试集上可比。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_builder.chain import canonical_chain, normalize_chain
from dataset_builder.executor import execute_chain
from dataset_builder.tables import load_tables
from dataset_builder.templates import categories_for
from finetune.dataset import sample_to_messages, tool_definitions, write_jsonl


def _stable_id(*parts: object) -> str:
    """确定性的 8 位样本编号。

    不能用内置 `hash()`：它默认按进程随机化（PYTHONHASHSEED），同一份输入
    每次生成都会得到不同的 sample_id，既无法对比两次构建的差异，也破坏可复现性。
    """
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return f"{int(digest[:8], 16) % 10**8:08d}"


def load_workbook_splits(root: Path) -> dict[str, str]:
    """从已有数据集读取「工作簿 -> 分区」映射。"""
    mapping: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = root / "splits" / f"{split}.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            mapping[str(sample["workbook_id"])] = split
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="变体链池 -> SFT 轨迹")
    parser.add_argument("--variants", default="build/variants.json")
    parser.add_argument("--data", default="datasets/2026-10-29")
    parser.add_argument("--out", default="build/sft-expanded")
    parser.add_argument(
        "--reuse-split-from",
        default="datasets/2026-10-29",
        help="沿用该数据集的工作簿划分；留空则按 80/10/10 重新划分",
    )
    parser.add_argument("--max-per-chain", type=int, default=0, help="每条链最多保留几条问法")
    parser.add_argument(
        "--no-templates",
        action="store_true",
        help="不使用规则模板问法（纯 LLM 链池应开启）",
    )
    parser.add_argument(
        "--limit-chains", type=int, default=0, help="只处理前 N 条链（小样验证）"
    )
    parser.add_argument(
        "--clarifications",
        default="",
        help="澄清样本 JSON；为澄清类样本构造「不调用工具、直接反问」的轨迹",
    )
    parser.add_argument(
        "--max-clarify-ratio",
        type=float,
        default=0.03,
        help="澄清样本占训练集的上限比例；过高的反问会让模型对什么都说\"请补充\"",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.variants).read_text(encoding="utf-8"))
    chain_list = payload["chains"]
    if args.limit_chains:
        chain_list = chain_list[: args.limit_chains]
    tables = {
        table.table_id: table for table in load_tables(Path(args.data) / "tables")
    }

    workbook_split = (
        load_workbook_splits(Path(args.reuse_split_from))
        if args.reuse_split_from
        else {}
    )

    # ---------- 展开为样本 ----------
    samples: list[dict] = []
    skipped_chains = 0
    for entry in chain_list:
        workbook = entry["workbook_id"]
        table = tables.get(workbook)
        if table is None:
            skipped_chains += 1
            continue
        steps = normalize_chain(entry["steps"], max_steps=32)
        outcome = execute_chain(table.frame, steps)
        if not outcome.ok or outcome.frame is None:
            skipped_chains += 1
            continue

        tasks = categories_for(steps)
        # 纯 LLM 链没有对应的规则模板问法，因此只用口语变体。
        questions = list(entry.get("variants", []))
        if args.no_templates:
            questions = list(entry.get("variants", []))
        elif entry.get("templates"):
            questions = [*entry["templates"], *questions]
        if args.max_per_chain:
            questions = questions[: args.max_per_chain]
        for index, item in enumerate(questions, start=1):
            samples.append(
                {
                    "workbook_id": workbook,
                    "sheet": table.sheet,
                    "user_request": item["question"],
                    "expected_response": item["answer"],
                    "expected_actions": steps,
                    "task_categories": tasks,
                    "style": item.get("style", ""),
                    "origin": entry.get("origin", "rule"),
                    "chain_key": canonical_chain(steps),
                    "sample_id": f"{workbook}-{_stable_id(canonical_chain(steps), index)}",
                }
            )

    # ---------- 澄清样本 ----------
    clarification_count = 0
    if args.clarifications:
        clarify_path = Path(args.clarifications)
        if not clarify_path.is_file():
            print(f"警告：找不到澄清样本 {clarify_path}")
        else:
            clarify = json.loads(clarify_path.read_text(encoding="utf-8"))
            usable = [item for item in clarify if item["workbook_id"] in tables]

            # 按上限比例抽样：澄清样本按类型分层轮转，避免只剩一两类。
            budget = int(len(samples) * args.max_clarify_ratio / (1 - args.max_clarify_ratio))
            if budget and len(usable) > budget:
                import random

                rng = random.Random(0)
                by_kind: dict[str, list[dict]] = defaultdict(list)
                for item in usable:
                    by_kind[item.get("kind", "unknown")].append(item)
                for items in by_kind.values():
                    rng.shuffle(items)
                picked: list[dict] = []
                index = 0
                kinds = sorted(by_kind)
                while len(picked) < budget:
                    progressed = False
                    for kind in kinds:
                        if index < len(by_kind[kind]) and len(picked) < budget:
                            picked.append(by_kind[kind][index])
                            progressed = True
                    if not progressed:
                        break
                    index += 1
                # 保序，便于复现
                usable = sorted(picked, key=lambda item: (item["workbook_id"], item.get("kind", "")))
                print(
                    f"澄清样本按 {args.max_clarify_ratio * 100:.0f}% 上限抽样："
                    f"{len(clarify)} -> {len(usable)}"
                )

            for item in usable:
                samples.append(
                    {
                        "workbook_id": item["workbook_id"],
                        "sheet": tables[item["workbook_id"]].sheet,
                        "user_request": item["user_request"],
                        "expected_response": item["expected_response"],
                        "expected_actions": [],
                        "task_categories": item.get("task_categories") or ["clarify"],
                        "style": item.get("style", "clarify"),
                        "origin": "clarify",
                        "chain_key": "",
                        "sample_id": (
                            f"{item['workbook_id']}-clarify-"
                            f"{_stable_id(item['user_request'], item['expected_response'])}"
                        ),
                    }
                )
                clarification_count += 1

    # ---------- 划分 ----------
    if workbook_split:
        groups: dict[str, list[dict]] = defaultdict(list)
        for sample in samples:
            split = workbook_split.get(sample["workbook_id"], "train")
            groups[split].append(sample)
        splits = {
            "train": groups.get("train", []),
            "val": groups.get("val", []),
            "test": groups.get("test", []),
        }
    else:
        from dataset_builder.split import stratified_split

        splits = stratified_split(samples, seed=0)

    # ---------- 构造轨迹 ----------
    tools = tool_definitions()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    stats: dict[str, dict] = {}
    for split, items in splits.items():
        records = []
        for sample in items:
            table = tables[sample["workbook_id"]]
            messages = sample_to_messages(sample, table.frame)
            records.append(
                {
                    "sample_id": sample["sample_id"],
                    "tools": tools,
                    "messages": messages,
                }
            )
        write_jsonl(records, out_dir / f"{split}.jsonl")
        total += len(records)
        multi = sum(
            1
            for record in records
            if sum(1 for message in record["messages"] if message.get("tool_calls")) > 1
        )
        styles = Counter(
            sample["style"] for sample in items
        )
        workbooks = len({sample["workbook_id"] for sample in items})
        stats[split] = {
            "records": len(records),
            "multi_step": multi,
            "workbooks": workbooks,
            "styles": dict(styles),
        }
        print(
            f"{split:<6} {len(records):>5} 条（多步 {multi}，工作簿 {workbooks}）"
            f" -> {out_dir / f'{split}.jsonl'}"
        )

    (out_dir / "tools.json").write_text(
        json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n合计 {total} 条；跳过的链 {skipped_chains} 条")
    if clarification_count:
        print(f"其中澄清样本 {clarification_count} 条")
    if workbook_split:
        print(f"划分沿用 {args.reuse_split_from}（保证与旧模型同测试集）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
