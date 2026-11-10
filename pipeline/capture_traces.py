"""重跑模型在指定样本上的轨迹，抓取带参数的完整决策序列。

用途：评测结果只保存了工具**名称**，而构造 DPO 偏好对需要 `arguments`。
这里重跑并保存每一步的 (工具名, 参数) 与当时的消息上下文。

默认抓取该模型**全部执行类失败**；`--only-duplicates` 只抓重复调用类
（第一版 DPO 用这个口径，当时重复调用是主要失败形态）。

用法：
    # 第二轮：抓 DPO 之后剩下的 57 条执行类失败
    python pipeline/capture_traces.py --adapter build/lora-v4-dpo \
        --samples build/exp-a-dpo.json,build/exp-b-dpo.json \
        --out build/traces-dpo.json

    # 第一版：只抓重复调用
    python pipeline/capture_traces.py --adapter build/lora-v4 \
        --samples build/exp-a-v4.json,build/exp-b-v4.json \
        --out build/traces-v4.json --only-duplicates
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
from dataset_builder.tables import load_tables
from finetune.eval_sft import extract_question, expected_calls
from finetune.evaluate_local import load_model


def failed_executions(
    result_paths: list[Path], *, only_duplicates: bool = False
) -> list[tuple[str, str]]:
    """从评测结果里挑出失败的执行类样本，返回 (组, sample_id)。

    `only_duplicates=True` 时只保留「同一工具被调用多次」的样本——第一版 DPO
    只针对这类失败（当时占多数）。DPO 之后重复调用降到 35 条，但参数写错的
    非重复失败还有 15 条，因此默认改为收集全部执行类失败。
    """
    picked: list[tuple[str, str]] = []
    for path in result_paths:
        group = path.stem.split("-")[1]  # exp-a-dpo -> a
        payload = json.loads(path.read_text(encoding="utf-8"))
        for result in payload["results"]:
            if result.get("passed") or result.get("kind") != "execute":
                continue
            tools = result.get("actual_tools") or []
            if only_duplicates and not any(
                count >= 2 for count in Counter(tools).values()
            ):
                continue
            picked.append((group, result["sample_id"]))
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description="抓取模型决策轨迹（含参数）")
    parser.add_argument("--adapter", default="build/lora-v4-dpo")
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--samples",
        default="build/exp-a-dpo.json,build/exp-b-dpo.json",
        help="评测结果文件，逗号分隔",
    )
    parser.add_argument("--out", default="build/traces-dpo.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--only-duplicates",
        action="store_true",
        help="只抓「同一工具被调用多次」的失败（第一版 DPO 的做法）",
    )
    args = parser.parse_args()

    result_paths = [Path(p.strip()) for p in args.samples.split(",") if p.strip()]
    targets = failed_executions(
        result_paths, only_duplicates=args.only_duplicates
    )
    if args.limit:
        targets = targets[: args.limit]
    scope = "重复调用失败" if args.only_duplicates else "执行类失败"
    print(f"待抓取 {len(targets)} 条（{scope}）")

    tables_by_group = {
        group: {t.table_id: t for t in load_tables(Path(f"datasets-holdout/tables-{group}"))}
        for group in {g for g, _ in targets}
    }
    records_by_key: dict[tuple[str, str], dict] = {}
    for group in {g for g, _ in targets}:
        path = Path(f"datasets-holdout/sft-{group}/test.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            records_by_key[(group, record["sample_id"])] = record

    import excel_ops

    llm = load_model(args.model, args.adapter, 256)
    previous_llm = agent.llm
    previous_output = excel_ops.OUTPUT_DIR
    agent.llm = llm
    excel_ops.OUTPUT_DIR = str(Path("build/capture-charts"))

    traces: list[dict] = []
    missing = 0
    try:
        for index, (group, sample_id) in enumerate(targets, start=1):
            record = records_by_key.get((group, sample_id))
            if record is None:
                missing += 1
                continue
            parts = sample_id.split("-")
            workbook = "-".join(parts[:2])
            table = tables_by_group[group].get(workbook)
            if table is None:
                missing += 1
                continue

            question = extract_question(record["messages"][1]["content"])
            expected = expected_calls(record)
            outcome = agent.invoke_agent(
                {
                    "df": table.frame,
                    "df_info": "",
                    "question": question,
                    "history": [],
                    "messages": [],
                    "current_df": table.frame.copy(),
                    "plans": [],
                    "tool_steps": 0,
                    "turns": 0,
                    "result_df": None,
                    "last_result_rows": None,
                    "final_answer": "",
                    "chart_file": "",
                    "code": "",
                    "user_prompt": "",
                }
            )

            calls = []
            for raw in outcome.get("plans", []):
                try:
                    entry = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                calls.append(
                    {
                        "tool": entry.get("tool"),
                        "args": entry.get("arguments", {}),
                        "ok": bool(entry.get("ok", True)),
                    }
                )

            # 把每次调用对应的工具观察对齐进来：构造偏好对时 prompt 需要含报错文本。
            observations: list[str] = []
            for message in outcome.get("messages", []):
                if getattr(message, "type", "") == "tool":
                    observations.append(str(message.content))
            for position, call in enumerate(calls):
                call["observation"] = (
                    observations[position] if position < len(observations) else ""
                )

            traces.append(
                {
                    "group": group,
                    "sample_id": sample_id,
                    "question": question,
                    "tools": record["tools"],
                    # 模型实际看到的提示，用于重建 DPO 的 prompt
                    "user_prompt": record["messages"][1]["content"],
                    "system_prompt": record["messages"][0]["content"],
                    "expected": expected,
                    "actual": calls,
                    "final_answer": str(outcome.get("final_answer", ""))[:400],
                }
            )
            print(
                f"  [{index}/{len(targets)}] {sample_id} "
                f"期望 {len(expected)} 步 / 实际 {len(calls)} 步",
                flush=True,
            )
    finally:
        excel_ops.OUTPUT_DIR = previous_output
        agent.llm = previous_llm

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(traces, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已保存 {len(traces)} 条轨迹 → {out}（跳过 {missing}）")

    # 复现率检查
    dup_again = sum(
        1
        for trace in traces
        if any(c >= 2 for c in Counter(c["tool"] for c in trace["actual"]).values())
    )
    print(f"重跑后仍出现重复调用的：{dup_again} / {len(traces)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
