"""演示 temperature 与「整条决策链是否可重复」的关系。

单步测试会低估分歧：第一步往往是最集中的决策（例如「先填充缺失值」几乎只有一种
合理选择），即使 temperature=1 也常年一致。真正的分歧来自**多步累积**——某一步
换了一个工具或参数，后续上下文随之完全不同。

因此这里跑完整 Agent 轨迹：同一批样本、同一 temperature，跑两遍，比较工具链。

用法：
    python pipeline/demo_temperature.py --runs 2 --samples 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent  # noqa: E402
from dataset_builder.tables import load_tables  # noqa: E402
from finetune.eval_sft import extract_question  # noqa: E402


def load_samples(split: Path, tables_dir: Path, limit: int):
    """取前 N 条执行类样本，配上对应表格。"""
    tables = {t.table_id: t for t in load_tables(tables_dir)}
    picked = []
    for line in split.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not any(m.get("tool_calls") for m in record["messages"]):
            continue
        parts = record["sample_id"].split("-")
        workbook = "-".join(parts[:2])
        table = tables.get(workbook)
        if table is None:
            continue
        picked.append((record, table))
        if len(picked) >= limit:
            break
    return picked


def run_once(table, question: str) -> list[str]:
    """跑一次 Agent，返回工具名序列。"""
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
    names = []
    for raw in outcome.get("plans", []):
        try:
            names.append(json.loads(raw).get("tool"))
        except json.JSONDecodeError:
            continue
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description="temperature 与多步轨迹可重复性")
    parser.add_argument("--split", default="datasets-holdout2/sft-a/test.jsonl")
    parser.add_argument("--tables-dir", default="datasets-holdout2/tables-a")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()

    from langchain.chat_models import init_chat_model
    from dotenv import load_dotenv

    load_dotenv()
    samples = load_samples(Path(args.split), Path(args.tables_dir), args.samples)

    import excel_ops

    previous_dir = excel_ops.OUTPUT_DIR
    excel_ops.OUTPUT_DIR = str(Path("build/demo-charts"))
    previous_llm = agent.llm

    try:
        for temperature in (0.0, 1.0):
            agent.llm = init_chat_model(
                model=os.getenv("DEEPSEEK_MODEL", "").strip(),
                model_provider="openai",
                temperature=temperature,
                api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
                base_url=os.getenv("DEEPSEEK_BASE_URL", "").strip(),
            )
            print(f"\n=== temperature={temperature} ===")
            identical = 0
            for record, table in samples:
                question = extract_question(record["messages"][1]["content"])
                chains = []
                for index in range(args.runs):
                    chain = run_once(table, question)
                    chains.append(chain)
                    print(f"  {record['sample_id']} 第{index + 1}次: {chain}")
                if len({tuple(c) for c in chains}) == 1:
                    identical += 1
                    print("    -> 完全一致")
                else:
                    print("    -> **不一致**")
            print(
                f"  整链可重复: {identical}/{len(samples)}"
                f" = {identical / len(samples) * 100:.0f}%"
            )
    finally:
        excel_ops.OUTPUT_DIR = previous_dir
        agent.llm = previous_llm
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
