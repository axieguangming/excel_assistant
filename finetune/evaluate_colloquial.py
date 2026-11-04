"""口语化 / 歧义 / 不可完成请求的评测。

与 `dataset_builder.eval` 的区别：问题来自手工改写（`evalsets/*.json`），
而不是规则模板，用于检验模型是否只是记住了固定句式。

两类判定：

- ``equivalent``：真值沿用对应规则样本的期望操作链与结果 CSV，
  比较最终结果表是否一致。允许模型自由选择新列名。
- ``unanswerable``：现有工具无法完成。正确行为是不调用执行类工具，
  而是说明限制或追问；``describe_data`` 属于只读概览，允许调用。

用法：
    python finetune/evaluate_colloquial.py --set evalsets/colloquial_2026-10.json
    python finetune/evaluate_colloquial.py --set ... --base-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import agent
from dataset_builder.eval import _initial_state, _plan_entries, compare_frames
from dataset_builder.pipeline import read_jsonl
from dataset_builder.tables import load_tables
from finetune.local_llm import LocalChatModel

DEFAULT_MODEL = "models/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = "build/lora-out"

# 会改变数据或产出图表、因此对“无法完成”的请求不应调用的工具。
EXECUTION_TOOLS = frozenset({
    "filter_rows",
    "sort_rows",
    "aggregate_data",
    "top_n_rows",
    "deduplicate_rows",
    "fill_missing_values",
    "add_computed_column",
    "create_chart",
})


def load_model(model_dir: str, adapter: str | None, max_new_tokens: int) -> LocalChatModel:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="auto",
        dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return LocalChatModel(model, tokenizer, max_new_tokens=max_new_tokens)


def _chart_ready(url: object) -> bool:
    if not isinstance(url, str) or not url.strip():
        return False
    import excel_ops

    filename = url.rstrip("/").rsplit("/", 1)[-1]
    return (Path(excel_ops.OUTPUT_DIR) / filename).is_file()


def run_item(
    item: dict,
    *,
    tables: dict,
    samples: dict,
    dataset_root: Path,
) -> dict:
    result: dict = {
        "id": item["id"],
        "kind": item["kind"],
        "style": item.get("style", ""),
        "question": item["question"],
        "passed": False,
        "failure": "",
        "planned_tools": [],
        "tool_errors": [],
        "turns": 0,
        "answer": "",
    }

    if item["kind"] == "equivalent":
        source = samples.get(item["source_sample"])
        if source is None:
            result["failure"] = "missing_source_sample"
            return result
        table = tables.get(source["workbook_id"])
        if table is None:
            result["failure"] = "missing_table"
            return result
        artifact = dataset_root / source["expected_result"]["artifact"]
        if not artifact.is_file():
            result["failure"] = "missing_artifact"
            return result
        expected_frame = pd.read_csv(artifact, encoding="utf-8-sig")
        expects_chart = any(
            step["tool"] == "create_chart" for step in source["expected_actions"]
        )
    else:
        table = tables.get(item["workbook_id"])
        if table is None:
            result["failure"] = "missing_table"
            return result
        expected_frame = None
        expects_chart = False

    outcome = agent.invoke_agent(_initial_state(table.frame, item["question"]))
    planned = _plan_entries(outcome)
    tools = [str(entry.get("tool")) for entry in planned]
    result["planned_tools"] = tools
    # 记录被工具自身拒绝的调用：区分"模型选错"与"参数不合法"。
    result["tool_errors"] = [
        {"tool": entry.get("tool"), "error": str(entry.get("error", ""))[:200]}
        for entry in planned
        if not entry.get("ok")
    ]
    result["turns"] = int(outcome.get("turns", 0))
    result["answer"] = str(outcome.get("final_answer", ""))[:300]

    if item["kind"] == "unanswerable":
        executed = [tool for tool in tools if tool in EXECUTION_TOOLS]
        if executed:
            result["failure"] = f"强行调用执行类工具: {executed}"
        else:
            result["passed"] = True
        return result

    if expects_chart and not _chart_ready(outcome.get("chart_file")):
        result["failure"] = "no_chart" if tools else "no_tool_call"
        return result

    actual = outcome.get("result_df")
    if actual is None:
        actual = outcome.get("current_df")
    if actual is None:
        result["failure"] = "no_result"
        return result

    matched, reason = compare_frames(expected_frame, actual)
    if matched:
        result["passed"] = True
    else:
        result["failure"] = reason
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="口语化评测集")
    parser.add_argument("--set", required=True, help="评测集 JSON 路径")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", help="报告输出路径")
    args = parser.parse_args()

    payload = json.loads(Path(args.set).read_text(encoding="utf-8"))
    dataset_root = Path(payload["dataset"])
    items = payload["items"]
    if args.limit:
        items = items[: args.limit]

    tables = {t.table_id: t for t in load_tables(dataset_root / "tables")}
    samples: dict[str, dict] = {}
    for split in ("train", "val", "test"):
        path = dataset_root / "splits" / f"{split}.jsonl"
        if path.is_file():
            for sample in read_jsonl(path):
                samples[str(sample["sample_id"])] = sample

    adapter = None if args.base_only else args.adapter
    llm = load_model(args.model, adapter, args.max_new_tokens)

    # 图表写到独立目录，避免污染正式 outputs。
    import excel_ops

    previous_output = excel_ops.OUTPUT_DIR
    chart_dir = dataset_root / "colloquial_charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    excel_ops.OUTPUT_DIR = str(chart_dir)

    previous_llm = agent.llm
    agent.llm = llm
    results: list[dict] = []
    try:
        for index, item in enumerate(items, start=1):
            outcome = run_item(
                item, tables=tables, samples=samples, dataset_root=dataset_root
            )
            results.append(outcome)
            mark = "OK  " if outcome["passed"] else "FAIL"
            print(
                f"[{index}/{len(items)}] {mark} {outcome['id']:8s} "
                f"{outcome['kind']:12s} tools={outcome['planned_tools']} "
                f"{outcome['failure']}"
            )
    finally:
        excel_ops.OUTPUT_DIR = previous_output
        agent.llm = previous_llm

    def rate(subset: list[dict]) -> str:
        if not subset:
            return "n/a"
        ok = sum(1 for item in subset if item["passed"])
        return f"{ok}/{len(subset)} = {ok / len(subset) * 100:.1f}%"

    equivalent = [r for r in results if r["kind"] == "equivalent"]
    unanswerable = [r for r in results if r["kind"] == "unanswerable"]

    print("\n=== 汇总 ===")
    print(f"  equivalent   语义等价请求  {rate(equivalent)}")
    print(f"  unanswerable 不可完成请求  {rate(unanswerable)}")
    print(f"  合计                        {rate(results)}")

    print("\n=== 失败明细 ===")
    for item in results:
        if not item["passed"]:
            print(f"  {item['id']} [{item['kind']}] {item['failure']}")
            print(f"    问题: {item['question']}")
            print(f"    工具: {item['planned_tools']}")
            for error in item.get("tool_errors", []):
                print(f"    工具报错: {error['tool']} -> {error['error']}")
            print(f"    答复: {item['answer'][:160]}")

    out_path = Path(args.out) if args.out else (
        dataset_root
        / f"eval-colloquial-{'base' if args.base_only else 'lora'}.json"
    )
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "set": args.set,
                    "model": args.model,
                    "adapter": adapter,
                },
                "summary": {
                    "equivalent": rate(equivalent),
                    "unanswerable": rate(unanswerable),
                    "total": rate(results),
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n报告：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
