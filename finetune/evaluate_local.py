"""用本地模型（基座 + 可选 LoRA 适配器）跑真实 Agent 评测。

本脚本读数据集批次（需要 `samples.jsonl` 与 results CSV），并驱动**应用真实的
Agent 图**（`agent.invoke_agent`）：真正执行多步工具调用、读取工具观察结果、
在达到轮数上限时强制总结，因此结果能代表线上行为。

若只有 SFT 轨迹目录，改用 `finetune/eval_sft.py`。

用法：
    python finetune/evaluate_local.py --data datasets/2026-10-29 --split test
    python finetune/evaluate_local.py --split test --base-only      # 基座基线
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_builder.eval import EvalConfig, evaluate
from finetune.local_llm import LocalChatModel

DEFAULT_MODEL = "models/Qwen2.5-1.5B-Instruct"
DEFAULT_ADAPTER = "build/lora-out"


def load_model(model_dir: str, adapter: str | None, max_new_tokens: int):
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
    return LocalChatModel(
        model,
        tokenizer,
        max_new_tokens=max_new_tokens,
        model_name=adapter or model_dir,
    )


def _breakdown(samples, expected_steps_by_id: dict[str, int]) -> dict[str, dict]:
    """按期望步数与难度分组统计成功率，用于定位多步链的短板。"""
    groups: dict[str, dict[str, int]] = {}
    for sample in samples:
        steps = expected_steps_by_id.get(sample.sample_id)
        key = f"{steps}_step" if steps is not None else sample.difficulty
        bucket = groups.setdefault(key, {"samples": 0, "success": 0})
        bucket["samples"] += 1
        bucket["success"] += int(sample.task_success)
    for bucket in groups.values():
        bucket["success_rate"] = round(bucket["success"] / bucket["samples"], 4)
    return dict(sorted(groups.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="在真实 Agent 图上评测本地模型")
    parser.add_argument("--data", default="datasets", help="数据集目录（自动取最新批次）")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, help="只评测前 N 条")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="基座权重目录")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="LoRA 适配器目录")
    parser.add_argument("--base-only", action="store_true", help="不加载适配器，跑基座基线")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--out", help="报告输出路径")
    args = parser.parse_args()

    from dataset_builder.pipeline import read_jsonl
    from dataset_builder.workspace import resolve_dataset_dir

    root = resolve_dataset_dir(args.data)
    print(f"数据集：{root}")

    split_path = root / "splits" / f"{args.split}.jsonl"
    expected_steps = {
        str(sample["sample_id"]): len(sample.get("expected_actions") or [])
        for sample in read_jsonl(split_path)
    }

    adapter = None if args.base_only else args.adapter
    print(f"模型：{args.model} 适配器：{adapter or '（无，基座基线）'}")
    llm = load_model(args.model, adapter, args.max_new_tokens)

    report = evaluate(
        data_dir=root,
        llm=llm,
        config=EvalConfig(split=args.split, limit=args.limit),
    )

    metrics = report.metrics
    payload = report.as_dict()
    payload["metrics"]["by_expected_steps"] = _breakdown(
        report.samples, expected_steps
    )

    out_path = Path(args.out) if args.out else (
        root / f"eval-local-{args.split}-{'base' if args.base_only else 'lora'}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 总览 ===")
    for key in (
        "samples",
        "task_success_rate",
        "task_success_unordered_rate",
        "tool_sequence_match_rate",
        "first_tool_match_rate",
        "args_match_rate",
        "hit_turn_limit_rate",
        "avg_tool_steps",
        "avg_turns",
        "chart_produced_rate",
    ):
        print(f"  {key:32s} {metrics.get(key)}")

    print("\n=== 按期望步数 ===")
    for key, bucket in payload["metrics"]["by_expected_steps"].items():
        print(
            f"  {key:10s} {bucket['success']:3d}/{bucket['samples']:<3d} "
            f"= {bucket['success_rate'] * 100:5.1f}%"
        )

    print("\n=== 按任务类别 ===")
    for key, bucket in metrics.get("by_category", {}).items():
        print(
            f"  {key:18s} {bucket['success']:3d}/{bucket['samples']:<3d} "
            f"= {bucket['success_rate'] * 100:5.1f}%"
        )

    failures = payload["failures"]
    print(f"\n=== 失败 {len(failures)} 条 ===")
    counts: dict[str, int] = {}
    for item in failures:
        reason = item["failure"].split(":")[0]
        counts[reason] = counts.get(reason, 0) + 1
    for reason, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:28s} {count}")

    print(f"\n报告：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
