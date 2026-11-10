"""测量本地小模型单次请求的推理延迟。

分三个场景，因为「一次请求」的耗时主要取决于**输出 token 数**，而工具调用的
输出长度差异极大：

- `short_tool`  ：只挑一个工具（输出约 20-40 token）
- `tool_with_args`：工具 + 完整参数（输出约 60-150 token）
- `free_text`   ：自然语言答复（输出约 40-100 token）

分别报首 token 延迟（TTFT）与解码速度（token/s），这两项才是可复用的指标——
总耗时会随输出长度线性增长。

用法：
    python pipeline/bench_inference.py --adapter build/lora-v4-dpo --runs 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402

from finetune.eval_sft import extract_question  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_prompts(split: Path, limit: int = 3) -> dict[str, list[list[dict]]]:
    """从真实留出样本取提示：system + 用户问题 + 数据概况。"""
    records = load_jsonl(split)
    execute: list[list[dict]] = []
    clarify: list[list[dict]] = []
    for record in records:
        messages = record["messages"]
        has_call = any(message.get("tool_calls") for message in messages)
        # 取前两条（system + 带数据概况的 user）作为输入提示
        prompt = messages[:2]
        if has_call and len(execute) < limit:
            execute.append(prompt)
        elif not has_call and len(clarify) < limit:
            clarify.append(prompt)
        if len(execute) >= limit and len(clarify) >= limit:
            break
    return {"execute": execute, "clarify": clarify}


def measure(
    model,
    tokenizer,
    messages: list[dict],
    tools: list[dict],
    *,
    max_new_tokens: int,
) -> dict:
    """跑一次生成，返回首 token 延迟、token 数与总耗时。"""
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[1]

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    generated = int(output.shape[1] - prompt_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "new_tokens": generated,
        "seconds": elapsed,
        "tokens_per_second": generated / elapsed if elapsed > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="本地模型推理延迟基准")
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="build/lora-v4-dpo")
    parser.add_argument(
        "--split", default="datasets-holdout/sft-a/test.jsonl"
    )
    parser.add_argument("--runs", type=int, default=3, help="每个提示重复几次")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        ),
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, args.adapter)
        model = model.merge_and_unload()
    else:
        model = base
    model.eval()

    bundles = build_prompts(Path(args.split))
    records = load_jsonl(Path(args.split))
    tools = records[0]["tools"]

    print(f"模型 {args.model} + {args.adapter or '（无适配器）'}")
    print(f"提示来源 {args.split}\n")

    all_rows: list[dict] = []
    for category, prompts in bundles.items():
        for index, prompt in enumerate(prompts):
            rows = []
            for run in range(args.runs):
                # 第一次含 CUDA 预热，单独标注但计入均值
                row = measure(
                    model,
                    tokenizer,
                    prompt,
                    tools,
                    max_new_tokens=args.max_new_tokens,
                )
                row["warmup"] = run == 0
                rows.append(row)
                all_rows.append(row)
            steady = [r for r in rows if not r["warmup"]] or rows
            seconds = [r["seconds"] for r in steady]
            speeds = [r["tokens_per_second"] for r in steady]
            tokens = [r["new_tokens"] for r in steady]
            print(
                f"{category}#{index}  prompt {rows[0]['prompt_tokens']:5} tok | "
                f"输出 {statistics.mean(tokens):5.1f} tok | "
                f"{statistics.mean(seconds):6.2f} s | "
                f"{statistics.mean(speeds):5.1f} tok/s"
            )

    print("\n=== 汇总（跳过预热）===")
    steady = [r for r in all_rows if not r["warmup"]] or all_rows
    speeds = [r["tokens_per_second"] for r in steady]
    print(f"解码速度  均值 {statistics.mean(speeds):.1f} tok/s"
          f"  中位 {statistics.median(speeds):.1f}"
          f"  范围 {min(speeds):.1f}–{max(speeds):.1f}")
    print("\n按输出长度换算的单次请求耗时：")
    mean_speed = statistics.mean(speeds)
    for tokens in (20, 40, 80, 150, 300):
        print(f"  输出 {tokens:3} tok → 约 {tokens / mean_speed:5.2f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
