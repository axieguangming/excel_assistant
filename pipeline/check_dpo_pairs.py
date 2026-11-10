"""DPO 训练前自检：确认偏好对里「模型确实答错」，而不是已经答对。

为什么必须先查
--------------
第一次做 DPO 时直接用训练集造了 600 对（「下一步」 vs 「原地重复」）。训练一开始
就 `acc=1.00`，`logp(chosen) ≈ 0`、`logp(rejected) ≈ -19`：

    初始 chosen 胜率 100%  →  没有任何可学信息  →  margin 十步内从 0.1 涨到 +33

这不是在学习，是把一个已经压到极低的概率再压低，属于纯过拟合。判据很简单：

    初始 chosen 胜率应接近 50%（模型自己也不确定哪个更好）
    接近 100% 说明偏好对选错了，训练只会浪费算力

第二轮的偏好对类型不同，预期也不同：

- `fix_after_error`：模型看到报错后原样重试 → 胜率应接近 50%
- `missing_call`：模型提前收尾没有调用工具 → 胜率应接近 0%（它主动选错了）

两类都远低于 100% 才算合格。

用法
----
    python pipeline/check_dpo_pairs.py \
        --pairs build/dpo-pairs-dpo.jsonl --init build/lora-v4-dpo
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from finetune.train_dpo import (  # noqa: E402
    PairDataset,
    completion_logprob,
    load_pairs,
)


def summarize(label: str, margins: list[float]) -> None:
    if not margins:
        print(f"  {label}: 无样本")
        return
    ordered = sorted(margins)
    wins = sum(1 for value in margins if value > 0)
    total = len(ordered)
    print(
        f"  {label:16} n={total:3}  margin 均值 {sum(margins) / total:+.3f}"
        f"  中位 {ordered[total // 2]:+.3f}"
        f"  min {ordered[0]:+.2f}  max {ordered[-1]:+.2f}"
        f"  胜率 {wins}/{total} = {wins / total:.0%}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="DPO 偏好对自检")
    parser.add_argument("--pairs", default="build/dpo-pairs-dpo.jsonl")
    parser.add_argument("--init", default="build/lora-v4-dpo")
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=0, help="只查前 N 对")
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
    model = PeftModel.from_pretrained(
        base, args.init, adapter_name="policy", is_trainable=True
    )
    model.load_adapter(args.init, adapter_name="ref", is_trainable=False)
    model.set_adapter("policy")
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    # 必须切到 train()：梯度检查点只在训练模式下生效，否则峰值显存会翻数倍。
    model.train()

    pairs = load_pairs(Path(args.pairs))
    if args.limit:
        pairs = pairs[: args.limit]
    dataset = PairDataset(pairs, tokenizer, args.max_length)
    print(f"偏好对 {len(dataset.items)} 条（跳过 {dataset.skipped}）\n")

    def logp(prompt_ids: list[int], completion_ids: list[int]):
        prompt = torch.tensor([prompt_ids], device="cuda")
        completion = torch.tensor([completion_ids], device="cuda")
        full = torch.cat([prompt, completion], dim=1)
        mask = torch.zeros_like(full, dtype=torch.bool)
        mask[:, prompt.shape[1]:] = True
        return completion_logprob(model, full, torch.ones_like(full), mask)

    by_kind: dict[str, list[float]] = defaultdict(list)
    margins: list[float] = []
    for item in dataset.items:
        with torch.no_grad():
            model.set_adapter("ref")
            ref_chosen = logp(item["prompt_ids"], item["chosen_ids"])
            ref_rejected = logp(item["prompt_ids"], item["rejected_ids"])
            model.set_adapter("policy")
            pol_chosen = logp(item["prompt_ids"], item["chosen_ids"])
            pol_rejected = logp(item["prompt_ids"], item["rejected_ids"])
        margin = float((pol_chosen - ref_chosen) - (pol_rejected - ref_rejected))
        margins.append(margin)
        by_kind[item["kind"]].append(margin)

    print("按类型分：")
    for kind, values in sorted(by_kind.items()):
        summarize(kind, values)
    print("\n合计：")
    summarize("全部", margins)

    wins = sum(1 for value in margins if value > 0)
    ratio = wins / len(margins) if margins else 0.0
    print()
    if ratio >= 0.9:
        print("结论：胜率过高，模型本来就倾向 chosen，DPO 几乎没有学习空间。")
        print("      请检查偏好对是否来自真实失败，而不是已通过的数据。")
        return 1
    print("结论：胜率远低于 100%，说明模型确实答错，可以开始训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
