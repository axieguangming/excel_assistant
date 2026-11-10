"""在 SFT 适配器之上做 DPO，教「有报错才重试 / 有下一步就别原地重复」。

为什么用双适配器而不是合并权重
------------------------------
DPO 的参考模型应当是**已经 SFT 过的模型**。若把参考模型设成原始基座，DPO 会
把 SFT 学到的东西往回拉。合并权重需要先以 bf16 加载模型再写盘 3GB，成本高；
这里直接加载两份相同的适配器：

    adapter_name="policy"     —— 参与训练
    adapter_name="ref"        —— 冻结，作为参考模型

初始时两者权重完全一致，等价于「参考模型 = SFT 模型」。

损失
----
    loss = -logsigmoid( beta * ( (logp_c - logp_ref_c) - (logp_r - logp_ref_r) ) )

只对补全部分（assistant 的工具调用）算 logprob，prompt 部分不算，这样信号完全
落在「选哪个动作」上。用 `logits_to_keep` 布尔掩码只投影补全位置，显存因此可控。

用法
----
    python finetune/train_dpo.py --pairs build/dpo-pairs.jsonl \
        --model models/Qwen2.5-1.5B-Instruct --init build/lora-v4 \
        --output build/lora-v4-dpo --epochs 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finetune.train_lora import (  # noqa: E402
    build_lora_config,
    resolve_model_source,
)

DEFAULT_MAX_LENGTH = 4096


def load_pairs(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class PairDataset:
    """把偏好对编码成 (prompt_ids, chosen_ids, rejected_ids) 的三元组。"""

    def __init__(self, pairs: list[dict], tokenizer, max_length: int):
        self.items: list[dict] = []
        self.skipped = 0
        for pair in pairs:
            encoded = self._encode(pair, tokenizer, max_length)
            if encoded is None:
                self.skipped += 1
                continue
            self.items.append(encoded)

    @staticmethod
    def _encode(pair: dict, tokenizer, max_length: int) -> dict | None:
        tools = pair["tools"]
        prompt = pair["prompt"]

        def template(messages: list[dict]) -> list[int]:
            out = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=False,
            )
            return list(out["input_ids"])

        prompt_ids = template(prompt)
        chosen_ids = template(prompt + [pair["chosen"]])[len(prompt_ids):]
        rejected_ids = template(prompt + [pair["rejected"]])[len(prompt_ids):]

        if not chosen_ids or not rejected_ids:
            return None
        if len(prompt_ids) + max(len(chosen_ids), len(rejected_ids)) > max_length:
            return None
        return {
            "pair_id": pair["pair_id"],
            "kind": pair["kind"],
            "prompt_ids": prompt_ids,
            "chosen_ids": chosen_ids,
            "rejected_ids": rejected_ids,
        }


def save_policy_adapter(model, output_dir: Path) -> None:
    """只保存 policy 适配器，并把文件放到目录根位。

    多适配器模式下 peft 会把每个适配器写到 `<output>/{name}/` 子目录，而
    `load_model` 期望 adapter_config.json 在根目录，因此这里把内容提升一级。
    """
    import shutil

    model.set_adapter("policy")
    model.save_pretrained(output_dir, selected_adapters=["policy"])
    nested = output_dir / "policy"
    if nested.is_dir():
        for item in nested.iterdir():
            shutil.move(str(item), str(output_dir / item.name))
        nested.rmdir()


def completion_logprob(model, input_ids, attention_mask, completion_mask):
    """返回补全部分的对数概率之和。

    位置 i 的 logits 预测第 i+1 个 token，因此把 `completion_mask` 右移一位作为
    `logits_to_keep`，只投影补全位置——与 SFT 的显存优化思路一致。
    """
    import torch
    import torch.nn.functional as F

    keep = torch.zeros_like(completion_mask)
    keep[:, :-1] = completion_mask[:, 1:]
    if not bool(keep.any()):
        return None

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=keep)
    logits = outputs.logits.float()
    targets = input_ids[:, 1:][completion_mask[:, 1:]]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return token_logp.sum()


def main() -> int:
    parser = argparse.ArgumentParser(description="DPO on top of an SFT adapter")
    parser.add_argument("--pairs", default="build/dpo-pairs.jsonl")
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--init",
        default="build/lora-v4",
        help="作为起点与参考模型的 SFT 适配器目录",
    )
    parser.add_argument("--output", default="build/lora-v4-dpo")
    parser.add_argument("--beta", type=float, default=0.1, help="KL 约束强度")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 对，用于冒烟")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    pair_path = Path(args.pairs)
    if not pair_path.is_file():
        print(f"找不到偏好对文件 {pair_path}，请先运行 pipeline/make_dpo_pairs.py")
        return 2
    init_dir = Path(args.init)
    if not init_dir.is_dir():
        print(f"找不到初始适配器 {init_dir}")
        return 2

    model_source, warning = resolve_model_source(args.model)
    if warning:
        print(warning)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"trust_remote_code": True, "device_map": "auto"}
    if not args.no_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    base = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

    # 第一份适配器：policy，参与训练
    model = PeftModel.from_pretrained(
        base, str(init_dir), adapter_name="policy", is_trainable=True
    )
    # 第二份适配器：ref，冻结。权重与 policy 初始值相同，因此等价于「参考模型 = SFT 模型」
    model.load_adapter(str(init_dir), adapter_name="ref", is_trainable=False)
    model.set_adapter("policy")

    for name, param in model.named_parameters():
        if "ref" in name.split("."):
            param.requires_grad_(False)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数：{trainable:,}")
    # 4-bit + PEFT 下梯度检查点需要两件事才会生效：
    #   1. 输入必须求导（enable_input_require_grads）
    #   2. 模型必须处于 train 模式 —— 检查点的实现判断 self.training，
    #      漏了这一步会导致激活全部保留，实测同样序列从 2.2GB 爆到 11.8GB。
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.config.use_cache = False
    model.train()

    def gpu_mem() -> str:
        if not torch.cuda.is_available():
            return "cpu"
        return f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB"

    print(f"加载完成，显存 {gpu_mem()}")

    pairs = load_pairs(pair_path)
    if args.limit:
        pairs = pairs[: args.limit]
    dataset = PairDataset(pairs, tokenizer, args.max_length)
    print(f"偏好对 {len(dataset.items)} 条（跳过 {dataset.skipped} 条：超长或缺补全）")
    if not dataset.items:
        print("没有可用的偏好对")
        return 1

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    total_steps = args.max_steps if args.max_steps > 0 else max(
        1, (len(dataset.items) // max(1, args.grad_accum)) * int(args.epochs)
    )
    total_samples = (
        args.max_steps * args.grad_accum if args.max_steps > 0
        else len(dataset.items) * int(args.epochs)
    )
    warmup = max(1, int(total_steps * 0.05))
    print(f"总步数 {total_steps}（每步 {args.grad_accum} 对，预热 {warmup} 步）")

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr * max(0.0, 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    import torch.nn.functional as F

    def encode(item: dict) -> tuple:
        p = torch.tensor([item["prompt_ids"]], device=device)
        c = torch.tensor([item["chosen_ids"]], device=device)
        r = torch.tensor([item["rejected_ids"]], device=device)
        pc = torch.cat([p, c], dim=1)
        pr = torch.cat([p, r], dim=1)
        mc = torch.zeros_like(pc, dtype=torch.bool)
        mc[:, p.shape[1]:] = True
        mr = torch.zeros_like(pr, dtype=torch.bool)
        mr[:, p.shape[1]:] = True
        return (
            pc, torch.ones_like(pc), mc,
            pr, torch.ones_like(pr), mr,
        )

    order = list(range(len(dataset.items)))
    step = 0
    seen = 0
    running = {"loss": 0.0, "margin": 0.0, "acc": 0.0, "n": 0}
    started = time.time()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    while seen < total_samples:
        accum_loss = None
        for _ in range(args.grad_accum):
            if seen >= total_samples:
                break
            item = dataset.items[order[seen % len(order)]]
            seen += 1
            pc, ac, mc, pr, ar, mr = encode(item)

            # 参考模型：冻结的 SFT 适配器
            model.set_adapter("ref")
            with torch.no_grad():
                ref_c = completion_logprob(model, pc, ac, mc)
                ref_r = completion_logprob(model, pr, ar, mr)
            model.set_adapter("policy")
            if ref_c is None or ref_r is None:
                continue

            pol_c = completion_logprob(model, pc, ac, mc)
            pol_r = completion_logprob(model, pr, ar, mr)
            if pol_c is None or pol_r is None:
                continue

            logits = args.beta * ((pol_c - ref_c) - (pol_r - ref_r))
            loss = -F.logsigmoid(logits) / args.grad_accum
            loss.backward()
            accum_loss = loss.detach().item() * args.grad_accum if accum_loss is None else accum_loss + loss.detach().item() * args.grad_accum

            running["loss"] += loss.detach().item() * args.grad_accum
            running["margin"] += logits.detach().item() / args.beta
            running["acc"] += 1.0 if logits.detach().item() > 0 else 0.0
            running["n"] += 1

        if accum_loss is None:
            continue

        for group in optimizer.param_groups:
            group["lr"] = lr_at(step)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % 5 == 0 or step == total_steps:
            n = max(1, running["n"])
            elapsed = time.time() - started
            print(
                f"  step {step}/{total_steps} "
                f"loss={running['loss']/n:.4f} "
                f"margin={running['margin']/n:+.4f} "
                f"acc={running['acc']/n:.2f} "
                f"lr={lr_at(step):.2e} "
                f"{elapsed:.0f}s",
                flush=True,
            )
            running = {"loss": 0.0, "margin": 0.0, "acc": 0.0, "n": 0}

        if args.save_every and step % args.save_every == 0:
            ckpt = output_dir / f"checkpoint-{step}"
            save_policy_adapter(model, ckpt)
            print(f"  已保存 {ckpt}", flush=True)

    save_policy_adapter(model, output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nDPO 适配器已保存到 {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
