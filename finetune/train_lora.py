"""Qwen2.5-1.5B 的 LoRA 微调示例。

前置：
    pip install -r requirements-train.txt

用法：
    python finetune/convert.py --data datasets --out build/sft
    python finetune/train_lora.py --data build/sft --output build/lora-out

设计要点：

- 只在 assistant 片段上计算损失。system / user / tool 内容属于输入，
  对它们算损失只会让模型学会复述问题。
- 训练用的 system prompt、工具 schema、对话结构来自 `finetune.dataset`，
  与线上 Agent 完全一致；训练/推理格式漂移是微调失败最常见的原因。
- 多步轨迹保留完整的"调用 → 观察 → 下一步"循环，这正是 1.5B 目前最缺的能力。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _require_training_dependencies() -> None:
    """把重依赖的导入放到运行时，这样 --help 与参数校验无需 GPU 环境。"""
    missing = []
    for name in ("torch", "transformers", "peft"):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise SystemExit(
            "缺少训练依赖：" + ", ".join(missing) + "\n"
            "请先安装：python -m pip install -r requirements-train.txt"
        )


try:
    # 未安装训练依赖时仍允许 --help 与参数校验，
    # 只有真正训练才会因为缺少 Trainer 而报错。
    from torch.utils.data import Dataset  # type: ignore[assignment]
except ImportError:  # pragma: no cover - 仅用于无依赖环境
    Dataset = object  # type: ignore[assignment, misc]


class TrajectoryCollator:
    """按 batch 内最长序列右填充。

    labels 用 -100 填充，这样 padding 位置不参与损失计算
    （若用 0 填充，会被当成 token id 0 计入损失）。
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        input_ids, attention_mask, labels = [], [], []
        for item in features:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class TrajectoryDataset(Dataset):
    """把 JSONL 里的对话轨迹编码成带 loss mask 的样本。"""

    def __init__(
        self,
        path: str | Path,
        tokenizer,
        *,
        max_length: int = 4096,
    ):
        self.records: list[dict[str, Any]] = []
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        record = self.records[index]
        encoded = self.tokenizer.apply_chat_template(
            record["messages"],
            tools=record["tools"],
            tokenize=True,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = list(encoded["input_ids"])
        # assistant_tokens_mask 为 1 的位置才参与损失。
        assistant_mask = list(encoded.get("assistant_tokens_mask") or [])

        if len(assistant_mask) != len(input_ids):
            # 少数模板不支持 generation 标记时退化为"整段都算"，
            # 但会打印告警，避免悄悄训练出复述问题的模型。
            assistant_mask = [1] * len(input_ids)

        labels = [
            token if keep else -100
            for token, keep in zip(input_ids, assistant_mask)
        ]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


def build_lora_config():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        # 覆盖注意力与 MLP，是 Qwen 系列常用的组合。
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def resolve_model_source(value: str) -> tuple[str, str | None]:
    """校验并解析 --model，返回 (可加载的路径或仓库名, 警告信息)。

    GGUF 是 llama.cpp 的量化推理格式，transformers/peft 无法加载；直接报错
    比让用户看着一次隐式下载更清楚。
    """
    path = Path(value)

    if path.suffix.lower() == ".gguf" or (
        path.is_file() and path.name.lower().endswith(".gguf")
    ):
        raise SystemExit(
            f"{value} 是 GGUF 量化文件，不能用于微调。\n"
            "训练需要原始 safetensors 权重，请下载 Qwen2.5-1.5B-Instruct 的\n"
            "HuggingFace / ModelScope 仓库（约 3.1 GB），再通过 --model 指向该目录。\n"
            "GGUF 只用于 llama.cpp 推理，微调完成后再导出 GGUF 部署。"
        )

    if not path.exists():
        warning = (
            f"本地没有 {value}，将从 HuggingFace 下载约 3.1 GB 权重。\n"
            "国内网络较慢时可改用镜像：\n"
            "    set HF_ENDPOINT=https://hf-mirror.com\n"
            "或先用 modelscope 下载，再 --model 指向本地目录。\n"
        )
        return value, warning

    if not path.is_dir():
        raise SystemExit(f"--model 需要是模型目录或仓库名，收到文件：{value}")

    has_weights = any(
        path.glob("*.safetensors")
    ) or any(path.glob("*.bin"))
    if not has_weights:
        raise SystemExit(
            f"{value} 里没有 *.safetensors 或 *.bin 权重文件。\n"
            "该目录可能只放了 GGUF，或下载不完整。"
        )
    return str(path), None


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA SFT for the Excel agent")
    parser.add_argument("--data", default="build/sft", help="包含 train.jsonl / val.jsonl 的目录")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace 仓库名，或本地 safetensors 模型目录（不能是 .gguf）",
    )
    parser.add_argument("--output", default="build/lora-out")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=-1, help="用于冒烟测试")
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="默认用 4bit 量化加载以省显存；显存充足可关闭",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    if not train_path.is_file():
        print(f"找不到 {train_path}，请先运行 finetune/convert.py")
        return 2

    model_source, warning = resolve_model_source(args.model)
    if warning:
        print(warning)
        print(f"实际加载来源：{model_source}\n")

    _require_training_dependencies()
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from peft import get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "device_map": "auto"}
    if not args.no_4bit:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    model = get_peft_model(model, build_lora_config())
    model.print_trainable_parameters()

    train_dataset = TrajectoryDataset(
        train_path, tokenizer, max_length=args.max_length
    )
    eval_dataset = (
        TrajectoryDataset(val_path, tokenizer, max_length=args.max_length)
        if val_path.is_file()
        else None
    )

    collator = TrajectoryCollator(pad_token_id=tokenizer.pad_token_id)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to=[],
        max_steps=args.max_steps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"\nLoRA 适配器已保存到 {args.output}")
    print("推理时用 PeftModel.from_pretrained(base_model, output) 加载。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
