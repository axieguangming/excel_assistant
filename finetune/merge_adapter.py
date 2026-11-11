"""把 LoRA 适配器合并进基座权重，输出完整的 HF safetensors 目录。

为什么需要这一步
----------------
`build/lora-v4-dpo2/` 是 LoRA 适配器（只含 A/B 低秩矩阵，约 220 MB），
llama.cpp 的 `convert_hf_to_gguf.py` 只接受完整权重目录，不能读适配器。
因此需要先把适配器合并进基座、保存成标准的 HF checkpoint。

为什么不用 4-bit 基座来合并
--------------------------
在 4-bit 量化权重上做 `merge_and_unload()`，peft 会给出警告
「Merge lora module to 4-bit linear may get different generations due to
rounding errors」——合并结果会带上量化误差，而且要再经过一次量化转换。
这里以 bf16 加载基座（1.5B 约 3 GB，走 CPU 即可），合并后精度损失最小，
后续转 GGUF 时只做一次目标量化。

用法
----
    # 合并第二轮 DPO 适配器（当前最优）
    python finetune/merge_adapter.py \
        --base models/Qwen2.5-1.5B-Instruct \
        --adapter build/lora-v4-dpo2 \
        --output build/merged-dpo2

    # 只校验不写盘
    python finetune/merge_adapter.py --adapter build/lora-v4-dpo2 --check-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 需要一并复制的配套文件：convert_hf_to_gguf.py 依赖 tokenizer 与 config。
COMPANION_FILES: tuple[str, ...] = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "special_tokens_map.json",
)


def check_adapter(adapter_dir: Path) -> dict:
    """读取适配器元信息，确认它是可合并的 LoRA。"""
    config_path = adapter_dir / "adapter_config.json"
    weight_path = adapter_dir / "adapter_model.safetensors"
    if not config_path.is_file():
        raise SystemExit(f"缺少 adapter_config.json：{adapter_dir}")
    if not weight_path.is_file():
        raise SystemExit(f"缺少 adapter_model.safetensors：{adapter_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA":
        raise SystemExit(f"不是 LoRA 适配器：peft_type={config.get('peft_type')!r}")

    size_mb = weight_path.stat().st_size / 1024 / 1024
    return {
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "target_modules": len(config.get("target_modules") or []),
        "base_model": config.get("base_model_name_or_path"),
        "weight_mb": round(size_mb, 1),
    }


def merge(base_dir: Path, adapter_dir: Path, output_dir: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"加载基座（bf16, CPU）：{base_dir}")
    base = AutoModelForCausalLM.from_pretrained(
        base_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )

    print(f"挂载适配器：{adapter_dir}")
    model = PeftModel.from_pretrained(base, adapter_dir)

    print("合并权重…")
    merged = model.merge_and_unload()
    merged.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"保存完整权重 -> {output_dir}")
    merged.save_pretrained(output_dir, safe_serialization=True)

    # tokenizer 与 chat template 必须与基座一致，转换脚本要用到。
    tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    for name in COMPANION_FILES:
        source = base_dir / name
        target = output_dir / name
        if source.is_file() and not target.is_file():
            shutil.copyfile(source, target)

    total_mb = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file()) / 1024 / 1024
    print(f"完成，输出目录共 {total_mb:.1f} MB")


def verify(output_dir: Path) -> None:
    """回读合并结果，确认能被正常加载且权重不是 4-bit 量化过的。"""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
    print(f"校验：架构 {config.architectures}，hidden {getattr(config, 'hidden_size', '?')}")

    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    dtype = next(model.parameters()).dtype
    count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"校验：参数 {count:.3f}B，dtype {dtype}")
    if dtype != torch.bfloat16:
        raise SystemExit(f"预期 bf16，实际 {dtype}——可能加载到了量化权重")


def main() -> int:
    parser = argparse.ArgumentParser(description="合并 LoRA 适配器到基座权重")
    parser.add_argument("--base", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="build/lora-v4-dpo2")
    parser.add_argument("--output", default="build/merged-dpo2")
    parser.add_argument("--check-only", action="store_true", help="只打印适配器信息并退出")
    args = parser.parse_args()

    adapter_dir = Path(args.adapter)
    info = check_adapter(adapter_dir)
    print("=== 适配器信息 ===")
    for key, value in info.items():
        print(f"  {key}: {value}")

    if args.check_only:
        return 0

    base_dir = Path(args.base)
    if not (base_dir / "config.json").is_file():
        raise SystemExit(f"基座目录缺少 config.json：{base_dir}")

    output_dir = Path(args.output)
    merge(base_dir, adapter_dir, output_dir)
    verify(output_dir)
    print("\n下一步（需要 llama.cpp）：")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {output_dir} \\")
    print("    --outfile models/excel-assistant-dpo2.gguf --outtype q4_k_m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
