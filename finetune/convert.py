"""把数据集分区转换成微调用的 JSONL。

用法：
    python finetune/convert.py --data datasets --out build/sft

输出：
    build/sft/train.jsonl
    build/sft/val.jsonl
    build/sft/tools.json      # 工具定义，便于核对
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# 允许直接运行 `python finetune/convert.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset_builder.workspace import resolve_dataset_dir  # noqa: E402
from finetune.dataset import convert_split, tool_definitions, write_jsonl  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert dataset splits to SFT JSONL")
    parser.add_argument("--data", default="datasets", help="数据集根目录或具体批次")
    parser.add_argument("--out", default="build/sft", help="输出目录")
    args = parser.parse_args()

    try:
        batch = resolve_dataset_dir(args.data)
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2

    tables_dir = batch / "tables"
    tools = tool_definitions()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split in ("train", "val", "test"):
        split_path = batch / "splits" / f"{split}.jsonl"
        if not split_path.is_file():
            continue
        records = convert_split(split_path, tables_dir, tools=tools)
        target = write_jsonl(records, out_dir / f"{split}.jsonl")
        total += len(records)
        multi = sum(
            1
            for record in records
            if sum(1 for m in record["messages"] if m.get("tool_calls")) > 1
        )
        print(f"{split:<6} {len(records):>4} 条（多步 {multi}）-> {target}")

    (out_dir / "tools.json").write_text(
        json.dumps(tools, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n合计 {total} 条；数据来自 {batch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
