"""把恢复轨迹合并进训练集，产出下一版 SFT 数据目录。

只扩充 train 分区：val／test 保持原样，这样新旧模型的评测可以直接对比。

用法：
    python pipeline/merge_recovery.py --base build/sft-expanded \
        --recovery build/recovery.jsonl --out build/sft-v4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

FILES = ("train.jsonl", "val.jsonl", "test.jsonl", "tools.json")


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description="合并恢复轨迹到训练集")
    parser.add_argument("--base", default="build/sft-expanded")
    parser.add_argument("--recovery", default="build/recovery.jsonl")
    parser.add_argument("--out", default="build/sft-v4")
    args = parser.parse_args()

    base = Path(args.base)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in FILES:
        source = base / name
        if source.is_file():
            shutil.copy2(source, out / name)

    train_path = out / "train.jsonl"
    recovery_path = Path(args.recovery)
    if not recovery_path.is_file():
        print(f"找不到恢复轨迹 {recovery_path}")
        return 1

    existing = {
        json.loads(line)["sample_id"]
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    added = 0
    with train_path.open("a", encoding="utf-8") as handle:
        for line in recovery_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record["sample_id"] in existing:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing.add(record["sample_id"])
            added += 1

    print(f"train : {count_lines(base / 'train.jsonl')} + {added} = {count_lines(train_path)}")
    for name in ("val.jsonl", "test.jsonl"):
        print(f"{name:6s}: {count_lines(out / name)}（保持不变）")
    print(f"输出 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
