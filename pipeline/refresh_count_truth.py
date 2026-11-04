"""重新生成因 count 语义变更而过期的结果真值。

`excel_ops._aggregate` 的 count 已从「该列非空值个数」改为「组内行数」。
因此含 count 的样本重新执行后，部分结果 CSV 的**数值**会变（形状不变）。

本脚本只重写发生变化的真值，其余样本保持原样：
- 重新执行期望操作链
- 与已存 CSV 比对；一致则跳过
- 不一致则覆盖 CSV、更新 sha256
- 同步更新 samples.jsonl 与 splits/*.jsonl 中的记录
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from dataset_builder.executor import execute_chain
from dataset_builder.tables import load_tables

ROOT = Path("datasets/2026-10-29")


def uses_count(sample: dict) -> bool:
    for step in sample.get("expected_actions") or []:
        if step.get("tool") != "aggregate_data":
            continue
        metrics = (step.get("args") or {}).get("metrics") or []
        if any(isinstance(m, dict) and m.get("function") == "count" for m in metrics):
            return True
    return False


def frames_equal(stored: pd.DataFrame, fresh: pd.DataFrame) -> bool:
    """按字符串比较，避免浮点表示差异造成误判。"""
    if stored.shape != fresh.shape:
        return False
    if list(stored.columns) != list(fresh.columns):
        return False
    return stored.astype(str).equals(fresh.astype(str))


def main() -> int:
    tables = {t.table_id: t for t in load_tables(ROOT / "tables")}
    samples = [
        json.loads(line)
        for line in (ROOT / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"样本总数 {len(samples)}")

    count_samples = [s for s in samples if uses_count(s)]
    print(f"含 count 的样本 {len(count_samples)}")

    updated: dict[str, str] = {}  # sample_id -> 新 sha256
    for sample in count_samples:
        artifact = ROOT / sample["expected_result"]["artifact"]
        if not artifact.is_file():
            print(f"  [跳过] {sample['sample_id']} 找不到真值文件")
            continue
        stored = pd.read_csv(artifact, encoding="utf-8-sig")

        table = tables.get(sample["workbook_id"])
        if table is None:
            print(f"  [跳过] {sample['sample_id']} 找不到表格")
            continue

        outcome = execute_chain(table.frame, sample["expected_actions"])
        if not outcome.ok or outcome.frame is None:
            print(f"  [失败] {sample['sample_id']}: {outcome.error}")
            continue

        fresh = outcome.frame
        if frames_equal(stored, fresh):
            continue

        artifact.parent.mkdir(parents=True, exist_ok=True)
        fresh.to_csv(artifact, index=False, encoding="utf-8-sig")
        digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
        sample["expected_result"]["sha256"] = digest
        sample["expected_result"]["row_count"] = int(len(fresh))
        sample["expected_result"]["columns"] = [str(c) for c in fresh.columns]
        updated[sample["sample_id"]] = digest
        print(
            f"  [更新] {sample['sample_id']} "
            f"{stored.shape} -> {fresh.shape}"
        )

    if not updated:
        print("\n没有需要更新的真值。")
        return 0

    # 写回 samples.jsonl
    (ROOT / "samples.jsonl").write_text(
        "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in samples),
        encoding="utf-8",
    )

    # 写回 splits（每个分区是样本副本，需要同步 sha256）
    split_updated = 0
    for path in sorted((ROOT / "splits").glob("*.jsonl")):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        changed = False
        for sample in rows:
            digest = updated.get(sample["sample_id"])
            if digest and sample["expected_result"].get("sha256") != digest:
                sample["expected_result"]["sha256"] = digest
                changed = True
        if changed:
            path.write_text(
                "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in rows),
                encoding="utf-8",
            )
            split_updated += 1

    print(f"\n更新真值 {len(updated)} 条；同步分区文件 {split_updated} 个")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
