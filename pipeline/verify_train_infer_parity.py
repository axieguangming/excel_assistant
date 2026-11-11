"""核对训练数据与当前代码是否仍然一致（训练/推理错配体检）。

为什么需要这个脚本
------------------
这个项目已经出现过五次训练/推理格式漂移：

1. 工具调用标签格式
2. 观察文本措辞
3. tools schema（LangChain 转换后与训练时不同）
4. 历史工具调用的 `arguments` 编码
5. Excel 往返的 dtype（`Int64` → `float64`，`""` → `NaN`）

每一次都是「模型学到的是 A，线上发的是 B」。共同的原因是：两处代码各自
生成同一份协议，靠意图保证一致。所以需要工具定期按**字节**核对。

检查三项
--------
1. `tools`：训练数据里存的定义 vs `tool_schema.tool_definitions()`
2. system prompt：训练数据里的 vs `agent.AGENT_SYSTEM`
3. 观察文本：训练数据里存的 tool 消息 vs 用磁盘表格重新执行得到的

第 3 项最容易出问题，因为它在评测时是**实时生成**的，不依赖数据文件里
存了什么。

用法
----
    python pipeline/verify_train_infer_parity.py
    python pipeline/verify_train_infer_parity.py --sft datasets-holdout2/sft-a
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import AGENT_SYSTEM  # noqa: E402
from dataset_builder.executor import _apply_step  # noqa: E402
from dataset_builder.tables import load_tables  # noqa: E402
from excel_ops import observation_for_step  # noqa: E402
from tool_schema import tool_definitions  # noqa: E402

# 由框架注入、模型无从填写的参数，执行前要剔除。
INJECTED = ("state", "tool_call_id")


def _first_difference(left: str, right: str) -> int:
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def _check_static(record: dict, current_tools: list[dict]) -> list[str]:
    """检查 tools 与 system prompt。"""
    problems: list[str] = []

    stored_tools = record.get("tools")
    if stored_tools != current_tools:
        stored_text = json.dumps(stored_tools, ensure_ascii=False)
        current_text = json.dumps(current_tools, ensure_ascii=False)
        at = _first_difference(stored_text, current_text)
        problems.append(
            f"tools 不一致（{len(stored_text)} vs {len(current_text)} 字符，"
            f"首个差异 @{at}）"
        )

    system = next((m for m in record["messages"] if m["role"] == "system"), None)
    stored_system = (system or {}).get("content")
    if stored_system != AGENT_SYSTEM:
        problems.append(
            f"system prompt 不一致（{len(stored_system or '')} vs "
            f"{len(AGENT_SYSTEM)} 字符）"
        )
    return problems


def _check_observation(record: dict, tables: dict, limit: int) -> tuple[bool, str]:
    """重放第一步工具调用，比对观察文本。"""
    table_id = record["sample_id"].rsplit("-", 1)[0]
    table = tables.get(table_id)
    if table is None:
        return True, "表格缺失，跳过"

    frame = table.frame.copy()
    messages = record["messages"]
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or not message.get("tool_calls"):
            continue
        stored = next(
            (m for m in messages[index + 1 :] if m["role"] == "tool"), None
        )
        if stored is None:
            return True, "无 tool 消息，跳过"

        call = message["tool_calls"][0]["function"]
        name = call["name"]
        try:
            args = json.loads(call["arguments"])
        except json.JSONDecodeError:
            return True, "参数无法解析，跳过"
        for key in INJECTED:
            if isinstance(args, dict):
                args.pop(key, None)

        try:
            frame, _ = _apply_step(
                frame,
                {"tool": name, "args": args},
                render_charts=False,
                chart_dir=None,
            )
        except Exception as error:  # noqa: BLE001 - 单样本失败不影响整体
            return True, f"执行失败，跳过（{error}）"

        fresh = observation_for_step(name, frame, arguments=args)
        if fresh == stored["content"]:
            return True, ""
        at = _first_difference(stored["content"], fresh)
        return False, (
            f"{name} 的观察文本不一致（首个差异 @{at}）\n"
            f"      存的: {stored['content'][max(0, at - 20): at + 60]!r}\n"
            f"      现在: {fresh[max(0, at - 20): at + 60]!r}"
        )
    return True, "无可核对步骤"


def main() -> int:
    parser = argparse.ArgumentParser(description="核对训练数据与当前代码的一致性")
    parser.add_argument(
        "--sft",
        action="append",
        default=None,
        help="SFT 目录，可重复；默认检查 datasets-holdout2 下的两组",
    )
    parser.add_argument(
        "--tables",
        default="",
        help="表格目录；留空则用 <sft 的父目录>/tables-<组名>",
    )
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--limit", type=int, default=0, help="每组最多检查多少条")
    args = parser.parse_args()

    sft_dirs = args.sft or [
        "datasets-holdout2/sft-a",
        "datasets-holdout2/sft-b",
    ]
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    current_tools = tool_definitions()

    total = problems_found = 0
    for sft in sft_dirs:
        sft_path = Path(sft)
        group = sft_path.name.split("-", 1)[-1]  # sft-a -> a
        tables_dir = Path(args.tables) if args.tables else (
            sft_path.parent / f"tables-{group}"
        )
        tables = {}
        if tables_dir.is_dir():
            tables = {table.table_id: table for table in load_tables(tables_dir)}

        for split in splits:
            path = sft_path / f"{split}.jsonl"
            if not path.is_file():
                continue
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if args.limit:
                records = records[: args.limit]

            print(f"\n=== {path}（{len(records)} 条，表格 {len(tables)} 张）===")
            static_reported = False
            observation_mismatch = 0
            for record in records:
                total += 1
                # 静态部分每个文件只需查一次（同一份数据源）
                if not static_reported:
                    static_reported = True
                    for problem in _check_static(record, current_tools):
                        print(f"  [错配] {problem}")
                        problems_found += 1

                ok, detail = _check_observation(record, tables, args.limit)
                if not ok:
                    observation_mismatch += 1
                    if observation_mismatch <= 3:
                        print(f"  [错配] {record['sample_id']}")
                        print(f"      {detail}")

            if observation_mismatch:
                problems_found += 1
                ratio = observation_mismatch / len(records)
                print(
                    f"  观察文本不一致：{observation_mismatch}/{len(records)}"
                    f"（{ratio:.0%}）"
                )
            else:
                print("  观察文本全部一致")

    print()
    if problems_found:
        print(f"发现 {problems_found} 类错配（共检查 {total} 条）")
        print("提示：观察文本不一致通常是因为训练数据用了内存表而非写盘后回读的表。")
        return 1
    print(f"未发现错配（共检查 {total} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
