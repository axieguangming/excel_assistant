"""构造 DPO 偏好对：教「有报错才重试」，直接惩罚「无理由重复」。

背景
----
v4 加了 400 条失败-修正轨迹后，「字段找不到就放弃」的缺陷消失了，但引入了
新问题：44 条评测失败是「重复调用同一工具」（最多连调 9 次触顶）。成因是
recovery 轨迹里 `A(错) → A(对)` 天然含连续的同一工具调用，模型学成了
「这个工具该多调几次」。v5 试图用插入 `describe_data` 打断该模式，结果更糟
（37 条循环且回答为空）。

SFT 很难表达「这个行为是错的」——它只能加强正确路径。DPO 直接对比正反样本，
更适合教这类「不要做什么」。

关键约束
--------
训练数据里有 15 条**合法**的连续重复（如用不同参数连填两列空值），因此不能
教「禁止重复」，只能教「禁止同工具 + 同参数的无效重复」。

两类偏好对
----------
1. `success_repeat`：前一步成功后，应当执行链中的下一步，而不是原地重复。
   - chosen   = 链中的下一个工具调用
   - rejected = 重复刚才那个工具调用（同工具同参数）
2. `error_fix`：工具报错后，应当读报错并修正参数，而不是原样重试或放弃。
   - chosen   = 修正后的调用
   - rejected = 原样重复错误调用

用法
----
    python pipeline/make_dpo_pairs.py --count 600 --out build/dpo-pairs.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_builder.tables import load_tables  # noqa: E402
from make_recovery_dataset import (  # noqa: E402
    _apply_step,
    iter_column_refs,
    make_wrong_name,
    set_by_path,
)


def tool_call_message(call_no: int, name: str, args: dict) -> dict:
    """与训练数据一致的 assistant 工具调用消息。"""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{call_no}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        ],
    }


def calls_in(record: dict) -> list[tuple[int, str, dict]]:
    """抽取轨迹里的 (消息下标, 工具名, 参数)。"""
    result: list[tuple[int, str, dict]] = []
    for index, message in enumerate(record["messages"]):
        for call in message.get("tool_calls") or []:
            function = call["function"]
            try:
                args = json.loads(function["arguments"])
            except json.JSONDecodeError:
                continue
            result.append((index, function["name"], args))
    return result


def renumber(messages: list[dict]) -> list[dict]:
    """把 tool_call id 与对应 tool 消息的 id 重排，避免与 chosen/rejected 冲突。"""
    out = copy.deepcopy(messages)
    mapping: dict[str, str] = {}
    counter = 0
    for message in out:
        for call in message.get("tool_calls") or []:
            counter += 1
            new_id = f"ctx_{counter}"
            mapping[call.get("id")] = new_id
            call["id"] = new_id
    for message in out:
        if message.get("role") == "tool" and message.get("tool_call_id") in mapping:
            message["tool_call_id"] = mapping[message["tool_call_id"]]
    return out


def build_success_repeat_pairs(record: dict, rng: random.Random) -> list[dict]:
    """同一条轨迹的每个「过渡点」都可以造一对：下一步 vs 原地重复。"""
    calls = calls_in(record)
    if len(calls) < 2:
        return []

    pairs: list[dict] = []
    for position in range(1, len(calls)):
        prev_index, prev_name, prev_args = calls[position - 1]
        next_index, next_name, next_args = calls[position]

        # 若下一步与上一步完全相同（同工具同参数），这根链本身就是重复，跳过
        if prev_name == next_name and prev_args == next_args:
            continue

        # prompt = 到上一步的 tool 观察为止
        prompt = renumber(record["messages"][: prev_index + 2])
        chosen = tool_call_message(1, next_name, next_args)
        rejected = tool_call_message(1, prev_name, prev_args)

        pairs.append(
            {
                "kind": "success_repeat",
                "tools": record["tools"],
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return pairs


def build_error_fix_pairs(
    record: dict,
    table,
    rng: random.Random,
) -> list[dict]:
    """在某个步骤上注入错误列名，造「修正 vs 原样重试」的偏好对。"""
    calls = calls_in(record)
    if not calls:
        return []

    existing = {str(column) for column in table.frame.columns}
    candidates: list[tuple[int, str, dict, tuple, str]] = []
    # 复用 recovery 的路径抽取逻辑
    for index, step in enumerate(record["_steps"]):
        for _, path, column in iter_column_refs([step]):
            if column in existing:
                candidates.append((index, step["tool"], step.get("args") or {}, path, column))
    if not candidates:
        return []

    rng.shuffle(candidates)
    for index, name, args, path, column in candidates:
        wrong = make_wrong_name(column, existing, rng)
        if wrong is None:
            continue
        broken_args = copy.deepcopy(args)
        set_by_path([{"args": broken_args}], 0, path, wrong)

        # 确认错误参数确实会失败
        try:
            _apply_step(
                table.frame.copy(),
                {"tool": name, "args": broken_args},
                render_charts=False,
                chart_dir=None,
            )
        except Exception as error:  # noqa: BLE001
            failure_text = f"工具执行失败：{error}"
        else:
            continue  # 错误参数竟然可行，换一个

        if "可用字段" not in failure_text and name != "create_chart":
            continue

        # prompt = 该步骤之前的上下文 + 错误调用 + 报错观察
        prompt_messages = record["messages"][: 2 + 2 * index]
        prompt = renumber(prompt_messages)
        prompt.append(tool_call_message(99, name, broken_args))
        prompt.append(
            {
                "role": "tool",
                "tool_call_id": "ctx_99",
                "name": name,
                "content": failure_text,
            }
        )

        return [
            {
                "kind": "error_fix",
                "tools": record["tools"],
                "prompt": prompt,
                "chosen": tool_call_message(1, name, args),
                "rejected": tool_call_message(1, name, broken_args),
            }
        ]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="构造 DPO 偏好对")
    parser.add_argument("--sft", default="build/sft-expanded/train.jsonl")
    parser.add_argument("--tables", default="datasets/2026-10-29/tables")
    parser.add_argument("--out", default="build/dpo-pairs.jsonl")
    parser.add_argument(
        "--count",
        type=int,
        default=600,
        help="目标对数。占比过高会过度强化，600 对约为训练集的 15%",
    )
    parser.add_argument(
        "--error-ratio",
        type=float,
        default=0.25,
        help="其中「报错后修正」类别的比例，其余为「成功后不重复」",
    )
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tables = {table.table_id: table for table in load_tables(Path(args.tables))}

    records: list[dict] = []
    for line in Path(args.sft).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    rng.shuffle(records)

    want_error = int(args.count * args.error_ratio)
    want_success = args.count - want_error

    success_pairs: list[dict] = []
    error_pairs: list[dict] = []
    rejects: Counter = Counter()

    for record in records:
        if len(success_pairs) >= want_success and len(error_pairs) >= want_error:
            break
        calls = calls_in(record)
        if len(calls) < 2:
            continue

        if len(success_pairs) < want_success:
            for pair in build_success_repeat_pairs(record, rng):
                if len(success_pairs) >= want_success:
                    break
                success_pairs.append(pair)

        if len(error_pairs) < want_error:
            workbook = record["sample_id"].split("-clarify")[0].rsplit("-", 1)[0]
            table = tables.get(workbook)
            if table is None:
                rejects["missing_table"] += 1
                continue
            # 复用 recovery 的步骤抽取：从消息里还原 steps
            steps = [
                {"tool": name, "args": args_}
                for _, name, args_ in calls
            ]
            record["_steps"] = steps
            built = build_error_fix_pairs(record, table, rng)
            record.pop("_steps", None)
            if built:
                error_pairs.extend(built)
            else:
                rejects["error_unavailable"] += 1

    chosen = success_pairs[:want_success] + error_pairs[:want_error]
    rng.shuffle(chosen)
    for index, pair in enumerate(chosen, start=1):
        pair["pair_id"] = f"dpo-{index:05d}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(pair, ensure_ascii=False) for pair in chosen) + "\n",
        encoding="utf-8",
    )

    kinds = Counter(pair["kind"] for pair in chosen)
    dup_bad = sum(
        1
        for pair in chosen
        if pair["chosen"]["tool_calls"][0]["function"]
        == pair["rejected"]["tool_calls"][0]["function"]
    )
    print(f"产出 {len(chosen)} 对 → {out}")
    print(f"类别：{dict(kinds)}")
    print(f"chosen 与 rejected 完全相同的坏样本：{dup_bad}")
    if rejects:
        print(f"跳过：{dict(rejects)}")

    if chosen:
        sample = chosen[0]
        print("\n--- 抽查 ---")
        print(f"kind = {sample['kind']}")
        print(f"prompt 消息数 = {len(sample['prompt'])}")
        print(f"chosen   = {sample['chosen']['tool_calls'][0]['function']['name']}"
              f"({sample['chosen']['tool_calls'][0]['function']['arguments'][:80]})")
        print(f"rejected = {sample['rejected']['tool_calls'][0]['function']['name']}"
              f"({sample['rejected']['tool_calls'][0]['function']['arguments'][:80]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
