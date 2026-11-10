"""构造「工具报错 → 读取报错 → 修正参数」的恢复轨迹。

背景
----
原有 3934 条训练轨迹里，工具报错 **0 次**。模型从未见过报错后的情况，因此
参数一写错就放弃并道歉，而不是像强模型那样读取报错里的「可用字段」重试。
留出集 B 组的失败里有 71% 属于这一类。

构造方式
--------
1. 取已校验通过的正确链
2. 把某一步的列名换成「近似但不存在的名字」（`运费` → `运费合计`）
3. 真实执行，抓到工具返回的报错文本（含可用字段清单）
4. 用正确列名重试该步，再继续执行剩余步骤
5. 把「错误调用 → 报错观察 → 修正调用 → 成功观察 → 最终答复」写入轨迹

用法
----
    python pipeline/make_recovery_dataset.py --count 400 --out build/recovery.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import excel_ops
from dataset_builder.executor import _apply_step
from dataset_builder.tables import load_tables
from finetune.dataset import (
    _chart_placeholder_url,
    build_human_message,
    render_observation,
    tool_definitions,
)

# 列名参数的路径：(工具名) -> 参数形状说明
COLUMN_KEYS = (
    "column",
    "group_by",
    "subset",
    "chart_x",
    "chart_y",
    "left",
    "right_column",
)

# 生成「近似但不存在的列名」的后缀/前缀，模拟模型常见的套错方式。
DERIVED_SUFFIXES = ("合计", "总额", "总和", "总计", "均值", "平均值", "数量")
DERIVED_PREFIXES = ("总", "平均", "全部")
SIMPLIFY_PATTERNS = (
    (r"编号$", "号"),
    (r"号$", ""),
    (r"代码$", "号"),
    (r"日期$", "时间"),
    (r"金额$", "数额"),
)

RUNTIME_ERRORS = (ValueError, TypeError, KeyError, OverflowError)


def iter_column_refs(steps: list[dict]):
    """产出 (步骤下标, 参数路径, 列名)。参数路径用于回写错误值。"""
    for index, step in enumerate(steps):
        args = step.get("args") or {}
        for key in COLUMN_KEYS:
            value = args.get(key)
            if isinstance(value, str):
                yield index, (key,), value
            elif isinstance(value, list):
                for pos, item in enumerate(value):
                    if isinstance(item, str):
                        yield index, (key, pos), item
        for pos, metric in enumerate(args.get("metrics") or []):
            if isinstance(metric.get("column"), str):
                yield index, ("metrics", pos, "column"), metric["column"]
        for pos, condition in enumerate(args.get("filters") or []):
            if isinstance(condition.get("column"), str):
                yield index, ("filters", pos, "column"), condition["column"]
        for pos, sort_key in enumerate(args.get("sort_by") or []):
            if isinstance(sort_key.get("column"), str):
                yield index, ("sort_by", pos, "column"), sort_key["column"]


def set_by_path(steps: list[dict], index: int, path: tuple, value) -> None:
    """把 steps[index] 的参数按 path 写回新值（就地修改）。"""
    target = steps[index].setdefault("args", {})
    if path[0] in ("metrics", "filters", "sort_by"):
        collection = target.get(path[0]) or []
        collection[path[1]][path[2]] = value
        return
    if len(path) == 1:
        target[path[0]] = value
    else:
        target[path[0]][path[1]] = value


def make_wrong_name(column: str, existing: set[str], rng: random.Random) -> str | None:
    """生成一个近似但不存在的列名；生成不出来就返回 None。"""
    candidates: list[str] = []

    simplified = column
    for pattern, replacement in SIMPLIFY_PATTERNS:
        if re.search(pattern, column):
            simplified = re.sub(pattern, replacement, column)
            break
    if simplified != column:
        candidates.append(simplified)

    for suffix in DERIVED_SUFFIXES:
        candidates.append(f"{column}{suffix}")
    for prefix in DERIVED_PREFIXES:
        candidates.append(f"{prefix}{column}")

    # 加单位括号这一常见写法
    candidates.append(f"{column}(元)")
    candidates.append(f"{column}_合计")

    rng.shuffle(candidates)
    for name in candidates:
        if name and name not in existing and len(name) <= 20:
            return name
    return None


def run_step(
    frame: pd.DataFrame,
    step: dict,
) -> tuple[pd.DataFrame | None, str, bool]:
    """执行单步，返回 (新表, 观察文本, 是否成功)。观察文本与线上格式一致。"""
    tool = str(step["tool"])
    arguments = dict(step.get("args") or {})
    try:
        new_frame, chart_url = _apply_step(
            frame.copy(), step, render_charts=False, chart_dir=None
        )
    except RUNTIME_ERRORS as error:
        return None, excel_ops.error_observation(error, tool_name=tool), False

    url = chart_url
    if url is None and tool == "create_chart":
        url = _chart_placeholder_url("recovery-chart")
    return new_frame, render_observation(
        new_frame, tool_name=tool, arguments=arguments, chart_url=url
    ), True


def build_record(
    sample_id: str,
    table,
    steps: list[dict],
    question: str,
    answer: str,
    *,
    error_index: int,
    error_path: tuple,
    wrong_name: str,
    correct_name: str,
) -> dict | None:
    """构造一条「报错→修正」轨迹。任一步执行失败则返回 None。

    关键设计：报错后先调 `describe_data` 查看字段，再重试原工具。
    这样做的原因来自 v4 的副作用：若直接「错一次→对一次」，轨迹里会出现
    同一工具连续调用，模型会学成「这个工具该连调多次」（实测 44 条失败，
    最多连调 9 次触顶）。插入只读查看后，轨迹里不再有连续重复的工具调用。

    额外好处：多余的 `describe_data` 在评分时会被 `align_read_only` 忽略，
    即使模型学成「多看一次」也不会因此扣分。
    """
    messages: list[dict] = [
        {"role": "system", "content": __import__("agent").AGENT_SYSTEM},
        {
            "role": "user",
            "content": build_human_message(table.frame, question),
        },
    ]

    current = table.frame
    call_no = 0
    for index, step in enumerate(steps):
        tool = str(step["tool"])

        if index == error_index:
            # 1) 先用错误列名调用一次，拿到报错
            broken = copy.deepcopy(step)
            set_by_path([broken], 0, error_path, wrong_name)
            _, failure_text, ok = run_step(current, broken)
            if ok:
                # 错误列名竟然执行成功，说明这个样本不可用
                return None
            call_no += 1
            messages.append(_assistant_call(call_no, tool, broken.get("args") or {}))
            messages.append(_tool_message(call_no, tool, failure_text))

            # 2) 先查看当前表的结构（只读），再决定怎么改
            call_no += 1
            messages.append(_assistant_call(call_no, "describe_data", {}))
            _, desc_text, desc_ok = run_step(
                current, {"tool": "describe_data", "args": {}}
            )
            if not desc_ok:
                return None
            messages.append(_tool_message(call_no, "describe_data", desc_text))

            # 3) 用正确列名重试
            call_no += 1
            messages.append(_assistant_call(call_no, tool, step.get("args") or {}))

        new_frame, observation, ok = run_step(current, step)
        if not ok or new_frame is None:
            return None

        if index != error_index:
            call_no += 1
            messages.append(_assistant_call(call_no, tool, step.get("args") or {}))

        messages.append(_tool_message(call_no, tool, observation))
        current = new_frame

    messages.append({"role": "assistant", "content": answer})
    return {
        "sample_id": sample_id,
        "tools": tool_definitions(),
        "messages": messages,
        "origin": "recovery",
    }


def has_consecutive_duplicate(record: dict) -> bool:
    """轨迹里是否存在「同一工具连续调用」。这类样本会教模型无效重复。"""
    names = [
        call["function"]["name"]
        for message in record["messages"]
        for call in (message.get("tool_calls") or [])
    ]
    return any(
        names[i] == names[i + 1] for i in range(len(names) - 1)
    )


def _assistant_call(call_no: int, tool: str, args: dict) -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{call_no}",
                "type": "function",
                "function": {
                    "name": tool,
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        ],
    }


def _tool_message(call_no: int, tool: str, content: str) -> dict:
    return {
        "role": "tool",
        "tool_call_id": f"call_{call_no}",
        "name": tool,
        "content": content,
    }


def error_mentions_fields(text: str) -> bool:
    """报错里必须带「可用字段」清单，否则模型无从恢复。"""
    return "可用字段" in text


def main() -> int:
    parser = argparse.ArgumentParser(description="构造工具报错后的恢复轨迹")
    parser.add_argument("--variants", default="build/variants-faithful-v2.json")
    parser.add_argument("--tables", default="datasets/2026-10-29/tables")
    parser.add_argument("--out", default="build/recovery.jsonl")
    parser.add_argument(
        "--count",
        type=int,
        default=200,
        help="目标条数。v4 用 400 条时占比 9.2%，过高的比例带来了重复调用的副作用，现降到约 200 条（4.5%）",
    )
    parser.add_argument("--seed", type=int, default=20261105)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tables = {table.table_id: table for table in load_tables(Path(args.tables))}
    payload = json.loads(Path(args.variants).read_text(encoding="utf-8"))
    chains = payload["chains"]
    rng.shuffle(chains)

    records: list[dict] = []
    rejects: Counter = Counter()
    tool_hits: Counter = Counter()

    for chain in chains:
        if len(records) >= args.count:
            break
        table = tables.get(str(chain.get("workbook_id")))
        if table is None:
            rejects["missing_table"] += 1
            continue
        steps = chain.get("steps") or []
        if not steps:
            continue
        variants = chain.get("variants") or []
        if not variants:
            continue

        existing = {str(column) for column in table.frame.columns}
        refs = [ref for ref in iter_column_refs(steps) if ref[2] in existing]
        if not refs:
            rejects["no_column_ref"] += 1
            continue

        # 错误注入位置随机化：插入 describe_data 后不再有连续重复的工具调用，
        # 因此可以把错误放在任意步骤，学到「链中任意一步写错都能恢复」。
        refs.sort(key=lambda item: (item[0], item[1]))
        rng.shuffle(refs)
        placed = False
        for error_index, error_path, correct_name in refs:
            wrong_name = make_wrong_name(correct_name, existing, rng)
            if wrong_name is None:
                continue
            variant = rng.choice(variants)
            question = str(variant.get("question") or "")
            answer = str(variant.get("answer") or "")
            if not question or not answer:
                continue
            record = build_record(
                f"{table.table_id}-recovery-{len(records) + 1:03d}",
                table,
                steps,
                question,
                answer,
                error_index=error_index,
                error_path=error_path,
                wrong_name=wrong_name,
                correct_name=correct_name,
            )
            if record is None:
                reject_reason = "execute_failed"
                # 确认报错文本里确实有可用字段清单
                continue
            # 校验报错观察含可用字段
            tool_texts = [m["content"] for m in record["messages"] if m["role"] == "tool"]
            if not any(error_mentions_fields(text) for text in tool_texts):
                rejects["no_field_list"] += 1
                continue
            # 目录：不得存在「同一工具连续调用」，否则会教出无效重复
            if has_consecutive_duplicate(record):
                rejects["consecutive_duplicate"] += 1
                continue
            records.append(record)
            tool_hits[steps[error_index]["tool"]] += 1
            placed = True
            break
        if not placed:
            rejects["no_error_generated"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    print(f"产出 {len(records)} 条恢复轨迹 → {out}")
    print(f"注入错误所在工具：{dict(tool_hits.most_common())}")
    if rejects:
        print(f"剔除：{dict(rejects)}")
    # 抽查一条
    if records:
        sample = records[0]
        print("\n--- 抽查 ---")
        for message in sample["messages"]:
            role = message["role"]
            if role == "system":
                continue
            if role == "assistant" and message.get("tool_calls"):
                call = message["tool_calls"][0]["function"]
                print(f"  [{role}] {call['name']}({call['arguments'][:80]})")
            elif role == "tool":
                print(f"  [tool] {message['content'].splitlines()[0][:110]}")
            else:
                print(f"  [{role}] {(message.get('content') or '')[:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
