"""用模型的真实失败轨迹构造 DPO 偏好对。

为什么不用训练集的「下一步」
----------------------------
第一版从训练集造了 600 对（下一步 vs 原地重复），实测 DPO 一开始就 `acc=1.00`：

    logp(chosen) ≈ 0    logp(rejected) ≈ -19     margin 初始 ≈ 0.1

说明 SFT 模型对训练集里的这些动作早已完全正确，**没有可学的新信息**，DPO 只能
把 rejected 的概率继续压低（margin 十步内涨到 +33），本质是过拟合噪声。

现在改为从模型**真实答错**的轨迹取样：
- v4 在留出集上有 44 条「重复调用同一工具」的失败，重跑可稳定复现
- 逐条检查发现真实根因：模型把**问题里的派生名当列名**
  （`column: "平均工时"`，而表里只有 `工时`），工具报错后它**忽略报错、原样重试 9 次**

偏好对因此是「看到报错后，修正参数」 vs 「原样重试」：

    prompt   = system + user + [失败调用 + 报错观察]
    chosen   = 期望链中该位置的正确调用
    rejected = 模型刚才那次失败调用（同工具同参数）

第二轮（DPO 之后仍有 57 条执行类失败）增加了另一类偏好：

    missing_call / no_call
    prompt   = system + user + 实际已完成的调用
    chosen   = 期望链里的下一个工具调用
    rejected = 模型直接给出的文字答复（「已经处理好了」）

这针对「少调一步却声称完成」与「一次工具都不调用就作答」。

用法
----
    python pipeline/make_dpo_pairs_from_traces.py \
        --traces build/traces-dpo.json --out build/dpo-pairs-dpo.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# pipeline 不是包，按目录加入路径后直接导入同目录模块。
sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_dpo_pairs import renumber, tool_call_message  # noqa: E402


def build_pair(trace: dict, *, repeat_index: int) -> dict | None:
    """在第 `repeat_index` 次调用处造一对「原样重试 vs 修正参数」。"""
    calls = trace["actual"]
    expected = trace["expected"]
    if repeat_index >= len(calls):
        return None
    failed = calls[repeat_index]
    if failed.get("ok"):
        return None

    # 找到该位置对应的期望动作。工具名不一致说明前缀就跑偏了，跳过。
    if repeat_index >= len(expected):
        return None
    target = expected[repeat_index]
    if target["tool"] != failed["tool"]:
        return None
    if target["args"] == failed["args"]:
        return None  # 期望与失败参数相同，不是参数写错

    # prompt = 到这次失败调用为止（含其报错观察）
    messages: list[dict] = [
        {"role": "system", "content": trace["system_prompt"]},
        {"role": "user", "content": trace["user_prompt"]},
    ]
    call_no = 0
    for position, call in enumerate(calls):
        call_no += 1
        messages.append(tool_call_message(call_no, call["tool"], call["args"]))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{call_no}",
                "name": call["tool"],
                "content": call.get("observation") or "（无观察）",
            }
        )
        if position == repeat_index:
            break

    prompt = renumber(messages)
    return {
        "kind": "fix_after_error",
        "source": "holdout_failure",
        "sample_id": trace["sample_id"],
        "tools": trace["tools"],
        "prompt": prompt,
        "chosen": tool_call_message(1, target["tool"], target["args"]),
        "rejected": tool_call_message(1, failed["tool"], failed["args"]),
    }


def build_silent_wrong_args_pair(trace: dict, *, position: int) -> dict | None:
    """工具名与位置都对、参数写错但**没有报错**的静默错误。

    这类失败不会产生 `ok=False`，因此 `build_pair` 抓不到，但它恰恰是最难发现的
    缺陷形态——工具接受了参数、返回了「操作成功」，只有最终结果与期望不符：

        aggregate_data 少写一个 metric（`总工时` 有、`总加班时长` 漏）
        add_computed_column 用 `单价` 而不是 `销售额` 作左操作数
        fill_missing_values 的 subset 少一列
        fill_value 写成 `"未 知"`（多了一个空格）

    偏好对是「这个参数」 vs 「模型写的那个参数」，prompt 只到出错调用之前。
    """
    actual = trace["actual"]
    expected = trace["expected"]
    if position >= len(actual) or position >= len(expected):
        return None
    call = actual[position]
    target = expected[position]
    if not call.get("ok"):
        return None  # 有报错的交给 build_pair，避免重复
    if target["tool"] != call["tool"]:
        return None
    if target["args"] == call["args"]:
        return None
    # 前缀必须逐位正确，否则这是链跑偏而不是参数写错。
    for index in range(position):
        if expected[index]["tool"] != actual[index]["tool"]:
            return None
        if not actual[index].get("ok"):
            return None

    return {
        "kind": "silent_wrong_args",
        "source": "holdout_failure",
        "sample_id": trace["sample_id"],
        "tools": trace["tools"],
        "prompt": prefix_context(trace, upto=position),
        "chosen": tool_call_message(1, target["tool"], target["args"]),
        "rejected": tool_call_message(1, call["tool"], call["args"]),
    }


def text_message(content: str) -> dict:
    """纯文本 assistant 消息：模型没调用工具就直接作答时作为「被拒」样本。"""
    return {"role": "assistant", "content": content}


def prefix_context(trace: dict, *, upto: int | None = None) -> list[dict]:
    """system + user + 前 `upto` 次实际调用及其观察。

    `upto=None` 表示包含全部调用（用于「已经做完一部分但提前收尾」的场景）。
    """
    messages: list[dict] = [
        {"role": "system", "content": trace["system_prompt"]},
        {"role": "user", "content": trace["user_prompt"]},
    ]
    calls = trace["actual"] if upto is None else trace["actual"][:upto]
    for position, call in enumerate(calls, start=1):
        messages.append(tool_call_message(position, call["tool"], call["args"]))
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{position}",
                "name": call["tool"],
                "content": call.get("observation") or "（无观察）",
            }
        )
    return renumber(messages)


def build_missing_call_pair(trace: dict) -> dict | None:
    """实际链是期望链的严格前缀 → 模型提前收尾，应当继续调用下一个工具。

    覆盖两类失败：
    - `actual` 为空（一次都没调用就说「已经处理好了」）
    - `actual` 少末尾步骤（典型是漏掉 `create_chart` 却声称图画好了）

    要求实际链每一步都成功且与期望前缀逐位一致，否则属于别的失败形态。
    """
    actual = trace["actual"]
    expected = trace["expected"]
    if not expected or len(actual) >= len(expected):
        return None
    if not all(call.get("ok") for call in actual):
        return None
    for index, call in enumerate(actual):
        if expected[index]["tool"] != call["tool"]:
            return None
        if expected[index]["args"] != call["args"]:
            return None

    answer = (trace.get("final_answer") or "").strip()
    if not answer:
        return None
    target = expected[len(actual)]
    return {
        "kind": "no_call" if not actual else "missing_call",
        "source": "holdout_failure",
        "sample_id": trace["sample_id"],
        "tools": trace["tools"],
        "prompt": prefix_context(trace),
        "chosen": tool_call_message(1, target["tool"], target["args"]),
        "rejected": text_message(answer),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从真实失败轨迹构造 DPO 偏好对")
    parser.add_argument("--traces", default="build/traces-dpo.json")
    parser.add_argument("--out", default="build/dpo-pairs-dpo.jsonl")
    parser.add_argument(
        "--per-trace",
        type=int,
        default=2,
        help="每条轨迹最多造几对修正对（同一错误重试多次时取前几次）",
    )
    parser.add_argument(
        "--kinds",
        default="fix_after_error,missing_call,silent_wrong_args",
        help=(
            "要构造的偏好对类型：fix_after_error=报错后修正；"
            "missing_call=漏步/未调用；silent_wrong_args=参数写错但不报错"
        ),
    )
    args = parser.parse_args()

    path = Path(args.traces)
    if not path.is_file():
        print(f"找不到轨迹文件 {path}，请先运行 pipeline/capture_traces.py")
        return 2
    traces = json.loads(path.read_text(encoding="utf-8"))
    kinds = {item.strip() for item in args.kinds.split(",") if item.strip()}

    pairs: list[dict] = []
    kinds_count: Counter = Counter()
    rejects: Counter = Counter()
    for trace in traces:
        built = 0
        if "fix_after_error" in kinds:
            for position, call in enumerate(trace["actual"]):
                if built >= args.per_trace:
                    break
                if call.get("ok"):
                    continue
                pair = build_pair(trace, repeat_index=position)
                if pair is None:
                    rejects["not_alignable"] += 1
                    continue
                pair["pair_id"] = f"realdpo-{len(pairs) + 1:04d}"
                pairs.append(pair)
                kinds_count[pair["kind"]] += 1
                built += 1
        if "missing_call" in kinds:
            pair = build_missing_call_pair(trace)
            if pair is not None:
                pair["pair_id"] = f"realdpo-{len(pairs) + 1:04d}"
                pairs.append(pair)
                kinds_count[pair["kind"]] += 1
                built += 1
        if "silent_wrong_args" in kinds:
            for position in range(len(trace["actual"])):
                pair = build_silent_wrong_args_pair(trace, position=position)
                if pair is None:
                    continue
                pair["pair_id"] = f"realdpo-{len(pairs) + 1:04d}"
                pairs.append(pair)
                kinds_count[pair["kind"]] += 1
                built += 1
                break  # 每条轨迹只取第一个静默错误，避免同类样本重复
        if built == 0:
            rejects["no_pair"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in pairs) + "\n",
        encoding="utf-8",
    )

    print(f"产出 {len(pairs)} 对 → {out}")
    if kinds_count:
        print(f"类型分布：{dict(kinds_count.most_common())}")
    if rejects:
        print(f"跳过：{dict(rejects)}")

    # 统计：chosen 与 rejected 差在哪（只对两边都是工具调用的对有意义）
    diff_fields: Counter = Counter()
    for pair in pairs:
        rejected_calls = pair["rejected"].get("tool_calls") or []
        if not rejected_calls:
            continue
        c = json.loads(pair["chosen"]["tool_calls"][0]["function"]["arguments"])
        r = json.loads(rejected_calls[0]["function"]["arguments"])
        for key in set(c) | set(r):
            if c.get(key) != r.get(key):
                diff_fields[key] += 1
    if diff_fields:
        print(f"chosen 与 rejected 不同的参数项：{dict(diff_fields.most_common())}")

    if pairs:
        sample = pairs[0]
        print("\n--- 抽查 ---")
        print(f"样本 {sample['sample_id']}（{sample['kind']}）")
        for message in sample["prompt"][-2:]:
            if message.get("tool_calls"):
                call = message["tool_calls"][0]["function"]
                print(f"  [assistant] {call['name']}({call['arguments'][:90]})")
            else:
                print(f"  [{message['role']}] {str(message.get('content'))[:110]}")
        chosen = sample["chosen"]["tool_calls"][0]["function"]
        print(f"  chosen  : {chosen['name']}({chosen['arguments'][:100]})")
        rejected_calls = sample["rejected"].get("tool_calls") or []
        if rejected_calls:
            rejected = rejected_calls[0]["function"]
            print(f"  rejected: {rejected['name']}({rejected['arguments'][:100]})")
        else:
            print(f"  rejected: [文本] {sample['rejected']['content'][:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
