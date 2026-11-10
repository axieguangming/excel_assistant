"""DPO 偏好对构造的单元测试。

关注三件事：
1. chosen 与 rejected 必须真的不同（否则没有对比信号）
2. `success_repeat` 的 rejected 必须是「上一步调用的原样重复」（同工具同参数）
3. 合法的连续同工具调用（参数不同）不能被当成反例
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module():
    spec = importlib.util.spec_from_file_location(
        "make_dpo_under_test", ROOT / "pipeline" / "make_dpo_pairs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tool_call(name: str, args: dict, call_id: str = "c") -> dict:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
        ],
    }


def tool_obs(name: str, content: str = "操作成功。", call_id: str = "c") -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


def make_record(chains: list[tuple[str, dict]]) -> dict:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "问题"},
    ]
    for index, (name, args) in enumerate(chains, start=1):
        messages.append(tool_call(name, args, f"c{index}"))
        messages.append(tool_obs(name, call_id=f"c{index}"))
    messages.append({"role": "assistant", "content": "完成。"})
    return {"sample_id": "wb-1-1", "tools": [], "messages": messages}


class SuccessRepeatPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_rejected_is_previous_call_repeated(self):
        record = make_record([
            ("aggregate_data", {"group_by": ["月份"]}),
            ("sort_rows", {"sort_by": [{"column": "合计"}]}),
        ])
        pairs = self.module.build_success_repeat_pairs(record, random.Random(0))
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        rejected = json.loads(pair["rejected"]["tool_calls"][0]["function"]["arguments"])
        chosen = json.loads(pair["chosen"]["tool_calls"][0]["function"]["arguments"])
        # rejected = 上一步（aggregate_data）
        self.assertEqual(pair["rejected"]["tool_calls"][0]["function"]["name"], "aggregate_data")
        self.assertEqual(rejected, {"group_by": ["月份"]})
        # chosen = 下一步
        self.assertEqual(pair["chosen"]["tool_calls"][0]["function"]["name"], "sort_rows")
        self.assertNotEqual(chosen, rejected)

    def test_creates_one_pair_per_transition(self):
        record = make_record([
            ("filter_rows", {"filters": []}),
            ("aggregate_data", {"group_by": ["a"]}),
            ("create_chart", {"chart_type": "bar", "chart_y": "x"}),
        ])
        pairs = self.module.build_success_repeat_pairs(record, random.Random(0))
        self.assertEqual(len(pairs), 2)

    def test_prompt_ends_with_previous_observation(self):
        record = make_record([
            ("aggregate_data", {"group_by": ["月份"]}),
            ("sort_rows", {"sort_by": [{"column": "合计"}]}),
        ])
        pair = self.module.build_success_repeat_pairs(record, random.Random(0))[0]
        self.assertEqual(pair["prompt"][-1]["role"], "tool")
        self.assertEqual(pair["prompt"][-1]["name"], "aggregate_data")

    def test_single_step_has_no_pair(self):
        record = make_record([("describe_data", {})])
        self.assertEqual(self.module.build_success_repeat_pairs(record, random.Random(0)), [])

    def test_identical_consecutive_call_is_skipped(self):
        """上一步与下一步完全相同（同工具同参数）时，不该拿它当过渡点。"""
        record = make_record([
            ("aggregate_data", {"group_by": ["月份"]}),
            ("aggregate_data", {"group_by": ["月份"]}),
        ])
        self.assertEqual(self.module.build_success_repeat_pairs(record, random.Random(0)), [])

    def test_legitimate_repeat_with_different_args_is_not_rejected(self):
        """合法的连续同工具调用（参数不同）不能被当作「无理由重复」反例。"""
        record = make_record([
            ("fill_missing_values", {"fill_strategy": "value", "subset": ["班次"], "fill_value": "未知"}),
            ("fill_missing_values", {"fill_strategy": "value", "subset": ["加班时长"], "fill_value": 0}),
            ("sort_rows", {"sort_by": [{"column": "工时"}]}),
        ])
        pairs = self.module.build_success_repeat_pairs(record, random.Random(0))
        self.assertEqual(len(pairs), 2)
        # 第一个过渡点：chosen 是第二次 fill（不同参数），rejected 是第一次 fill
        chosen_args = json.loads(pairs[0]["chosen"]["tool_calls"][0]["function"]["arguments"])
        rejected_args = json.loads(pairs[0]["rejected"]["tool_calls"][0]["function"]["arguments"])
        self.assertNotEqual(chosen_args, rejected_args)


class PromptIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_tool_call_ids_are_unique_in_prompt(self):
        """prompt 里的 tool_call id 必须与后续 chosen/rejected 不冲突。"""
        record = make_record([
            ("aggregate_data", {"group_by": ["月份"]}),
            ("sort_rows", {"sort_by": [{"column": "合计"}]}),
            ("create_chart", {"chart_type": "bar", "chart_y": "x"}),
        ])
        pair = self.module.build_success_repeat_pairs(record, random.Random(0))[1]
        ids = [
            call["id"]
            for message in pair["prompt"]
            for call in (message.get("tool_calls") or [])
        ]
        self.assertEqual(len(ids), len(set(ids)), "prompt 内 tool_call id 重复")
        self.assertNotIn("call_1", ids, "prompt 不应沿用 chosen/rejected 的 id")

    def test_tool_messages_match_their_call_ids(self):
        record = make_record([
            ("aggregate_data", {"group_by": ["月份"]}),
            ("sort_rows", {"sort_by": [{"column": "合计"}]}),
        ])
        pair = self.module.build_success_repeat_pairs(record, random.Random(0))[0]
        call_ids = {
            call["id"]
            for message in pair["prompt"]
            for call in (message.get("tool_calls") or [])
        }
        for message in pair["prompt"]:
            if message["role"] == "tool":
                self.assertIn(message["tool_call_id"], call_ids)


if __name__ == "__main__":
    unittest.main()
