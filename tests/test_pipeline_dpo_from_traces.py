"""真实失败轨迹 → DPO 偏好对 的单元测试。

核心断言：
1. prompt 必须以「失败调用 + 报错观察」结尾（模型要看到错误）
2. chosen 是期望参数，rejected 是模型当时的错误参数
3. 参数本来就正确、或工具名对不上位置时，不产出偏好对
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))


def load_module():
    spec = importlib.util.spec_from_file_location(
        "pairs_from_traces_under_test",
        ROOT / "pipeline" / "make_dpo_pairs_from_traces.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trace(
    *,
    actual: list[dict],
    expected: list[dict],
    final_answer: str = "",
) -> dict:
    return {
        "group": "a",
        "sample_id": "wb-20260301-001",
        "question": "按班次汇总",
        "system_prompt": "你是 Excel Agent。",
        "user_prompt": "工作表数据概况：...\n本轮用户问题：按班次汇总平均工时",
        "tools": [],
        "expected": expected,
        "actual": actual,
        "final_answer": final_answer,
    }


ERR = "工具执行失败：找不到字段：['平均工时']。可用字段：['班次','工时','加班时长']"


class BuildPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def _failing(self):
        return trace(
            actual=[
                {"tool": "aggregate_data", "args": {"metrics": [{"column": "平均工时"}]},
                 "ok": False, "observation": ERR},
            ],
            expected=[
                {"tool": "aggregate_data", "args": {"metrics": [{"column": "工时"}]}},
            ],
        )

    def test_builds_pair_with_error_in_prompt(self):
        pair = self.module.build_pair(self._failing(), repeat_index=0)
        self.assertIsNotNone(pair)
        # prompt 最后两条：失败调用 + 报错观察
        self.assertEqual(pair["prompt"][-2]["role"], "assistant")
        self.assertEqual(pair["prompt"][-1]["role"], "tool")
        self.assertIn("可用字段", pair["prompt"][-1]["content"])

    def test_chosen_is_expected_and_rejected_is_model_output(self):
        pair = self.module.build_pair(self._failing(), repeat_index=0)
        chosen = json.loads(pair["chosen"]["tool_calls"][0]["function"]["arguments"])
        rejected = json.loads(pair["rejected"]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(chosen["metrics"][0]["column"], "工时")
        self.assertEqual(rejected["metrics"][0]["column"], "平均工时")
        self.assertNotEqual(chosen, rejected)

    def test_successful_call_is_skipped(self):
        ok_trace = trace(
            actual=[{"tool": "aggregate_data", "args": {"a": 1}, "ok": True, "observation": "ok"}],
            expected=[{"tool": "aggregate_data", "args": {"a": 1}}],
        )
        self.assertIsNone(self.module.build_pair(ok_trace, repeat_index=0))

    def test_tool_name_mismatch_is_skipped(self):
        mismatched = trace(
            actual=[{"tool": "filter_rows", "args": {"x": 1}, "ok": False, "observation": ERR}],
            expected=[{"tool": "aggregate_data", "args": {"x": 2}}],
        )
        self.assertIsNone(self.module.build_pair(mismatched, repeat_index=0))

    def test_identical_args_is_skipped(self):
        """期望参数与失败参数相同，说明不是「参数写错」，没有对比信号。"""
        same = trace(
            actual=[{"tool": "aggregate_data", "args": {"a": 1}, "ok": False, "observation": ERR}],
            expected=[{"tool": "aggregate_data", "args": {"a": 1}}],
        )
        self.assertIsNone(self.module.build_pair(same, repeat_index=0))

    def test_out_of_range_index_returns_none(self):
        self.assertIsNone(self.module.build_pair(self._failing(), repeat_index=5))

    def test_prompt_ids_are_renumbered(self):
        pair = self.module.build_pair(self._failing(), repeat_index=0)
        ids = [
            call["id"]
            for message in pair["prompt"]
            for call in (message.get("tool_calls") or [])
        ]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("call_1", ids, "prompt 不应与 chosen/rejected 的 id 冲突")

    def test_second_step_failure_keeps_prefix(self):
        """若失败发生在第 2 步，prompt 必须包含第 1 步的调用与观察。"""
        two_steps = trace(
            actual=[
                {"tool": "fill_missing_values", "args": {"fill_value": 0}, "ok": True,
                 "observation": "操作成功。当前结果为 40 行 × 3 列。"},
                {"tool": "aggregate_data", "args": {"metrics": [{"column": "总工时"}]},
                 "ok": False, "observation": ERR},
            ],
            expected=[
                {"tool": "fill_missing_values", "args": {"fill_value": 0}},
                {"tool": "aggregate_data", "args": {"metrics": [{"column": "工时"}]}},
            ],
        )
        pair = self.module.build_pair(two_steps, repeat_index=1)
        self.assertIsNotNone(pair)
        # system + user + 第1步调用 + 第1步观察 + 第2步调用 + 报错 = 6 条
        self.assertEqual(len(pair["prompt"]), 6)
        self.assertEqual(pair["prompt"][2]["role"], "assistant")
        self.assertEqual(pair["prompt"][3]["role"], "tool")
        self.assertIn("操作成功", pair["prompt"][3]["content"])
        self.assertIn("可用字段", pair["prompt"][-1]["content"])


class MissingCallPairTests(unittest.TestCase):
    """第二轮新增：漏步 / 一次都没调用就作答。"""

    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def _prefix_trace(self, *, final_answer: str = "已经处理好了。"):
        return trace(
            actual=[
                {"tool": "aggregate_data", "args": {"metrics": [{"column": "工时"}]},
                 "ok": True, "observation": "操作成功。当前结果为 5 行 × 2 列。"},
            ],
            expected=[
                {"tool": "aggregate_data", "args": {"metrics": [{"column": "工时"}]}},
                {"tool": "create_chart", "args": {"chart_type": "bar", "chart_y": "工时合计"}},
            ],
            final_answer=final_answer,
        )

    def test_builds_pair_when_final_step_missing(self):
        pair = self.module.build_missing_call_pair(self._prefix_trace())
        self.assertIsNotNone(pair)
        self.assertEqual(pair["kind"], "missing_call")

    def test_chosen_is_next_expected_call(self):
        pair = self.module.build_missing_call_pair(self._prefix_trace())
        chosen = pair["chosen"]["tool_calls"][0]["function"]
        self.assertEqual(chosen["name"], "create_chart")
        args = json.loads(chosen["arguments"])
        self.assertEqual(args["chart_type"], "bar")

    def test_rejected_is_text_answer_not_tool_call(self):
        """漏步的负样本是「直接说完成」的文本，而不是另一次工具调用。"""
        pair = self.module.build_missing_call_pair(self._prefix_trace())
        self.assertNotIn("tool_calls", pair["rejected"])
        self.assertEqual(pair["rejected"]["role"], "assistant")
        self.assertIn("已经处理好了", pair["rejected"]["content"])

    def test_prompt_keeps_completed_prefix(self):
        pair = self.module.build_missing_call_pair(self._prefix_trace())
        # system + user + 已完成的 1 步调用 + 观察 = 4 条
        self.assertEqual(len(pair["prompt"]), 4)
        self.assertIn("操作成功", pair["prompt"][-1]["content"])

    def test_kind_is_no_call_when_nothing_was_called(self):
        empty = trace(
            actual=[],
            expected=[{"tool": "describe_data", "args": {}}],
            final_answer="这表看起来不错。",
        )
        pair = self.module.build_missing_call_pair(empty)
        self.assertIsNotNone(pair)
        self.assertEqual(pair["kind"], "no_call")
        self.assertEqual(pair["prompt"][-1]["role"], "user")

    def test_complete_chain_is_skipped(self):
        complete = trace(
            actual=[
                {"tool": "aggregate_data", "args": {"a": 1}, "ok": True, "observation": "ok"},
                {"tool": "create_chart", "args": {"b": 2}, "ok": True, "observation": "ok"},
            ],
            expected=[
                {"tool": "aggregate_data", "args": {"a": 1}},
                {"tool": "create_chart", "args": {"b": 2}},
            ],
            final_answer="完成。",
        )
        self.assertIsNone(self.module.build_missing_call_pair(complete))

    def test_failed_prefix_is_skipped(self):
        """前缀里有失败调用时属于 fix_after_error 的范畴，不重复造对。"""
        broken = trace(
            actual=[
                {"tool": "aggregate_data", "args": {"a": 1}, "ok": False, "observation": ERR},
            ],
            expected=[
                {"tool": "aggregate_data", "args": {"a": 1}},
                {"tool": "create_chart", "args": {"b": 2}},
            ],
            final_answer="完成。",
        )
        self.assertIsNone(self.module.build_missing_call_pair(broken))

    def test_prefix_arg_mismatch_is_skipped(self):
        drifted = trace(
            actual=[
                {"tool": "aggregate_data", "args": {"a": 999}, "ok": True, "observation": "ok"},
            ],
            expected=[
                {"tool": "aggregate_data", "args": {"a": 1}},
                {"tool": "create_chart", "args": {"b": 2}},
            ],
            final_answer="完成。",
        )
        self.assertIsNone(self.module.build_missing_call_pair(drifted))

    def test_empty_answer_is_skipped(self):
        """没有文字答复就没有可对比的负样本。"""
        self.assertIsNone(
            self.module.build_missing_call_pair(self._prefix_trace(final_answer="   "))
        )


class SilentWrongArgsTests(unittest.TestCase):
    """参数写错但工具没报错——最难发现的失败形态。"""

    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def _silent(self):
        return trace(
            actual=[
                {"tool": "aggregate_data",
                 "args": {"metrics": [{"column": "单价", "function": "sum"}]},
                 "ok": True, "observation": "操作成功。当前结果为 5 行 × 2 列。"},
            ],
            expected=[
                {"tool": "aggregate_data",
                 "args": {"metrics": [{"column": "销售额", "function": "sum"}]}},
            ],
        )

    def test_builds_pair_even_without_error(self):
        pair = self.module.build_silent_wrong_args_pair(self._silent(), position=0)
        self.assertIsNotNone(pair)
        self.assertEqual(pair["kind"], "silent_wrong_args")

    def test_chosen_and_rejected_differ_only_in_args(self):
        pair = self.module.build_silent_wrong_args_pair(self._silent(), position=0)
        chosen = pair["chosen"]["tool_calls"][0]["function"]
        rejected = pair["rejected"]["tool_calls"][0]["function"]
        self.assertEqual(chosen["name"], rejected["name"])
        self.assertNotEqual(chosen["arguments"], rejected["arguments"])
        self.assertIn("销售额", chosen["arguments"])
        self.assertIn("单价", rejected["arguments"])

    def test_prompt_excludes_the_wrong_call(self):
        """prompt 必须停在出错之前，否则模型看到的是自己的错误输出。"""
        pair = self.module.build_silent_wrong_args_pair(self._silent(), position=0)
        self.assertEqual(len(pair["prompt"]), 2)  # system + user
        self.assertEqual(pair["prompt"][-1]["role"], "user")

    def test_error_case_is_left_to_fix_after_error(self):
        """有报错的样本交给 build_pair，避免同一样本造出两对冲突的偏好。"""
        failing = trace(
            actual=[
                {"tool": "aggregate_data", "args": {"a": 1}, "ok": False, "observation": ERR},
            ],
            expected=[{"tool": "aggregate_data", "args": {"a": 2}}],
        )
        self.assertIsNone(self.module.build_silent_wrong_args_pair(failing, position=0))

    def test_identical_args_is_skipped(self):
        same = trace(
            actual=[{"tool": "aggregate_data", "args": {"a": 1}, "ok": True, "observation": "ok"}],
            expected=[{"tool": "aggregate_data", "args": {"a": 1}}],
        )
        self.assertIsNone(self.module.build_silent_wrong_args_pair(same, position=0))

    def test_broken_prefix_is_skipped(self):
        """前缀里有报错时，问题不在当前位置，不该在这里造对。"""
        broken = trace(
            actual=[
                {"tool": "fill_missing_values", "args": {"x": 1}, "ok": False, "observation": ERR},
                {"tool": "aggregate_data", "args": {"a": 1}, "ok": True, "observation": "ok"},
            ],
            expected=[
                {"tool": "fill_missing_values", "args": {"x": 1}},
                {"tool": "aggregate_data", "args": {"a": 2}},
            ],
        )
        self.assertIsNone(self.module.build_silent_wrong_args_pair(broken, position=1))

    def test_second_position_keeps_first_call(self):
        two_steps = trace(
            actual=[
                {"tool": "fill_missing_values", "args": {"fill_value": 0}, "ok": True,
                 "observation": "操作成功。"},
                {"tool": "aggregate_data", "args": {"metrics": [{"column": "单价"}]},
                 "ok": True, "observation": "操作成功。"},
            ],
            expected=[
                {"tool": "fill_missing_values", "args": {"fill_value": 0}},
                {"tool": "aggregate_data", "args": {"metrics": [{"column": "销售额"}]}},
            ],
        )
        pair = self.module.build_silent_wrong_args_pair(two_steps, position=1)
        self.assertIsNotNone(pair)
        self.assertEqual(len(pair["prompt"]), 4)  # system + user + 第1步调用 + 观察
        self.assertIn("操作成功", pair["prompt"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
