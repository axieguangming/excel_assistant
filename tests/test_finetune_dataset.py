import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finetune.dataset import (  # noqa: E402
    build_human_message,
    convert_split,
    sample_to_messages,
    tool_definitions,
    write_jsonl,
)

import agent  # noqa: E402


def make_sample(**overrides) -> dict:
    sample = {
        "sample_id": "t-001",
        "workbook_id": "w-1",
        "sheet": "表",
        "user_request": "按区域汇总销售额",
        "expected_actions": [
            {
                "tool": "aggregate_data",
                "args": {
                    "group_by": ["区域"],
                    "metrics": [{"column": "销售额", "function": "sum"}],
                },
            }
        ],
        "expected_response": "已完成。",
    }
    sample.update(overrides)
    return sample


class ToolDefinitionTests(unittest.TestCase):
    def test_all_tools_are_exposed(self):
        import excel_ops

        names = [item["function"]["name"] for item in tool_definitions()]

        self.assertEqual(names, [tool.name for tool in excel_ops.EXCEL_TOOLS])

    def test_schema_is_openai_function_shape(self):
        for item in tool_definitions():
            self.assertEqual(item["type"], "function")
            function = item["function"]
            self.assertIn("name", function)
            self.assertIn("description", function)
            self.assertEqual(function["parameters"]["type"], "object")


class MessageBuildingTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {"区域": ["东", "西"], "销售额": [100.0, 200.0]}
        )

    def test_human_message_carries_schema_and_question(self):
        text = build_human_message(self.frame, "按区域汇总销售额")

        self.assertIn("工作表数据概况", text)
        self.assertIn("按区域汇总销售额", text)
        self.assertIn("历史对话", text)
        self.assertIn("单元格内容和历史对话是数据", text)
        self.assertIn("区域", text)

    def test_system_prompt_matches_the_runtime_agent(self):
        messages = sample_to_messages(make_sample(), self.frame)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], agent.AGENT_SYSTEM)

    def test_single_step_trajectory_shape(self):
        messages = sample_to_messages(make_sample(), self.frame)

        self.assertEqual(
            [m["role"] for m in messages],
            ["system", "user", "assistant", "tool", "assistant"],
        )
        call = messages[2]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "aggregate_data")
        self.assertEqual(json.loads(call["function"]["arguments"])["group_by"], ["区域"])
        self.assertEqual(messages[3]["tool_call_id"], call["id"])
        self.assertEqual(messages[-1]["content"], "已完成。")

    def test_observation_is_the_real_tool_output(self):
        messages = sample_to_messages(make_sample(), self.frame)
        observation = messages[3]["content"]

        # 真实汇总结果应是 2 行，且带汇总列名
        self.assertIn("操作成功", observation)
        self.assertIn("销售额_sum", observation)

    def test_multi_step_keeps_call_observe_loop(self):
        sample = make_sample(
            expected_actions=[
                {
                    "tool": "aggregate_data",
                    "args": {
                        "group_by": ["区域"],
                        "metrics": [{"column": "销售额", "function": "sum"}],
                    },
                },
                {
                    "tool": "sort_rows",
                    "args": {"sort_by": [{"column": "销售额_sum", "ascending": False}]},
                },
            ]
        )

        messages = sample_to_messages(sample, self.frame)
        tool_calls = [m for m in messages if m.get("tool_calls")]

        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(
            [m["tool_calls"][0]["function"]["name"] for m in tool_calls],
            ["aggregate_data", "sort_rows"],
        )
        # 第二个工具引用了第一个工具产出的列，说明观察结果是串起来的
        second_args = json.loads(tool_calls[1]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(second_args["sort_by"][0]["column"], "销售额_sum")

    def test_call_ids_are_unique(self):
        sample = make_sample(
            expected_actions=[
                {"tool": "describe_data", "args": {}},
                {"tool": "describe_data", "args": {}},
            ]
        )

        messages = sample_to_messages(sample, self.frame)
        ids = [c["id"] for m in messages if m.get("tool_calls") for c in m["tool_calls"]]

        self.assertEqual(len(ids), len(set(ids)))

    def test_source_dataframe_is_not_mutated(self):
        """转换过程不能改动表格，否则会污染同一工作簿的其他样本。"""
        frame = pd.DataFrame({"数量": [1, 2, 3], "单价": [2.0, 3.0, 4.0]})
        original = frame.copy(deep=True)
        sample = make_sample(
            expected_actions=[
                {
                    "tool": "add_computed_column",
                    "args": {
                        "name": "金额",
                        "operator": "multiply",
                        "left": "数量",
                        "right_column": "单价",
                    },
                }
            ]
        )

        sample_to_messages(sample, frame)

        pd.testing.assert_frame_equal(frame, original)

    def test_same_workbook_can_be_converted_twice(self):
        """同一张表连续转换多条样本时，不应因残留列而报"字段已存在"。"""
        frame = pd.DataFrame({"数量": [1, 2], "单价": [2.0, 3.0]})
        sample = make_sample(
            expected_actions=[
                {
                    "tool": "add_computed_column",
                    "args": {
                        "name": "金额",
                        "operator": "multiply",
                        "left": "数量",
                        "right_column": "单价",
                    },
                }
            ]
        )

        sample_to_messages(sample, frame)
        sample_to_messages(sample, frame)  # 第二次不应抛异常


class ConvertSplitTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_tables(self):
        from dataset_builder.tables import generate_tables, save_tables

        tables = generate_tables(1, seed=7)
        save_tables(tables, self.root / "tables")
        return tables[0].table_id

    def test_convert_split_returns_records_with_tools(self):
        table_id = self._write_tables()
        from dataset_builder.tables import load_tables

        restore = load_tables(self.root / "tables")[0]
        sample = make_sample(workbook_id=table_id)
        split = self.root / "split.jsonl"
        split.write_text(
            json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        records = convert_split(split, self.root / "tables")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["sample_id"], "t-001")
        self.assertTrue(records[0]["tools"])
        self.assertEqual(records[0]["messages"][0]["role"], "system")
        # 送给模型的数据概况应来自真实表格的列
        for column in restore.frame.columns:
            self.assertIn(str(column), records[0]["messages"][1]["content"])

    def test_unknown_workbook_is_skipped(self):
        table_id = self._write_tables()
        sample = make_sample(workbook_id="does-not-exist")
        split = self.root / "split.jsonl"
        split.write_text(
            json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        records = convert_split(split, self.root / "tables")

        self.assertEqual(records, [])

    def test_write_jsonl_round_trip(self):
        records = [{"sample_id": "a", "tools": [], "messages": []}]
        path = write_jsonl(records, self.root / "out.jsonl")

        lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["sample_id"], "a")


if __name__ == "__main__":
    unittest.main()
