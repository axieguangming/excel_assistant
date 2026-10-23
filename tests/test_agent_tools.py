import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from langchain_core.messages import AIMessage

import agent
import excel_ops
import app as api_app


def tool_call(name, args, call_id):
    return AIMessage(
        content="",
        tool_calls=[{
            "name": name,
            "args": args,
            "id": call_id,
            "type": "tool_call",
        }],
    )


class ScriptedToolCallingModel:
    def __init__(self, responses):
        self.responses = iter(responses)

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        response = next(self.responses)
        return AIMessage(content=response) if isinstance(response, str) else response


class AgentToolWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.previous_llm = agent.llm
        self.previous_logging = os.environ.get("TRAIN_DATA_LOGGING")
        os.environ.pop("TRAIN_DATA_LOGGING", None)

    def tearDown(self):
        agent.llm = self.previous_llm
        if self.previous_logging is None:
            os.environ.pop("TRAIN_DATA_LOGGING", None)
        else:
            os.environ["TRAIN_DATA_LOGGING"] = self.previous_logging

    def _state(self, question):
        df = pd.DataFrame({
            "区域": ["东", "西", "东", "东"],
            "年份": [2025, 2025, 2024, 2025],
            "销售额": [100, 80, 50, 25],
        })
        return {
            "df": df,
            "df_info": f"字段：{list(df.columns)}",
            "question": question,
            "history": [],
            "messages": [],
            "current_df": df.copy(),
            "plans": [],
            "tool_steps": 0,
            "turns": 0,
            "result_df": None,
            "final_answer": "",
            "chart_file": "",
            "code": "",
            "user_prompt": "",
        }

    def test_agent_sequentially_filters_aggregates_sorts_and_charts(self):
        agent.llm = ScriptedToolCallingModel([
            tool_call("filter_rows", {
                "filters": [{
                    "column": "年份",
                    "operator": "eq",
                    "value": 2025,
                }],
            }, "filter-1"),
            tool_call("aggregate_data", {
                "group_by": ["区域"],
                "metrics": [{"column": "销售额", "function": "sum"}],
            }, "aggregate-1"),
            tool_call("sort_rows", {
                "sort_by": [{"column": "销售额_sum", "ascending": False}],
            }, "sort-1"),
            tool_call("create_chart", {
                "chart_type": "bar",
                "chart_x": "区域",
                "chart_y": "销售额_sum",
                "chart_title": "2025 区域销售额",
            }, "chart-1"),
            AIMessage(content="2025 年东区销售额 125，西区 80。"),
        ])

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(excel_ops, "OUTPUT_DIR", output_dir):
                result = agent.invoke_agent(self._state("筛选2025年，按区域汇总销售额、降序排序并画柱状图"))

            chart_name = result["chart_file"].rsplit("/", 1)[-1]
            self.assertTrue(os.path.isfile(os.path.join(output_dir, chart_name)))
            os.remove(os.path.join(output_dir, chart_name))

        self.assertEqual(result["result_df"]["区域"].tolist(), ["东", "西"])
        self.assertEqual(result["result_df"]["销售额_sum"].tolist(), [125, 80])
        self.assertEqual(result["tool_steps"], 4)
        self.assertEqual(result["final_answer"], "2025 年东区销售额 125，西区 80。")
        self.assertEqual(len(result["plans"]), 4)

    def test_agent_adds_computed_column_before_aggregate(self):
        agent.llm = ScriptedToolCallingModel([
            tool_call("add_computed_column", {
                "name": "含税销售额",
                "operator": "multiply",
                "left": "销售额",
                "right_value": 1.13,
                "round_digits": 2,
            }, "compute-1"),
            tool_call("aggregate_data", {
                "group_by": ["区域"],
                "metrics": [{"column": "含税销售额", "function": "sum"}],
            }, "aggregate-1"),
            AIMessage(content="已按区域汇总含税销售额。"),
        ])

        result = agent.invoke_agent(self._state("给销售额加13%税后按区域汇总"))

        self.assertEqual(result["result_df"]["区域"].tolist(), ["东", "西"])
        self.assertAlmostEqual(result["result_df"]["含税销售额_sum"].tolist()[0], 197.75)
        self.assertAlmostEqual(result["result_df"]["含税销售额_sum"].tolist()[1], 90.4)
        self.assertEqual(result["tool_steps"], 2)

    def test_tool_error_is_returned_to_model_and_corrected(self):
        agent.llm = ScriptedToolCallingModel([
            tool_call("filter_rows", {
                "filters": [{"column": "不存在", "operator": "eq", "value": "x"}],
            }, "bad-filter"),
            tool_call("aggregate_data", {
                "group_by": ["区域"],
                "metrics": [{"column": "销售额", "function": "sum"}],
            }, "good-aggregate"),
            AIMessage(content="按区域统计已完成。"),
        ])

        result = agent.invoke_agent(self._state("按区域汇总销售额"))

        self.assertEqual(result["result_df"]["销售额_sum"].tolist(), [175, 80])
        self.assertEqual(result["tool_steps"], 2)
        tool_messages = [message for message in result["messages"] if message.type == "tool"]
        self.assertIn("找不到字段", tool_messages[0].content)
        self.assertIn("操作成功", tool_messages[1].content)

    def test_optional_tool_arguments_use_their_declared_defaults(self):
        df = pd.DataFrame({"区域": ["东", None, "西"], "销售额": [100, 80, 50]})
        state = self._state("补齐区域后统计销售额")
        state["df"] = df
        state["df_info"] = f"字段：{list(df.columns)}"
        agent.llm = ScriptedToolCallingModel([
            tool_call("fill_missing_values", {
                "fill_strategy": "ffill",
            }, "fill-1"),
            tool_call("aggregate_data", {
                "group_by": ["区域"],
                "metrics": [{"column": "销售额", "function": "sum"}],
            }, "aggregate-1"),
            AIMessage(content="汇总完成。"),
        ])

        result = agent.invoke_agent(state)

        self.assertEqual(result["result_df"]["销售额_sum"].tolist(), [180, 50])

    def test_aggregate_tool_defaults_to_a_single_overall_group(self):
        agent.llm = ScriptedToolCallingModel([
            tool_call("aggregate_data", {
                "metrics": [{"column": "销售额", "function": "sum"}],
            }, "aggregate-all"),
            AIMessage(content="总销售额为 255。"),
        ])

        result = agent.invoke_agent(self._state("计算总销售额"))

        self.assertEqual(result["result_df"].loc[0, "销售额_sum"], 255)

    def test_app_run_agent_keeps_the_existing_response_contract(self):
        agent.llm = ScriptedToolCallingModel([
            tool_call("top_n_rows", {
                "sort_by": [{"column": "销售额", "ascending": False}],
                "limit": 1,
            }, "top-1"),
            AIMessage(content="最高销售额为 100。"),
        ])

        result = api_app.run_agent(
            self._state("销售额最高的一行")["df"],
            "销售额最高的一行",
            [],
        )

        self.assertEqual(result["result_df"]["销售额"].tolist(), [100])
        self.assertEqual(result["final_answer"], "最高销售额为 100。")
        self.assertIn("top_n_rows", result["code"])

    def test_agent_rejects_multiple_tool_calls_in_one_turn(self):
        parallel_response = AIMessage(content="", tool_calls=[
            {
                "name": "describe_data",
                "args": {},
                "id": "parallel-1",
                "type": "tool_call",
            },
            {
                "name": "deduplicate_rows",
                "args": {"subset": []},
                "id": "parallel-2",
                "type": "tool_call",
            },
        ])
        agent.llm = ScriptedToolCallingModel([
            parallel_response,
            tool_call("describe_data", {}, "describe-1"),
            AIMessage(content="已完成描述统计。"),
        ])

        result = agent.invoke_agent(self._state("描述表格"))

        self.assertEqual(result["tool_steps"], 1)
        self.assertIn("每轮只能调用一个工具", result["messages"][2].content)

    def test_maximum_agent_turns_force_a_final_response(self):
        agent.llm = ScriptedToolCallingModel([
            *[
                tool_call("describe_data", {}, f"describe-{index}")
                for index in range(agent.MAX_AGENT_TURNS)
            ],
            AIMessage(content="达到操作步骤上限，已总结现有结果。"),
        ])

        result = agent.invoke_agent(self._state("描述表格"))

        self.assertEqual(result["turns"], agent.MAX_AGENT_TURNS)
        self.assertEqual(result["tool_steps"], agent.MAX_AGENT_TURNS - 1)
        self.assertIn("步骤上限", result["final_answer"])

    def test_training_sample_logging_is_opt_in(self):
        with tempfile.TemporaryDirectory() as data_dir:
            with patch.dict(os.environ, {"TRAIN_DATA_DIR": data_dir}, clear=False):
                excel_ops._log_tool_sample(
                    {"user_prompt": "workbook sample"},
                    '{"tool":"describe_data"}',
                    True,
                    "",
                )
                self.assertFalse(os.path.exists(os.path.join(data_dir, "codegen_log.jsonl")))

                os.environ["TRAIN_DATA_LOGGING"] = "1"
                excel_ops._log_tool_sample(
                    {"user_prompt": "authorized sample"},
                    '{"tool":"describe_data"}',
                    True,
                    "",
                )

            with open(os.path.join(data_dir, "codegen_log.jsonl"), encoding="utf-8") as file:
                self.assertIn("authorized sample", file.read())


if __name__ == "__main__":
    unittest.main()
