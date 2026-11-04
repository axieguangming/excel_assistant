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
        # 模型运行时是进程级单例，测试间必须清理，避免相互影响。
        from local_model_runtime import set_runtime

        self.previous_runtime = __import__(
            "local_model_runtime", fromlist=["get_runtime"]
        ).get_runtime()
        set_runtime(None)

    def tearDown(self):
        from local_model_runtime import set_runtime

        set_runtime(self.previous_runtime)
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

    def test_chat_returns_actionable_error_when_local_model_is_offline(self):
        """本地模型服务未启动时，接口应返回可操作的 503 而不是 500。"""
        import asyncio

        from fastapi import HTTPException

        from schemas import ChatRequest

        df = pd.DataFrame({"区域": ["东"], "销售额": [100]})
        api_app.SESSIONS["offline-session"] = {"df": df, "history": []}
        self.addCleanup(api_app.SESSIONS.pop, "offline-session", None)

        async def fail_immediately(*args, **kwargs):
            raise ConnectionError("Connection refused")

        with (
            patch.object(api_app, "run_in_threadpool", fail_immediately),
            patch("load_model.current_provider", return_value="local"),
        ):
            with self.assertRaises(HTTPException) as captured:
                asyncio.run(
                    api_app.chat(
                        ChatRequest(session_id="offline-session", question="汇总销售额")
                    )
                )

        self.assertEqual(captured.exception.status_code, 503)
        self.assertIn("本地模型服务", captured.exception.detail)

    def test_chat_propagates_unrelated_errors(self):
        import asyncio

        from schemas import ChatRequest

        df = pd.DataFrame({"区域": ["东"], "销售额": [100]})
        api_app.SESSIONS["broken-session"] = {"df": df, "history": []}
        self.addCleanup(api_app.SESSIONS.pop, "broken-session", None)

        async def fail_immediately(*args, **kwargs):
            raise KeyError("unexpected")

        with patch.object(api_app, "run_in_threadpool", fail_immediately):
            with self.assertRaises(KeyError):
                asyncio.run(
                    api_app.chat(
                        ChatRequest(session_id="broken-session", question="汇总销售额")
                    )
                )

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


class ModelStatusEndpointTests(unittest.TestCase):
    """验证界面依赖的两个模型接口。"""

    def setUp(self):
        from local_model_runtime import set_runtime

        self.previous_runtime = __import__(
            "local_model_runtime", fromlist=["get_runtime"]
        ).get_runtime()
        self.previous_llm = agent.llm
        set_runtime(None)
        self.addCleanup(set_runtime, self.previous_runtime)
        self.addCleanup(setattr, agent, "llm", self.previous_llm)

    def _runtime(self, state: str, **extra):
        class StubRuntime:
            def __init__(self):
                self.manifest = type("M", (), {"is_configured": state != "no_manifest"})()
                self.downloader = type(
                    "D", (), {"is_installed": lambda self_: state == "ready"}
                )()
                self.download_calls = 0

            def status(self):
                return {"provider": "local", "state": state, **extra}

            def start_download(self):
                self.download_calls += 1
                return True

        return StubRuntime()

    def test_status_returns_disabled_outside_local_mode(self):
        import asyncio
        from unittest.mock import patch

        with patch.object(api_app, "get_runtime", return_value=None), patch.object(
            api_app, "current_provider", return_value="remote"
        ):
            payload = asyncio.run(api_app.model_status())

        self.assertEqual(payload["state"], "disabled")
        self.assertFalse(payload["can_download"])

    def test_status_delegates_to_runtime(self):
        import asyncio

        from local_model_runtime import set_runtime

        runtime = self._runtime("needs_download", can_download=True, message="下载吧")
        set_runtime(runtime)

        payload = asyncio.run(api_app.model_status())

        self.assertEqual(payload["state"], "needs_download")
        self.assertEqual(payload["message"], "下载吧")

    def test_download_endpoint_rejects_remote_mode(self):
        import asyncio

        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as captured:
            asyncio.run(api_app.model_download())

        self.assertEqual(captured.exception.status_code, 400)

    def test_download_endpoint_starts_download(self):
        import asyncio

        from local_model_runtime import set_runtime

        runtime = self._runtime("needs_download")
        set_runtime(runtime)

        payload = asyncio.run(api_app.model_download())

        self.assertTrue(payload["started"])
        self.assertEqual(runtime.download_calls, 1)

    def test_download_endpoint_is_noop_when_model_exists(self):
        import asyncio

        from local_model_runtime import set_runtime

        runtime = self._runtime("ready")
        set_runtime(runtime)

        payload = asyncio.run(api_app.model_download())

        self.assertFalse(payload["started"])
        self.assertEqual(runtime.download_calls, 0)

    def test_download_endpoint_requires_a_download_url(self):
        import asyncio

        from fastapi import HTTPException
        from local_model_runtime import set_runtime

        set_runtime(self._runtime("no_manifest"))

        with self.assertRaises(HTTPException) as captured:
            asyncio.run(api_app.model_download())

        self.assertEqual(captured.exception.status_code, 400)

    def test_chat_refuses_while_model_is_not_ready(self):
        import asyncio

        from fastapi import HTTPException
        from local_model_runtime import set_runtime

        df = pd.DataFrame({"区域": ["东"], "销售额": [100]})
        api_app.SESSIONS["pending-model"] = {"df": df, "history": []}
        self.addCleanup(api_app.SESSIONS.pop, "pending-model", None)
        set_runtime(self._runtime("needs_download"))

        with self.assertRaises(HTTPException) as captured:
            asyncio.run(
                api_app.chat(
                    api_app.ChatRequest(session_id="pending-model", question="汇总")
                )
            )

        self.assertEqual(captured.exception.status_code, 503)
        self.assertIn("模型下载", captured.exception.detail)


class AmbiguityGuidanceTests(unittest.TestCase):
    """系统提示必须要求模型在歧义时澄清，而不是自行假设。"""

    def test_prompt_requires_count_semantics_clarification(self):
        # 「订单数」既可理解为记录条数，也可理解为去重编号个数；
        # 两者的结果在有重复/空值时不同，必须问清而不是猜。
        self.assertIn("计数口径", agent.AGENT_SYSTEM)
        self.assertIn("去重", agent.AGENT_SYSTEM)
        self.assertIn("记录条数", agent.AGENT_SYSTEM)

    def test_prompt_requires_aggregation_function_clarification(self):
        self.assertIn("统计方式", agent.AGENT_SYSTEM)

    def test_prompt_requires_metric_column_clarification(self):
        self.assertIn("指标所指", agent.AGENT_SYSTEM)

    def test_prompt_discourages_restating_known_conditions(self):
        # 澄清要短，只问缺的那一点。
        self.assertIn("不要复述已知条件", agent.AGENT_SYSTEM)

    def test_prompt_still_forbids_claiming_unexecuted_work(self):
        # 新增歧义规则不能削弱原有的"不得谎称已完成"约束。
        self.assertIn("不得声称未执行的操作已经完成", agent.AGENT_SYSTEM)


if __name__ == "__main__":
    unittest.main()
