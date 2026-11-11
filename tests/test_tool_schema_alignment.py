"""训练/推理 schema 一致性的回归测试。

背景：GGUF 部署后小模型正确率从 85% 掉到 55%。根因不是量化，而是推理侧
LangChain 的 `bind_tools()` 把工具 schema 重新转换了一遍，与训练数据里
发给模型的 schema 不一致（9 个工具 0 个相同，5,704 → 4,558 字符）。

这些测试锁住三件事：
1. 请求体里的 `tools` 与 `tool_schema.tool_definitions()` 完全相等；
2. `describe_data` 的注入参数保持训练时的形态（不要"顺手清理"）；
3. 认证、`extra_body` 等透传能力没被包装层破坏。
"""

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import excel_ops
from schema_llm import TrainingSchemaChatModel, fold_tool_history, strip_injected_params
from tool_schema import describe_tool, tool_definitions


class ToolSchemaTests(unittest.TestCase):
    def test_definitions_cover_all_tools(self):
        names = [item["function"]["name"] for item in tool_definitions()]
        self.assertEqual(names, [tool.name for tool in excel_ops.EXCEL_TOOLS])
        self.assertEqual(len(names), 9)

    def test_describe_data_keeps_training_shape(self):
        """`describe_data` 没有 args_schema，训练时就是两个注入参数。"""
        definition = next(
            item
            for item in tool_definitions()
            if item["function"]["name"] == "describe_data"
        )
        properties = definition["function"]["parameters"]["properties"]
        self.assertEqual(sorted(properties), ["state", "tool_call_id"])

    def test_describe_tool_matches_batch_output(self):
        single = describe_tool(excel_ops.EXCEL_TOOLS[0])
        self.assertEqual(single, tool_definitions()[0])

    def test_strip_injected_only_touches_named_params(self):
        cleaned = strip_injected_params(
            {
                "type": "object",
                "properties": {"state": {"type": "object"}, "top": {"type": "integer"}},
                "required": ["state", "top"],
            }
        )
        self.assertEqual(sorted(cleaned["properties"]), ["top"])
        self.assertEqual(cleaned["required"], ["top"])

    def test_strip_injected_drops_empty_required(self):
        cleaned = strip_injected_params(
            {
                "type": "object",
                "properties": {"state": {"type": "object"}},
                "required": ["state"],
            }
        )
        self.assertNotIn("required", cleaned)


class TrainingSchemaChatModelTests(unittest.TestCase):
    def _payload(self, wrapper_factory):
        """走一次 invoke，拦下真正要发出去的请求体。"""

        class FakeOpenAI:
            def __init__(self):
                self.bound_tools = None
                self.kwargs = {}

            def bind(self, **kwargs):
                clone = FakeOpenAI()
                clone.bound_tools = kwargs.get("tools")
                clone.kwargs = kwargs
                return clone

            def invoke(self, messages, **kwargs):
                return {"tools": self.bound_tools, "kwargs": self.kwargs}

        inner = FakeOpenAI()
        wrapper = wrapper_factory(inner)
        result = wrapper.bind_tools([]).invoke(["hi"])
        return result

    def test_bind_tools_sends_training_schema(self):
        result = self._payload(lambda inner: TrainingSchemaChatModel(inner))
        self.assertEqual(result["tools"], tool_definitions())
        self.assertEqual(
            len(json.dumps(result["tools"], ensure_ascii=False)),
            len(json.dumps(tool_definitions(), ensure_ascii=False)),
        )

    def test_bind_tools_ignores_passed_tool_objects(self):
        result = self._payload(lambda inner: TrainingSchemaChatModel(inner))
        self.assertEqual(
            [item["function"]["name"] for item in result["tools"]],
            [tool.name for tool in excel_ops.EXCEL_TOOLS],
        )

    def test_strict_is_not_forwarded(self):
        result = self._payload(lambda inner: TrainingSchemaChatModel(inner))
        self.assertNotIn("strict", result["kwargs"])

    def test_strip_injected_is_opt_in(self):
        result = self._payload(
            lambda inner: TrainingSchemaChatModel(inner, strip_injected=True)
        )
        describe = next(
            item
            for item in result["tools"]
            if item["function"]["name"] == "describe_data"
        )
        self.assertEqual(describe["function"]["parameters"]["properties"], {})


class FoldToolHistoryTests(unittest.TestCase):
    """历史折叠：让 llama.cpp 渲染出与训练相同的 prompt。

    背景：训练数据里 `arguments` 是字符串，Qwen 模板的 `tojson` 会二次编码。
    llama-server 先解析回对象再渲染，第二轮起就偏离训练分布，表现是重复
    上一步工具调用。同 20 条样本：transformers 85%，未折叠的 llama-server 70%。
    """

    def _folded(self, messages):
        return fold_tool_history(messages)

    def test_tool_call_is_folded_into_content(self):
        folded = self._folded(
            [
                SystemMessage("sys"),
                HumanMessage("问题"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "aggregate_data",
                            "args": {"group_by": "部门"},
                            "id": "call_1",
                        }
                    ],
                ),
            ]
        )
        self.assertEqual(len(folded), 3)
        self.assertIsInstance(folded[2], AIMessage)
        self.assertFalse(folded[2].tool_calls)
        self.assertIn('<tool_call>', folded[2].content)
        self.assertIn('"name": "aggregate_data"', folded[2].content)

    def test_arguments_are_double_encoded_like_training(self):
        """训练数据里 arguments 是字符串，模板 tojson 再编码一次。"""
        folded = self._folded(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "x", "args": {"a": 1}, "id": "c"}],
                )
            ]
        )
        text = folded[0].content
        # 期望出现 \"a\": 1 这种被转义的形式
        self.assertIn('\\"a\\": 1', text)
        self.assertIn('"arguments": "', text)

    def test_chinese_arguments_are_not_escaped(self):
        folded = self._folded(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "x", "args": {"列": "销售额"}, "id": "c"}],
                )
            ]
        )
        self.assertIn("销售额", folded[0].content)
        self.assertNotIn("\\u", folded[0].content)

    def test_tool_message_becomes_human_with_tool_response(self):
        folded = self._folded([ToolMessage(content="结果", tool_call_id="c")])
        self.assertIsInstance(folded[0], HumanMessage)
        self.assertIn("<tool_response>", folded[0].content)
        self.assertIn("结果", folded[0].content)

    def test_plain_messages_pass_through(self):
        messages = [SystemMessage("s"), HumanMessage("h"), AIMessage(content="a")]
        self.assertEqual(self._folded(messages), messages)

    def test_content_and_tool_calls_both_kept(self):
        folded = self._folded(
            [
                AIMessage(
                    content="先汇总",
                    tool_calls=[{"name": "x", "args": {}, "id": "c"}],
                )
            ]
        )
        self.assertTrue(folded[0].content.startswith("先汇总"))
        self.assertIn('<tool_call>', folded[0].content)


class FoldWiringTests(unittest.TestCase):
    def _capture(self, fold: bool):
        captured = {}

        class FakeOpenAI:
            def bind(self, **kwargs):
                return FakeOpenAI()

            def invoke(self, messages, **kwargs):
                captured["messages"] = messages
                return "ok"

        wrapper = TrainingSchemaChatModel(FakeOpenAI(), fold_tool_history=fold)
        wrapper.bind_tools([]).invoke(
            [
                HumanMessage("q"),
                AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "c"}]),
                ToolMessage(content="r", tool_call_id="c"),
            ]
        )
        return captured["messages"]

    def test_fold_off_sends_raw_messages(self):
        messages = self._capture(False)
        self.assertTrue(messages[1].tool_calls)
        self.assertIsInstance(messages[2], ToolMessage)

    def test_fold_on_rewrites_messages(self):
        messages = self._capture(True)
        self.assertFalse(messages[1].tool_calls)
        self.assertIsInstance(messages[2], HumanMessage)

    def test_fold_survives_bind_tools(self):
        """bind_tools 返回新实例时不能丢掉折叠开关。"""
        captured = {}

        class FakeOpenAI:
            def bind(self, **kwargs):
                return FakeOpenAI()

            def invoke(self, messages, **kwargs):
                captured["messages"] = messages
                return "ok"

        wrapper = TrainingSchemaChatModel(FakeOpenAI(), fold_tool_history=True)
        wrapper.bind_tools([]).invoke(
            [AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "c"}])]
        )
        self.assertFalse(captured["messages"][0].tool_calls)


class InitLlmSchemaWiringTests(unittest.TestCase):
    def test_init_llm_returns_training_schema_wrapper(self):
        from unittest.mock import patch

        class FakeOpenAI:
            def bind(self, **kwargs):
                clone = FakeOpenAI()
                clone.tools = kwargs.get("tools")
                return clone

            def invoke(self, messages, **kwargs):
                return {'tools': self.tools}

        with (
            patch("load_model.init_chat_model", return_value=FakeOpenAI()) as factory,
            patch.dict(
                "os.environ",
                {
                    "LLM_PROVIDER": "local",
                    "LOCAL_MODEL": "test-model",
                    "LOCAL_BASE_URL": "http://127.0.0.1:8080/v1",
                },
            ),
        ):
            from load_model import init_llm

            llm = init_llm()

        self.assertIsInstance(llm, TrainingSchemaChatModel)
        self.assertTrue(factory.called)
        # 真的绑定一次，确认发出的是训练格式的 schema
        self.assertEqual(llm.bind_tools([]).invoke([])["tools"], tool_definitions())


if __name__ == "__main__":
    unittest.main()
