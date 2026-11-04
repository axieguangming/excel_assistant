"""本地模型封装的纯逻辑测试：不加载权重，只验证消息转换与工具调用解析。"""

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from finetune.local_llm import _extract_tool_call, _to_chat_messages


class ChatMessageConversionTests(unittest.TestCase):
    def test_plain_conversation_keeps_roles(self):
        messages = _to_chat_messages([
            SystemMessage(content="系统"),
            HumanMessage(content="问题"),
            AIMessage(content="答复"),
        ])
        self.assertEqual(
            [item["role"] for item in messages], ["system", "user", "assistant"]
        )
        self.assertEqual(messages[0]["content"], "系统")

    def test_tool_call_is_serialized_as_json_string(self):
        message = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "filter_rows",
                    "args": {"filters": [{"column": "区域", "operator": "eq", "value": "华东"}]},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        )
        converted = _to_chat_messages([message])[0]
        self.assertEqual(converted["role"], "assistant")
        call = converted["tool_calls"][0]
        self.assertEqual(call["id"], "call_1")
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "filter_rows")
        # 与训练数据一致：arguments 是 JSON 字符串而不是字典。
        self.assertIsInstance(call["function"]["arguments"], str)
        self.assertEqual(
            json.loads(call["function"]["arguments"]),
            {"filters": [{"column": "区域", "operator": "eq", "value": "华东"}]},
        )

    def test_tool_message_keeps_call_id_and_name(self):
        converted = _to_chat_messages([
            ToolMessage(content="操作成功", tool_call_id="call_1", name="filter_rows")
        ])[0]
        self.assertEqual(converted["role"], "tool")
        self.assertEqual(converted["tool_call_id"], "call_1")
        self.assertEqual(converted["name"], "filter_rows")


class ToolCallExtractionTests(unittest.TestCase):
    def test_extracts_call_and_preface(self):
        text = (
            "我来统计一下。\n<tool_call>\n"
            '{"name": "aggregate_data", "arguments": {"metrics": []}}\n'
            "</tool_call><|im_end|>"
        )
        call, preface = _extract_tool_call(text)
        self.assertEqual(call["name"], "aggregate_data")
        self.assertEqual(call["arguments"], {"metrics": []})
        self.assertEqual(preface, "我来统计一下。")

    def test_handles_nested_json_in_arguments(self):
        payload = {
            "name": "filter_rows",
            "arguments": {
                "filters": [
                    {"column": "金额", "operator": "gt", "value": 100},
                    {"column": "区域", "operator": "eq", "value": "华东"},
                ]
            },
        }
        text = "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"
        call, preface = _extract_tool_call(text)
        self.assertEqual(call, payload)
        self.assertEqual(preface, "")

    def test_text_without_tool_call_is_returned_whole(self):
        call, preface = _extract_tool_call("这个需求无法用现有工具完成。")
        self.assertIsNone(call)
        self.assertEqual(preface, "这个需求无法用现有工具完成。")

    def test_malformed_payload_is_not_treated_as_call(self):
        call, preface = _extract_tool_call("<tool_call>{not json}</tool_call>")
        self.assertIsNone(call)
        self.assertIn("not json", preface)

    def test_unclosed_tag_still_parses(self):
        call, _ = _extract_tool_call('<tool_call>\n{"name": "describe_data", "arguments": {}}')
        self.assertEqual(call["name"], "describe_data")


if __name__ == "__main__":
    unittest.main()
