import json
import unittest

from dataset_builder.llm import (
    DatasetLLM,
    LLMChainProposer,
    LLMQuestionParser,
    LLMQuestionWriter,
    extract_json,
    tool_catalog,
)
from dataset_builder.tables import generate_table


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeChatModel:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages: list[object] = []

    def invoke(self, messages):
        self.messages.append(messages)
        return FakeResponse(self.replies.pop(0))


class ExtractJsonTests(unittest.TestCase):
    def test_extracts_fenced_array(self):
        payload = extract_json('说明如下：\n```json\n[{"steps": []}]\n```\n完成')

        self.assertEqual(payload, [{"steps": []}])

    def test_extracts_object_with_nested_brackets(self):
        payload = extract_json('结果：{"steps": [{"tool": "describe_data"}], "note": "}"}')

        self.assertEqual(payload["steps"][0]["tool"], "describe_data")
        self.assertEqual(payload["note"], "}")

    def test_rejects_text_without_json(self):
        with self.assertRaisesRegex(ValueError, "找不到合法 JSON"):
            extract_json("这里没有结构化内容")

    def test_rejects_empty_reply(self):
        with self.assertRaisesRegex(ValueError, "为空"):
            extract_json("   ")


class ToolCatalogTests(unittest.TestCase):
    def test_catalog_lists_every_tool_with_schema(self):
        catalog = json.loads(tool_catalog())

        names = {entry["tool"] for entry in catalog}
        self.assertIn("filter_rows", names)
        self.assertIn("add_computed_column", names)
        computed = next(item for item in catalog if item["tool"] == "add_computed_column")
        self.assertIn("operator", computed["args_schema"]["properties"])


class LLMAdapterTests(unittest.TestCase):
    def setUp(self):
        self.table = generate_table("sales_orders", 3)

    def test_chain_proposer_returns_candidate_chains(self):
        model = FakeChatModel([json.dumps({"chains": [{"steps": [{"tool": "describe_data"}]}]})])
        proposer = LLMChainProposer(DatasetLLM(model))

        chains = proposer.propose(self.table, 1)

        self.assertEqual(chains, [{"steps": [{"tool": "describe_data"}]}])
        prompt = model.messages[0][-1].content
        self.assertIn(self.table.table_id, prompt)
        self.assertIn("describe_data", prompt)

    def test_chain_proposer_rejects_non_list_payload(self):
        model = FakeChatModel([json.dumps({"note": "no chains"})])
        proposer = LLMChainProposer(DatasetLLM(model))

        with self.assertRaises(ValueError):
            proposer.propose(self.table, 1)

    def test_question_writer_returns_question_and_answer(self):
        model = FakeChatModel([json.dumps({"question": "按区域汇总销售额", "answer": "已完成"})])
        writer = LLMQuestionWriter(DatasetLLM(model))

        written = writer.write(self.table, [{"tool": "describe_data", "args": {}}])

        self.assertEqual(written["question"], "按区域汇总销售额")
        self.assertEqual(written["answer"], "已完成")

    def test_question_writer_requires_both_fields(self):
        model = FakeChatModel([json.dumps({"question": "", "answer": "已完成"})])
        writer = LLMQuestionWriter(DatasetLLM(model))

        with self.assertRaisesRegex(ValueError, "为空"):
            writer.write(self.table, [{"tool": "describe_data", "args": {}}])

    def test_chain_proposer_accepts_default_tool_set(self):
        model = FakeChatModel([json.dumps([{"steps": [{"tool": "describe_data"}]}])])
        proposer = LLMChainProposer(DatasetLLM(model))

        chains = proposer.propose(self.table, 1, allowed_tools=None)

        self.assertEqual(len(chains), 1)

    def test_question_parser_returns_steps(self):
        model = FakeChatModel([json.dumps({"steps": [{"tool": "describe_data"}]})])
        parser = LLMQuestionParser(DatasetLLM(model))

        payload = parser.parse(self.table, "看一下统计概况")

        self.assertEqual(payload["steps"][0]["tool"], "describe_data")


if __name__ == "__main__":
    unittest.main()
