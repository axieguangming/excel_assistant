"""口语化评测集的逻辑测试：不加载模型，只验证评测集与判定规则。"""

import json
from pathlib import Path
import unittest

from finetune.evaluate_colloquial import EXECUTION_TOOLS

import excel_ops


EVALSET = Path("evalsets/colloquial_2026-10.json")


class ExecutionToolClassificationTests(unittest.TestCase):
    def test_describe_is_not_treated_as_execution(self):
        # describe_data 只读，不可完成请求里调用它属于可接受的澄清动作。
        self.assertNotIn("describe_data", EXECUTION_TOOLS)

    def test_every_data_mutating_tool_is_classified_as_execution(self):
        for tool in excel_ops.EXCEL_TOOLS:
            if tool.name == "describe_data":
                continue
            self.assertIn(tool.name, EXECUTION_TOOLS, f"{tool.name} 未归类")

    def test_classification_only_references_real_tools(self):
        known = {tool.name for tool in excel_ops.EXCEL_TOOLS}
        self.assertTrue(EXECUTION_TOOLS <= known, EXECUTION_TOOLS - known)


class EvalSetShapeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(EVALSET.read_text(encoding="utf-8"))

    def test_items_have_unique_ids(self):
        ids = [item["id"] for item in self.payload["items"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_only_known_kinds(self):
        for item in self.payload["items"]:
            self.assertIn(item["kind"], {"equivalent", "unanswerable"})

    def test_equivalent_items_reference_a_source_sample(self):
        for item in self.payload["items"]:
            if item["kind"] == "equivalent":
                self.assertTrue(item.get("source_sample"), item["id"])

    def test_unanswerable_items_reference_a_workbook(self):
        for item in self.payload["items"]:
            if item["kind"] == "unanswerable":
                self.assertTrue(item.get("workbook_id"), item["id"])

    def test_questions_are_not_verbatim_template_phrases(self):
        # 模板句式的固定开头；口语化问题不应带上它们，否则就退化成同质评测。
        prefixes = (
            "请对",
            "在",
            "对",
            "基于",
            "帮我处理一下",
        )
        for item in self.payload["items"]:
            question = item["question"]
            self.assertNotIn("执行以下处理", question, item["id"])
            self.assertFalse(
                question.startswith("对预算执行") or question.startswith("请对预算执行"),
                f"{item['id']} 仍是模板句式：{question}",
            )
            self.assertFalse(any(question == p for p in prefixes), item["id"])


if __name__ == "__main__":
    unittest.main()
