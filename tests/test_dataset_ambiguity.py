"""歧义检测与澄清措辞的单元测试（不调用模型）。

重点验证「歧义必须是可验证的」：两个候选在数据上确实给出不同结果时才算歧义，
否则不该反问用户。
"""

import unittest

import pandas as pd

from dataset_builder.ambiguity import (
    CLARIFY_TEMPLATES,
    Ambiguity,
    detect,
)
from dataset_builder.tables import SyntheticTable


def make_table(frame: pd.DataFrame, domain: str = "sales_orders") -> SyntheticTable:
    return SyntheticTable(
        table_id=f"{domain}-test",
        domain=domain,
        sheet="测试",
        frame=frame,
    )


class CountSemanticsTests(unittest.TestCase):
    def test_no_ambiguity_when_identifier_is_unique(self):
        # 订单号唯一时，count 与 nunique 结果相同，反问用户是多余的。
        frame = pd.DataFrame({
            "订单号": ["A1", "A2", "A3"],
            "销售额": [10.0, 20.0, 30.0],
            "区域": ["东", "西", "东"],
        })
        kinds = [a.kind for a in detect(make_table(frame))]
        self.assertNotIn("count_semantics", kinds)

    def test_ambiguity_when_identifier_has_duplicates(self):
        frame = pd.DataFrame({
            "订单号": ["A1", "A1", "A2"],
            "销售额": [10.0, 20.0, 30.0],
            "区域": ["东", "西", "东"],
        })
        found = [a for a in detect(make_table(frame)) if a.kind == "count_semantics"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].evidence["rows"], 3)
        self.assertEqual(found[0].evidence["distinct"], 2)
        self.assertEqual(found[0].options[0], "订单号")

    def test_clarification_names_both_options(self):
        frame = pd.DataFrame({
            "订单号": ["A1", "A1", "A2"],
            "销售额": [10.0, 20.0, 30.0],
            "区域": ["东", "西", "东"],
        })
        ambiguity = next(
            a for a in detect(make_table(frame)) if a.kind == "count_semantics"
        )
        text = ambiguity.render(0)
        self.assertIn("订单号", text)
        self.assertIn("记录条数", text)


class MetricColumnTests(unittest.TestCase):
    def test_sales_domain_detects_amount_ambiguity(self):
        frame = pd.DataFrame({
            "单价": [1.0, 2.0],
            "销售额": [10.0, 20.0],
            "区域": ["东", "西"],
        })
        found = [a for a in detect(make_table(frame)) if a.kind == "metric_column"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].vague_term, "金额")
        self.assertEqual(set(found[0].options), {"单价", "销售额"})

    def test_missing_candidate_column_skips_ambiguity(self):
        # 表里只有单价、没有销售额时，「金额」不存在真正的二义性。
        frame = pd.DataFrame({"单价": [1.0, 2.0], "区域": ["东", "西"]})
        kinds = [a.kind for a in detect(make_table(frame))]
        self.assertNotIn("metric_column", kinds)


class MissingValueTests(unittest.TestCase):
    def test_reports_actual_missing_count_in_clarification(self):
        frame = pd.DataFrame({
            "单价": [1.0, None, 3.0, None],
            "销售额": [10.0, 20.0, 30.0, 40.0],
            "区域": ["东", "西", "东", "西"],
        })
        ambiguity = next(
            a for a in detect(make_table(frame)) if a.kind == "missing_value"
        )
        self.assertEqual(ambiguity.evidence["missing"], 2)
        text = ambiguity.render(0)
        self.assertIn("2", text)
        self.assertIn("单价", text)
        # 不能出现"空值 个空值"这类未替换占位符的写法
        self.assertNotIn("空值 个空值", text)

    def test_no_ambiguity_when_no_missing_values(self):
        frame = pd.DataFrame({
            "单价": [1.0, 2.0],
            "销售额": [10.0, 20.0],
            "区域": ["东", "西"],
        })
        kinds = [a.kind for a in detect(make_table(frame))]
        self.assertNotIn("missing_value", kinds)


class TimeRangeTests(unittest.TestCase):
    def test_long_span_is_ambiguous(self):
        frame = pd.DataFrame({
            "日期": pd.to_datetime(["2025-01-01", "2025-12-01"]),
            "销售额": [10.0, 20.0],
        })
        found = [a for a in detect(make_table(frame)) if a.kind == "time_range"]
        self.assertEqual(len(found), 1)
        self.assertGreater(found[0].evidence["span_days"], 30)

    def test_short_span_is_not_ambiguous(self):
        frame = pd.DataFrame({
            "日期": pd.to_datetime(["2025-01-01", "2025-01-15"]),
            "销售额": [10.0, 20.0],
        })
        kinds = [a.kind for a in detect(make_table(frame))]
        self.assertNotIn("time_range", kinds)


class ClarificationContractTests(unittest.TestCase):
    def test_every_template_is_short_enough(self):
        # 澄清只问缺的那一点，不应变成一段解释。
        for kind, templates in CLARIFY_TEMPLATES.items():
            for template in templates:
                self.assertLess(len(template), 80, f"{kind} 模板过长")

    def test_templates_end_with_question(self):
        for kind, templates in CLARIFY_TEMPLATES.items():
            for template in templates:
                self.assertTrue(
                    template.rstrip().endswith(("？", "?", "。")),
                    f"{kind} 模板缺少收尾标点：{template}",
                )

    def test_render_cycles_through_templates(self):
        ambiguity = Ambiguity(
            kind="metric_column",
            vague_term="金额",
            options=("单价", "销售额"),
            clarifications=CLARIFY_TEMPLATES["metric_column"],
        )
        first = ambiguity.render(0)
        second = ambiguity.render(1)
        self.assertNotEqual(first, second)
        self.assertEqual(ambiguity.render(len(CLARIFY_TEMPLATES["metric_column"])), first)

    def test_matches_accepts_synonyms(self):
        ambiguity = Ambiguity(
            kind="chart_type",
            vague_term="画个图",
            options=("销售额", "图表"),
            clarifications=CLARIFY_TEMPLATES["chart_type"],
            accept_terms=("画个图", "图表", "可视化"),
        )
        self.assertTrue(ambiguity.matches("帮我做个可视化"))
        self.assertTrue(ambiguity.matches("画个图看看"))
        self.assertFalse(ambiguity.matches("按区域汇总销售额"))


if __name__ == "__main__":
    unittest.main()
