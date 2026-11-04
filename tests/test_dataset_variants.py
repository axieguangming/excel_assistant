"""口语化变体生成、忠实度比对、链质量检查的单元测试（不调用模型）。"""

import unittest

from dataset_builder.fidelity import (
    is_semantically_equivalent,
    normalize_for_comparison,
)
from dataset_builder.quality import chain_quality_warnings, is_identifier_column
from dataset_builder.variants import (
    ColloquialVariantWriter,
    _check,
    _counting_columns as counting_columns,
    generated_column_names,
    internal_column_names,
    looks_internal_name,
)


class GeneratedColumnTests(unittest.TestCase):
    def test_collects_computed_column_name(self):
        steps = [
            {
                "tool": "add_computed_column",
                "args": {"name": "预算金额_x_差异", "operator": "multiply"},
            }
        ]
        self.assertEqual(generated_column_names(steps), ["预算金额_x_差异"])

    def test_collects_explicit_aggregate_names(self):
        steps = [
            {
                "tool": "aggregate_data",
                "args": {
                    "group_by": ["城市"],
                    "metrics": [{"column": "满意度", "function": "sum", "name": "满意度_sum"}],
                },
            }
        ]
        self.assertEqual(generated_column_names(steps), ["满意度_sum"])

    def test_ignores_unnamed_metrics(self):
        # 未显式命名时由代码拼出列名，但问题生成阶段不必当作"已知列"。
        steps = [
            {
                "tool": "aggregate_data",
                "args": {"metrics": [{"column": "满意度", "function": "sum"}]},
            }
        ]
        self.assertEqual(generated_column_names(steps), [])


class InternalNameTests(unittest.TestCase):
    def test_underscore_names_are_internal(self):
        for name in ("销售额_sum", "预算金额_x_差异", "推荐意愿_add_100", "数量_mean"):
            self.assertTrue(looks_internal_name(name), name)

    def test_natural_business_names_are_not_internal(self):
        # LLM 生成的链用自然业务命名，用户在提问里说出来是完全正常的。
        for name in ("销售额合计", "平均满意度", "总工时", "订单数", "库存金额"):
            self.assertFalse(looks_internal_name(name), name)

    def test_internal_column_names_filters_natural_names(self):
        steps = [
            {
                "tool": "aggregate_data",
                "args": {
                    "metrics": [
                        {"column": "销售额", "function": "sum", "name": "销售额合计"},
                        {"column": "数量", "function": "mean", "name": "数量_mean"},
                    ]
                },
            }
        ]
        self.assertEqual(internal_column_names(steps), ["数量_mean"])


class VariantCheckTests(unittest.TestCase):
    def test_accepts_natural_question(self):
        reason = _check(
            "各城市满意度加起来是多少，画个饼图看看占比",
            "已经处理好了，共 5 行 2 列。",
            leaked=[],
        )
        self.assertEqual(reason, "")

    def test_rejects_tool_name(self):
        reason = _check(
            "用 aggregate_data 汇总满意度",
            "完成，5 行 2 列。",
            leaked=[],
        )
        self.assertIn("工具名", reason)

    def test_rejects_leaked_generated_column(self):
        reason = _check(
            "按城市汇总满意度_sum",
            "完成，5 行 2 列。",
            leaked=["满意度_sum"],
        )
        self.assertIn("泄漏生成列名", reason)

    def test_rejects_template_phrasing(self):
        reason = _check(
            "请对问卷执行以下处理：按 城市 分组统计 满意度 的合计",
            "完成，5 行 2 列。",
            leaked=[],
        )
        self.assertIn("模板句式", reason)

    def test_axis_wording_is_allowed(self):
        # 真实用户指定图表轴时也会说"横轴"，不能因此判为模板。
        reason = _check(
            "按城市汇总满意度画柱状图，横轴放城市，纵轴放总和",
            "完成，4 行 2 列。",
            leaked=[],
        )
        self.assertEqual(reason, "")

    def test_rejects_answer_without_counts(self):
        reason = _check(
            "按城市汇总满意度总和",
            "已经处理好了。",
            leaked=[],
        )
        self.assertIn("行数/列数", reason)


class CountingAmbiguityTests(unittest.TestCase):
    """计数指标的问法不得出现「X数」这类口径歧义说法。

    这类说法既可能指行数也可能指去重个数。实测强模型会因此先反问
    （符合 `agent.AGENT_SYSTEM` 的要求），却被「期望直接执行」的评测判为失败。
    """

    def test_rejects_identifier_count_phrase(self):
        reason = _check(
            "按站点汇总运费合计和运单数",
            "完成，4 行 2 列。",
            leaked=[],
            count_columns=["运单数"],
        )
        self.assertIn("计数口径歧义", reason)

    def test_accepts_explicit_row_count_phrasing(self):
        reason = _check(
            "按站点汇总运费合计，并数一下有多少条记录",
            "完成，4 行 2 列。",
            leaked=[],
            count_columns=["运单数"],
        )
        self.assertEqual(reason, "")

    def test_ambiguity_check_skipped_when_no_count_metric(self):
        # 链里没有 count/nunique 时，不应因为问题里出现「个数」就拒绝。
        reason = _check(
            "按分类看销售额合计和平均单价",
            "完成，5 行 3 列。",
            leaked=[],
        )
        self.assertEqual(reason, "")

    def test_rejects_dedupe_count_phrase(self):
        reason = _check(
            "按合作等级统计客户数",
            "完成，3 行 2 列。",
            leaked=[],
            nunique_columns=["客户数"],
        )
        self.assertIn("计数口径歧义", reason)

    def test_counting_columns_extraction(self):
        steps = [
            {
                "tool": "aggregate_data",
                "args": {
                    "group_by": ["站点"],
                    "metrics": [
                        {"column": "运费", "function": "sum", "name": "运费合计"},
                        {"column": "运单编号", "function": "count", "name": "运单数"},
                        {"column": "客户代码", "function": "nunique", "name": "客户数"},
                    ],
                },
            }
        ]
        counting, distinct = counting_columns(steps)
        self.assertEqual(counting, ["运单数"])
        self.assertEqual(distinct, ["客户数"])


class FidelityTests(unittest.TestCase):
    def test_missing_metric_name_is_equivalent(self):
        expected = [
            {
                "tool": "aggregate_data",
                "args": {"group_by": ["城市"], "metrics": [{"column": "满意度", "function": "sum", "name": "满意度_sum"}]},
            }
        ]
        parsed = [
            {
                "tool": "aggregate_data",
                "args": {"group_by": ["城市"], "metrics": [{"column": "满意度", "function": "sum"}]},
            }
        ]
        self.assertTrue(is_semantically_equivalent(expected, parsed))

    def test_different_aggregation_function_is_not_equivalent(self):
        expected = [
            {
                "tool": "aggregate_data",
                "args": {"group_by": ["城市"], "metrics": [{"column": "满意度", "function": "sum"}]},
            }
        ]
        parsed = [
            {
                "tool": "aggregate_data",
                "args": {"group_by": ["城市"], "metrics": [{"column": "满意度", "function": "mean"}]},
            }
        ]
        self.assertFalse(is_semantically_equivalent(expected, parsed))

    def test_redundant_sort_before_top_n_is_equivalent(self):
        sort_step = {"tool": "sort_rows", "args": {"sort_by": [{"column": "满意度", "ascending": False}]}}
        top_step = {
            "tool": "top_n_rows",
            "args": {"sort_by": [{"column": "满意度", "ascending": False}], "limit": 10},
        }
        self.assertTrue(
            is_semantically_equivalent([top_step], [sort_step, top_step])
        )

    def test_different_sort_direction_is_not_equivalent(self):
        sort_step = {"tool": "sort_rows", "args": {"sort_by": [{"column": "满意度", "ascending": True}]}}
        top_step = {
            "tool": "top_n_rows",
            "args": {"sort_by": [{"column": "满意度", "ascending": False}], "limit": 10},
        }
        self.assertFalse(
            is_semantically_equivalent([top_step], [sort_step, top_step])
        )

    def test_chart_title_difference_is_equivalent(self):
        base = {"tool": "create_chart", "args": {"chart_type": "bar", "chart_y": "销售额"}}
        titled = {
            "tool": "create_chart",
            "args": {"chart_type": "bar", "chart_y": "销售额", "chart_title": "销售额对比"},
        }
        self.assertTrue(is_semantically_equivalent([base], [titled]))

    def test_computed_column_name_difference_is_equivalent(self):
        left = {
            "tool": "add_computed_column",
            "args": {"name": "推荐意愿_add_100", "operator": "add", "left": "推荐意愿", "right_value": 100.0},
        }
        right = {
            "tool": "add_computed_column",
            "args": {"name": "推荐意愿_加100", "operator": "add", "left": "推荐意愿", "right_value": 100.0},
        }
        self.assertTrue(is_semantically_equivalent([left], [right]))

    def test_operator_difference_is_not_equivalent(self):
        left = {
            "tool": "add_computed_column",
            "args": {"name": "x", "operator": "add", "left": "推荐意愿", "right_value": 100.0},
        }
        right = {
            "tool": "add_computed_column",
            "args": {"name": "x", "operator": "subtract", "left": "推荐意愿", "right_value": 100.0},
        }
        self.assertFalse(is_semantically_equivalent([left], [right]))

    def test_count_on_different_columns_is_equivalent(self):
        # count 是「组内行数」，与所选列无关，因此两者结果相同。
        expected = [
            {
                "tool": "aggregate_data",
                "args": {
                    "group_by": ["反馈"],
                    "metrics": [
                        {"column": "满意度", "function": "mean", "name": "平均满意度"},
                        {"column": "受访者编号", "function": "count", "name": "反馈数"},
                    ],
                },
            }
        ]
        parsed = [
            {
                "tool": "aggregate_data",
                "args": {
                    "group_by": ["反馈"],
                    "metrics": [
                        {"column": "满意度", "function": "mean"},
                        {"column": "反馈", "function": "count"},
                    ],
                },
            }
        ]
        self.assertTrue(is_semantically_equivalent(expected, parsed))

    def test_nunique_on_different_columns_is_not_equivalent(self):
        # 去重计数依赖具体列，不能抹平。
        expected = [
            {
                "tool": "aggregate_data",
                "args": {
                    "group_by": ["区域"],
                    "metrics": [{"column": "科目", "function": "nunique"}],
                },
            }
        ]
        parsed = [
            {
                "tool": "aggregate_data",
                "args": {
                    "group_by": ["区域"],
                    "metrics": [{"column": "城市", "function": "nunique"}],
                },
            }
        ]
        self.assertFalse(is_semantically_equivalent(expected, parsed))


class ChainQualityTests(unittest.TestCase):
    def test_flags_pie_without_category(self):
        warnings = chain_quality_warnings(
            [{"tool": "create_chart", "args": {"chart_type": "pie", "chart_y": "推荐意愿"}}],
            total_rows=90,
        )
        self.assertTrue(any("pie_without_category" in w for w in warnings))

    def test_pie_with_category_is_fine(self):
        warnings = chain_quality_warnings(
            [
                {
                    "tool": "create_chart",
                    "args": {"chart_type": "pie", "chart_y": "x", "chart_x": "城市"},
                }
            ]
        )
        self.assertFalse(any("pie_without_category" in w for w in warnings))

    def test_flags_sum_on_rating_scale(self):
        warnings = chain_quality_warnings(
            [
                {
                    "tool": "aggregate_data",
                    "args": {"group_by": ["城市"], "metrics": [{"column": "满意度", "function": "sum"}]},
                }
            ]
        )
        self.assertTrue(any("sum_on_rating_scale" in w for w in warnings))

    def test_mean_on_rating_scale_is_fine(self):
        warnings = chain_quality_warnings(
            [
                {
                    "tool": "aggregate_data",
                    "args": {"group_by": ["城市"], "metrics": [{"column": "满意度", "function": "mean"}]},
                }
            ]
        )
        self.assertFalse(any("sum_on_rating_scale" in w for w in warnings))

    def test_flags_grouping_by_filtered_single_value(self):
        warnings = chain_quality_warnings(
            [
                {
                    "tool": "filter_rows",
                    "args": {"filters": [{"column": "受访者编号", "operator": "eq", "value": "R501"}]},
                },
                {
                    "tool": "aggregate_data",
                    "args": {"group_by": ["受访者编号"], "metrics": [{"column": "满意度", "function": "count"}]},
                },
            ]
        )
        self.assertTrue(any("degenerate_group" in w for w in warnings))

    def test_flags_top_n_covering_whole_table(self):
        warnings = chain_quality_warnings(
            [{"tool": "top_n_rows", "args": {"sort_by": [{"column": "金额", "ascending": False}], "limit": 50}}],
            total_rows=48,
        )
        self.assertTrue(any("top_n_covers_whole_table" in w for w in warnings))

    def test_top_n_below_table_size_is_fine(self):
        warnings = chain_quality_warnings(
            [{"tool": "top_n_rows", "args": {"sort_by": [{"column": "金额", "ascending": False}], "limit": 10}}],
            total_rows=48,
        )
        self.assertFalse(any("top_n_covers_whole_table" in w for w in warnings))

    def test_flags_numeric_aggregation_on_identifier(self):
        warnings = chain_quality_warnings(
            [
                {
                    "tool": "aggregate_data",
                    "args": {"group_by": ["区域"], "metrics": [{"column": "订单号", "function": "sum"}]},
                }
            ]
        )
        self.assertTrue(any("numeric_op_on_identifier" in w for w in warnings))

    def test_count_on_identifier_is_allowed(self):
        warnings = chain_quality_warnings(
            [
                {
                    "tool": "aggregate_data",
                    "args": {"group_by": ["区域"], "metrics": [{"column": "订单号", "function": "count"}]},
                }
            ]
        )
        self.assertFalse(any("numeric_op_on_identifier" in w for w in warnings))

    def test_flags_arithmetic_on_identifier(self):
        warnings = chain_quality_warnings(
            [
                {
                    "tool": "add_computed_column",
                    "args": {
                        "name": "x",
                        "operator": "multiply",
                        "left": "员工编号",
                        "right_column": "工时",
                    },
                }
            ]
        )
        self.assertTrue(any("arithmetic_on_identifier" in w for w in warnings))

    def test_flags_cross_dimension_subtraction(self):
        # 运费（元）减毛重（千克）没有业务含义。
        warnings = chain_quality_warnings(
            [
                {
                    "tool": "add_computed_column",
                    "args": {
                        "name": "x",
                        "operator": "subtract",
                        "left": "运费",
                        "right_column": "毛重",
                    },
                }
            ]
        )
        self.assertTrue(any("cross_dimension_arithmetic" in w for w in warnings))

    def test_same_dimension_subtraction_is_fine(self):
        warnings = chain_quality_warnings(
            [
                {
                    "tool": "add_computed_column",
                    "args": {
                        "name": "毛利",
                        "operator": "subtract",
                        "left": "销售额",
                        "right_column": "单价",
                    },
                }
            ]
        )
        self.assertFalse(any("cross_dimension" in w or "derived" in w for w in warnings))

    def test_flags_arithmetic_on_derived_metric(self):
        # 单位运费是「元/千克」的派生值，与运费直接相减无意义。
        warnings = chain_quality_warnings(
            [
                {
                    "tool": "add_computed_column",
                    "args": {
                        "name": "x",
                        "operator": "subtract",
                        "left": "运费",
                        "right_column": "单位运费",
                    },
                }
            ]
        )
        self.assertTrue(any("arithmetic_on_derived_metric" in w for w in warnings))

    def test_division_across_dimensions_is_fine(self):
        # 除法正是用来构造比率，不应被拦。
        warnings = chain_quality_warnings(
            [
                {
                    "tool": "add_computed_column",
                    "args": {
                        "name": "单位运费",
                        "operator": "divide",
                        "left": "运费",
                        "right_column": "毛重",
                    },
                }
            ]
        )
        self.assertFalse(any("cross_dimension" in w for w in warnings))


class IdentifierColumnTests(unittest.TestCase):
    def test_recognizes_common_identifier_names(self):
        for name in (
            "订单号",
            "员工编号",
            "受访者编号",
            "商品编码",
            "商品代码",
            "order_id",
            "SKU_ID",
        ):
            self.assertTrue(is_identifier_column(name), name)

    def test_does_not_flag_business_measures(self):
        for name in ("销售额", "数量", "单价", "工时", "满意度", "差异"):
            self.assertFalse(is_identifier_column(name), name)


class PromptContractTests(unittest.TestCase):
    def test_prompt_requires_explicit_aggregation_wording(self):
        # 汇总口径歧义是实测最主要的忠实度失分点，提示词必须显式约束。
        system = ColloquialVariantWriter.system
        self.assertIn("合计", system)
        self.assertIn("平均", system)
        self.assertIn("绝不能只说", system)

    def test_prompt_forbids_internal_column_names(self):
        system = ColloquialVariantWriter.system
        self.assertIn("带下划线的内部列名", system)
        self.assertIn("xxx_sum", system)

    def test_prompt_allows_natural_business_terms(self):
        # LLM 链的列名是自然业务用语（如"销售额合计"），应当允许出现在提问里。
        self.assertIn("销售额合计", ColloquialVariantWriter.system)


if __name__ == "__main__":
    unittest.main()
