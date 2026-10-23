import unittest

from dataset_builder.chain import (
    ChainValidationError,
    canonical_chain,
    chains_equivalent,
    normalize_chain,
)


class ChainNormalizationTests(unittest.TestCase):
    def test_filter_and_aggregate_chain_is_normalized(self):
        steps = normalize_chain(
            {
                "steps": [
                    {
                        "tool": "filter_rows",
                        "args": {
                            "filters": [
                                {"column": "区域", "operator": "eq", "value": "华东"}
                            ]
                        },
                    },
                    {
                        "tool": "aggregate_data",
                        "args": {
                            "group_by": ["区域"],
                            "metrics": [{"column": "销售额", "function": "sum"}],
                        },
                    },
                ]
            }
        )

        self.assertEqual([step["tool"] for step in steps], ["filter_rows", "aggregate_data"])
        self.assertEqual(steps[0]["args"]["filters"][0]["column"], "区域")
        self.assertEqual(steps[1]["args"]["group_by"], ["区域"])

    def test_bare_list_is_accepted(self):
        steps = normalize_chain([{"tool": "describe_data"}])

        self.assertEqual(steps, [{"tool": "describe_data", "args": {}}])

    def test_unknown_tool_is_rejected(self):
        with self.assertRaisesRegex(ChainValidationError, "未知工具"):
            normalize_chain([{"tool": "pivot_table", "args": {}}])

    def test_invalid_arguments_are_rejected(self):
        with self.assertRaisesRegex(ChainValidationError, "参数无效"):
            normalize_chain(
                [{"tool": "top_n_rows", "args": {"sort_by": [], "limit": 5}}]
            )

    def test_limit_out_of_range_is_rejected(self):
        with self.assertRaisesRegex(ChainValidationError, "参数无效"):
            normalize_chain(
                [
                    {
                        "tool": "top_n_rows",
                        "args": {
                            "sort_by": [{"column": "金额", "ascending": False}],
                            "limit": 0,
                        },
                    }
                ]
            )

    def test_describe_data_rejects_arguments(self):
        with self.assertRaisesRegex(ChainValidationError, "不接受参数"):
            normalize_chain([{"tool": "describe_data", "args": {"column": "金额"}}])

    def test_empty_chain_is_rejected(self):
        with self.assertRaisesRegex(ChainValidationError, "不能为空"):
            normalize_chain([])

    def test_step_limit_is_enforced(self):
        steps = [{"tool": "describe_data"} for _ in range(4)]
        with self.assertRaisesRegex(ChainValidationError, "超过上限"):
            normalize_chain(steps, max_steps=3)

    def test_allowed_tools_restriction(self):
        with self.assertRaisesRegex(ChainValidationError, "不允许的工具"):
            normalize_chain([{"tool": "create_chart", "args": {"chart_type": "bar", "chart_y": "值"}}], allowed_tools=("describe_data",))

    def test_allowed_tools_do_not_mask_unknown_tools(self):
        with self.assertRaisesRegex(ChainValidationError, "未知工具"):
            normalize_chain([{"tool": "pivot_table", "args": {}}], allowed_tools=("pivot_table",))


class ChainComparisonTests(unittest.TestCase):
    def test_equal_chains_match_regardless_of_key_order(self):
        left = normalize_chain(
            [{"tool": "sort_rows", "args": {"sort_by": [{"column": "金额", "ascending": False}]}}]
        )
        right = normalize_chain(
            [{"tool": "sort_rows", "args": {"sort_by": [{"ascending": False, "column": "金额"}]}}]
        )

        self.assertTrue(chains_equivalent(left, right))

    def test_numeric_tolerance_is_applied(self):
        left = [{"tool": "add_computed_column", "args": {"name": "x", "operator": "multiply", "left": "a", "right_value": 1.0}}]
        right = [{"tool": "add_computed_column", "args": {"name": "x", "operator": "multiply", "left": "a", "right_value": 1.0000001}}]

        self.assertTrue(chains_equivalent(left, right))
        self.assertFalse(chains_equivalent(left, right, tolerance=1e-9))

    def test_different_tool_order_does_not_match(self):
        left = normalize_chain([{"tool": "describe_data"}, {"tool": "sort_rows", "args": {"sort_by": [{"column": "a"}]}}])
        right = normalize_chain([{"tool": "sort_rows", "args": {"sort_by": [{"column": "a"}]}}, {"tool": "describe_data"}])

        self.assertFalse(chains_equivalent(left, right))

    def test_canonical_form_is_stable(self):
        first = normalize_chain([{"tool": "deduplicate_rows", "args": {"subset": ["a"]}}])
        second = normalize_chain([{"args": {"subset": ["a"]}, "tool": "deduplicate_rows"}])

        self.assertEqual(canonical_chain(first), canonical_chain(second))


if __name__ == "__main__":
    unittest.main()
