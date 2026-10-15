import unittest

import pandas as pd
from pydantic import ValidationError

import excel_ops
from schemas import (
    FillMissingInput,
    FilterCondition,
    Metric,
    SortKey,
    TopNRowsInput,
)


class ExcelOperationTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "地区": ["北区", "南区", "北区", "南区"],
            "销售额": [100, 80, 150, 120],
            "产品": ["A", "A", "B", "B"],
        })

    def test_filter_uses_numeric_comparison(self):
        result = excel_ops._apply_filters(self.df, [
            FilterCondition(column="销售额", operator="gt", value=100),
        ])

        self.assertEqual(result["销售额"].tolist(), [150, 120])

    def test_filter_contains_treats_value_as_literal(self):
        frame = pd.DataFrame({"名称": ["A+B", "AB", None]})
        result = excel_ops._apply_filters(
            frame,
            [FilterCondition(column="名称", operator="contains", value="+")],
        )

        self.assertEqual(result["名称"].tolist(), ["A+B"])

    def test_filter_supports_date_comparison(self):
        frame = pd.DataFrame({
            "日期": pd.to_datetime(["2025-01-01", "2025-03-01"]),
        })
        result = excel_ops._apply_filters(
            frame,
            [FilterCondition(column="日期", operator="gte", value="2025-02-01")],
        )

        self.assertEqual(result["日期"].tolist(), [pd.Timestamp("2025-03-01")])

    def test_grouped_aggregate_keeps_missing_group(self):
        frame = pd.DataFrame({"组": ["甲", None, "甲"], "值": [1, 2, 3]})
        result = excel_ops._aggregate(
            frame,
            ["组"],
            [Metric(column="值", function="sum")],
        )

        self.assertEqual(result["值_sum"].tolist(), [4, 2])

    def test_global_aggregate(self):
        result = excel_ops._aggregate(
            self.df,
            [],
            [
                Metric(column="销售额", function="sum"),
                Metric(column="销售额", function="mean"),
            ],
        )

        self.assertEqual(result.loc[0, "销售额_sum"], 450)
        self.assertEqual(result.loc[0, "销售额_mean"], 112.5)

    def test_top_n_sorts_descending_and_limits_rows(self):
        result = excel_ops._top_n_rows(
            self.df,
            [SortKey(column="销售额", ascending=False)],
            2,
        )

        self.assertEqual(result["销售额"].tolist(), [150, 120])

    def test_sort_supports_multiple_keys(self):
        frame = pd.DataFrame({"组": ["甲", "甲", "乙"], "值": [2, 1, 1]})
        result = excel_ops._sort_rows(
            frame,
            [
                SortKey(column="组", ascending=True),
                SortKey(column="值", ascending=False),
            ],
        )

        self.assertEqual(list(zip(result["组"], result["值"])), [("乙", 1), ("甲", 2), ("甲", 1)])

    def test_deduplicate_uses_selected_columns(self):
        frame = pd.DataFrame({"组": ["甲", "甲", "乙"], "值": [1, 2, 3]})
        result = excel_ops._deduplicate_rows(frame, ["组"])

        self.assertEqual(result["值"].tolist(), [1, 3])

    def test_fill_forward_only_selected_columns(self):
        frame = pd.DataFrame({"组": ["甲", None, "乙"], "值": [1, None, 3]})
        result = excel_ops._fill_missing_values(frame, "ffill", ["组"], None)

        self.assertEqual(result["组"].tolist(), ["甲", "甲", "乙"])
        self.assertTrue(pd.isna(result.loc[1, "值"]))

    def test_unknown_column_is_reported(self):
        with self.assertRaisesRegex(ValueError, "找不到字段"):
            excel_ops._sort_rows(self.df, [SortKey(column="不存在")])

    def test_top_n_requires_sort_keys(self):
        with self.assertRaises(ValidationError):
            TopNRowsInput.model_validate({"limit": 5})

    def test_sum_rejects_non_numeric_field(self):
        frame = pd.DataFrame({"值": ["未知", "缺失"]})
        with self.assertRaisesRegex(ValueError, "不包含可用于 sum 的数值"):
            excel_ops._aggregate(
                frame,
                [],
                [Metric(column="值", function="sum")],
            )

    def test_fill_value_requires_a_scalar(self):
        with self.assertRaises(ValidationError):
            FillMissingInput(fill_strategy="value")

    def test_tool_executor_does_not_modify_source_dataframe(self):
        frame = pd.DataFrame({"值": [1, None]})
        original = frame.copy(deep=True)
        state = {"current_df": frame}

        excel_ops._execute_dataframe_tool(
            state,
            "tool-call",
            "fill_missing_values",
            {"fill_strategy": "value", "fill_value": 0},
            lambda df: excel_ops._fill_missing_values(df, "value", [], 0),
        )
        pd.testing.assert_frame_equal(frame, original)


if __name__ == "__main__":
    unittest.main()
