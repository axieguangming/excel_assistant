import unittest
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

import excel_ops
from schemas import (
    ComputedColumnInput,
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

    def test_computed_column_multiplies_two_fields(self):
        frame = pd.DataFrame({"数量": [2, 3], "单价": [10.5, 4]})
        result = excel_ops._computed_column(
            frame,
            ComputedColumnInput(
                name="金额",
                operator="multiply",
                left="数量",
                right_column="单价",
            ),
        )

        self.assertEqual(result["金额"].tolist(), [21.0, 12.0])

    def test_computed_column_rounds_and_accepts_constant(self):
        frame = pd.DataFrame({"销售额": [100, 250]})
        result = excel_ops._computed_column(
            frame,
            ComputedColumnInput(
                name="提成",
                operator="multiply",
                left="销售额",
                right_value=0.07,
                round_digits=2,
            ),
        )

        self.assertEqual(result["提成"].tolist(), [7.0, 17.5])

    def test_computed_column_rejects_zero_divisor(self):
        frame = pd.DataFrame({"销售额": [10, 20], "数量": [2, 0]})
        with self.assertRaisesRegex(ValueError, "不能作为除数"):
            excel_ops._computed_column(
                frame,
                ComputedColumnInput(
                    name="单价",
                    operator="divide",
                    left="销售额",
                    right_column="数量",
                ),
            )
        with self.assertRaisesRegex(ValueError, "除数不能为 0"):
            excel_ops._computed_column(
                frame,
                ComputedColumnInput(
                    name="占比",
                    operator="divide",
                    left="销售额",
                    right_value=0,
                ),
            )

    def test_computed_column_rejects_existing_column_name(self):
        with self.assertRaisesRegex(ValueError, "已存在"):
            excel_ops._computed_column(
                self.df,
                ComputedColumnInput(
                    name="销售额",
                    operator="add",
                    left="销售额",
                    right_value=1,
                ),
            )

    def test_computed_column_rejects_non_numeric_field(self):
        frame = pd.DataFrame({"产品": ["A", "B"]})
        with self.assertRaisesRegex(ValueError, "不包含可用于计算的数值"):
            excel_ops._computed_column(
                frame,
                ComputedColumnInput(
                    name="新值",
                    operator="add",
                    left="产品",
                    right_value=1,
                ),
            )

    def test_computed_column_requires_exactly_one_right_operand(self):
        for kwargs in (
            {},
            {"right_column": "单价", "right_value": 2},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValidationError):
                    ComputedColumnInput(
                        name="金额",
                        operator="multiply",
                        left="数量",
                        **kwargs,
                    )

    def test_computed_column_rejects_date_field(self):
        frame = pd.DataFrame({"日期": pd.to_datetime(["2025-01-01", "2025-02-01"])})
        with self.assertRaisesRegex(ValueError, "日期或时间类型"):
            excel_ops._computed_column(
                frame,
                ComputedColumnInput(
                    name="偏移日期",
                    operator="add",
                    left="日期",
                    right_value=1,
                ),
            )

    def test_computed_column_is_registered_as_agent_tool(self):
        self.assertIn("add_computed_column", {item.name for item in excel_ops.EXCEL_TOOLS})

    def test_chart_handles_missing_category_values(self):
        """类别列有空值时也必须能画图（matplotlib 拒绝把 NaN 当类别）。"""
        import tempfile

        from schemas import CreateChartInput

        frame = pd.DataFrame(
            {"区域": ["东", None, "西"], "销售额": [100.0, 50.0, 80.0]}
        )
        previous = excel_ops.OUTPUT_DIR
        with tempfile.TemporaryDirectory() as directory:
            excel_ops.OUTPUT_DIR = directory
            try:
                for chart_type in ("bar", "line", "pie"):
                    with self.subTest(chart_type=chart_type):
                        url = excel_ops._create_chart(
                            frame,
                            CreateChartInput(
                                chart_type=chart_type,
                                chart_x="区域",
                                chart_y="销售额",
                            ),
                        )
                        self.assertTrue(url.startswith("/api/chart/"))
                        filename = url.rsplit("/", 1)[-1]
                        self.assertTrue(
                            (Path(directory) / filename).is_file(),
                            "图表文件未生成",
                        )
            finally:
                excel_ops.OUTPUT_DIR = previous

    def test_category_labels_replace_missing_values(self):
        labels = excel_ops._category_labels(pd.Series(["东", None, "西"]))

        self.assertEqual(labels.tolist(), ["东", "(空)", "西"])
        self.assertTrue(all(isinstance(value, str) for value in labels))

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
