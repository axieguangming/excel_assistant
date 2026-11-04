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

    def test_grouped_count_is_row_count_not_non_null_count(self):
        # count 表示「该组有多少条记录」。含空值的列不能少算，
        # 否则「有多少条反馈」会因模型选哪一列而得到不同结果。
        frame = pd.DataFrame({
            "反馈": ["好", None, "一般", None],
            "编号": ["A", "B", "C", "D"],
        })
        metrics = [Metric(column="反馈", function="count", name="条数")]

        result = excel_ops._aggregate(frame, ["编号"], metrics)

        self.assertEqual(result["条数"].tolist(), [1, 1, 1, 1])

    def test_count_does_not_depend_on_chosen_column(self):
        frame = pd.DataFrame({
            "反馈": ["好", None, "一般", None],
            "编号": ["A", "B", "C", "D"],
        })
        on_nullable = excel_ops._aggregate(
            frame, [], [Metric(column="反馈", function="count")]
        )
        on_complete = excel_ops._aggregate(
            frame, [], [Metric(column="编号", function="count")]
        )

        self.assertEqual(on_nullable.loc[0, "反馈_count"], 4)
        self.assertEqual(
            on_nullable.loc[0, "反馈_count"], on_complete.loc[0, "编号_count"]
        )

    def test_grouped_count_equals_sum_of_group_sizes(self):
        frame = pd.DataFrame({
            "组": ["甲", "甲", "乙", None],
            "值": [1, None, 3, 4],
        })
        result = excel_ops._aggregate(
            frame,
            ["组"],
            [Metric(column="值", function="count", name="行数")],
        )

        self.assertEqual(sorted(result["行数"].tolist()), [1, 1, 2])

    def test_nunique_still_ignores_nulls(self):
        # 去重计数问的是「有多少种」，空值不算一种。
        frame = pd.DataFrame({"值": ["a", None, "a", "b"]})
        result = excel_ops._aggregate(
            frame, [], [Metric(column="值", function="nunique")]
        )

        self.assertEqual(result.loc[0, "值_nunique"], 2)

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


class ReadOnlyToolTests(unittest.TestCase):
    """describe_data 只读：不覆盖 current_df，但 result_df 与行数仍要更新。

    回归背景：早期实现里 describe_data 会把 current_df 换成 9×12 的统计摘要，
    导致「先看一眼数据」之后所有工具都报「找不到字段」。
    """

    def setUp(self):
        self.frame = pd.DataFrame({
            "地区": ["北区", "南区", "北区"],
            "销售额": [100, 80, 150],
        })

    def _run(self, read_only: bool):
        state = {"current_df": self.frame.copy(), "tool_steps": 0, "plans": []}
        command = excel_ops._execute_dataframe_tool(
            state,
            "tool-call",
            "describe_data",
            {},
            excel_ops._describe_data,
            read_only=read_only,
        )
        return command.update

    def test_read_only_tool_keeps_current_dataframe(self):
        update = self._run(read_only=True)

        self.assertNotIn("current_df", update)
        self.assertEqual(list(update["result_df"].columns)[0], "字段")

    def test_read_only_tool_records_result_rows(self):
        update = self._run(read_only=True)

        self.assertEqual(update["last_result_rows"], len(update["result_df"]))

    def test_writable_tool_still_replaces_current_dataframe(self):
        update = self._run(read_only=False)

        self.assertIn("current_df", update)
        self.assertIs(update["current_df"], update["result_df"])

    def test_describe_does_not_break_following_tools(self):
        # 真实场景：「先看一眼数据，再做汇总」。
        state = {"current_df": self.frame.copy(), "tool_steps": 0, "plans": []}
        describe = excel_ops._execute_dataframe_tool(
            state, "c1", "describe_data", {}, excel_ops._describe_data, read_only=True
        )
        state.update(describe.update)

        aggregate = excel_ops._execute_dataframe_tool(
            state,
            "c2",
            "aggregate_data",
            {"group_by": ["地区"], "metrics": []},
            lambda df: excel_ops._aggregate(df, ["地区"], [Metric(column="销售额", function="sum")]),
        )

        self.assertEqual(
            aggregate.update["result_df"]["销售额_sum"].tolist(), [250, 80]
        )


if __name__ == "__main__":
    unittest.main()
