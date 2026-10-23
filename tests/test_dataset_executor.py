import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dataset_builder.executor import execute_chain


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "区域": ["东", "西", "东", "西"],
                "年份": [2025, 2025, 2024, 2025],
                "销售额": [100, 80, 50, 25],
                "成本": [40, 30, 20, 10],
            }
        )

    def test_filter_then_aggregate_produces_ground_truth(self):
        outcome = execute_chain(
            self.frame,
            [
                {"tool": "filter_rows", "args": {"filters": [{"column": "年份", "operator": "eq", "value": 2025}]}},
                {
                    "tool": "aggregate_data",
                    "args": {
                        "group_by": ["区域"],
                        "metrics": [{"column": "销售额", "function": "sum"}],
                    },
                },
            ],
        )

        self.assertTrue(outcome.ok)
        assert outcome.frame is not None
        self.assertEqual(outcome.frame["区域"].tolist(), ["东", "西"])
        self.assertEqual(outcome.frame["销售额_sum"].tolist(), [100, 105])
        self.assertEqual([step.rows for step in outcome.steps], [3, 2])
        self.assertTrue(all(step.ok for step in outcome.steps))

    def test_computed_column_can_feed_later_aggregate(self):
        outcome = execute_chain(
            self.frame,
            [
                {
                    "tool": "add_computed_column",
                    "args": {
                        "name": "毛利",
                        "operator": "subtract",
                        "left": "销售额",
                        "right_column": "成本",
                    },
                },
                {
                    "tool": "aggregate_data",
                    "args": {
                        "group_by": ["区域"],
                        "metrics": [{"column": "毛利", "function": "sum"}],
                    },
                },
            ],
        )

        self.assertTrue(outcome.ok)
        assert outcome.frame is not None
        self.assertEqual(outcome.frame["毛利_sum"].tolist(), [90, 65])

    def test_unknown_column_reports_failing_step(self):
        outcome = execute_chain(
            self.frame,
            [
                {"tool": "sort_rows", "args": {"sort_by": [{"column": "不存在"}]}},
                {"tool": "describe_data", "args": {}},
            ],
        )

        self.assertFalse(outcome.ok)
        self.assertIn("找不到字段", outcome.error or "")
        self.assertEqual(len(outcome.steps), 1)
        self.assertFalse(outcome.steps[0].ok)

    def test_source_frame_is_not_modified(self):
        original = self.frame.copy(deep=True)
        execute_chain(
            self.frame,
            [{"tool": "fill_missing_values", "args": {"fill_strategy": "value", "fill_value": 0}}],
        )

        pd.testing.assert_frame_equal(self.frame, original)

    def test_chart_requires_numeric_value_column(self):
        frame = pd.DataFrame({"名称": ["a", "b"]})
        outcome = execute_chain(
            frame,
            [{"tool": "create_chart", "args": {"chart_type": "bar", "chart_y": "名称"}}],
        )

        self.assertFalse(outcome.ok)
        self.assertIn("没有可用数值", outcome.error or "")

    def test_chart_does_not_change_current_frame(self):
        outcome = execute_chain(
            self.frame,
            [{"tool": "create_chart", "args": {"chart_type": "bar", "chart_x": "区域", "chart_y": "销售额"}}],
        )

        self.assertTrue(outcome.ok)
        assert outcome.frame is not None
        pd.testing.assert_frame_equal(outcome.frame.reset_index(drop=True), self.frame)

    def test_chart_can_be_rendered_to_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            outcome = execute_chain(
                self.frame,
                [{"tool": "create_chart", "args": {"chart_type": "bar", "chart_x": "区域", "chart_y": "销售额"}}],
                render_charts=True,
                chart_dir=Path(directory),
            )

            self.assertTrue(outcome.ok)
            self.assertTrue(outcome.chart)
            chart_name = outcome.chart.rsplit("/", 1)[-1]
            self.assertTrue((Path(directory) / chart_name).is_file())

    def test_describe_data_changes_the_working_frame(self):
        outcome = execute_chain(self.frame, [{"tool": "describe_data", "args": {}}])

        self.assertTrue(outcome.ok)
        assert outcome.frame is not None
        self.assertIn("字段", outcome.frame.columns)


if __name__ == "__main__":
    unittest.main()
