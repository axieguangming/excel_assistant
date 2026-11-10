"""训练数据与线上工具的一致性测试。

回归背景：`create_chart` 的观察文本曾经在训练侧写成数据框预览，而线上返回的是
「图表生成成功：地址」。两者不一致导致模型在训练中「看不见」画图这一步：
- 画图前后上下文完全相同，token 损失惩罚极轻，容易被跳过
- 866 条以画图结尾的样本中，实测只有 50% 通过
"""

import unittest

import pandas as pd

import excel_ops
from finetune.dataset import sample_to_messages
from schemas import CreateChartInput


class ObservationConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({"部门": ["A", "B"], "金额": [1.0, 2.0]})

    def _observations(self, actions: list[dict]) -> list[str]:
        sample = {
            "user_request": "测试",
            "expected_actions": actions,
            "expected_response": "完成。",
        }
        return [
            message["content"]
            for message in sample_to_messages(sample, self.frame)
            if message["role"] == "tool"
        ]

    def test_chart_observation_reports_success_not_data_preview(self):
        observations = self._observations([
            {
                "tool": "create_chart",
                "args": {"chart_type": "bar", "chart_y": "金额", "chart_x": "部门"},
            },
        ])
        self.assertEqual(len(observations), 1)
        self.assertTrue(observations[0].startswith("图表生成成功："))
        self.assertIn("/api/chart/chart_", observations[0])
        self.assertIn("图表设置：", observations[0])
        # 关键：不能退化成结果行预览，否则模型以为这一步没发生。
        self.assertNotIn("样例：", observations[0])

    def test_chart_observation_matches_tool_implementation(self):
        chart = CreateChartInput(chart_type="bar", chart_y="金额", chart_x="部门")
        expected = excel_ops.chart_observation(chart, "/api/chart/chart_test.png")
        observations = self._observations([
            {
                "tool": "create_chart",
                "args": {"chart_type": "bar", "chart_y": "金额", "chart_x": "部门"},
            },
        ])
        # 地址是随机生成的，比较格式而非具体值。
        head = observations[0].rsplit("\n", 1)[0]
        self.assertEqual(head.split("：", 1)[0], expected.split("：", 1)[0])
        self.assertIn("chart_type", observations[0])

    def test_chart_step_is_distinguishable_from_previous_step(self):
        """画图前后的观察必须不同，否则模型无法判断该步是否发生。"""
        observations = self._observations([
            {"tool": "describe_data", "args": {}},
            {
                "tool": "create_chart",
                "args": {"chart_type": "bar", "chart_y": "金额", "chart_x": "部门"},
            },
        ])
        self.assertEqual(len(observations), 2)
        self.assertNotEqual(observations[0], observations[1])

    def test_describe_data_uses_read_only_wording(self):
        observations = self._observations([{"tool": "describe_data", "args": {}}])
        self.assertTrue(observations[0].startswith("查看成功（只读"))
        self.assertNotIn("操作成功", observations[0])

    def test_dataframe_tools_keep_preview(self):
        observations = self._observations([
            {
                "tool": "aggregate_data",
                "args": {
                    "group_by": ["部门"],
                    "metrics": [{"column": "金额", "function": "sum", "name": "合计"}],
                },
            },
        ])
        self.assertTrue(observations[0].startswith("操作成功。当前结果"))
        self.assertIn("样例：", observations[0])

    def test_describe_data_does_not_break_following_steps(self):
        """只读工具之后，后续步骤仍应作用在原数据上。"""
        observations = self._observations([
            {"tool": "describe_data", "args": {}},
            {
                "tool": "aggregate_data",
                "args": {
                    "group_by": ["部门"],
                    "metrics": [{"column": "金额", "function": "sum", "name": "合计"}],
                },
            },
        ])
        self.assertEqual(len(observations), 2)
        self.assertTrue(observations[0].startswith("查看成功（只读"))
        # 若 describe 被当成长表操作，这里会因找不到「部门」而抛错。
        self.assertIn("合计", observations[1])

    def test_observation_for_step_dispatches_by_tool(self):
        read_only = excel_ops.observation_for_step("describe_data", self.frame)
        chart = excel_ops.observation_for_step(
            "create_chart",
            self.frame,
            arguments={"chart_type": "bar", "chart_y": "金额"},
            chart_url="/api/chart/chart_x.png",
        )
        plain = excel_ops.observation_for_step("filter_rows", self.frame)

        self.assertTrue(read_only.startswith("查看成功（只读"))
        self.assertTrue(chart.startswith("图表生成成功："))
        self.assertIn("/api/chart/chart_x.png", chart)
        self.assertTrue(plain.startswith("操作成功。当前结果"))


class DeterminismTests(unittest.TestCase):
    """同一份输入必须每次生成完全相同的数据，否则无法对比两次构建的差异。"""

    def test_chart_url_is_deterministic(self):
        from finetune.dataset import _chart_placeholder_url

        first = _chart_placeholder_url("sample-1-2")
        second = _chart_placeholder_url("sample-1-2")
        self.assertEqual(first, second)
        self.assertRegex(first, r"^/api/chart/chart_[0-9a-f]{12}\.png$")

    def test_chart_url_differs_across_samples(self):
        from finetune.dataset import _chart_placeholder_url

        self.assertNotEqual(
            _chart_placeholder_url("sample-1-1"),
            _chart_placeholder_url("sample-2-1"),
        )

    def test_sample_messages_are_reproducible(self):
        frame = pd.DataFrame({"部门": ["A", "B"], "金额": [1.0, 2.0]})
        sample = {
            "sample_id": "wb-001",
            "user_request": "画个图",
            "expected_actions": [
                {
                    "tool": "create_chart",
                    "args": {"chart_type": "bar", "chart_y": "金额", "chart_x": "部门"},
                },
            ],
            "expected_response": "完成。",
        }
        first = sample_to_messages(sample, frame)
        second = sample_to_messages(sample, frame)
        self.assertEqual(
            [m.get("content") for m in first],
            [m.get("content") for m in second],
        )


class ErrorObservationTests(unittest.TestCase):
    """失败文本必须与线上工具一致，否则「报错后恢复」样本学不到真实行为。"""

    def test_dataframe_tool_prefix(self):
        self.assertEqual(
            excel_ops.error_observation(ValueError("找不到字段：['x']")),
            "工具执行失败：找不到字段：['x']",
        )

    def test_create_chart_keeps_legacy_prefix(self):
        # create_chart 在 v2 训练与评测中已固定为「图表生成失败」，不能改动。
        self.assertEqual(
            excel_ops.error_observation(ValueError("x"), tool_name="create_chart"),
            "图表生成失败：x",
        )

    def test_live_tool_uses_same_text(self):
        """真实工具抛错时返回的文本必须与 error_observation 一致。"""
        frame = pd.DataFrame({"部门": ["A"], "金额": [1.0]})
        state = {"current_df": frame, "plans": [], "tool_steps": 0}
        command = excel_ops._execute_dataframe_tool(
            state,
            "call-1",
            "aggregate_data",
            {"group_by": ["不存在的列"]},
            lambda df: excel_ops._aggregate(
                df,
                ["不存在的列"],
                [__import__("schemas").Metric(column="金额", function="sum")],
            ),
        )
        content = command.update["messages"][0].content
        self.assertTrue(content.startswith("工具执行失败："))
        self.assertIn("可用字段", content)


if __name__ == "__main__":
    unittest.main()
