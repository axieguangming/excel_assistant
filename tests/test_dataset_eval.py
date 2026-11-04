import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from langchain_core.messages import AIMessage

import agent
import excel_ops
from dataset_builder.eval import (
    EvalConfig,
    EvalReport,
    compare_frames,
    evaluate,
    evaluate_sample,
    failure_breakdown,
    format_summary,
    write_report,
)
from dataset_builder.pipeline import (
    BuildConfig,
    build_dataset,
    read_jsonl,
    write_build_result,
)
from dataset_builder.tables import generate_tables, load_tables, save_tables
from dataset_builder.templates import TemplateChainProposer, TemplateQuestionWriter
from schemas import Metric


def tool_call(name, args, call_id):
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": call_id, "type": "tool_call"}
        ],
    )


class ScriptedModel:
    """按期望操作链回放工具调用的假模型，用于验证评测接线。"""

    def __init__(self, plan):
        self.plan = plan

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        index = sum(1 for message in messages if message.type == "ai")
        if index < len(self.plan):
            return tool_call(self.plan[index]["tool"], self.plan[index]["args"], f"c{index}")
        return AIMessage(content="已完成。")


class WrongToolModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content="我不确定，请补充信息。")


class CompareFramesTests(unittest.TestCase):
    def test_identical_frames_match(self):
        frame = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        matched, reason = compare_frames(frame, frame.copy())

        self.assertTrue(matched)
        self.assertEqual(reason, "")

    def test_float_representation_difference_is_tolerated(self):
        expected = pd.read_csv(
            pd.io.common.StringIO("v\n100.0\n250.5\n"), encoding="utf-8-sig"
        )
        actual = pd.DataFrame({"v": [100, 250.5]})

        matched, _ = compare_frames(expected, actual)
        self.assertTrue(matched)

    def test_large_number_with_ulp_difference_is_tolerated(self):
        # 汇总大额金额时 float64 的 ULP 约 3.8e-6；固定 6 位小数的旧实现会
        # 把 27103570497.56 与 27103570497.559998 判成不同值。
        expected = pd.DataFrame({"v": [27103570497.560001]})
        actual = pd.DataFrame({"v": [27103570497.559998]})

        matched, reason = compare_frames(expected, actual)
        self.assertTrue(matched, reason)

    def test_genuinely_different_large_numbers_are_reported(self):
        expected = pd.DataFrame({"v": [27103570497.56]})
        actual = pd.DataFrame({"v": [27103570498.56]})

        matched, reason = compare_frames(expected, actual)
        self.assertFalse(matched)
        self.assertEqual(reason, "数据内容不同")

    def test_row_order_difference_is_reported_separately(self):
        expected = pd.DataFrame({"a": [1, 2]})
        actual = pd.DataFrame({"a": [2, 1]})

        matched, reason = compare_frames(expected, actual)
        self.assertFalse(matched)
        self.assertEqual(reason, "行内容一致但顺序不同")

    def test_shape_difference_is_reported(self):
        expected = pd.DataFrame({"a": [1, 2, 3]})
        actual = pd.DataFrame({"a": [1, 2]})

        matched, reason = compare_frames(expected, actual)
        self.assertFalse(matched)
        self.assertIn("形状不同", reason)

    def test_different_values_are_reported(self):
        matched, reason = compare_frames(
            pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2]})
        )

        self.assertFalse(matched)
        self.assertEqual(reason, "数据内容不同")


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.previous_llm = agent.llm
        self.tables = generate_tables(2, seed=301)
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

        save_tables(self.tables, self.root / "tables")
        self.restored = load_tables(self.root / "tables")
        self.result = build_dataset(
            self.restored,
            proposer=TemplateChainProposer(),
            writer=TemplateQuestionWriter(),
            config=BuildConfig(samples_per_table=2, max_steps=3),
            out_dir=self.root,
        )
        write_build_result(self.result, self.root)

    def tearDown(self):
        agent.llm = self.previous_llm
        self._tmp.cleanup()

    def _tables_by_id(self):
        return {table.table_id: table for table in self.restored}

    def test_perfect_model_scores_full_marks(self):
        first = read_jsonl(self.root / "splits" / "train.jsonl")[0]
        report = evaluate(
            data_dir=self.root,
            llm=ScriptedModel(first["expected_actions"]),
            config=EvalConfig(split="train", limit=1),
        )

        self.assertTrue(report.samples[0].task_success, report.samples[0].failure)
        self.assertEqual(report.metrics["task_success_rate"], 1.0)

    def test_sample_with_correct_plan_matches_expectation(self):
        sample = self.result.samples[0]
        agent.llm = ScriptedModel(sample["expected_actions"])

        outcome = evaluate_sample(
            sample, self._tables_by_id(), dataset_root=self.root
        )

        self.assertTrue(outcome.task_success, outcome.failure)
        self.assertTrue(outcome.tool_sequence_match)
        self.assertTrue(outcome.args_match)
        self.assertTrue(outcome.first_tool_match)
        self.assertEqual(outcome.failure, "")

    def test_model_without_tool_calls_reports_no_result(self):
        sample = self.result.samples[0]
        agent.llm = WrongToolModel()

        outcome = evaluate_sample(
            sample, self._tables_by_id(), dataset_root=self.root
        )

        self.assertFalse(outcome.task_success)
        self.assertEqual(outcome.failure, "no_result")
        self.assertFalse(outcome.tool_sequence_match)

    def test_missing_table_is_reported(self):
        sample = dict(self.result.samples[0])
        sample["workbook_id"] = "does-not-exist"

        outcome = evaluate_sample(sample, self._tables_by_id(), dataset_root=self.root)

        self.assertFalse(outcome.task_success)
        self.assertEqual(outcome.failure, "missing_table")

    def test_missing_artifact_is_reported(self):
        sample = dict(self.result.samples[0])
        sample["expected_result"] = dict(sample["expected_result"])
        sample["expected_result"]["artifact"] = "results/nope.csv"

        outcome = evaluate_sample(sample, self._tables_by_id(), dataset_root=self.root)

        self.assertFalse(outcome.task_success)
        self.assertIn("missing_artifact", outcome.failure)


    def test_missing_split_raises_clear_error(self):
        empty_root = Path(self._tmp.name) / "empty"
        empty_root.mkdir()
        with self.assertRaises(FileNotFoundError):
            evaluate(
                data_dir=empty_root,
                llm=ScriptedModel([]),
                config=EvalConfig(split="test"),
            )

    def test_empty_split_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "没有样本"):
            evaluate(
                data_dir=self.root,
                llm=ScriptedModel([]),
                config=EvalConfig(split="test", limit=0),
            )

    def test_metrics_and_report_are_serializable(self):
        class MixedModel:
            """第一条正确、其余直接结束，用于产生成功与失败混合的报告。"""

            def __init__(self, first_plan):
                self.first_plan = first_plan
                self.calls = 0

            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                if self.calls:
                    return AIMessage(content="完成。")
                self.calls += 1
                if not self.first_plan:
                    return AIMessage(content="无法完成。")
                step = self.first_plan[0]
                return tool_call(step["tool"], step["args"], "c0")

        report = evaluate(
            data_dir=self.root,
            llm=MixedModel(self.result.samples[0]["expected_actions"]),
            config=EvalConfig(split="train", limit=2),
        )

        metrics = report.metrics
        self.assertEqual(metrics["samples"], len(report.samples))
        self.assertTrue(0.0 <= metrics["task_success_rate"] <= 1.0)
        self.assertIn("by_category", metrics)

        payload = report.as_dict()
        json.dumps(payload, ensure_ascii=False)
        self.assertIn("failures", payload)

        path = write_report(report, self.root / "eval.json")
        written = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(written["metrics"]["samples"], metrics["samples"])

        summary = format_summary(report)
        self.assertIn("任务成功率", summary)
        self.assertTrue(all(isinstance(reason, str) for reason, _ in failure_breakdown(report)))

    def test_llm_is_restored_after_evaluation(self):
        sentinel = object()
        agent.llm = sentinel
        with self.assertRaisesRegex(ValueError, "没有样本"):
            evaluate(
                data_dir=self.root,
                llm=ScriptedModel([]),
                config=EvalConfig(split="test", limit=0),
            )

        self.assertIs(agent.llm, sentinel)

    def test_llm_is_restored_when_model_raises(self):
        class ExplodingModel:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                raise RuntimeError("模型崩溃")

        sentinel = object()
        agent.llm = sentinel
        with self.assertRaisesRegex(RuntimeError, "模型崩溃"):
            evaluate(
                data_dir=self.root,
                llm=ExplodingModel(),
                config=EvalConfig(split="train", limit=1),
            )

        self.assertIs(agent.llm, sentinel)




class ChartVerdictTests(unittest.TestCase):
    """图表任务必须按"是否真的画出图"判定，而不是"是否产生结果表"。"""

    def setUp(self):
        self.previous_llm = agent.llm
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        tables = generate_tables(1, seed=401)
        save_tables(tables, self.root / "tables")
        self.table = load_tables(self.root / "tables")[0]

        # 期望值取自真实工具输出，保证"完整执行应判定成功"这一断言本身可靠。
        expected_frame = excel_ops._aggregate(
            self.table.frame,
            ["区域"],
            [Metric(column="销售额", function="sum")],
        )
        artifact = self.root / "results" / "chart-sample.csv"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        expected_frame.to_csv(artifact, index=False, encoding="utf-8-sig")

        self.sample = {
            "sample_id": "chart-1",
            "workbook_id": self.table.table_id,
            "user_request": "按区域汇总销售额并画柱状图",
            "task_categories": ["aggregate", "chart"],
            "difficulty": "multi_step",
            "expected_actions": [
                {
                    "tool": "aggregate_data",
                    "args": {
                        "group_by": ["区域"],
                        "metrics": [{"column": "销售额", "function": "sum"}],
                    },
                },
                {
                    "tool": "create_chart",
                    "args": {
                        "chart_type": "bar",
                        "chart_x": "区域",
                        "chart_y": "销售额_sum",
                    },
                },
            ],
            "expected_result": {"artifact": "results/chart-sample.csv", "row_count": 2},
        }

        self.previous_output = excel_ops.OUTPUT_DIR
        self.chart_dir = self.root / "charts"
        self.chart_dir.mkdir(exist_ok=True)
        excel_ops.OUTPUT_DIR = str(self.chart_dir)

    def tearDown(self):
        excel_ops.OUTPUT_DIR = self.previous_output
        agent.llm = self.previous_llm
        self._tmp.cleanup()

    def _agent(self, calls):
        class Model:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                index = sum(1 for message in messages if message.type == "ai")
                if index < len(calls):
                    name, args = calls[index]
                    return tool_call(name, args, f"c{index}")
                return AIMessage(content="已完成。")

        return Model()

    def _evaluate(self):
        return evaluate_sample(
            self.sample,
            {self.table.table_id: self.table},
            dataset_root=self.root,
        )

    def test_failed_chart_tool_is_reported_as_no_chart(self):
        """create_chart 执行失败时，原因应是 no_chart，而不是含混的 no_result。"""
        agent.llm = self._agent(
            [("create_chart", {"chart_type": "bar", "chart_x": "区域", "chart_y": "不存在的列"})]
        )

        outcome = self._evaluate()

        self.assertTrue(outcome.expects_chart)
        self.assertFalse(outcome.chart_produced)
        self.assertEqual(outcome.failure, "no_chart")
        self.assertFalse(outcome.task_success)

    def test_no_tool_call_at_all_is_distinguished(self):
        agent.llm = WrongToolModel()

        outcome = self._evaluate()

        self.assertEqual(outcome.failure, "no_tool_call")

    def test_chart_without_preceding_aggregate_fails_on_data(self):
        """画出图但少了汇总步骤：有图、数据不对，应单独报告。"""
        agent.llm = self._agent(
            [("create_chart", {"chart_type": "bar", "chart_x": "区域", "chart_y": "销售额"})]
        )

        outcome = self._evaluate()

        self.assertTrue(outcome.chart_produced, outcome.failure)
        self.assertFalse(outcome.task_success)
        self.assertEqual(outcome.failure, "chart_ok_but_data_mismatch")

    def test_complete_chart_task_counts_as_success(self):
        agent.llm = self._agent(
            [
                (
                    "aggregate_data",
                    {
                        "group_by": ["区域"],
                        "metrics": [{"column": "销售额", "function": "sum"}],
                    },
                ),
                (
                    "create_chart",
                    {"chart_type": "bar", "chart_x": "区域", "chart_y": "销售额_sum"},
                ),
            ]
        )

        outcome = self._evaluate()

        self.assertTrue(outcome.chart_produced, outcome.failure)
        self.assertTrue(outcome.task_success, outcome.failure)
        self.assertEqual(outcome.failure, "")

    def test_chart_metrics_include_production_rate(self):
        agent.llm = self._agent(
            [("create_chart", {"chart_type": "bar", "chart_x": "区域", "chart_y": "销售额"})]
        )

        report = EvalReport(config={})
        report.samples.append(self._evaluate())

        metrics = report.metrics
        self.assertEqual(metrics["chart_samples"], 1)
        self.assertEqual(metrics["chart_produced_rate"], 1.0)

    def test_non_chart_samples_do_not_affect_chart_rate(self):
        agent.llm = self._agent([("describe_data", {})])
        non_chart = dict(self.sample)
        non_chart["expected_actions"] = [{"tool": "describe_data", "args": {}}]
        non_chart["task_categories"] = ["describe"]

        result = evaluate_sample(
            non_chart, {self.table.table_id: self.table}, dataset_root=self.root
        )
        report = EvalReport(config={})
        report.samples.append(result)

        self.assertFalse(result.expects_chart)
        self.assertEqual(report.metrics["chart_samples"], 0)


if __name__ == "__main__":
    unittest.main()
