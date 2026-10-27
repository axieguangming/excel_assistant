import json
import tempfile
import unittest
from pathlib import Path

from dataset_builder.pipeline import (
    BuildConfig,
    build_dataset,
    read_jsonl,
    replay_samples,
    write_build_result,
)
from dataset_builder.split import coverage_summary, find_leaks, stratified_split
from dataset_builder.tables import generate_tables, load_tables, save_tables
from dataset_builder.templates import TemplateChainProposer, TemplateQuestionWriter


class StaticProposer:
    def __init__(self, chains):
        self.chains = chains

    def propose(self, table, count, *, max_steps=4, allowed_tools=None):
        return self.chains


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeChatModel:
    """按脚本顺序返回回复，用于离线验证大模型适配层接线。"""

    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, messages):
        return FakeResponse(self.replies.pop(0))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tables = generate_tables(4, seed=101)
        self.config = BuildConfig(samples_per_table=3, max_steps=3)

    def test_template_pipeline_produces_verified_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            result = build_dataset(
                self.tables,
                proposer=TemplateChainProposer(),
                writer=TemplateQuestionWriter(),
                config=self.config,
                out_dir=directory,
            )

            self.assertGreater(len(result.samples), 0)
            self.assertEqual(len(result.samples), self.config.samples_per_table * len(self.tables))
            for sample in result.samples:
                self.assertEqual(sample["verification"]["state"], "auto_passed")
                self.assertEqual(sample["verification"]["execution"], "passed")
                self.assertEqual(
                    sample["verification"]["question_consistency"], "structural"
                )
                self.assertTrue(sample["expected_actions"])
                self.assertTrue(sample["user_request"])
                artifact = Path(directory) / sample["expected_result"]["artifact"]
                self.assertTrue(artifact.is_file())
                self.assertEqual(
                    sample["expected_result"]["row_count"],
                    len(read_csv_rows(artifact)),
                )

            report = result.report
            self.assertEqual(report["samples"], len(result.samples))
            self.assertEqual(report["tables"], len(self.tables))

    def test_split_has_no_leakage_and_covers_every_partition(self):
        with tempfile.TemporaryDirectory() as directory:
            result = build_dataset(
                self.tables,
                proposer=TemplateChainProposer(),
                writer=TemplateQuestionWriter(),
                config=self.config,
                out_dir=directory,
            )

        self.assertEqual(find_leaks(result.splits), [])
        for name, samples in result.splits.items():
            self.assertTrue(samples, f"{name} 分区为空")
        total = sum(len(samples) for samples in result.splits.values())
        self.assertEqual(total, len(result.samples))

    def test_replay_matches_recorded_step_results(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = save_tables(self.tables, Path(directory) / "tables")
            self.assertTrue(index_path.is_file())
            restored = load_tables(Path(directory) / "tables")

            # 与 CLI 一致：用 Excel 往返后的表格建样本。
            result = build_dataset(
                restored,
                proposer=TemplateChainProposer(),
                writer=TemplateQuestionWriter(),
                config=self.config,
                out_dir=directory,
            )

        report = replay_samples(
            result.samples, {table.table_id: table for table in restored}
        )

        self.assertEqual(report["mismatches"], [])
        self.assertEqual(report["checked"], len(result.samples))

    def test_samples_are_built_on_excel_round_tripped_tables(self):
        """真值必须基于用户上传 Excel 后读到的数据，而不是内存中的生成结果。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_tables(self.tables, root / "tables")
            restored = load_tables(root / "tables")

            result = build_dataset(
                restored,
                proposer=TemplateChainProposer(),
                writer=TemplateQuestionWriter(),
                config=self.config,
                out_dir=root,
            )

            self.assertTrue(any((root / "tables").glob("*.xlsx")))
            self.assertGreater(len(result.samples), 0)
            sample = result.samples[0]
            artifact = root / sample["expected_result"]["artifact"]
            self.assertTrue(artifact.is_file())

            # 每一步的行数都不能超过 Excel 里读到的原始行数（筛选/汇总只会减少），
            # 且最后一步的行数要与落盘的结果文件一致。
            table = next(t for t in restored if t.table_id == sample["workbook_id"])
            source_rows = int(len(table.frame))
            for step in sample["step_results"]:
                self.assertLessEqual(step["rows"], max(source_rows, 1))

            self.assertEqual(
                sample["step_results"][-1]["rows"],
                sample["expected_result"]["row_count"],
            )
            written_rows = len(
                artifact.read_text(encoding="utf-8-sig").splitlines()
            ) - 1
            self.assertEqual(written_rows, sample["expected_result"]["row_count"])

    def test_build_result_is_written_as_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            result = build_dataset(
                self.tables,
                proposer=TemplateChainProposer(),
                writer=TemplateQuestionWriter(),
                config=self.config,
                out_dir=directory,
            )
            paths = write_build_result(result, directory)

            written = read_jsonl(paths["samples"])
            self.assertEqual(len(written), len(result.samples))
            self.assertEqual(written[0]["sample_id"], result.samples[0]["sample_id"])
            for split_name in ("train", "val", "test"):
                self.assertTrue(paths[split_name].is_file())
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            self.assertEqual(report["samples"], len(result.samples))

    def test_invalid_chain_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = build_dataset(
                self.tables[:1],
                proposer=StaticProposer([{"steps": [{"tool": "pivot_table", "args": {}}]}]),
                writer=TemplateQuestionWriter(),
                config=self.config,
                out_dir=directory,
            )

        self.assertEqual(result.samples, [])
        self.assertEqual(result.rejects[0]["reason"], "invalid_chain")

    def test_execution_failure_is_rejected(self):
        chain = [
            {"tool": "sort_rows", "args": {"sort_by": [{"column": "不存在"}]}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            result = build_dataset(
                self.tables[:1],
                proposer=StaticProposer([{"steps": chain}]),
                writer=TemplateQuestionWriter(),
                config=self.config,
                out_dir=directory,
            )

        self.assertEqual(result.samples, [])
        self.assertEqual(result.rejects[0]["reason"], "execution_failed")
        self.assertIn("找不到字段", result.rejects[0]["detail"])

    def test_question_mismatch_is_rejected(self):
        chain = [{"tool": "describe_data", "args": {}}]
        with tempfile.TemporaryDirectory() as directory:
            result = build_dataset(
                self.tables[:1],
                proposer=StaticProposer([{"steps": chain}]),
                writer=TemplateQuestionWriter(),
                parser=lambda table, question: {
                    "steps": [{"tool": "deduplicate_rows", "args": {}}]
                },
                config=self.config,
                out_dir=directory,
            )

        self.assertEqual(result.samples, [])
        self.assertEqual(result.rejects[0]["reason"], "question_mismatch")

    def test_unexecutable_parse_is_rejected(self):
        chain = [{"tool": "describe_data", "args": {}}]
        with tempfile.TemporaryDirectory() as directory:
            result = build_dataset(
                self.tables[:1],
                proposer=StaticProposer([{"steps": chain}]),
                writer=TemplateQuestionWriter(),
                parser=lambda table, question, feedback="": {
                    "steps": [
                        {
                            "tool": "sort_rows",
                            "args": {"sort_by": [{"column": "不存在的列"}]},
                        }
                    ]
                },
                config=self.config,
                out_dir=directory,
            )

        self.assertEqual(result.samples, [])
        self.assertIn("无法执行", result.rejects[0]["detail"])

    def test_parser_is_retried_with_feedback(self):
        chain = [{"tool": "describe_data", "args": {}}]
        feedbacks: list[str] = []

        def parser(table, question, feedback=""):
            feedbacks.append(feedback)
            if not feedback:
                return {"steps": [{"tool": "describe_data", "args": {"bad": 1}}]}
            return {"steps": chain}

        with tempfile.TemporaryDirectory() as directory:
            result = build_dataset(
                self.tables[:1],
                proposer=StaticProposer([{"steps": chain}]),
                writer=TemplateQuestionWriter(),
                parser=parser,
                config=self.config,
                out_dir=directory,
            )

        self.assertEqual(len(result.samples), 1)
        self.assertEqual(len(feedbacks), 2)
        self.assertTrue(feedbacks[0] == "")
        self.assertIn("不接受参数", feedbacks[1])

    def test_matching_parser_accepts_the_sample(self):
        chain = [{"tool": "describe_data", "args": {}}]
        with tempfile.TemporaryDirectory() as directory:
            result = build_dataset(
                self.tables[:1],
                proposer=StaticProposer([{"steps": chain}]),
                writer=TemplateQuestionWriter(),
                parser=lambda table, question: {"steps": chain},
                config=self.config,
                out_dir=directory,
            )

        self.assertEqual(len(result.samples), 1)
        self.assertEqual(
            result.samples[0]["verification"]["question_consistency"], "llm_verified"
        )

    def test_disabling_question_verification_marks_needs_review(self):
        chain = [{"tool": "describe_data", "args": {}}]
        with tempfile.TemporaryDirectory() as directory:
            result = build_dataset(
                self.tables[:1],
                proposer=StaticProposer([{"steps": chain}]),
                writer=TemplateQuestionWriter(),
                config=BuildConfig(
                    samples_per_table=1, max_steps=3, verify_questions=False
                ),
                out_dir=directory,
            )

        self.assertEqual(
            result.samples[0]["verification"]["state"], "needs_review"
        )
        self.assertEqual(
            result.samples[0]["verification"]["question_consistency"], "unverified"
        )

    def test_llm_components_are_wired_into_the_pipeline(self):
        from dataset_builder.llm import (
            DatasetLLM,
            LLMChainProposer,
            LLMQuestionParser,
            LLMQuestionWriter,
        )

        chain = [{"tool": "describe_data", "args": {}}]
        model = FakeChatModel(
            [
                json.dumps([{"steps": chain}]),
                json.dumps({"question": "请给出这份数据的统计概况", "answer": "已生成统计概况。"}),
                json.dumps({"steps": chain}),
            ]
        )
        parser = LLMQuestionParser(DatasetLLM(model))

        with tempfile.TemporaryDirectory() as directory:
            result = build_dataset(
                self.tables[:1],
                proposer=LLMChainProposer(DatasetLLM(model)),
                writer=LLMQuestionWriter(DatasetLLM(model)),
                parser=lambda table, question: parser.parse(table, question),
                config=BuildConfig(samples_per_table=1, max_steps=2),
                out_dir=directory,
            )

        self.assertEqual(len(result.samples), 1)
        sample = result.samples[0]
        self.assertEqual(sample["user_request"], "请给出这份数据的统计概况")
        self.assertEqual(
            sample["verification"]["question_consistency"], "llm_verified"
        )
        self.assertEqual(sample["provenance"]["proposer"], "LLMChainProposer")
        self.assertEqual(model.replies, [])


class SplitTests(unittest.TestCase):
    def test_groups_are_not_shared_between_splits(self):
        samples = [
            {
                "sample_id": f"s{index}",
                "workbook_id": f"w{index // 3}",
                "task_categories": ["aggregate"],
            }
            for index in range(30)
        ]
        splits = stratified_split(samples, seed=3)

        self.assertEqual(find_leaks(splits), [])
        self.assertEqual(sum(len(items) for items in splits.values()), len(samples))

    def test_small_group_counts_still_receive_partitions(self):
        samples = [
            {
                "sample_id": f"s{index}",
                "workbook_id": f"w{index}",
                "task_categories": ["sort"],
            }
            for index in range(3)
        ]
        splits = stratified_split(samples, seed=1)

        self.assertTrue(all(splits[name] for name in ("train", "val", "test")))

    def test_invalid_ratios_are_rejected(self):
        with self.assertRaises(ValueError):
            stratified_split([], ratios=(0.0, 0.0, 0.0))

    def test_empty_input_returns_empty_splits(self):
        splits = stratified_split([])

        self.assertEqual(splits, {"train": [], "val": [], "test": []})

    def test_eval_splits_get_at_least_two_workbooks(self):
        samples = [
            {
                "sample_id": f"s{book}-{index}",
                "workbook_id": f"w{book}",
                "task_categories": ["filter"],
            }
            for book in range(12)
            for index in range(3)
        ]
        splits = stratified_split(samples, seed=0, min_eval_workbooks=2)

        for name in ("val", "test"):
            workbooks = {item["workbook_id"] for item in splits[name]}
            self.assertGreaterEqual(len(workbooks), 2, f"{name} 工作簿过少")
        self.assertTrue(splits["train"])

    def test_test_split_maximises_category_coverage(self):
        """测试集应尽量覆盖多种任务类别，而不是随机落在一类上。"""
        samples = []
        category_sets = [
            ["aggregate", "chart", "sort"],
            ["filter", "top_n"],
            ["dedupe", "fill_missing"],
            ["computed_column", "describe"],
            ["aggregate"],
            ["sort"],
            ["chart"],
            ["filter"],
            ["top_n"],
            ["describe"],
            ["computed_column"],
            ["dedupe"],
        ]
        for index, categories in enumerate(category_sets):
            samples.append(
                {
                    "sample_id": f"s{index}",
                    "workbook_id": f"w{index}",
                    "task_categories": categories,
                }
            )

        splits = stratified_split(samples, seed=7, min_eval_workbooks=2)
        summary = coverage_summary(splits)

        self.assertGreaterEqual(len(summary["test"]["categories"]), 5)
        self.assertGreaterEqual(len(summary["val"]["categories"]), 3)

    def test_one_workbook_is_left_for_training(self):
        samples = [
            {"sample_id": "s0", "workbook_id": "w0", "task_categories": ["sort"]},
            {"sample_id": "s1", "workbook_id": "w1", "task_categories": ["filter"]},
            {"sample_id": "s2", "workbook_id": "w2", "task_categories": ["chart"]},
        ]
        splits = stratified_split(samples, seed=0, min_eval_workbooks=2)

        self.assertGreaterEqual(len({item["workbook_id"] for item in splits["train"]}), 1)

    def test_coverage_summary_reports_counts(self):
        samples = [
            {"sample_id": "s0", "workbook_id": "w0", "task_categories": ["chart"]},
            {"sample_id": "s1", "workbook_id": "w0", "task_categories": ["chart", "sort"]},
        ]
        summary = coverage_summary({"train": samples, "val": [], "test": []})

        self.assertEqual(summary["train"]["samples"], 2)
        self.assertEqual(summary["train"]["workbooks"], 1)
        self.assertEqual(summary["train"]["charts"], 2)
        self.assertEqual(summary["train"]["categories"], ["chart", "sort"])


def read_csv_rows(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()[1:]


if __name__ == "__main__":
    unittest.main()
