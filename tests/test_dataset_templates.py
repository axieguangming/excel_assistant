import unittest

from dataset_builder.chain import normalize_chain
from dataset_builder.tables import DOMAINS, generate_tables
from dataset_builder.templates import (
    TemplateChainProposer,
    TemplateQuestionWriter,
    categories_for,
    summarize_steps,
)


class TemplateProposerTests(unittest.TestCase):
    def test_every_generated_chain_passes_schema_validation(self):
        for table in generate_tables(len(DOMAINS), seed=21):
            chains = TemplateChainProposer().propose(table, 8, max_steps=4)
            self.assertTrue(chains)
            for chain in chains:
                with self.subTest(table=table.table_id):
                    steps = normalize_chain(chain, max_steps=4)
                    self.assertTrue(steps)

    def test_candidates_cover_distinct_categories(self):
        table = generate_tables(1, seed=21, domains=["sales_orders"])[0]
        chains = TemplateChainProposer().propose(table, 20, max_steps=4)
        signatures = {tuple(categories_for(chain)) for chain in chains}

        self.assertIn(("chart",), signatures)
        self.assertIn(("sort", "top_n"), signatures)
        self.assertIn(("aggregate",), signatures)

    def test_respects_allowed_tools(self):
        table = generate_tables(1, seed=21, domains=["sales_orders"])[0]
        chains = TemplateChainProposer().propose(
            table, 5, max_steps=4, allowed_tools=("describe_data",)
        )

        self.assertTrue(chains)
        for chain in chains:
            self.assertEqual([step["tool"] for step in chain], ["describe_data"])

    def test_max_steps_limits_chain_length(self):
        table = generate_tables(1, seed=4, domains=["sales_orders"])[0]
        chains = TemplateChainProposer().propose(table, 8, max_steps=1)

        self.assertTrue(all(len(chain) == 1 for chain in chains))


class TemplateQuestionWriterTests(unittest.TestCase):
    def test_question_and_answer_include_step_summary(self):
        table = generate_tables(1, seed=21, domains=["sales_orders"])[0]
        steps = [
            {
                "tool": "top_n_rows",
                "args": {"sort_by": [{"column": "销售额", "ascending": False}], "limit": 10},
            }
        ]

        written = TemplateQuestionWriter().write(
            table, steps, rows=10, columns=["销售额"]
        )

        self.assertIn("取前 10 条", written["question"])
        self.assertNotIn("按降序 按", written["question"])
        self.assertIn("10 行 × 1 列", written["answer"])

    def test_summary_covers_multi_step_chain(self):
        steps = [
            {"tool": "filter_rows", "args": {"filters": [{"column": "区域", "operator": "eq", "value": "华东"}]}},
            {
                "tool": "aggregate_data",
                "args": {"group_by": ["区域"], "metrics": [{"column": "销售额", "function": "sum"}]},
            },
            {"tool": "sort_rows", "args": {"sort_by": [{"column": "销售额_sum", "ascending": False}]}},
        ]

        summary = summarize_steps(steps)

        self.assertIn("筛选出 区域 等于 华东", summary)
        self.assertIn("按 区域 分组统计 销售额 的合计", summary)
        self.assertIn("销售额_sum（降序）", summary)


if __name__ == "__main__":
    unittest.main()
