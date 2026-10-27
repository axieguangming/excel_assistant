import json
import unittest

from dataset_builder.chain import canonical_chain, normalize_chain
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
        self.assertIn(("filter", "top_n"), signatures)
        self.assertIn(("aggregate",), signatures)
        self.assertIn(("aggregate", "chart"), signatures)

    def test_candidate_pool_is_large_enough_for_bulk_generation(self):
        """规则生成要能支撑上千条样本，候选链不能只有十几种。"""
        table = generate_tables(1, seed=20261026, domains=["sales_orders"])[0]

        chains = TemplateChainProposer().propose(table, 1000, max_steps=4)

        self.assertGreater(len(chains), 200)

    def test_candidates_include_multi_step_chains(self):
        table = generate_tables(1, seed=20261026, domains=["sales_orders"])[0]

        chains = TemplateChainProposer().propose(table, 1000, max_steps=4)
        multi = [chain for chain in chains if len(chain) >= 2]

        self.assertGreater(len(multi), 50)

    def test_candidates_vary_parameters_not_just_tool_names(self):
        table = generate_tables(1, seed=20261026, domains=["sales_orders"])[0]

        chains = TemplateChainProposer().propose(table, 1000, max_steps=4)
        top_n_limits = {
            step["args"]["limit"]
            for chain in chains
            for step in chain
            if step["tool"] == "top_n_rows"
        }
        filter_values = {
            condition["value"]
            for chain in chains
            for step in chain
            for condition in step.get("args", {}).get("filters", [])
            if "value" in condition
        }

        self.assertGreater(len(top_n_limits), 1)
        self.assertGreater(len(filter_values), 1)

    def test_candidates_have_no_exact_duplicates(self):
        table = generate_tables(1, seed=20261026, domains=["budget"])[0]

        chains = TemplateChainProposer().propose(table, 1000, max_steps=4)
        keys = [canonical_chain(chain) for chain in chains]

        self.assertEqual(len(keys), len(set(keys)))

    def test_respects_allowed_tools(self):
        table = generate_tables(1, seed=21, domains=["sales_orders"])[0]
        chains = TemplateChainProposer().propose(
            table, 5, max_steps=4, allowed_tools=("describe_data",)
        )

        self.assertTrue(chains)
        for chain in chains:
            self.assertEqual([step["tool"] for step in chain], ["describe_data"])

    def test_chart_candidates_include_aggregate_before_plotting(self):
        """绘图任务要覆盖"先分组汇总再画图"，因为原始行粒度常与图表语义不符。"""
        table = generate_tables(1, seed=20261026, domains=["sales_orders"])[0]

        chains = TemplateChainProposer().propose(table, 2000, max_steps=4)
        chart_chains = [
            chain for chain in chains
            if any(step["tool"] == "create_chart" for step in chain)
        ]
        aggregated = [
            chain for chain in chart_chains
            if any(step["tool"] == "aggregate_data" for step in chain)
        ]

        self.assertGreater(len(chart_chains), 50)
        self.assertGreater(len(aggregated), len(chart_chains) // 2)

    def test_chart_candidates_still_include_single_step(self):
        table = generate_tables(1, seed=20261026, domains=["sales_orders"])[0]

        chains = TemplateChainProposer().propose(table, 2000, max_steps=4)
        single_chart = [
            chain for chain in chains
            if len(chain) == 1 and chain[0]["tool"] == "create_chart"
        ]

        self.assertGreater(len(single_chart), 10)

    def test_aggregate_chart_uses_aggregated_column_name(self):
        table = generate_tables(1, seed=20261026, domains=["sales_orders"])[0]

        chains = TemplateChainProposer().propose(table, 2000, max_steps=4)
        pairs = [
            (chain[0]["args"], chain[-1]["args"])
            for chain in chains
            if len(chain) == 2
            and chain[0]["tool"] == "aggregate_data"
            and chain[1]["tool"] == "create_chart"
        ]

        self.assertTrue(pairs)
        for aggregate_args, chart_args in pairs:
            produced = {
                metric.get("name")
                or f"{metric['column']}_{metric['function']}"
                for metric in aggregate_args["metrics"]
            }
            self.assertIn(chart_args["chart_y"], produced)

    def test_chart_candidates_cover_multiple_chart_types(self):
        table = generate_tables(1, seed=20261026, domains=["sales_orders"])[0]

        chains = TemplateChainProposer().propose(table, 2000, max_steps=4)
        types = {
            step["args"]["chart_type"]
            for chain in chains
            for step in chain
            if step["tool"] == "create_chart"
        }

        self.assertIn("bar", types)
        self.assertIn("pie", types)
        self.assertIn("line", types)

    def test_chart_after_sort_is_generated(self):
        table = generate_tables(1, seed=20261026, domains=["sales_orders"])[0]

        chains = TemplateChainProposer().propose(table, 2000, max_steps=4)
        sorted_charts = [
            chain for chain in chains
            if [step["tool"] for step in chain]
            == ["aggregate_data", "sort_rows", "create_chart"]
        ]

        self.assertTrue(sorted_charts)

    def test_all_empty_columns_are_excluded_from_candidates(self):
        """全空列在 pandas 里会被推断成 float64，不能当作数值列参与任务生成。"""
        import pandas as pd

        from dataset_builder.tables import SyntheticTable
        from dataset_builder.templates import _columns_by_kind, _numeric_columns

        frame = pd.DataFrame(
            {
                "区域": ["东", "西", "南"],
                "销售额": [100.0, 200.0, 300.0],
                "备注": [None, None, None],
            }
        )
        table = SyntheticTable(
            table_id="t-empty", domain="sales_orders", sheet="表", frame=frame
        )

        self.assertEqual(_numeric_columns(table), ["销售额"])
        self.assertNotIn("备注", _columns_by_kind(table, "text"))
        self.assertNotIn("备注", _columns_by_kind(table, "numeric"))

    def test_candidates_never_reference_all_empty_columns(self):
        import pandas as pd

        from dataset_builder.tables import SyntheticTable

        frame = pd.DataFrame(
            {
                "区域": ["东", "西", "南", "北"],
                "销售额": [100.0, 200.0, 300.0, 400.0],
                "数量": [1, 2, 3, 4],
                "备注": [None, None, None, None],
            }
        )
        table = SyntheticTable(
            table_id="t-empty2", domain="sales_orders", sheet="表", frame=frame
        )

        chains = TemplateChainProposer().propose(table, 500, max_steps=4)
        blob = json.dumps(chains, ensure_ascii=False)

        self.assertNotIn('"备注"', blob)

    def test_single_step_chart_skips_high_duplication_grouping(self):
        """明细表按类别直接绘图没有意义，应交给 aggregate→chart 覆盖。"""
        import pandas as pd

        from dataset_builder.tables import SyntheticTable

        frame = pd.DataFrame(
            {
                "销售员": ["甲", "乙", "丙", "丁"] * 25,
                "销售额": [100.0 + i for i in range(100)],
            }
        )
        table = SyntheticTable(
            table_id="t-dup", domain="sales_orders", sheet="表", frame=frame
        )

        chains = TemplateChainProposer().propose(table, 500, max_steps=4)
        single_with_x = [
            chain for chain in chains
            if len(chain) == 1
            and chain[0]["tool"] == "create_chart"
            and chain[0]["args"].get("chart_x") == "销售员"
        ]
        aggregated_chart = [
            chain for chain in chains
            if [step["tool"] for step in chain] == ["aggregate_data", "create_chart"]
            and chain[-1]["args"].get("chart_x") == "销售员"
        ]

        self.assertEqual(single_with_x, [])
        self.assertTrue(aggregated_chart)

    def test_single_step_chart_allowed_when_rows_match_categories(self):
        """行数与类别数接近时（每行就是一个类别），单步绘图是合理的。"""
        import pandas as pd

        from dataset_builder.tables import SyntheticTable

        frame = pd.DataFrame(
            {
                "区域": ["东", "西", "南", "北"],
                "销售额": [100.0, 200.0, 300.0, 400.0],
            }
        )
        table = SyntheticTable(
            table_id="t-small", domain="sales_orders", sheet="表", frame=frame
        )

        chains = TemplateChainProposer().propose(table, 500, max_steps=4)
        single_with_x = [
            chain for chain in chains
            if len(chain) == 1
            and chain[0]["tool"] == "create_chart"
            and chain[0]["args"].get("chart_x") == "区域"
        ]

        self.assertTrue(single_with_x)

    def test_single_step_chart_excludes_identifier_columns(self):
        """订单号这类每行都不同的标识符不适合做分类轴。"""
        import pandas as pd

        from dataset_builder.tables import SyntheticTable

        rows = 40
        frame = pd.DataFrame(
            {
                "订单号": [f"SO{1000 + i}" for i in range(rows)],
                "区域": ["东", "西"] * (rows // 2),
                "销售额": [100.0 + i for i in range(rows)],
            }
        )
        table = SyntheticTable(
            table_id="t-ids", domain="sales_orders", sheet="表", frame=frame
        )

        chains = TemplateChainProposer().propose(table, 500, max_steps=4)
        id_axis = [
            chain for chain in chains
            if len(chain) == 1
            and chain[0]["tool"] == "create_chart"
            and chain[0]["args"].get("chart_x") == "订单号"
        ]

        self.assertEqual(id_axis, [])

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

    def test_variants_produce_different_wordings(self):
        table = generate_tables(1, seed=21, domains=["sales_orders"])[0]
        steps = [
            {"tool": "describe_data", "args": {}},
        ]
        writer = TemplateQuestionWriter(variants=3, seed=0)

        variants = writer.write_variants(table, steps, rows=10, columns=["a"])

        questions = [item["question"] for item in variants]
        self.assertGreaterEqual(len(questions), 2)
        self.assertEqual(len(set(questions)), len(questions))
        # 同一操作链的变体共享同一份预期答复。
        self.assertEqual(len({item["answer"] for item in variants}), 1)

    def test_default_writer_returns_single_question(self):
        table = generate_tables(1, seed=21, domains=["sales_orders"])[0]

        written = TemplateQuestionWriter().write(table, [{"tool": "describe_data", "args": {}}])

        self.assertIn("question", written)
        self.assertIn("answer", written)

    def test_question_templates_are_all_used_across_tables(self):
        tables = generate_tables(5, seed=21)
        steps = [{"tool": "describe_data", "args": {}}]
        writer = TemplateQuestionWriter(variants=3, seed=0)

        questions = {
            item["question"]
            for table in tables
            for item in writer.write_variants(table, steps)
        }

        self.assertGreaterEqual(len(questions), len(tables))

    def test_no_double_punctuation_in_questions(self):
        table = generate_tables(1, seed=21, domains=["sales_orders"])[0]
        steps = [{"tool": "describe_data", "args": {}}]

        for index in range(7):
            question = TemplateQuestionWriter(variants=7, seed=index).write(
                table, steps
            )["question"]
            self.assertNotIn("。。", question)
            self.assertNotIn("，。", question)


if __name__ == "__main__":
    unittest.main()
