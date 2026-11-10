"""恢复轨迹构造脚本的单元测试。

关注三件事：
1. 生成的错误列名确实不在表里（否则样本无效）
2. 报错观察含「可用字段」清单（模型才能恢复）
3. 轨迹结构是「错误调用 → 报错 → 修正调用 → 成功」
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module():
    spec = importlib.util.spec_from_file_location(
        "make_recovery_under_test", ROOT / "pipeline" / "make_recovery_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WrongNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_generated_name_is_absent_from_columns(self):
        existing = {"满意度", "城市", "年龄段"}
        rng = random.Random(0)
        for _ in range(30):
            name = self.module.make_wrong_name("满意度", existing, rng)
            if name is None:
                continue
            self.assertNotIn(name, existing)

    def test_generated_name_looks_like_derived_metric(self):
        """错误名要像模型真实会犯的错（X合计 / 平均X / 简化名）。"""
        existing = {"满意度", "城市"}
        rng = random.Random(1)
        names = {
            self.module.make_wrong_name("满意度", existing, rng) for _ in range(20)
        }
        names.discard(None)
        joined = "".join(names)
        self.assertTrue(
            any(token in joined for token in ("合计", "总和", "总计", "平均", "总", "全部", "_")),
            f"错误名不像是派生指标：{names}",
        )

    def test_returns_none_when_no_variant_available(self):
        # 列名已经带上所有可能后缀时，仍应能生成或明确返回 None，不能撞已有的列
        existing = {"x"}
        rng = random.Random(2)
        name = self.module.make_wrong_name("x", existing, rng)
        self.assertTrue(name is None or name not in existing)


class ColumnRefTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_extracts_all_argument_shapes(self):
        steps = [
            {"tool": "filter_rows", "args": {"filters": [{"column": "城市"}]}},
            {
                "tool": "aggregate_data",
                "args": {
                    "group_by": ["部门"],
                    "metrics": [{"column": "金额", "function": "sum", "name": "合计"}],
                },
            },
            {"tool": "sort_rows", "args": {"sort_by": [{"column": "合计"}]}},
            {"tool": "deduplicate_rows", "args": {"subset": ["编号"]}},
            {"tool": "create_chart", "args": {"chart_x": "部门", "chart_y": "合计"}},
        ]
        found = {name for _, _, name in self.module.iter_column_refs(steps)}
        self.assertEqual(found, {"城市", "部门", "金额", "合计", "编号"})

    def test_set_by_path_rewrites_nested_value(self):
        steps = [
            {
                "tool": "aggregate_data",
                "args": {"metrics": [{"column": "金额", "function": "sum"}]},
            }
        ]
        self.module.set_by_path(steps, 0, ("metrics", 0, "column"), "金额合计")
        self.assertEqual(steps[0]["args"]["metrics"][0]["column"], "金额合计")


class RecoveryRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def _table(self):
        class Table:
            table_id = "wb-001"
            frame = pd.DataFrame({"部门": ["A", "B", "A"], "金额": [1.0, 2.0, 3.0]})

        return Table()

    def _steps(self):
        return [
            {
                "tool": "aggregate_data",
                "args": {
                    "group_by": ["部门"],
                    "metrics": [{"column": "金额", "function": "sum", "name": "金额合计"}],
                },
            }
        ]

    def test_record_has_error_then_fix(self):
        record = self.module.build_record(
            "wb-001-recovery-001",
            self._table(),
            self._steps(),
            "按部门汇总金额",
            "已完成汇总。",
            error_index=0,
            error_path=("metrics", 0, "column"),
            wrong_name="金额合计",
            correct_name="金额",
        )
        self.assertIsNotNone(record)
        calls = [
            (m["tool_calls"][0]["function"]["name"],
             json.loads(m["tool_calls"][0]["function"]["arguments"]))
            for m in record["messages"]
            if m.get("tool_calls")
        ]
        # 结构：错名调用 → describe_data 查看字段 → 正确调用
        self.assertEqual(calls[0][0], "aggregate_data")
        self.assertEqual(calls[0][1]["metrics"][0]["column"], "金额合计")
        self.assertEqual(calls[1][0], "describe_data")
        self.assertEqual(calls[2][0], "aggregate_data")
        self.assertEqual(calls[2][1]["metrics"][0]["column"], "金额")

    def test_no_consecutive_duplicate_tool_calls(self):
        """报错与重试之间必须隔一步查看，否则模型会学成「连调同一工具」。

        v4 用 400 条「错一次→对一次」的样本，导致 44 条评测失败是
        「重复调用同一工具」（最多连调 9 次触顶）。
        """
        record = self.module.build_record(
            "wb-001-recovery-004",
            self._table(),
            self._steps(),
            "按部门汇总金额",
            "已完成汇总。",
            error_index=0,
            error_path=("metrics", 0, "column"),
            wrong_name="金额合计",
            correct_name="金额",
        )
        self.assertFalse(self.module.has_consecutive_duplicate(record))

    def test_consecutive_duplicate_detector(self):
        """自检函数本身要能识别出连续重复。"""
        bad = {
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "x"}}]},
                {"role": "assistant", "tool_calls": [{"function": {"name": "x"}}]},
            ]
        }
        good = {
            "messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "x"}}]},
                {"role": "assistant", "tool_calls": [{"function": {"name": "y"}}]},
            ]
        }
        self.assertTrue(self.module.has_consecutive_duplicate(bad))
        self.assertFalse(self.module.has_consecutive_duplicate(good))

    def test_describe_observation_says_read_only(self):
        """插入的查看步骤必须是只读措辞，与线上工具一致。"""
        record = self.module.build_record(
            "wb-001-recovery-005",
            self._table(),
            self._steps(),
            "按部门汇总金额",
            "已完成汇总。",
            error_index=0,
            error_path=("metrics", 0, "column"),
            wrong_name="金额合计",
            correct_name="金额",
        )
        describe_texts = [
            m["content"]
            for m in record["messages"]
            if m["role"] == "tool" and m.get("name") == "describe_data"
        ]
        self.assertTrue(describe_texts)
        self.assertTrue(describe_texts[0].startswith("查看成功（只读"))

    def test_error_message_includes_available_columns(self):
        record = self.module.build_record(
            "wb-001-recovery-002",
            self._table(),
            self._steps(),
            "按部门汇总金额",
            "已完成汇总。",
            error_index=0,
            error_path=("metrics", 0, "column"),
            wrong_name="金额合计",
            correct_name="金额",
        )
        tool_texts = [m["content"] for m in record["messages"] if m["role"] == "tool"]
        self.assertTrue(any("可用字段" in text for text in tool_texts))
        self.assertTrue(any(text.startswith("工具执行失败：") for text in tool_texts))

    def test_returns_none_when_wrong_name_actually_works(self):
        """错误列名若真的存在，样本必须丢弃，不能产出错的训练信号。"""
        record = self.module.build_record(
            "wb-001-recovery-003",
            self._table(),
            self._steps(),
            "按部门汇总金额",
            "已完成汇总。",
            error_index=0,
            error_path=("metrics", 0, "column"),
            wrong_name="金额",  # 本来就是对的名字
            correct_name="金额",
        )
        self.assertIsNone(record)


if __name__ == "__main__":
    unittest.main()
