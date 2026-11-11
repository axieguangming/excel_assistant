"""训练数据必须基于「写盘后回读」的表格来构造。

为什么这条很重要
----------------
评测时表格是从磁盘读的（`load_tables`），所以推理侧生成的观察文本用的是
回读后的 dtype。而 Excel 往返会改变 dtype：

    内存  整数列 Int64    → 观察文本里是 "1"
    回读  整数列 float64  → 观察文本里是 "1.0"
    内存  缺失值 ""       → 观察文本里是 ""
    回读  缺失值 NaN      → 观察文本里是 "NaN"

如果训练数据用 `generate_tables()` 返回的**内存表**构造，模型学到的是
"1" 和 ""，线上看到的却是 "1.0" 和 "NaN"——典型的训练/推理错配。

本项目实测：394 个首步观察里只有 129 个文本一致（33%）。当前它还没造成
通过率损失（判定用数值等价比较），但一旦观察文本引入格式化，代价立刻显现。

`dataset_builder/cli.py` 一直是正确做法（写盘后回读）；
`pipeline/build_holdout_eval.py` 曾经漏了这一步，这些测试锁住它。
"""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

from excel_ops import observation_for_step  # noqa: E402
from dataset_builder.tables import generate_tables, load_tables, save_tables  # noqa: E402


class PersistedTablesTests(unittest.TestCase):
    def test_memory_and_reloaded_tables_differ(self):
        """先证明这个缺陷是真实存在的，否则下面的测试没有意义。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = generate_tables(2, seed=11, domains=("survey",))
            save_tables(memory, root, table_format="xlsx")
            reloaded = {table.table_id: table for table in load_tables(root)}

            differences = 0
            for table in memory:
                fresh = reloaded[table.table_id]
                if observation_for_step(
                    "filter_rows", table.frame, arguments={}
                ) != observation_for_step("filter_rows", fresh.frame, arguments={}):
                    differences += 1

            self.assertGreater(
                differences,
                0,
                "预期内存表与回读表的观察文本不同；若相同，说明该缺陷已不存在，"
                "可以考虑放宽这条约束。",
            )

    def test_generate_persisted_tables_matches_disk(self):
        """build_holdout_eval 的入口必须返回回读后的表。"""
        from build_holdout_eval import generate_persisted_tables

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tables-a"
            tables = generate_persisted_tables(
                root, count=2, seed=11, domains=("survey",)
            )
            fresh = {table.table_id: table for table in load_tables(root)}

            self.assertEqual(len(tables), len(fresh))
            for table in tables:
                self.assertEqual(table.table_id, table.table_id)
                disk = fresh[table.table_id]
                # dtype 必须与磁盘一致
                self.assertEqual(
                    [str(dtype) for dtype in table.frame.dtypes],
                    [str(dtype) for dtype in disk.frame.dtypes],
                    f"{table.table_id} 的 dtype 与磁盘不一致",
                )
                # 观察文本也必须一致——这才是真正影响模型的东西
                self.assertEqual(
                    observation_for_step("filter_rows", table.frame, arguments={}),
                    observation_for_step("filter_rows", disk.frame, arguments={}),
                    f"{table.table_id} 的观察文本与磁盘不一致",
                )


class ObservationDriftTests(unittest.TestCase):
    def test_observation_text_depends_on_dtype(self):
        """把「dtype 会改变观察文本」这件事写成断言，防止有人悄悄改回去。"""
        import pandas as pd

        frame = pd.DataFrame(
            {"名称": ["a", "b"], "数量": pd.array([1, 2], dtype="Int64")}
        )
        as_float = frame.copy()
        as_float["数量"] = as_float["数量"].astype("float64")

        left = observation_for_step("filter_rows", frame, arguments={})
        right = observation_for_step("filter_rows", as_float, arguments={})

        self.assertIn("1", left)
        self.assertIn("1.0", right)
        self.assertNotEqual(left, right)


if __name__ == "__main__":
    unittest.main()
