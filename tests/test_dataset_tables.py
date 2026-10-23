import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dataset_builder.tables import (
    DOMAINS,
    generate_table,
    generate_tables,
    load_tables,
    save_tables,
)


class SyntheticTableTests(unittest.TestCase):
    def test_generation_is_deterministic_for_a_seed(self):
        first = generate_table("sales_orders", 7)
        second = generate_table("sales_orders", 7)

        self.assertEqual(first.table_id, second.table_id)
        pd.testing.assert_frame_equal(first.frame, second.frame)

    def test_generation_varies_across_seeds(self):
        first = generate_table("sales_orders", 1)
        second = generate_table("sales_orders", 2)

        self.assertNotEqual(first.table_id, second.table_id)
        self.assertFalse(first.frame.equals(second.frame))

    def test_every_domain_is_supported(self):
        tables = generate_tables(len(DOMAINS))

        self.assertEqual({table.domain for table in tables}, set(DOMAINS))
        for table in tables:
            self.assertGreater(len(table.frame), 0)
            self.assertTrue(table.sheet)

    def test_profile_reports_column_kinds_and_samples(self):
        table = generate_table("sales_orders", 11)
        profile = table.profile()

        self.assertEqual(profile["table_id"], table.table_id)
        self.assertEqual(profile["rows"], len(table.frame))
        kinds = {column["name"]: column["kind"] for column in profile["columns"]}
        self.assertEqual(kinds["日期"], "datetime")
        self.assertEqual(kinds["数量"], "numeric")
        self.assertEqual(kinds["区域"], "text")
        self.assertTrue(all(column["samples"] for column in profile["columns"] if column["non_null"]))

    def test_unknown_domain_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知的仿真场景"):
            generate_table("payroll", 1)

    def test_defects_appear_across_a_batch(self):
        tables = generate_tables(20, seed=1)
        has_missing = any(table.frame.isna().any().any() for table in tables)
        has_duplicates = any(table.frame.duplicated().any() for table in tables)

        self.assertTrue(has_missing)
        self.assertTrue(has_duplicates)

    def test_save_and_load_round_trip_keeps_datetime_and_text(self):
        tables = generate_tables(3, seed=5)
        with tempfile.TemporaryDirectory() as directory:
            save_tables(tables, directory)
            restored = load_tables(directory)

        self.assertEqual([table.table_id for table in restored], [table.table_id for table in tables])
        for original, loaded in zip(tables, restored):
            self.assertEqual(list(original.frame.columns), list(loaded.frame.columns))
            for column in original.frame.columns:
                if pd.api.types.is_datetime64_any_dtype(original.frame[column]):
                    self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded.frame[column]))
            self.assertEqual(len(original.frame), len(loaded.frame))

    def test_load_requires_an_index(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_tables(Path(directory))


if __name__ == "__main__":
    unittest.main()
