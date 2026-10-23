import json
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

    def test_excel_is_the_default_format(self):
        with tempfile.TemporaryDirectory() as directory:
            save_tables(generate_tables(2, seed=3), directory)
            files = sorted(path.name for path in Path(directory).iterdir())
            payload = json.loads((Path(directory) / "index.json").read_text(encoding="utf-8"))

        self.assertTrue(any(name.endswith(".xlsx") for name in files))
        self.assertFalse(any(name.endswith(".csv") for name in files))
        self.assertTrue(all(entry["format"] == "xlsx" for entry in payload["tables"]))
        self.assertTrue(all(entry["path"].endswith(".xlsx") for entry in payload["tables"]))

    def test_excel_round_trip_preserves_values(self):
        tables = generate_tables(2, seed=9)
        with tempfile.TemporaryDirectory() as directory:
            save_tables(tables, directory)
            restored = load_tables(directory)

        for original, loaded in zip(tables, restored):
            self.assertEqual(loaded.frame.shape, original.frame.shape)
            for column in original.frame.columns:
                before = original.frame[column]
                after = loaded.frame[column]
                self.assertEqual(
                    int(after.notna().sum()),
                    int(before.notna().sum()),
                    f"{column} 的非空数量在读回后发生变化",
                )

    def test_excel_sheet_name_matches_the_written_sheet(self):
        table = generate_tables(1, seed=4, domains=["inventory"])[0]
        with tempfile.TemporaryDirectory() as directory:
            save_tables([table], directory)
            path = next(Path(directory).glob("*.xlsx"))
            with pd.ExcelFile(path, engine="openpyxl") as book:
                sheets = book.sheet_names
            payload = json.loads((Path(directory) / "index.json").read_text(encoding="utf-8"))

        self.assertEqual(sheets, [payload["tables"][0]["sheet_name"]])
        self.assertEqual(sheets, [table.sheet])

    def test_long_or_illegal_sheet_names_are_sanitized(self):
        from dataset_builder.tables import _excel_sheet_name

        self.assertEqual(_excel_sheet_name("a/b:c*d?e[f]g"), "a_b_c_d_e_f_g")
        self.assertEqual(len(_excel_sheet_name("表" * 40)), 31)
        self.assertEqual(_excel_sheet_name("   "), "工作表")

    def test_csv_format_is_still_supported(self):
        tables = generate_tables(2, seed=6)
        with tempfile.TemporaryDirectory() as directory:
            save_tables(tables, directory, table_format="csv")
            payload = json.loads((Path(directory) / "index.json").read_text(encoding="utf-8"))
            restored = load_tables(directory)

        self.assertTrue(all(entry["format"] == "csv" for entry in payload["tables"]))
        self.assertEqual([table.table_id for table in restored], [table.table_id for table in tables])
        for original, loaded in zip(tables, restored):
            for column in original.frame.columns:
                if pd.api.types.is_datetime64_any_dtype(original.frame[column]):
                    self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded.frame[column]))

    def test_legacy_index_without_format_is_read_as_csv(self):
        tables = generate_tables(1, seed=12)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tables[0].frame.to_csv(root / "legacy.csv", index=False, encoding="utf-8-sig")
            (root / "index.json").write_text(
                json.dumps(
                    {
                        "tables": [
                            {
                                "table_id": tables[0].table_id,
                                "domain": tables[0].domain,
                                "sheet": tables[0].sheet,
                                "path": "legacy.csv",
                                "rows": len(tables[0].frame),
                                "source": {"type": "synthetic"},
                            }
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            restored = load_tables(root)

        self.assertEqual(len(restored), 1)
        self.assertEqual(len(restored[0].frame), len(tables[0].frame))

    def test_unsupported_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "不支持的表格格式"):
                save_tables(generate_tables(1, seed=1), directory, table_format="parquet")

    def test_load_requires_an_index(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_tables(Path(directory))


if __name__ == "__main__":
    unittest.main()
