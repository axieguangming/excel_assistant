import tempfile
import unittest
from datetime import date
from pathlib import Path

from dataset_builder.workspace import (
    dated_run_dir,
    is_dataset_dir,
    latest_dataset_dir,
    list_dataset_dirs,
    resolve_dataset_dir,
)


def make_dataset(path: Path, marker: str = "samples.jsonl") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / marker).write_text("{}\n", encoding="utf-8")
    return path


class DatedRunDirTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_first_run_uses_today(self):
        target = dated_run_dir(self.root, day=date(2026, 10, 26))

        self.assertEqual(target, self.root / "2026-10-26")
        self.assertFalse(target.exists())

    def test_second_run_same_day_gets_a_suffix(self):
        make_dataset(dated_run_dir(self.root, day=date(2026, 10, 26)))

        target = dated_run_dir(self.root, day=date(2026, 10, 26))

        self.assertEqual(target.name, "2026-10-26-2")

    def test_suffixes_increment_without_overwriting(self):
        day = date(2026, 10, 26)
        make_dataset(dated_run_dir(self.root, day=day))
        make_dataset(dated_run_dir(self.root, day=day))

        self.assertEqual(dated_run_dir(self.root, day=day).name, "2026-10-26-3")

    def test_empty_directory_is_reused(self):
        (self.root / "2026-10-26").mkdir()

        target = dated_run_dir(self.root, day=date(2026, 10, 26))

        self.assertEqual(target.name, "2026-10-26")

    def test_different_days_do_not_collide(self):
        make_dataset(dated_run_dir(self.root, day=date(2026, 10, 25)))

        target = dated_run_dir(self.root, day=date(2026, 10, 26))

        self.assertEqual(target.name, "2026-10-26")

    def test_base_directory_is_created_on_demand(self):
        nested = self.root / "a" / "b"

        target = dated_run_dir(nested, day=date(2026, 10, 26))

        self.assertEqual(target.parent.name, "b")
        self.assertFalse(target.exists())


class ListingTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_only_dated_directories_are_listed(self):
        make_dataset(self.root / "2026-10-25")
        make_dataset(self.root / "2026-10-26")
        (self.root / "notes").mkdir()
        (self.root / "random.txt").write_text("x", encoding="utf-8")

        names = [path.name for path in list_dataset_dirs(self.root)]

        self.assertEqual(names, ["2026-10-25", "2026-10-26"])

    def test_suffix_ordering_is_numeric(self):
        for name in ("2026-10-26", "2026-10-26-2", "2026-10-26-10"):
            make_dataset(self.root / name)

        names = [path.name for path in list_dataset_dirs(self.root)]

        self.assertEqual(names, ["2026-10-26", "2026-10-26-2", "2026-10-26-10"])

    def test_latest_skips_directories_without_samples(self):
        make_dataset(self.root / "2026-10-25")
        (self.root / "2026-10-27").mkdir()

        latest = latest_dataset_dir(self.root)

        assert latest is not None
        self.assertEqual(latest.name, "2026-10-25")

    def test_latest_returns_none_when_empty(self):
        self.assertIsNone(latest_dataset_dir(self.root))

    def test_legacy_flat_layout_is_still_recognised(self):
        make_dataset(self.root)

        self.assertTrue(is_dataset_dir(self.root))
        self.assertEqual(latest_dataset_dir(self.root), self.root)

    def test_non_directory_base_is_handled(self):
        self.assertEqual(list_dataset_dirs(self.root / "missing"), [])


class ResolveDatasetDirTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_explicit_dataset_directory_is_returned_as_is(self):
        explicit = make_dataset(self.root / "2026-10-25")
        make_dataset(self.root / "2026-10-26")

        self.assertEqual(resolve_dataset_dir(explicit), explicit)

    def test_base_directory_resolves_to_latest(self):
        make_dataset(self.root / "2026-10-25")
        make_dataset(self.root / "2026-10-26")

        self.assertEqual(resolve_dataset_dir(self.root).name, "2026-10-26")

    def test_missing_dataset_raises_with_guidance(self):
        with self.assertRaisesRegex(FileNotFoundError, "没有找到数据集"):
            resolve_dataset_dir(self.root)

    def test_dated_run_dir_allocates_next_suffix_when_today_taken(self):
        existing = dated_run_dir(self.root)
        make_dataset(existing)

        target = dated_run_dir(self.root)

        self.assertEqual(target.name, f"{existing.name}-2")


if __name__ == "__main__":
    unittest.main()
