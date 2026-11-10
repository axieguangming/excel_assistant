"""轨迹抓取脚本的目标筛选逻辑测试。

只测纯函数 `failed_executions`：它决定「哪些失败样本值得重跑抓轨迹」。
第一版只抓重复调用类；DPO 之后重复降到 35 条、参数写错的还有 15 条，
因此默认改为收集全部执行类失败。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))


def load_module():
    spec = importlib.util.spec_from_file_location(
        "capture_traces_under_test",
        ROOT / "pipeline" / "capture_traces.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_results(path: Path, results: list[dict]) -> None:
    path.write_text(
        json.dumps({"results": results}, ensure_ascii=False), encoding="utf-8"
    )


class FailedExecutionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _build(self) -> Path:
        path = self.dir / "exp-a-dpo.json"
        write_results(
            path,
            [
                # 通过 → 不要
                {"sample_id": "ok-1", "kind": "execute", "passed": True,
                 "actual_tools": ["aggregate_data"]},
                # 失败且重复调用 → 要
                {"sample_id": "dup-1", "kind": "execute", "passed": False,
                 "actual_tools": ["aggregate_data"] * 4},
                # 失败但不重复 → 默认要，only_duplicates 时不要
                {"sample_id": "args-1", "kind": "execute", "passed": False,
                 "actual_tools": ["aggregate_data", "sort_rows"]},
                # 澄清类失败 → 不要（本脚本只抓执行类）
                {"sample_id": "clar-1", "kind": "clarify", "passed": False,
                 "actual_tools": []},
            ],
        )
        return path

    def test_default_collects_all_execute_failures(self):
        picked = self.module.failed_executions([self._build()])
        self.assertEqual([sid for _, sid in picked], ["dup-1", "args-1"])

    def test_only_duplicates_filters_out_single_call_failures(self):
        picked = self.module.failed_executions(
            [self._build()], only_duplicates=True
        )
        self.assertEqual([sid for _, sid in picked], ["dup-1"])

    def test_group_is_parsed_from_filename(self):
        picked = self.module.failed_executions([self._build()])
        self.assertTrue(all(group == "a" for group, _ in picked))

    def test_b_group_file(self):
        path = self.dir / "exp-b-dpo.json"
        write_results(
            path,
            [
                {"sample_id": "b-1", "kind": "execute", "passed": False,
                 "actual_tools": ["filter_rows", "filter_rows"]},
            ],
        )
        self.assertEqual(
            self.module.failed_executions([path]), [("b", "b-1")]
        )

    def test_missing_actual_tools_is_kept_as_failure(self):
        """一次工具都没调用的失败样本没有 actual_tools 字段，也要抓到。"""
        path = self.dir / "exp-a-dpo.json"
        write_results(
            path,
            [{"sample_id": "none-1", "kind": "execute", "passed": False}],
        )
        self.assertEqual(self.module.failed_executions([path]), [("a", "none-1")])


if __name__ == "__main__":
    unittest.main()
