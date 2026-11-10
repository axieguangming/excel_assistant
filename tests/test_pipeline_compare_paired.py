"""配对比较（McNemar 精确检验）的单元测试。

核心断言：
1. 只统计两个版本都有的样本（避免按错 id 对比）
2. p 值只来自不一致样本，且对调两个版本后不变
3. 回归与改进分别计入 lost / gained
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))


def load_module():
    spec = importlib.util.spec_from_file_location(
        "compare_paired_under_test",
        ROOT / "pipeline" / "compare_paired.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class McNemarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_no_discordant_samples_is_not_significant(self):
        self.assertEqual(self.module.mcnemar_exact(0, 0), 1.0)

    def test_symmetric(self):
        """对调两个版本不应改变 p 值。"""
        self.assertAlmostEqual(
            self.module.mcnemar_exact(2, 11), self.module.mcnemar_exact(11, 2)
        )

    def test_balanced_discordant_is_not_significant(self):
        self.assertEqual(self.module.mcnemar_exact(5, 5), 1.0)

    def test_known_value_for_two_versus_eleven(self):
        """v4 → DPO 的真实不一致分布：2 条回退、11 条改进。"""
        self.assertAlmostEqual(self.module.mcnemar_exact(2, 11), 0.0225, places=4)

    def test_one_sided_extreme_is_significant(self):
        self.assertLess(self.module.mcnemar_exact(0, 6), 0.05)

    def test_p_value_never_exceeds_one(self):
        for lost in range(0, 12):
            for gained in range(0, 12):
                self.assertLessEqual(self.module.mcnemar_exact(lost, gained), 1.0)


class WilsonIntervalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_bounds_contain_estimate(self):
        low, high = self.module.wilson_interval(334, 396)
        self.assertLess(low, 334 / 396)
        self.assertGreater(high, 334 / 396)

    def test_empty_sample(self):
        self.assertEqual(self.module.wilson_interval(0, 0), (0.0, 0.0))

    def test_small_sample_is_wider_than_large(self):
        small = self.module.wilson_interval(8, 10)
        large = self.module.wilson_interval(800, 1000)
        self.assertGreater(small[1] - small[0], large[1] - large[0])


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_counts_gained_and_lost(self):
        baseline = {"x1": True, "x2": True, "x3": False, "x4": False}
        candidate = {"x1": True, "x2": False, "x3": True, "x4": False}
        stats = self.module.compare(baseline, candidate, label="t")
        self.assertEqual(stats["both"], 1)
        self.assertEqual(stats["neither"], 1)
        self.assertEqual(stats["gained"], 1)
        self.assertEqual(stats["lost"], 1)
        self.assertEqual(stats["gained_ids"], ["x3"])
        self.assertEqual(stats["lost_ids"], ["x2"])
        self.assertEqual(stats["n"], 4)
        self.assertAlmostEqual(stats["delta"], 0.0)

    def test_only_shared_samples_are_compared(self):
        """候选集少一条样本时，不应把它算成回退。"""
        baseline = {"x1": True, "x2": True, "x3": False}
        candidate = {"x1": True, "x2": True}
        stats = self.module.compare(baseline, candidate, label="t")
        self.assertEqual(stats["n"], 2)
        self.assertEqual(stats["lost"], 0)
        self.assertEqual(stats["delta"], 0.0)

    def test_delta_is_percentage_points(self):
        baseline = {"a": False, "b": False, "c": True, "d": True}
        candidate = {"a": True, "b": True, "c": True, "d": True}
        stats = self.module.compare(baseline, candidate, label="t")
        self.assertAlmostEqual(stats["delta"], 0.5)


if __name__ == "__main__":
    unittest.main()
