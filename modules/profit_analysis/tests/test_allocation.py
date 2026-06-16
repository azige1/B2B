from __future__ import annotations

import sys
import unittest
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(MODULE_ROOT / "src"))

from profit_analysis import allocate_integer_plan  # noqa: E402


class AllocationTests(unittest.TestCase):
    def test_allocation_preserves_group_total(self) -> None:
        allocated = allocate_integer_plan(
            plan_qty=100,
            item_ids=["S", "M", "L"],
            demand_scores=[1.0, 2.0, 1.0],
        )

        self.assertEqual(sum(allocated.values()), 100)
        self.assertEqual(allocated, {"S": 25, "M": 50, "L": 25})

    def test_zero_scores_fall_back_to_even_allocation(self) -> None:
        allocated = allocate_integer_plan(
            plan_qty=10,
            item_ids=["S", "M", "L"],
            demand_scores=[0.0, 0.0, 0.0],
        )

        self.assertEqual(sum(allocated.values()), 10)
        self.assertEqual(sorted(allocated.values()), [3, 3, 4])


if __name__ == "__main__":
    unittest.main()
