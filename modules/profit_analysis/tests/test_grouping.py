from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(MODULE_ROOT / "src"))

from profit_analysis import (  # noqa: E402
    aggregate_hurdle_outputs,
    build_item_demand_gap_scores,
)


class GroupDemandTests(unittest.TestCase):
    def test_hurdle_outputs_aggregate_to_group(self) -> None:
        output = aggregate_hurdle_outputs(
            probabilities=[0.2, 0.5],
            conditional_quantities=[10.0, 20.0],
        )

        self.assertAlmostEqual(output.probability_positive, 0.6)
        self.assertAlmostEqual(output.expected_qty, 12.0)
        self.assertAlmostEqual(output.conditional_qty, 20.0)

    def test_group_expected_quantity_matches_item_sum(self) -> None:
        output = aggregate_hurdle_outputs(
            probabilities=np.array([0.0, 1.0, 0.25]),
            conditional_quantities=np.array([10.0, 4.0, 8.0]),
        )

        self.assertAlmostEqual(output.probability_positive, 1.0)
        self.assertAlmostEqual(output.expected_qty, 6.0)
        self.assertAlmostEqual(output.conditional_qty, 6.0)

    def test_demand_gap_scores_remove_existing_inventory(self) -> None:
        scores = build_item_demand_gap_scores(
            probabilities=[0.5, 0.8, 0.2],
            conditional_quantities=[20.0, 10.0, 5.0],
            current_inventory=[2.0, 10.0, 0.0],
            positive_multiplier_mean=1.5,
        )

        self.assertEqual(scores, [13.0, 2.0, 1.5])


if __name__ == "__main__":
    unittest.main()
