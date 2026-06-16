from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd
import tempfile
import json


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(MODULE_ROOT / "src"))

from profit_analysis import (  # noqa: E402
    DemandScenarioCalibration,
    ModelOutput,
    fit_demand_scenario_calibration,
    load_demand_scenario_calibration,
    probability_calibration_metrics,
)


class DemandCalibrationTests(unittest.TestCase):
    def test_calibration_learns_probability_and_quantity_multipliers(self) -> None:
        rows = []
        for probability, actual, prediction in [
            (0.1, 0.0, 1.0),
            (0.2, 0.0, 1.0),
            (0.7, 1.0, 2.0),
            (0.8, 2.0, 2.0),
            (0.9, 3.0, 2.0),
            (1.0, 4.0, 2.0),
        ]:
            rows.append(
                {
                    "true_replenish_qty": actual,
                    "ai_pred_prob": probability,
                    "ai_pred_qty_open": prediction,
                }
            )
        calibration = fit_demand_scenario_calibration(pd.DataFrame(rows))

        self.assertLess(calibration.calibrate_probability(0.15), 0.5)
        self.assertGreater(calibration.calibrate_probability(0.95), 0.5)
        self.assertEqual(calibration.positive_multipliers, (0.875, 1.25, 1.625))

        scenarios = calibration.build_scenarios(
            ModelOutput("SKU", "2026-02-15", 0.95, 30.0),
            horizon_days=30,
        )
        self.assertAlmostEqual(sum(row.probability for row in scenarios), 1.0)
        self.assertEqual(
            [row.demand_qty for row in scenarios[1:]],
            [26.25, 37.5, 48.75],
        )

        metrics = probability_calibration_metrics(
            pd.DataFrame(rows),
            calibration,
        )
        self.assertLessEqual(
            metrics["calibrated_brier_score"],
            metrics["raw_brier_score"],
        )

    def test_loads_nested_calibration_payload(self) -> None:
        calibration = DemandScenarioCalibration(
            probability_x=(0.0, 1.0),
            probability_y=(0.1, 0.9),
            positive_multipliers=(0.5, 1.0, 1.5),
            positive_weights=(0.25, 0.5, 0.25),
            calibration_rows=10,
            positive_calibration_rows=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(
                json.dumps({"calibration": calibration.to_dict()}),
                encoding="utf-8",
            )
            loaded = load_demand_scenario_calibration(path)

        self.assertEqual(loaded, calibration)


if __name__ == "__main__":
    unittest.main()
