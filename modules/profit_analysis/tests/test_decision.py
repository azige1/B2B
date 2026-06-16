from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(MODULE_ROOT / "src"))

from profit_analysis import (  # noqa: E402
    DemandScenarioCalibration,
    ProfitAnalysisQualityError,
    build_skc_decision_batch,
)


def _calibration() -> DemandScenarioCalibration:
    return DemandScenarioCalibration(
        probability_x=(0.0, 1.0),
        probability_y=(0.0, 1.0),
        positive_multipliers=(0.6, 1.0, 1.5),
        positive_weights=(0.25, 0.5, 0.25),
        calibration_rows=100,
        positive_calibration_rows=20,
    )


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction = pd.DataFrame(
        [
            {
                "sku_id": "S",
                "style_id": "STYLE",
                "snapshot_date": "2026-02-15",
                "pred_prob_positive": 0.9,
                "pred_qty_30d": 80.0,
            },
            {
                "sku_id": "M",
                "style_id": "STYLE",
                "snapshot_date": "2026-02-15",
                "pred_prob_positive": 0.8,
                "pred_qty_30d": 60.0,
            },
        ]
    )
    inventory = pd.DataFrame(
        [
            {
                "sku_id": "S",
                "snapshot_date": "2026-02-15",
                "current_inventory": 0.0,
                "lead_time_days": 0,
            },
            {
                "sku_id": "M",
                "snapshot_date": "2026-02-15",
                "current_inventory": 0.0,
                "lead_time_days": 0,
            },
        ]
    )
    economics = pd.DataFrame(
        [
            {
                "sku_id": sku_id,
                "unit_cost": 10.0,
                "unit_price": 100.0,
                "holding_cost_per_unit_per_day": 0.0,
                "salvage_value_per_unit": 0.0,
                "cost_source": "client_style_cost",
            }
            for sku_id in ["S", "M"]
        ]
    )
    return prediction, inventory, economics


class SKCDecisionBatchTests(unittest.TestCase):
    def test_builds_skc_plan_and_preserves_allocation_total(self) -> None:
        prediction, inventory, economics = _inputs()
        batch = build_skc_decision_batch(
            prediction,
            inventory,
            economics,
            _calibration(),
            horizon_days=45,
        )

        self.assertEqual(len(batch.skc_recommendations), 1)
        plan_qty = batch.skc_recommendations.iloc[0]["recommended_plan_qty"]
        self.assertGreaterEqual(plan_qty, 100.0)
        self.assertEqual(
            batch.sku_allocations["allocated_plan_qty"].sum(),
            plan_qty,
        )
        self.assertEqual(batch.quality_report["status"], "passed")

    def test_rejects_missing_inventory_row(self) -> None:
        prediction, inventory, economics = _inputs()
        inventory = inventory[inventory["sku_id"] == "S"]

        with self.assertRaises(ProfitAnalysisQualityError) as context:
            build_skc_decision_batch(
                prediction,
                inventory,
                economics,
                _calibration(),
            )

        self.assertEqual(
            context.exception.quality_report["missing_inventory_rows"],
            1,
        )

    def test_rejects_duplicate_prediction_key(self) -> None:
        prediction, inventory, economics = _inputs()
        prediction = pd.concat([prediction, prediction.iloc[[0]]], ignore_index=True)

        with self.assertRaises(ProfitAnalysisQualityError) as context:
            build_skc_decision_batch(
                prediction,
                inventory,
                economics,
                _calibration(),
            )

        self.assertGreater(
            context.exception.quality_report["duplicate_prediction_rows"],
            0,
        )

    def test_uses_economics_style_when_prediction_style_is_blank(self) -> None:
        prediction, inventory, economics = _inputs()
        prediction["style_id"] = ""
        economics["style_id"] = "STYLE"

        batch = build_skc_decision_batch(
            prediction,
            inventory,
            economics,
            _calibration(),
        )

        self.assertEqual(batch.skc_recommendations.iloc[0]["style_id"], "STYLE")

    def test_rejects_conflicting_style_mapping(self) -> None:
        prediction, inventory, economics = _inputs()
        economics["style_id"] = "OTHER_STYLE"

        with self.assertRaises(ProfitAnalysisQualityError) as context:
            build_skc_decision_batch(
                prediction,
                inventory,
                economics,
                _calibration(),
            )

        self.assertEqual(
            context.exception.quality_report["conflicting_style_id_skus"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
