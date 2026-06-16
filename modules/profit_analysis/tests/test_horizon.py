from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(MODULE_ROOT / "src"))

from profit_analysis import (  # noqa: E402
    build_daily_demand_curves,
    build_horizon_calibration_frame,
    build_horizon_demand,
)


class HorizonDemandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.daily = pd.DataFrame(
            [
                {"date": "2026-01-01", "sku_id": "A", "qty_replenish": 99},
                {"date": "2026-01-02", "sku_id": "A", "qty_replenish": 1},
                {"date": "2026-01-02", "sku_id": "A", "qty_replenish": 2},
                {"date": "2026-01-03", "sku_id": "A", "qty_replenish": 4},
                {"date": "2026-01-04", "sku_id": "A", "qty_replenish": 8},
                {"date": "2026-01-02", "sku_id": "B", "qty_replenish": 5},
            ]
        )

    def test_horizon_total_excludes_anchor_and_includes_end_day(self) -> None:
        context = pd.DataFrame(
            [
                {"anchor_date": "2026-01-01", "sku_id": "A"},
                {"anchor_date": "2026-01-01", "sku_id": "B"},
                {"anchor_date": "2026-01-01", "sku_id": "C"},
            ]
        )

        result = build_horizon_demand(context, self.daily, horizon_days=2)

        self.assertEqual(result["demand_qty"].tolist(), [7.0, 5.0, 0.0])

    def test_daily_curves_preserve_timing_and_duplicate_sums(self) -> None:
        curves = build_daily_demand_curves(
            self.daily,
            anchors=["2026-01-01"],
            horizon_days=3,
        )

        self.assertEqual(curves[("2026-01-01", "A")], [3.0, 4.0, 8.0])
        self.assertEqual(curves[("2026-01-01", "B")], [5.0, 0.0, 0.0])

    def test_calibration_quantity_is_scaled_to_target_horizon(self) -> None:
        context = pd.DataFrame(
            [
                {
                    "anchor_date": "2026-01-01",
                    "sku_id": "A",
                    "ai_pred_prob": 0.8,
                    "ai_pred_qty_open": 10.0,
                }
            ]
        )

        result = build_horizon_calibration_frame(
            context,
            self.daily,
            horizon_days=3,
            source_quantity_horizon_days=2,
        )

        self.assertEqual(float(result.loc[0, "true_replenish_qty"]), 15.0)
        self.assertEqual(float(result.loc[0, "ai_pred_qty_open"]), 15.0)


if __name__ == "__main__":
    unittest.main()
