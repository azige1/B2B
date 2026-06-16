from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(MODULE_ROOT / "src"))

from profit_analysis import (  # noqa: E402
    build_economics_config,
    build_profit_input_frame,
    build_inventory_snapshot,
    infer_actual_qty_col,
    infer_prediction_column_spec,
)


class BuilderDefaultTests(unittest.TestCase):
    def test_client_export_prediction_columns_are_inferred(self) -> None:
        client_export = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "锚点日期": "2025-12-01",
                    "模型补货概率": 0.8,
                    "最终预测量": 70.0,
                    "验证期实际补货量": 60.0,
                }
            ]
        )

        spec = infer_prediction_column_spec(client_export)

        self.assertEqual(spec.snapshot_date_col, "锚点日期")
        self.assertEqual(spec.prob_col, "模型补货概率")
        self.assertEqual(spec.qty_col, "最终预测量")
        self.assertEqual(infer_actual_qty_col(client_export), "验证期实际补货量")

    def test_client_feedback_defaults_are_used_without_override_csv(self) -> None:
        prediction_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "snapshot_date": "2026-04-10",
                    "pred_prob_positive": 0.8,
                    "pred_qty_30d": 70.0,
                }
            ]
        )
        product_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "style_id": "STYLE",
                    "category": "outerwear",
                    "price_tag": 700.0,
                }
            ]
        )
        clean_inventory_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "qty_stock": 5.0,
                    "inventory_date": "2026-04-10",
                }
            ]
        )

        inventory = build_inventory_snapshot(prediction_df, clean_inventory_df, product_df)
        economics = build_economics_config(prediction_df, product_df)

        self.assertEqual(float(inventory.loc[0, "min_batch_qty"]), 100.0)
        self.assertEqual(float(inventory.loc[0, "increment_batch_qty"]), 10.0)
        self.assertEqual(float(economics.loc[0, "unit_cost"]), 100.0)
        self.assertEqual(float(economics.loc[0, "salvage_value_per_unit"]), 0.0)
        self.assertEqual(float(economics.loc[0, "target_sell_through_rate"]), 0.85)
        self.assertEqual(int(economics.loc[0, "lifecycle_days"]), 45)

    def test_lifecycle_v0_feature_table_is_accepted_by_economics_builder(self) -> None:
        prediction_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "snapshot_date": "2026-04-10",
                    "pred_prob_positive": 0.8,
                    "pred_qty_30d": 70.0,
                }
            ]
        )
        product_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "style_id": "STYLE",
                    "category": "outerwear",
                    "price_tag": 700.0,
                }
            ]
        )
        lifecycle_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "launch_date": "2026-03-06",
                    "estimated_lifecycle_end_date": "2026-04-19",
                    "lifecycle_days_assumption": 45,
                }
            ]
        )

        economics = build_economics_config(prediction_df, product_df, lifecycle_df=lifecycle_df)

        self.assertEqual(economics.loc[0, "launch_date"], "2026-03-06")
        self.assertEqual(economics.loc[0, "lifecycle_end_date"], "2026-04-19")
        self.assertEqual(int(economics.loc[0, "lifecycle_days"]), 45)

    def test_style_id_cost_overrides_price_tag_fallback(self) -> None:
        prediction_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "snapshot_date": "2026-04-10",
                    "pred_prob_positive": 0.8,
                    "pred_qty_30d": 70.0,
                }
            ]
        )
        product_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "style_id": "ak10801537",
                    "category": "outerwear",
                    "price_tag": 700.0,
                }
            ]
        )
        style_cost_df = pd.DataFrame(
            [
                {
                    "style_id": "AK10801537",
                    "unit_cost": 135.5,
                    "cost_source": "client_style_cost_2024_2026_max",
                    "cost_record_count": 2,
                    "cost_conflict_flag": 1,
                }
            ]
        )

        economics = build_economics_config(
            prediction_df,
            product_df,
            style_cost_df=style_cost_df,
        )

        self.assertEqual(float(economics.loc[0, "unit_cost"]), 135.5)
        self.assertEqual(
            economics.loc[0, "cost_source"],
            "client_style_cost_2024_2026_max",
        )
        self.assertEqual(int(economics.loc[0, "cost_record_count"]), 2)
        self.assertEqual(int(economics.loc[0, "cost_conflict_flag"]), 1)

    def test_inventory_builder_never_uses_inventory_after_snapshot_date(self) -> None:
        prediction_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "snapshot_date": "2025-12-01",
                    "pred_prob_positive": 0.8,
                    "pred_qty_30d": 70.0,
                }
            ]
        )
        product_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "style_id": "STYLE",
                    "category": "outerwear",
                    "price_tag": 700.0,
                }
            ]
        )
        inventory_df = pd.DataFrame(
            [
                {"sku_id": "SKU", "qty_stock": 7.0, "inventory_date": "2025-11-30"},
                {"sku_id": "SKU", "qty_stock": 99.0, "inventory_date": "2026-01-30"},
            ]
        )

        inventory = build_inventory_snapshot(prediction_df, inventory_df, product_df)

        self.assertEqual(float(inventory.loc[0, "current_inventory"]), 7.0)
        self.assertEqual(inventory.loc[0, "inventory_source_date"], "2025-11-30")
        self.assertEqual(int(inventory.loc[0, "inventory_snapshot_present"]), 1)

    def test_daily_inventory_features_are_resolved_at_exact_anchor(self) -> None:
        prediction_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "snapshot_date": "2026-02-15",
                    "pred_prob_positive": 0.8,
                    "pred_qty_30d": 70.0,
                }
            ]
        )
        product_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "style_id": "STYLE",
                    "category": "outerwear",
                    "price_tag": 700.0,
                }
            ]
        )
        inventory_df = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "date": "2026-02-15",
                    "qty_total_stock": 20.0,
                    "snapshot_present": 1,
                },
                {
                    "sku_id": "SKU",
                    "date": "2026-02-15",
                    "qty_total_stock": 30.0,
                    "snapshot_present": 1,
                },
                {
                    "sku_id": "SKU",
                    "date": "2026-02-16",
                    "qty_total_stock": 90.0,
                    "snapshot_present": 1,
                },
            ]
        )

        inventory = build_inventory_snapshot(prediction_df, inventory_df, product_df)

        self.assertEqual(float(inventory.loc[0, "current_inventory"]), 30.0)
        self.assertEqual(inventory.loc[0, "inventory_source_date"], "2026-02-15")
        self.assertEqual(int(inventory.loc[0, "inventory_snapshot_present"]), 1)

    def test_profit_input_frame_joins_inventory_by_sku_and_snapshot_date(self) -> None:
        predictions = pd.DataFrame(
            [
                {"sku_id": "SKU", "snapshot_date": "2026-02-15", "pred_prob_positive": 0.8, "pred_qty_30d": 10.0},
                {"sku_id": "SKU", "snapshot_date": "2026-02-24", "pred_prob_positive": 0.7, "pred_qty_30d": 8.0},
            ]
        )
        inventory = pd.DataFrame(
            [
                {"sku_id": "SKU", "snapshot_date": "2026-02-15", "current_inventory": 20.0},
                {"sku_id": "SKU", "snapshot_date": "2026-02-24", "current_inventory": 12.0},
            ]
        )
        economics = pd.DataFrame(
            [
                {
                    "sku_id": "SKU",
                    "unit_cost": 10.0,
                    "unit_price": 20.0,
                    "holding_cost_per_unit_per_day": 0.0,
                    "salvage_value_per_unit": 0.0,
                }
            ]
        )

        merged = build_profit_input_frame(predictions, inventory, economics)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged["current_inventory"].tolist(), [20.0, 12.0])


if __name__ == "__main__":
    unittest.main()
