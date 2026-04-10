import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
MODULE_SRC = MODULE_ROOT / "src"
sys.path.append(str(MODULE_SRC))

from profit_analysis import (
    Economics,
    InventoryState,
    ModelOutput,
    build_profit_input_frame,
    load_economics_config,
    load_inventory_snapshot,
    load_prediction_snapshot,
    recommend_replenishment_plans,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run profit analysis V1 on normalized CSV snapshots.")
    parser.add_argument(
        "--prediction-csv",
        default=str(MODULE_ROOT / "config" / "profit_analysis_prediction_snapshot_template.csv"),
        help="CSV with normalized prediction snapshot fields.",
    )
    parser.add_argument(
        "--inventory-csv",
        default=str(MODULE_ROOT / "config" / "profit_analysis_inventory_snapshot_template.csv"),
        help="CSV with normalized inventory snapshot fields.",
    )
    parser.add_argument(
        "--economics-csv",
        default=str(MODULE_ROOT / "config" / "profit_analysis_economics_config_template.csv"),
        help="CSV with normalized economics config fields.",
    )
    parser.add_argument(
        "--policy",
        default="balanced",
        choices=["conservative", "balanced", "aggressive"],
        help="Ranking policy for candidate plans.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports" / "profit_analysis"),
        help="Directory to store summary CSV and detail JSON outputs.",
    )
    return parser.parse_args()


def build_objects(row):
    model_output = ModelOutput(
        sku_id=row["sku_id"],
        snapshot_date=row["snapshot_date"],
        pred_prob_positive=row["pred_prob_positive"],
        pred_qty_30d=row["pred_qty_30d"],
        prediction_version=row.get("prediction_version"),
    )
    inventory_state = InventoryState(
        sku_id=row["sku_id"],
        snapshot_date=row["snapshot_date"],
        current_inventory=row["current_inventory"],
        inbound_within_30d=row.get("inbound_within_30d", 0.0),
        lead_time_days=row.get("lead_time_days", 0),
        min_batch_qty=row.get("min_batch_qty"),
        max_replenish_qty=row.get("max_replenish_qty"),
        safety_stock_qty=row.get("safety_stock_qty"),
        last_decision_date=row.get("last_decision_date"),
    )
    economics = Economics(
        sku_id=row["sku_id"],
        unit_cost=row["unit_cost"],
        unit_price=row["unit_price"],
        holding_cost_per_unit_per_day=row["holding_cost_per_unit_per_day"],
        salvage_value_per_unit=row["salvage_value_per_unit"],
        stockout_penalty_per_unit=row.get("stockout_penalty_per_unit", 0.0),
        other_fixed_cost=row.get("other_fixed_cost", 0.0),
        lifecycle_end_date=row.get("lifecycle_end_date"),
    )
    return model_output, inventory_state, economics


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prediction_df = load_prediction_snapshot(args.prediction_csv)
    inventory_df = load_inventory_snapshot(args.inventory_csv)
    economics_df = load_economics_config(args.economics_csv)
    work_df = build_profit_input_frame(prediction_df, inventory_df, economics_df)

    if work_df.empty:
        raise ValueError("No joined rows found across prediction, inventory, and economics inputs.")

    details = []
    summary_rows = []
    for row in work_df.to_dict(orient="records"):
        model_output, inventory_state, economics = build_objects(row)
        rec = recommend_replenishment_plans(
            model_output=model_output,
            inventory_state=inventory_state,
            economics=economics,
            policy=args.policy,
        )
        details.append(rec)
        best_balanced = rec["best_balanced_plan"] or {}
        best_profit = rec["best_profit_plan"] or {}
        lowest_risk = rec["lowest_risk_plan"] or {}
        summary_rows.append(
            {
                "sku_id": row["sku_id"],
                "snapshot_date": row["snapshot_date"],
                "policy": args.policy,
                "pred_prob_positive": row["pred_prob_positive"],
                "pred_qty_30d": row["pred_qty_30d"],
                "current_inventory": row["current_inventory"],
                "inbound_within_30d": row.get("inbound_within_30d", 0.0),
                "best_balanced_plan_qty": best_balanced.get("plan_qty"),
                "best_balanced_expected_profit": best_balanced.get("expected_profit"),
                "best_balanced_stockout_rate": best_balanced.get("stockout_rate"),
                "best_balanced_expected_leftover_qty": best_balanced.get("expected_leftover_qty"),
                "best_profit_plan_qty": best_profit.get("plan_qty"),
                "lowest_risk_plan_qty": lowest_risk.get("plan_qty"),
            }
        )

    stamp = time.strftime("%Y%m%d_%H%M")
    summary_path = os.path.join(args.output_dir, f"profit_analysis_summary_{args.policy}_{stamp}.csv")
    detail_path = os.path.join(args.output_dir, f"profit_analysis_detail_{args.policy}_{stamp}.json")

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    with open(detail_path, "w", encoding="utf-8") as fh:
        json.dump(details, fh, ensure_ascii=False, indent=2)

    print(f"[OK] summary -> {os.path.relpath(summary_path, str(PROJECT_ROOT))}")
    print(f"[OK] detail  -> {os.path.relpath(detail_path, str(PROJECT_ROOT))}")
    print(f"[OK] rows    -> {len(summary_rows)}")


if __name__ == "__main__":
    main()
