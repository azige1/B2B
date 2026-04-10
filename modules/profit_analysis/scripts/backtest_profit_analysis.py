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
    CandidatePlan,
    Economics,
    InventoryState,
    ModelOutput,
    PredictionColumnSpec,
    build_economics_config,
    build_inventory_snapshot,
    infer_actual_qty_col,
    infer_prediction_column_spec,
    load_policy_defaults,
    normalize_prediction_snapshot,
    realize_replenishment_plan,
    recommend_replenishment_plans,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Backtest profit-analysis strategies against realized 30-day demand.")
    parser.add_argument("--source-csv", required=True, help="Raw prediction/eval detail CSV with realized demand column.")
    parser.add_argument("--actual-col", default=None, help="Optional override for actual demand column.")
    parser.add_argument("--sku-col", default=None, help="Optional override for SKU column.")
    parser.add_argument("--date-col", default=None, help="Optional override for snapshot date column.")
    parser.add_argument("--prob-col", default=None, help="Optional override for probability column.")
    parser.add_argument("--qty-col", default=None, help="Optional override for predicted quantity column.")
    parser.add_argument("--prediction-version-col", default=None, help="Optional version column.")
    parser.add_argument("--prediction-version", default=None, help="Optional fixed prediction version tag.")
    parser.add_argument(
        "--inventory-source",
        default=str(PROJECT_ROOT / "data" / "silver" / "clean_inventory.csv"),
        help="Clean inventory source CSV.",
    )
    parser.add_argument(
        "--products-source",
        default=str(PROJECT_ROOT / "data" / "silver" / "clean_products.csv"),
        help="Clean products source CSV.",
    )
    parser.add_argument(
        "--wide-table-source",
        default=str(PROJECT_ROOT / "data" / "gold" / "wide_table_sku.csv"),
        help="Wide table source CSV for inbound proxy.",
    )
    parser.add_argument(
        "--lifecycle-source",
        default=str(PROJECT_ROOT / "data_warehouse" / "dim_product" / "product_info_latest1.csv"),
        help="Lifecycle source CSV.",
    )
    parser.add_argument(
        "--defaults-csv",
        default=str(MODULE_ROOT / "config" / "profit_analysis_business_defaults_template.csv"),
        help="Business defaults CSV.",
    )
    parser.add_argument(
        "--policy",
        default="balanced",
        choices=["conservative", "balanced", "aggressive"],
        help="Which profit module recommendation policy to backtest.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports" / "profit_analysis_backtest"),
        help="Directory for backtest outputs.",
    )
    return parser.parse_args()


def _load_optional_csv(path: str, usecols: list[str] | None = None) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path, usecols=usecols)


def _build_prediction_spec(args, source_df: pd.DataFrame) -> PredictionColumnSpec:
    if any([args.sku_col, args.date_col, args.prob_col, args.qty_col]):
        if not all([args.sku_col, args.date_col, args.prob_col, args.qty_col]):
            raise ValueError("If any explicit column override is used, sku/date/prob/qty columns must all be provided.")
        return PredictionColumnSpec(
            sku_id_col=args.sku_col,
            snapshot_date_col=args.date_col,
            prob_col=args.prob_col,
            qty_col=args.qty_col,
            prediction_version_col=args.prediction_version_col,
            prediction_version=args.prediction_version,
        )

    spec = infer_prediction_column_spec(source_df, prediction_version=args.prediction_version)
    if args.prediction_version_col:
        spec = PredictionColumnSpec(
            sku_id_col=spec.sku_id_col,
            snapshot_date_col=spec.snapshot_date_col,
            prob_col=spec.prob_col,
            qty_col=spec.qty_col,
            prediction_version_col=args.prediction_version_col,
            prediction_version=args.prediction_version,
        )
    return spec


def _summarize_strategy(df: pd.DataFrame) -> dict:
    rows = len(df)
    total_available = float(df["available_qty"].sum())
    total_sold = float(df["sold_qty"].sum())
    total_leftover = float(df["leftover_qty"].sum())
    total_lost = float(df["lost_sales_qty"].sum())
    return {
        "rows": rows,
        "mean_plan_qty": float(df["plan_qty"].mean()) if rows else 0.0,
        "mean_realized_profit": float(df["realized_profit"].mean()) if rows else 0.0,
        "total_realized_profit": float(df["realized_profit"].sum()),
        "stockout_rate": float(df["stockout_flag"].mean()) if rows else 0.0,
        "leftover_rate": float((df["leftover_qty"] > 0).mean()) if rows else 0.0,
        "sell_through_rate": float(total_sold / max(total_available, 1e-9)),
        "lost_sales_rate": float(total_lost / max(df["actual_demand_qty"].sum(), 1e-9)),
        "leftover_share_of_supply": float(total_leftover / max(total_available, 1e-9)),
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    source_df = pd.read_csv(args.source_csv)
    actual_col = args.actual_col or infer_actual_qty_col(source_df)
    if actual_col not in source_df.columns:
        raise ValueError(f"actual demand column not found: {actual_col}")

    prediction_spec = _build_prediction_spec(args, source_df)
    prediction_snapshot = normalize_prediction_snapshot(source_df, spec=prediction_spec)
    actual_df = source_df.loc[:, [prediction_spec.sku_id_col, prediction_spec.snapshot_date_col, actual_col]].copy()
    actual_df = actual_df.rename(
        columns={
            prediction_spec.sku_id_col: "sku_id",
            prediction_spec.snapshot_date_col: "snapshot_date",
            actual_col: "actual_demand_qty",
        }
    )
    actual_df["sku_id"] = actual_df["sku_id"].astype(str)
    actual_df["snapshot_date"] = pd.to_datetime(actual_df["snapshot_date"]).dt.strftime("%Y-%m-%d")
    actual_df["actual_demand_qty"] = pd.to_numeric(actual_df["actual_demand_qty"], errors="coerce").fillna(0.0).clip(lower=0.0)

    clean_inventory = pd.read_csv(args.inventory_source)
    products = pd.read_csv(args.products_source)
    wide_table = _load_optional_csv(args.wide_table_source, usecols=["sku_id", "date", "qty_inbound"])
    lifecycle = _load_optional_csv(args.lifecycle_source, usecols=["NO", "PL_CYCLE", "LISTING_DATE"])
    defaults_df = load_policy_defaults(args.defaults_csv)

    inventory_snapshot = build_inventory_snapshot(
        prediction_df=prediction_snapshot,
        clean_inventory_df=clean_inventory,
        product_df=products,
        defaults_df=defaults_df,
        wide_table_df=wide_table,
    )
    economics_config = build_economics_config(
        prediction_df=prediction_snapshot,
        product_df=products,
        defaults_df=defaults_df,
        lifecycle_df=lifecycle,
    )

    work_df = prediction_snapshot.merge(actual_df, on=["sku_id", "snapshot_date"], how="inner")
    work_df = work_df.merge(inventory_snapshot, on=["sku_id", "snapshot_date"], how="inner")
    work_df = work_df.merge(economics_config, on="sku_id", how="inner")
    if work_df.empty:
        raise ValueError("No rows left after joining prediction, actual, inventory, and economics inputs.")

    detail_rows = []
    strategy_frames = {
        "baseline_0": [],
        "baseline_direct": [],
        "profit_module_v1": [],
    }

    for row in work_df.to_dict(orient="records"):
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

        rec = recommend_replenishment_plans(
            model_output=model_output,
            inventory_state=inventory_state,
            economics=economics,
            policy=args.policy,
        )
        recommended_plan_qty = float((rec["best_balanced_plan"] or {}).get("plan_qty", 0.0))
        direct_plan_qty = max(
            float(row["pred_qty_30d"]) - float(row["current_inventory"]) - float(row.get("inbound_within_30d", 0.0)),
            0.0,
        )

        plan_defs = {
            "baseline_0": CandidatePlan(plan_qty=0.0, policy="baseline_0"),
            "baseline_direct": CandidatePlan(plan_qty=direct_plan_qty, policy="baseline_direct"),
            "profit_module_v1": CandidatePlan(plan_qty=recommended_plan_qty, policy=args.policy),
        }

        for strategy, plan in plan_defs.items():
            realized = realize_replenishment_plan(
                model_output=model_output,
                inventory_state=inventory_state,
                economics=economics,
                plan=plan,
                actual_demand_qty=row["actual_demand_qty"],
            )
            out = {
                "strategy": strategy,
                "sku_id": row["sku_id"],
                "snapshot_date": row["snapshot_date"],
                "prediction_version": row.get("prediction_version"),
                "pred_prob_positive": row["pred_prob_positive"],
                "pred_qty_30d": row["pred_qty_30d"],
                "actual_demand_qty": row["actual_demand_qty"],
                "current_inventory": row["current_inventory"],
                "inbound_within_30d": row.get("inbound_within_30d", 0.0),
                "available_qty": float(row["current_inventory"]) + float(row.get("inbound_within_30d", 0.0)) + float(realized.plan_qty),
                "expected_profit_proxy": (
                    float((rec["best_balanced_plan"] or {}).get("expected_profit", 0.0))
                    if strategy == "profit_module_v1"
                    else None
                ),
                **realized.to_dict(),
            }
            detail_rows.append(out)
            strategy_frames[strategy].append(out)

    detail_df = pd.DataFrame(detail_rows)
    summary = {
        "source_csv": os.path.relpath(args.source_csv, str(PROJECT_ROOT)),
        "policy": args.policy,
        "rows": int(len(work_df)),
        "actual_col": actual_col,
        "strategies": {
            name: _summarize_strategy(pd.DataFrame(rows))
            for name, rows in strategy_frames.items()
        },
    }
    if summary["strategies"]["baseline_direct"]["total_realized_profit"] != 0:
        base_total = summary["strategies"]["baseline_direct"]["total_realized_profit"]
        profit_total = summary["strategies"]["profit_module_v1"]["total_realized_profit"]
        summary["profit_module_vs_direct_profit_lift"] = float((profit_total - base_total) / abs(base_total))

    stamp = time.strftime("%Y%m%d_%H%M")
    detail_path = os.path.join(args.output_dir, f"profit_backtest_detail_{args.policy}_{stamp}.csv")
    summary_path = os.path.join(args.output_dir, f"profit_backtest_summary_{args.policy}_{stamp}.json")

    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"[OK] detail  -> {os.path.relpath(detail_path, str(PROJECT_ROOT))}")
    print(f"[OK] summary -> {os.path.relpath(summary_path, str(PROJECT_ROOT))}")
    print(f"[OK] rows    -> {len(work_df)}")


if __name__ == "__main__":
    main()
