import argparse
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
    PredictionColumnSpec,
    build_economics_config,
    build_inventory_snapshot,
    load_policy_defaults,
    normalize_prediction_snapshot,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Build normalized profit-analysis input snapshots from repository data.")
    parser.add_argument("--prediction-csv", required=True, help="Upstream prediction CSV to normalize.")
    parser.add_argument("--sku-col", default="sku_id", help="Prediction CSV SKU column.")
    parser.add_argument("--date-col", default="snapshot_date", help="Prediction CSV snapshot date column.")
    parser.add_argument("--prob-col", default="pred_prob_positive", help="Prediction CSV positive-probability column.")
    parser.add_argument("--qty-col", default="pred_qty_30d", help="Prediction CSV 30-day quantity column.")
    parser.add_argument("--prediction-version-col", default=None, help="Optional version column in prediction CSV.")
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
        "--output-dir",
        default=str(PROJECT_ROOT / "reports" / "profit_analysis_inputs"),
        help="Directory for normalized output snapshots.",
    )
    return parser.parse_args()


def _load_optional_csv(path: str, usecols: list[str] | None = None) -> pd.DataFrame | None:
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path, usecols=usecols)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    prediction_raw = pd.read_csv(args.prediction_csv)
    prediction_spec = PredictionColumnSpec(
        sku_id_col=args.sku_col,
        snapshot_date_col=args.date_col,
        prob_col=args.prob_col,
        qty_col=args.qty_col,
        prediction_version_col=args.prediction_version_col,
        prediction_version=args.prediction_version,
    )
    prediction_snapshot = normalize_prediction_snapshot(prediction_raw, spec=prediction_spec)

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

    stamp = time.strftime("%Y%m%d_%H%M")
    pred_path = os.path.join(args.output_dir, f"prediction_snapshot_{stamp}.csv")
    inv_path = os.path.join(args.output_dir, f"inventory_snapshot_{stamp}.csv")
    econ_path = os.path.join(args.output_dir, f"economics_config_{stamp}.csv")

    prediction_snapshot.to_csv(pred_path, index=False, encoding="utf-8-sig")
    inventory_snapshot.to_csv(inv_path, index=False, encoding="utf-8-sig")
    economics_config.to_csv(econ_path, index=False, encoding="utf-8-sig")

    print(f"[OK] prediction -> {os.path.relpath(pred_path, str(PROJECT_ROOT))}")
    print(f"[OK] inventory  -> {os.path.relpath(inv_path, str(PROJECT_ROOT))}")
    print(f"[OK] economics  -> {os.path.relpath(econ_path, str(PROJECT_ROOT))}")
    print(f"[OK] rows       -> {len(prediction_snapshot)}")


if __name__ == "__main__":
    main()
