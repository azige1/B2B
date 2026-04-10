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

from profit_analysis import PredictionColumnSpec, infer_prediction_column_spec, normalize_prediction_snapshot


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize current model prediction outputs into profit-analysis prediction_snapshot format.")
    parser.add_argument("--source-csv", required=True, help="Prediction output CSV to normalize.")
    parser.add_argument("--output-csv", default=None, help="Optional output path. Defaults under reports/profit_analysis_inputs/.")
    parser.add_argument("--sku-col", default=None, help="Optional override for SKU column.")
    parser.add_argument("--date-col", default=None, help="Optional override for snapshot date column.")
    parser.add_argument("--prob-col", default=None, help="Optional override for probability column.")
    parser.add_argument("--qty-col", default=None, help="Optional override for quantity column.")
    parser.add_argument("--prediction-version-col", default=None, help="Optional version column in source CSV.")
    parser.add_argument("--prediction-version", default=None, help="Optional fixed prediction version tag.")
    return parser.parse_args()


def main():
    args = parse_args()
    source_df = pd.read_csv(args.source_csv)

    if any([args.sku_col, args.date_col, args.prob_col, args.qty_col]):
        if not all([args.sku_col, args.date_col, args.prob_col, args.qty_col]):
            raise ValueError("If any explicit column override is used, sku/date/prob/qty columns must all be provided.")
        spec = PredictionColumnSpec(
            sku_id_col=args.sku_col,
            snapshot_date_col=args.date_col,
            prob_col=args.prob_col,
            qty_col=args.qty_col,
            prediction_version_col=args.prediction_version_col,
            prediction_version=args.prediction_version,
        )
    else:
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

    normalized = normalize_prediction_snapshot(source_df, spec=spec)

    output_csv = args.output_csv
    if not output_csv:
        out_dir = PROJECT_ROOT / "reports" / "profit_analysis_inputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M")
        stem = Path(args.source_csv).stem
        output_csv = str(out_dir / f"{stem}_prediction_snapshot_{stamp}.csv")
    else:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    normalized.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"[OK] source   -> {os.path.relpath(args.source_csv, str(PROJECT_ROOT))}")
    print(f"[OK] output   -> {os.path.relpath(output_csv, str(PROJECT_ROOT))}")
    print(f"[OK] rows     -> {len(normalized)}")
    print(f"[OK] mapping  -> sku={spec.sku_id_col}, date={spec.snapshot_date_col}, prob={spec.prob_col}, qty={spec.qty_col}")


if __name__ == "__main__":
    main()
