from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
sys.path.append(str(MODULE_ROOT / "src"))

from profit_analysis import (  # noqa: E402
    ProfitAnalysisQualityError,
    build_skc_decision_batch,
    load_demand_scenario_calibration,
    load_economics_config,
    load_inventory_snapshot,
    load_prediction_snapshot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run production-style SKC profit decisions and allocate each "
            "recommended plan to SKU rows."
        )
    )
    parser.add_argument("--prediction-csv", required=True)
    parser.add_argument("--inventory-csv", required=True)
    parser.add_argument("--economics-csv", required=True)
    parser.add_argument(
        "--calibration-json",
        default=str(
            MODULE_ROOT
            / "config"
            / "demand_scenario_calibration_h45_20260612.json"
        ),
    )
    parser.add_argument(
        "--policy",
        default="balanced",
        choices=["conservative", "balanced", "aggressive"],
    )
    parser.add_argument("--horizon-days", type=int, default=45)
    parser.add_argument("--min-batch-qty", type=float, default=100.0)
    parser.add_argument("--increment-batch-qty", type=float, default=10.0)
    parser.add_argument("--max-fallback-cost-rate", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "reports" / "profit_analysis_snapshot"),
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Optional stable suffix for output files. If omitted, the runner "
            "uses the current timestamp."
        ),
    )
    return parser.parse_args()


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _resolve_run_id(value: str | None) -> str:
    if value is None or not str(value).strip():
        return time.strftime("%Y%m%d_%H%M%S")
    raw = str(value).strip()
    for char in ["\\", "/", ":", "*", "?", '"', "<", ">", "|", " "]:
        raw = raw.replace(char, "_")
    return raw


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = _resolve_run_id(args.run_id)
    quality_path = output_dir / f"quality_report_{stamp}.json"
    manifest_path = output_dir / f"run_manifest_{stamp}.json"

    manifest = {
        "status": "running",
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_csv": str(Path(args.prediction_csv).resolve()),
        "inventory_csv": str(Path(args.inventory_csv).resolve()),
        "economics_csv": str(Path(args.economics_csv).resolve()),
        "calibration_json": str(Path(args.calibration_json).resolve()),
        "policy": args.policy,
        "horizon_days": args.horizon_days,
        "min_batch_qty": args.min_batch_qty,
        "increment_batch_qty": args.increment_batch_qty,
        "max_fallback_cost_rate": args.max_fallback_cost_rate,
        "run_id": stamp,
    }
    try:
        prediction = load_prediction_snapshot(args.prediction_csv)
        inventory = load_inventory_snapshot(args.inventory_csv)
        economics = load_economics_config(args.economics_csv)
        with Path(args.calibration_json).open("r", encoding="utf-8") as handle:
            calibration_metadata = json.load(handle)
        calibration_horizon = calibration_metadata.get("horizon_days")
        if (
            calibration_horizon is not None
            and int(calibration_horizon) != args.horizon_days
        ):
            raise ValueError(
                "calibration horizon does not match requested horizon: "
                f"{calibration_horizon} != {args.horizon_days}"
            )
        calibration = load_demand_scenario_calibration(args.calibration_json)
        batch = build_skc_decision_batch(
            prediction,
            inventory,
            economics,
            calibration,
            policy=args.policy,
            horizon_days=args.horizon_days,
            min_batch_qty=args.min_batch_qty,
            increment_batch_qty=args.increment_batch_qty,
            max_fallback_cost_rate=args.max_fallback_cost_rate,
        )
    except ProfitAnalysisQualityError as exc:
        manifest["status"] = "failed_quality_gate"
        manifest["error"] = str(exc)
        _write_json(quality_path, exc.quality_report)
        _write_json(manifest_path, manifest)
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        _write_json(manifest_path, manifest)
        raise

    skc_path = output_dir / f"skc_recommendations_{stamp}.csv"
    sku_path = output_dir / f"sku_allocations_{stamp}.csv"
    detail_path = output_dir / f"recommendation_details_{stamp}.json"
    batch.skc_recommendations.to_csv(
        skc_path, index=False, encoding="utf-8-sig"
    )
    batch.sku_allocations.to_csv(
        sku_path, index=False, encoding="utf-8-sig"
    )
    _write_json(detail_path, batch.recommendation_details)
    _write_json(quality_path, batch.quality_report)
    manifest.update(
        {
            "status": "passed",
            "skc_recommendations": str(skc_path.resolve()),
            "sku_allocations": str(sku_path.resolve()),
            "recommendation_details": str(detail_path.resolve()),
            "quality_report": str(quality_path.resolve()),
            "skc_rows": int(len(batch.skc_recommendations)),
            "sku_rows": int(len(batch.sku_allocations)),
        }
    )
    _write_json(manifest_path, manifest)

    print(f"[OK] SKC recommendations -> {skc_path}")
    print(f"[OK] SKU allocations      -> {sku_path}")
    print(f"[OK] quality report       -> {quality_path}")
    print(
        "[OK] positive SKC plans   -> "
        f"{batch.quality_report['positive_plan_skc_rows']}"
    )


if __name__ == "__main__":
    main()
