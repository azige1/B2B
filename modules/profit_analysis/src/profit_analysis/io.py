from __future__ import annotations

from pathlib import Path

import pandas as pd


def _ensure_columns(df: pd.DataFrame, required: list[str], label: str) -> pd.DataFrame:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")
    return df


def load_prediction_snapshot(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["sku_id", "snapshot_date", "pred_prob_positive", "pred_qty_30d"]
    return _ensure_columns(df, required, "prediction_snapshot").copy()


def load_inventory_snapshot(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["sku_id", "snapshot_date", "current_inventory"]
    df = _ensure_columns(df, required, "inventory_snapshot").copy()

    optional_defaults = {
        "inbound_within_30d": 0.0,
        "lead_time_days": 0,
        "min_batch_qty": pd.NA,
        "max_replenish_qty": pd.NA,
        "safety_stock_qty": pd.NA,
        "last_decision_date": pd.NA,
    }
    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def load_economics_config(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = [
        "sku_id",
        "unit_cost",
        "unit_price",
        "holding_cost_per_unit_per_day",
        "salvage_value_per_unit",
    ]
    df = _ensure_columns(df, required, "economics_config").copy()

    optional_defaults = {
        "stockout_penalty_per_unit": 0.0,
        "other_fixed_cost": 0.0,
        "lifecycle_end_date": pd.NA,
    }
    for col, default in optional_defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def build_profit_input_frame(
    prediction_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    economics_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = prediction_df.merge(inventory_df, on="sku_id", how="inner", suffixes=("_pred", "_inv"))
    merged = merged.merge(economics_df, on="sku_id", how="inner")

    if "snapshot_date_pred" in merged.columns:
        merged["snapshot_date"] = merged["snapshot_date_pred"]
    elif "snapshot_date" not in merged.columns and "snapshot_date_inv" in merged.columns:
        merged["snapshot_date"] = merged["snapshot_date_inv"]

    return merged
