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
        "inventory_snapshot_present": 0,
        "inventory_source_date": pd.NA,
        "inbound_source_date": pd.NA,
        "lead_time_days": 0,
        "min_batch_qty": pd.NA,
        "increment_batch_qty": pd.NA,
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
        "launch_date": pd.NA,
        "lifecycle_end_date": pd.NA,
        "target_sell_through_rate": 0.85,
        "lifecycle_days": 45,
        "cost_source": pd.NA,
        "cost_record_count": 0,
        "cost_conflict_flag": 0,
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
    inventory_join_keys = ["sku_id"]
    if "snapshot_date" in prediction_df.columns and "snapshot_date" in inventory_df.columns:
        inventory_join_keys.append("snapshot_date")
    merged = prediction_df.merge(
        inventory_df,
        on=inventory_join_keys,
        how="inner",
        suffixes=("_pred", "_inv"),
    )
    merged = merged.merge(economics_df, on="sku_id", how="inner")

    if "snapshot_date_pred" in merged.columns:
        merged["snapshot_date"] = merged["snapshot_date_pred"]
    elif "snapshot_date" not in merged.columns and "snapshot_date_inv" in merged.columns:
        merged["snapshot_date"] = merged["snapshot_date_inv"]

    for field in [
        "style_id",
        "category",
        "launch_date",
        "lifecycle_end_date",
        "cost_source",
        "cost_record_count",
        "cost_conflict_flag",
    ]:
        if field in merged.columns:
            continue
        candidates = [f"{field}_x", f"{field}_y", f"{field}_pred", f"{field}_inv"]
        available = [col for col in candidates if col in merged.columns]
        if available:
            merged[field] = merged[available[0]]
            for source_col in available[1:]:
                missing = merged[field].isna()
                if pd.api.types.is_object_dtype(merged[field]) or pd.api.types.is_string_dtype(
                    merged[field]
                ):
                    missing |= merged[field].fillna("").astype(str).str.strip().eq("")
                merged.loc[missing, field] = merged.loc[missing, source_col]

    return merged
