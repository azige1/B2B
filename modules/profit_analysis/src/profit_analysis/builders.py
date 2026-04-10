from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_FIELDS = [
    "lead_time_days",
    "min_batch_qty",
    "max_replenish_qty",
    "safety_stock_ratio",
    "unit_price_ratio_to_price_tag",
    "unit_cost_ratio_to_price_tag",
    "holding_cost_ratio_per_day_to_unit_cost",
    "salvage_ratio_to_unit_cost",
    "stockout_penalty_per_unit",
    "other_fixed_cost",
]

DEFAULT_DEFAULTS = {
    "lead_time_days": 7.0,
    "min_batch_qty": 1.0,
    "max_replenish_qty": pd.NA,
    "safety_stock_ratio": 0.0,
    "unit_price_ratio_to_price_tag": 1.0,
    "unit_cost_ratio_to_price_tag": 0.35,
    "holding_cost_ratio_per_day_to_unit_cost": 0.001,
    "salvage_ratio_to_unit_cost": 0.30,
    "stockout_penalty_per_unit": 0.0,
    "other_fixed_cost": 0.0,
}


@dataclass(frozen=True)
class PredictionColumnSpec:
    sku_id_col: str = "sku_id"
    snapshot_date_col: str = "snapshot_date"
    prob_col: str = "pred_prob_positive"
    qty_col: str = "pred_qty_30d"
    prediction_version_col: str | None = None
    prediction_version: str | None = None


def infer_prediction_column_spec(
    prediction_df: pd.DataFrame,
    prediction_version: str | None = None,
) -> PredictionColumnSpec:
    column_sets = {
        "sku_id_col": ["sku_id", "SKU_ID"],
        "snapshot_date_col": ["snapshot_date", "anchor_date", "date"],
        "prob_col": ["pred_prob_positive", "ai_pred_prob", "pred_prob", "prob"],
        "qty_col": ["pred_qty_30d", "ai_pred_qty", "pred_qty", "qty_pred"],
        "prediction_version_col": ["prediction_version", "exp_id", "model_id"],
    }

    resolved: dict[str, str | None] = {}
    cols = set(prediction_df.columns)
    for key, candidates in column_sets.items():
        match = next((name for name in candidates if name in cols), None)
        resolved[key] = match

    required_keys = ["sku_id_col", "snapshot_date_col", "prob_col", "qty_col"]
    missing = [key for key in required_keys if not resolved.get(key)]
    if missing:
        raise ValueError(
            "Could not infer prediction columns automatically. "
            f"Missing mappings for: {missing}. "
            f"Available columns: {list(prediction_df.columns)}"
        )

    return PredictionColumnSpec(
        sku_id_col=str(resolved["sku_id_col"]),
        snapshot_date_col=str(resolved["snapshot_date_col"]),
        prob_col=str(resolved["prob_col"]),
        qty_col=str(resolved["qty_col"]),
        prediction_version_col=resolved["prediction_version_col"],
        prediction_version=prediction_version,
    )


def infer_actual_qty_col(prediction_df: pd.DataFrame) -> str:
    candidates = [
        "true_replenish_qty",
        "actual_qty_30d",
        "true_qty",
        "actual_qty",
        "qty_true",
    ]
    cols = set(prediction_df.columns)
    match = next((name for name in candidates if name in cols), None)
    if not match:
        raise ValueError(
            "Could not infer actual demand column automatically. "
            f"Available columns: {list(prediction_df.columns)}"
        )
    return match


def load_policy_defaults(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["scope_type", "scope_key"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"policy defaults missing required columns: {missing}")

    out = df.copy()
    out["scope_type"] = out["scope_type"].astype(str).str.lower()
    out["scope_key"] = out["scope_key"].fillna("*").astype(str)
    for field in DEFAULT_FIELDS:
        if field not in out.columns:
            out[field] = pd.NA
    return out


def normalize_prediction_snapshot(
    prediction_df: pd.DataFrame,
    spec: PredictionColumnSpec | None = None,
) -> pd.DataFrame:
    spec = spec or PredictionColumnSpec()
    required = [
        spec.sku_id_col,
        spec.snapshot_date_col,
        spec.prob_col,
        spec.qty_col,
    ]
    missing = [col for col in required if col not in prediction_df.columns]
    if missing:
        raise ValueError(f"prediction input missing required columns: {missing}")

    out = prediction_df.copy().rename(
        columns={
            spec.sku_id_col: "sku_id",
            spec.snapshot_date_col: "snapshot_date",
            spec.prob_col: "pred_prob_positive",
            spec.qty_col: "pred_qty_30d",
        }
    )
    keep_cols = ["sku_id", "snapshot_date", "pred_prob_positive", "pred_qty_30d"]
    if spec.prediction_version_col and spec.prediction_version_col in out.columns:
        out = out.rename(columns={spec.prediction_version_col: "prediction_version"})
        keep_cols.append("prediction_version")
    else:
        out["prediction_version"] = spec.prediction_version
        keep_cols.append("prediction_version")

    out = out.loc[:, keep_cols].copy()
    out["sku_id"] = out["sku_id"].astype(str)
    out["snapshot_date"] = pd.to_datetime(out["snapshot_date"]).dt.strftime("%Y-%m-%d")
    out["pred_prob_positive"] = pd.to_numeric(out["pred_prob_positive"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["pred_qty_30d"] = pd.to_numeric(out["pred_qty_30d"], errors="coerce").fillna(0.0).clip(lower=0.0)
    return out


def _latest_inventory(clean_inventory_df: pd.DataFrame) -> pd.DataFrame:
    required = ["sku_id", "qty_stock", "inventory_date"]
    missing = [col for col in required if col not in clean_inventory_df.columns]
    if missing:
        raise ValueError(f"clean_inventory input missing required columns: {missing}")

    out = clean_inventory_df.copy()
    out["inventory_date"] = pd.to_datetime(out["inventory_date"], errors="coerce")
    out["qty_stock"] = pd.to_numeric(out["qty_stock"], errors="coerce").fillna(0.0)
    out = out.sort_values(["sku_id", "inventory_date"]).drop_duplicates("sku_id", keep="last")
    return out.loc[:, ["sku_id", "qty_stock", "inventory_date"]].rename(
        columns={
            "qty_stock": "current_inventory",
            "inventory_date": "inventory_source_date",
        }
    )


def _latest_inbound_proxy(wide_table_df: pd.DataFrame | None) -> pd.DataFrame:
    if wide_table_df is None:
        return pd.DataFrame(columns=["sku_id", "inbound_within_30d", "inbound_source_date"])

    required = ["sku_id", "date", "qty_inbound"]
    missing = [col for col in required if col not in wide_table_df.columns]
    if missing:
        raise ValueError(f"wide_table input missing required columns: {missing}")

    out = wide_table_df.loc[:, required].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["qty_inbound"] = pd.to_numeric(out["qty_inbound"], errors="coerce").fillna(0.0)
    out = (
        out.groupby(["sku_id", "date"], as_index=False)["qty_inbound"]
        .sum()
        .sort_values(["sku_id", "date"])
        .drop_duplicates("sku_id", keep="last")
    )
    return out.rename(
        columns={
            "qty_inbound": "inbound_within_30d",
            "date": "inbound_source_date",
        }
    )


def _prepare_product_lookup(product_df: pd.DataFrame) -> pd.DataFrame:
    required = ["sku_id", "style_id", "category", "price_tag"]
    missing = [col for col in required if col not in product_df.columns]
    if missing:
        raise ValueError(f"product input missing required columns: {missing}")

    base_cols = ["sku_id", "style_id", "category", "price_tag"]
    optional = ["sub_category", "product_name", "season", "band", "series", "qty_first_order"]
    keep = base_cols + [col for col in optional if col in product_df.columns]
    out = product_df.loc[:, keep].copy()
    out["sku_id"] = out["sku_id"].astype(str)
    out["style_id"] = out["style_id"].astype(str)
    out["category"] = out["category"].fillna("UNKNOWN").astype(str)
    out["price_tag"] = pd.to_numeric(out["price_tag"], errors="coerce").fillna(0.0)
    return out.drop_duplicates("sku_id", keep="last")


def _prepare_lifecycle_lookup(lifecycle_df: pd.DataFrame | None) -> pd.DataFrame:
    if lifecycle_df is None:
        return pd.DataFrame(columns=["sku_id", "lifecycle_end_date"])

    required = ["NO", "PL_CYCLE", "LISTING_DATE"]
    missing = [col for col in required if col not in lifecycle_df.columns]
    if missing:
        raise ValueError(f"lifecycle input missing required columns: {missing}")

    out = lifecycle_df.loc[:, required].copy().rename(columns={"NO": "sku_id"})
    out["sku_id"] = out["sku_id"].astype(str)
    out["PL_CYCLE"] = pd.to_numeric(out["PL_CYCLE"], errors="coerce")
    out["LISTING_DATE"] = pd.to_datetime(out["LISTING_DATE"], errors="coerce")
    out["lifecycle_end_date"] = out["LISTING_DATE"] + pd.to_timedelta(out["PL_CYCLE"], unit="D")
    out["lifecycle_end_date"] = out["lifecycle_end_date"].dt.strftime("%Y-%m-%d")
    return out.loc[:, ["sku_id", "lifecycle_end_date"]].drop_duplicates("sku_id", keep="last")


def _defaults_index(defaults_df: pd.DataFrame | None) -> dict[tuple[str, str], dict]:
    if defaults_df is None or defaults_df.empty:
        return {}
    lookup: dict[tuple[str, str], dict] = {}
    for row in defaults_df.to_dict(orient="records"):
        key = (str(row["scope_type"]).lower(), str(row["scope_key"]))
        lookup[key] = row
    return lookup


def _resolve_defaults(lookup: dict[tuple[str, str], dict], sku_id: str, style_id: str, category: str) -> dict:
    merged = dict(DEFAULT_DEFAULTS)
    keys = [
        ("global", "*"),
        ("category", category),
        ("style_id", style_id),
        ("sku_id", sku_id),
    ]
    for key in keys:
        row = lookup.get(key)
        if not row:
            continue
        for field in DEFAULT_FIELDS:
            value = row.get(field)
            if pd.notna(value):
                merged[field] = value
    return merged


def build_inventory_snapshot(
    prediction_df: pd.DataFrame,
    clean_inventory_df: pd.DataFrame,
    product_df: pd.DataFrame,
    defaults_df: pd.DataFrame | None = None,
    wide_table_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pred = prediction_df.copy()
    products = _prepare_product_lookup(product_df)
    inventory = _latest_inventory(clean_inventory_df)
    inbound = _latest_inbound_proxy(wide_table_df)
    defaults_lookup = _defaults_index(defaults_df)

    work = pred.merge(products.loc[:, ["sku_id", "style_id", "category"]], on="sku_id", how="left")
    work = work.merge(inventory, on="sku_id", how="left")
    work = work.merge(inbound, on="sku_id", how="left")
    work["current_inventory"] = pd.to_numeric(work["current_inventory"], errors="coerce").fillna(0.0)
    work["inbound_within_30d"] = pd.to_numeric(work["inbound_within_30d"], errors="coerce").fillna(0.0)
    work["style_id"] = work["style_id"].fillna("").astype(str)
    work["category"] = work["category"].fillna("UNKNOWN").astype(str)

    rows = []
    for row in work.to_dict(orient="records"):
        defaults = _resolve_defaults(
            defaults_lookup,
            sku_id=str(row["sku_id"]),
            style_id=str(row["style_id"]),
            category=str(row["category"]),
        )
        safety_stock_qty = round(float(row["pred_qty_30d"]) * float(defaults["safety_stock_ratio"]))
        rows.append(
            {
                "sku_id": row["sku_id"],
                "snapshot_date": row["snapshot_date"],
                "current_inventory": float(row["current_inventory"]),
                "inbound_within_30d": float(row["inbound_within_30d"]),
                "lead_time_days": int(round(float(defaults["lead_time_days"]))),
                "min_batch_qty": float(defaults["min_batch_qty"]),
                "max_replenish_qty": defaults["max_replenish_qty"],
                "safety_stock_qty": float(safety_stock_qty),
                "last_decision_date": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def build_economics_config(
    prediction_df: pd.DataFrame,
    product_df: pd.DataFrame,
    defaults_df: pd.DataFrame | None = None,
    lifecycle_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pred = prediction_df.copy()
    products = _prepare_product_lookup(product_df)
    lifecycle = _prepare_lifecycle_lookup(lifecycle_df)
    defaults_lookup = _defaults_index(defaults_df)

    work = pred.merge(products, on="sku_id", how="left")
    work = work.merge(lifecycle, on="sku_id", how="left")
    work["style_id"] = work["style_id"].fillna("").astype(str)
    work["category"] = work["category"].fillna("UNKNOWN").astype(str)
    work["price_tag"] = pd.to_numeric(work["price_tag"], errors="coerce").fillna(0.0)

    rows = []
    for row in work.to_dict(orient="records"):
        defaults = _resolve_defaults(
            defaults_lookup,
            sku_id=str(row["sku_id"]),
            style_id=str(row["style_id"]),
            category=str(row["category"]),
        )
        unit_price = float(row["price_tag"]) * float(defaults["unit_price_ratio_to_price_tag"])
        unit_cost = float(row["price_tag"]) * float(defaults["unit_cost_ratio_to_price_tag"])
        holding_cost = unit_cost * float(defaults["holding_cost_ratio_per_day_to_unit_cost"])
        salvage_value = unit_cost * float(defaults["salvage_ratio_to_unit_cost"])
        rows.append(
            {
                "sku_id": row["sku_id"],
                "unit_cost": float(unit_cost),
                "unit_price": float(unit_price),
                "holding_cost_per_unit_per_day": float(holding_cost),
                "salvage_value_per_unit": float(salvage_value),
                "stockout_penalty_per_unit": float(defaults["stockout_penalty_per_unit"]),
                "other_fixed_cost": float(defaults["other_fixed_cost"]),
                "lifecycle_end_date": row.get("lifecycle_end_date", pd.NA),
            }
        )
    return pd.DataFrame(rows)
