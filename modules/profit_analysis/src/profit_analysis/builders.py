from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_FIELDS = [
    "lead_time_days",
    "min_batch_qty",
    "increment_batch_qty",
    "max_replenish_qty",
    "safety_stock_ratio",
    "unit_price_ratio_to_price_tag",
    "unit_cost_ratio_to_price_tag",
    "holding_cost_ratio_per_day_to_unit_cost",
    "salvage_ratio_to_unit_cost",
    "stockout_penalty_per_unit",
    "other_fixed_cost",
    "target_sell_through_rate",
    "lifecycle_days_assumption",
    "cost_fallback_rule",
]

DEFAULT_DEFAULTS = {
    "lead_time_days": 7.0,
    "min_batch_qty": 100.0,
    "increment_batch_qty": 10.0,
    "max_replenish_qty": pd.NA,
    "safety_stock_ratio": 0.0,
    "unit_price_ratio_to_price_tag": 1.0,
    "unit_cost_ratio_to_price_tag": 1.0 / 7.0,
    "holding_cost_ratio_per_day_to_unit_cost": 0.0,
    "salvage_ratio_to_unit_cost": 0.0,
    "stockout_penalty_per_unit": 0.0,
    "other_fixed_cost": 0.0,
    "target_sell_through_rate": 0.85,
    "lifecycle_days_assumption": 45.0,
    "cost_fallback_rule": "price_tag_div_7_when_cost_missing",
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
        "snapshot_date_col": ["snapshot_date", "anchor_date", "date", "锚点日期"],
        "prob_col": ["pred_prob_positive", "ai_pred_prob", "pred_prob", "prob", "模型补货概率"],
        "qty_col": ["pred_qty_30d", "ai_pred_qty", "pred_qty", "qty_pred", "最终预测量"],
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
        "验证期实际补货量",
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


def _as_of_rows(
    prediction_df: pd.DataFrame,
    source_df: pd.DataFrame,
    source_date_col: str,
    value_col: str,
    output_value_col: str,
    output_date_col: str,
    present_col: str | None = None,
    output_present_col: str | None = None,
) -> pd.DataFrame:
    predictions = prediction_df.loc[:, ["sku_id", "snapshot_date"]].drop_duplicates().copy()
    predictions["sku_id"] = predictions["sku_id"].astype(str)
    predictions["snapshot_date"] = pd.to_datetime(predictions["snapshot_date"], errors="coerce")

    source = source_df.loc[
        :,
        ["sku_id", source_date_col, value_col] + ([present_col] if present_col else []),
    ].copy()
    source["sku_id"] = source["sku_id"].astype(str)
    source[source_date_col] = pd.to_datetime(source[source_date_col], errors="coerce")
    source[value_col] = pd.to_numeric(source[value_col], errors="coerce").fillna(0.0)
    if present_col:
        source[present_col] = pd.to_numeric(source[present_col], errors="coerce").fillna(0.0)
    source = source.dropna(subset=[source_date_col])

    aggregations = {value_col: "max"}
    if present_col:
        aggregations[present_col] = "max"
    source = source.groupby(["sku_id", source_date_col], as_index=False).agg(aggregations)

    resolved = []
    for snapshot_date, pred_rows in predictions.groupby("snapshot_date", dropna=False):
        block = pred_rows.copy()
        if pd.isna(snapshot_date):
            block[output_value_col] = 0.0
            block[output_date_col] = pd.NaT
            if output_present_col and present_col:
                block[output_present_col] = 0
            resolved.append(block)
            continue

        eligible = source[source[source_date_col] <= snapshot_date]
        eligible = (
            eligible.sort_values(["sku_id", source_date_col])
            .drop_duplicates("sku_id", keep="last")
            .rename(
                columns={
                    value_col: output_value_col,
                    source_date_col: output_date_col,
                    **({present_col: output_present_col} if present_col and output_present_col else {}),
                }
            )
        )
        keep = ["sku_id", output_value_col, output_date_col]
        if output_present_col and present_col:
            keep.append(output_present_col)
        resolved.append(block.merge(eligible.loc[:, keep], on="sku_id", how="left"))

    out = pd.concat(resolved, ignore_index=True)
    out[output_value_col] = pd.to_numeric(out[output_value_col], errors="coerce").fillna(0.0)
    if output_present_col and present_col:
        out[output_present_col] = (
            pd.to_numeric(out[output_present_col], errors="coerce").fillna(0).astype(int)
        )
    out["snapshot_date"] = out["snapshot_date"].dt.strftime("%Y-%m-%d")
    out[output_date_col] = pd.to_datetime(out[output_date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    return out


def _inventory_as_of_predictions(
    prediction_df: pd.DataFrame,
    clean_inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    if {"inventory_date", "qty_stock"}.issubset(clean_inventory_df.columns):
        return _as_of_rows(
            prediction_df=prediction_df,
            source_df=clean_inventory_df,
            source_date_col="inventory_date",
            value_col="qty_stock",
            output_value_col="current_inventory",
            output_date_col="inventory_source_date",
        ).assign(inventory_snapshot_present=lambda df: df["inventory_source_date"].notna().astype(int))

    if {"date", "qty_total_stock"}.issubset(clean_inventory_df.columns):
        present_col = "snapshot_present" if "snapshot_present" in clean_inventory_df.columns else None
        out = _as_of_rows(
            prediction_df=prediction_df,
            source_df=clean_inventory_df,
            source_date_col="date",
            value_col="qty_total_stock",
            output_value_col="current_inventory",
            output_date_col="inventory_source_date",
            present_col=present_col,
            output_present_col="inventory_snapshot_present",
        )
        if present_col is None:
            out["inventory_snapshot_present"] = out["inventory_source_date"].notna().astype(int)
        return out

    raise ValueError(
        "inventory input must contain either "
        "['sku_id', 'inventory_date', 'qty_stock'] or "
        "['sku_id', 'date', 'qty_total_stock']"
    )


def _inbound_as_of_predictions(
    prediction_df: pd.DataFrame,
    wide_table_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if wide_table_df is None:
        out = prediction_df.loc[:, ["sku_id", "snapshot_date"]].drop_duplicates().copy()
        out["inbound_within_30d"] = 0.0
        out["inbound_source_date"] = pd.NA
        return out

    required = ["sku_id", "date", "qty_inbound"]
    missing = [col for col in required if col not in wide_table_df.columns]
    if missing:
        raise ValueError(f"wide_table input missing required columns: {missing}")

    daily = (
        wide_table_df.loc[:, required]
        .assign(qty_inbound=lambda df: pd.to_numeric(df["qty_inbound"], errors="coerce").fillna(0.0))
        .groupby(["sku_id", "date"], as_index=False)["qty_inbound"]
        .sum()
    )
    return _as_of_rows(
        prediction_df=prediction_df,
        source_df=daily,
        source_date_col="date",
        value_col="qty_inbound",
        output_value_col="inbound_within_30d",
        output_date_col="inbound_source_date",
    )


def _prepare_product_lookup(product_df: pd.DataFrame) -> pd.DataFrame:
    required = ["sku_id", "style_id", "category", "price_tag"]
    missing = [col for col in required if col not in product_df.columns]
    if missing:
        raise ValueError(f"product input missing required columns: {missing}")

    base_cols = ["sku_id", "style_id", "category", "price_tag"]
    optional = [
        "sub_category",
        "product_name",
        "season",
        "band",
        "series",
        "qty_first_order",
        "unit_cost",
        "cost",
        "standard_cost",
        "purchase_cost",
    ]
    keep = base_cols + [col for col in optional if col in product_df.columns]
    out = product_df.loc[:, keep].copy()
    out["sku_id"] = out["sku_id"].astype(str)
    out["style_id"] = out["style_id"].astype(str)
    out["category"] = out["category"].fillna("UNKNOWN").astype(str)
    out["price_tag"] = pd.to_numeric(out["price_tag"], errors="coerce").fillna(0.0)
    for cost_col in ["unit_cost", "cost", "standard_cost", "purchase_cost"]:
        if cost_col in out.columns:
            out[cost_col] = pd.to_numeric(out[cost_col], errors="coerce")
    return out.drop_duplicates("sku_id", keep="last")


def _prepare_style_cost_lookup(style_cost_df: pd.DataFrame | None) -> pd.DataFrame:
    columns = [
        "style_id",
        "style_unit_cost",
        "style_cost_source",
        "cost_record_count",
        "cost_conflict_flag",
    ]
    if style_cost_df is None or style_cost_df.empty:
        return pd.DataFrame(columns=columns)
    required = ["style_id", "unit_cost"]
    missing = [col for col in required if col not in style_cost_df.columns]
    if missing:
        raise ValueError(f"style cost input missing required columns: {missing}")

    out = style_cost_df.copy()
    out["style_id"] = out["style_id"].astype("string").str.strip().str.upper()
    out["unit_cost"] = pd.to_numeric(out["unit_cost"], errors="coerce")
    out = out[out["style_id"].notna() & (out["unit_cost"] > 0)].copy()
    if "cost_source" not in out.columns:
        out["cost_source"] = "client_style_cost"
    if "cost_record_count" not in out.columns:
        out["cost_record_count"] = 1
    if "cost_conflict_flag" not in out.columns:
        out["cost_conflict_flag"] = 0
    out = out.sort_values("unit_cost").drop_duplicates("style_id", keep="last")
    return out.rename(
        columns={
            "unit_cost": "style_unit_cost",
            "cost_source": "style_cost_source",
        }
    ).loc[:, columns]


def _prepare_lifecycle_lookup(lifecycle_df: pd.DataFrame | None) -> pd.DataFrame:
    if lifecycle_df is None:
        return pd.DataFrame(columns=["sku_id", "launch_date", "lifecycle_end_date", "lifecycle_days"])

    df = lifecycle_df.copy()
    rename_map = {}
    if "NO" in df.columns and "sku_id" not in df.columns:
        rename_map["NO"] = "sku_id"
    if "LISTING_DATE" in df.columns and "launch_date" not in df.columns:
        rename_map["LISTING_DATE"] = "launch_date"
    if "PL_CYCLE" in df.columns and "lifecycle_days" not in df.columns:
        rename_map["PL_CYCLE"] = "lifecycle_days"
    if "estimated_lifecycle_end_date" in df.columns and "lifecycle_end_date" not in df.columns:
        rename_map["estimated_lifecycle_end_date"] = "lifecycle_end_date"
    if "lifecycle_days_assumption" in df.columns and "lifecycle_days" not in df.columns:
        rename_map["lifecycle_days_assumption"] = "lifecycle_days"
    df = df.rename(columns=rename_map)

    if "sku_id" not in df.columns:
        raise ValueError(
            "lifecycle input missing required key column: sku_id. "
            "Use src/analysis/build_launch_date_lifecycle_v0_features.py to expand style-level launch dates to SKU level."
        )
    if "launch_date" not in df.columns and "lifecycle_end_date" not in df.columns:
        raise ValueError(
            "lifecycle input must include launch_date, lifecycle_end_date, "
            "or legacy LISTING_DATE/PL_CYCLE columns."
        )

    keep_cols = [
        col
        for col in ["sku_id", "launch_date", "lifecycle_end_date", "lifecycle_days"]
        if col in df.columns
    ]
    out = df.loc[:, keep_cols].copy()
    out["sku_id"] = out["sku_id"].astype(str)
    if "launch_date" in out.columns:
        out["launch_date"] = pd.to_datetime(out["launch_date"], errors="coerce")
    else:
        out["launch_date"] = pd.NaT
    if "lifecycle_days" in out.columns:
        out["lifecycle_days"] = pd.to_numeric(out["lifecycle_days"], errors="coerce")
    else:
        out["lifecycle_days"] = pd.NA

    if "lifecycle_end_date" in out.columns:
        out["lifecycle_end_date"] = pd.to_datetime(out["lifecycle_end_date"], errors="coerce")
    else:
        out["lifecycle_end_date"] = pd.NaT

    needs_end = out["lifecycle_end_date"].isna() & out["launch_date"].notna() & out["lifecycle_days"].notna()
    out.loc[needs_end, "lifecycle_end_date"] = (
        out.loc[needs_end, "launch_date"]
        + pd.to_timedelta(out.loc[needs_end, "lifecycle_days"].astype(int) - 1, unit="D")
    )

    needs_days = out["lifecycle_days"].isna() & out["launch_date"].notna() & out["lifecycle_end_date"].notna()
    out.loc[needs_days, "lifecycle_days"] = (
        out.loc[needs_days, "lifecycle_end_date"] - out.loc[needs_days, "launch_date"]
    ).dt.days + 1

    out["launch_date"] = out["launch_date"].dt.strftime("%Y-%m-%d")
    out["lifecycle_end_date"] = out["lifecycle_end_date"].dt.strftime("%Y-%m-%d")
    return out.loc[:, ["sku_id", "launch_date", "lifecycle_end_date", "lifecycle_days"]].drop_duplicates("sku_id", keep="last")


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
    inventory = _inventory_as_of_predictions(pred, clean_inventory_df)
    inbound = _inbound_as_of_predictions(pred, wide_table_df)
    defaults_lookup = _defaults_index(defaults_df)

    work = pred.merge(products.loc[:, ["sku_id", "style_id", "category"]], on="sku_id", how="left")
    work = work.merge(inventory, on=["sku_id", "snapshot_date"], how="left")
    work = work.merge(inbound, on=["sku_id", "snapshot_date"], how="left")
    work["current_inventory"] = pd.to_numeric(work["current_inventory"], errors="coerce").fillna(0.0)
    work["inbound_within_30d"] = pd.to_numeric(work["inbound_within_30d"], errors="coerce").fillna(0.0)
    work["inventory_snapshot_present"] = (
        pd.to_numeric(work["inventory_snapshot_present"], errors="coerce").fillna(0).astype(int)
    )
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
                "style_id": row["style_id"],
                "category": row["category"],
                "snapshot_date": row["snapshot_date"],
                "current_inventory": float(row["current_inventory"]),
                "inbound_within_30d": float(row["inbound_within_30d"]),
                "inventory_snapshot_present": int(row["inventory_snapshot_present"]),
                "inventory_source_date": row.get("inventory_source_date", pd.NA),
                "inbound_source_date": row.get("inbound_source_date", pd.NA),
                "lead_time_days": int(round(float(defaults["lead_time_days"]))),
                "min_batch_qty": float(defaults["min_batch_qty"]),
                "increment_batch_qty": float(defaults["increment_batch_qty"]),
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
    style_cost_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pred = prediction_df.copy()
    products = _prepare_product_lookup(product_df)
    lifecycle = _prepare_lifecycle_lookup(lifecycle_df)
    style_costs = _prepare_style_cost_lookup(style_cost_df)
    defaults_lookup = _defaults_index(defaults_df)

    work = pred.merge(products, on="sku_id", how="left")
    work["style_id"] = work["style_id"].fillna("").astype(str).str.strip().str.upper()
    work = work.merge(style_costs, on="style_id", how="left")
    work = work.merge(lifecycle, on="sku_id", how="left")
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
        explicit_cost = pd.NA
        for cost_col in ["unit_cost", "cost", "standard_cost", "purchase_cost"]:
            if cost_col in row and pd.notna(row[cost_col]) and float(row[cost_col]) > 0:
                explicit_cost = float(row[cost_col])
                break
        style_unit_cost = (
            float(row["style_unit_cost"])
            if "style_unit_cost" in row
            and pd.notna(row["style_unit_cost"])
            and float(row["style_unit_cost"]) > 0
            else pd.NA
        )
        unit_cost = (
            float(explicit_cost)
            if pd.notna(explicit_cost)
            else float(style_unit_cost)
            if pd.notna(style_unit_cost)
            else float(row["price_tag"]) * float(defaults["unit_cost_ratio_to_price_tag"])
        )
        cost_source = (
            "product_explicit_cost"
            if pd.notna(explicit_cost)
            else str(row.get("style_cost_source", "client_style_cost"))
            if pd.notna(style_unit_cost)
            else str(defaults["cost_fallback_rule"])
        )
        holding_cost = unit_cost * float(defaults["holding_cost_ratio_per_day_to_unit_cost"])
        salvage_value = unit_cost * float(defaults["salvage_ratio_to_unit_cost"])
        lifecycle_days = (
            int(round(float(row["lifecycle_days"])))
            if "lifecycle_days" in row and pd.notna(row["lifecycle_days"])
            else int(round(float(defaults["lifecycle_days_assumption"])))
        )
        rows.append(
            {
                "sku_id": row["sku_id"],
                "style_id": row["style_id"],
                "category": row["category"],
                "unit_cost": float(unit_cost),
                "unit_price": float(unit_price),
                "holding_cost_per_unit_per_day": float(holding_cost),
                "salvage_value_per_unit": float(salvage_value),
                "stockout_penalty_per_unit": float(defaults["stockout_penalty_per_unit"]),
                "other_fixed_cost": float(defaults["other_fixed_cost"]),
                "launch_date": row.get("launch_date", pd.NA),
                "lifecycle_end_date": row.get("lifecycle_end_date", pd.NA),
                "target_sell_through_rate": float(defaults["target_sell_through_rate"]),
                "lifecycle_days": max(lifecycle_days, 1),
                "cost_source": cost_source,
                "cost_record_count": int(row.get("cost_record_count", 0) or 0)
                if pd.notna(row.get("cost_record_count", pd.NA))
                else 0,
                "cost_conflict_flag": int(row.get("cost_conflict_flag", 0) or 0)
                if pd.notna(row.get("cost_conflict_flag", pd.NA))
                else 0,
            }
        )
    return pd.DataFrame(rows).drop_duplicates("sku_id", keep="last")
