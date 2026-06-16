from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def prepare_daily_demand(
    demand_df: pd.DataFrame,
    date_col: str = "date",
    sku_col: str = "sku_id",
    quantity_col: str = "qty_replenish",
) -> pd.DataFrame:
    required = [date_col, sku_col, quantity_col]
    missing = [col for col in required if col not in demand_df.columns]
    if missing:
        raise ValueError(f"demand input missing required columns: {missing}")

    daily = demand_df.loc[:, required].copy()
    daily = daily.rename(
        columns={
            date_col: "date",
            sku_col: "sku_id",
            quantity_col: "demand_qty",
        }
    )
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily["sku_id"] = daily["sku_id"].astype(str)
    daily["demand_qty"] = pd.to_numeric(
        daily["demand_qty"], errors="coerce"
    ).fillna(0.0).clip(lower=0.0)
    daily = daily.dropna(subset=["date"])
    return daily.groupby(["date", "sku_id"], as_index=False)["demand_qty"].sum()


def build_horizon_demand(
    context_df: pd.DataFrame,
    daily_demand_df: pd.DataFrame,
    horizon_days: int,
    anchor_col: str = "anchor_date",
    sku_col: str = "sku_id",
) -> pd.DataFrame:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    required = [anchor_col, sku_col]
    missing = [col for col in required if col not in context_df.columns]
    if missing:
        raise ValueError(f"context input missing required columns: {missing}")

    context = context_df.loc[:, required].copy()
    context = context.rename(columns={anchor_col: "anchor_date", sku_col: "sku_id"})
    context["anchor_date"] = pd.to_datetime(context["anchor_date"], errors="coerce")
    context["sku_id"] = context["sku_id"].astype(str)
    if context["anchor_date"].isna().any():
        raise ValueError("context input contains invalid anchor dates.")

    daily = prepare_daily_demand(
        daily_demand_df,
        date_col="date",
        sku_col="sku_id",
        quantity_col=(
            "demand_qty"
            if "demand_qty" in daily_demand_df.columns
            else "qty_replenish"
        ),
    )
    rows: list[pd.DataFrame] = []
    for anchor in context["anchor_date"].drop_duplicates().sort_values():
        window = daily[
            (daily["date"] > anchor)
            & (daily["date"] <= anchor + pd.Timedelta(days=horizon_days))
        ]
        totals = window.groupby("sku_id", as_index=False)["demand_qty"].sum()
        anchor_context = context[context["anchor_date"] == anchor].copy()
        anchor_context = anchor_context.merge(totals, on="sku_id", how="left")
        anchor_context["demand_qty"] = anchor_context["demand_qty"].fillna(0.0)
        rows.append(anchor_context)
    return pd.concat(rows, ignore_index=True) if rows else context.assign(demand_qty=0.0)


def build_daily_demand_curves(
    daily_demand_df: pd.DataFrame,
    anchors: Iterable[str],
    horizon_days: int,
) -> dict[tuple[str, str], list[float]]:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    daily = prepare_daily_demand(
        daily_demand_df,
        date_col="date",
        sku_col="sku_id",
        quantity_col=(
            "demand_qty"
            if "demand_qty" in daily_demand_df.columns
            else "qty_replenish"
        ),
    )
    lookup = {
        (row.date.strftime("%Y-%m-%d"), str(row.sku_id)): float(row.demand_qty)
        for row in daily.itertuples(index=False)
    }
    curves: dict[tuple[str, str], list[float]] = {}
    for anchor_text in anchors:
        anchor = pd.Timestamp(anchor_text)
        skus = daily.loc[
            (daily["date"] > anchor)
            & (daily["date"] <= anchor + pd.Timedelta(days=horizon_days)),
            "sku_id",
        ].unique()
        normalized_anchor = anchor.strftime("%Y-%m-%d")
        for sku_id in skus:
            curves[(normalized_anchor, str(sku_id))] = [
                lookup.get(
                    (
                        (anchor + pd.Timedelta(days=offset)).strftime("%Y-%m-%d"),
                        str(sku_id),
                    ),
                    0.0,
                )
                for offset in range(1, horizon_days + 1)
            ]
    return curves


def build_horizon_calibration_frame(
    prediction_context_df: pd.DataFrame,
    daily_demand_df: pd.DataFrame,
    horizon_days: int,
    source_quantity_horizon_days: int = 30,
) -> pd.DataFrame:
    required = ["anchor_date", "sku_id", "ai_pred_prob", "ai_pred_qty_open"]
    missing = [col for col in required if col not in prediction_context_df.columns]
    if missing:
        raise ValueError(f"prediction context missing required columns: {missing}")
    if source_quantity_horizon_days <= 0:
        raise ValueError("source_quantity_horizon_days must be positive.")

    context = prediction_context_df.loc[:, required].copy()
    context["anchor_date"] = pd.to_datetime(
        context["anchor_date"], errors="coerce"
    )
    context["sku_id"] = context["sku_id"].astype(str)
    if context["anchor_date"].isna().any():
        raise ValueError("prediction context contains invalid anchor dates.")
    actual = build_horizon_demand(
        context_df=context,
        daily_demand_df=daily_demand_df,
        horizon_days=horizon_days,
    )
    out = context.merge(actual, on=["anchor_date", "sku_id"], how="left")
    out["true_replenish_qty"] = out["demand_qty"].fillna(0.0)
    out["ai_pred_prob"] = pd.to_numeric(
        out["ai_pred_prob"], errors="coerce"
    ).fillna(0.0)
    out["ai_pred_qty_open"] = (
        pd.to_numeric(out["ai_pred_qty_open"], errors="coerce").fillna(0.0)
        * (float(horizon_days) / float(source_quantity_horizon_days))
    )
    return out.loc[
        :,
        [
            "anchor_date",
            "sku_id",
            "true_replenish_qty",
            "ai_pred_prob",
            "ai_pred_qty_open",
        ],
    ]
