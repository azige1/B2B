from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .allocation import allocate_integer_plan
from .calibration import DemandScenarioCalibration
from .core import (
    Economics,
    InventoryState,
    ModelOutput,
    build_default_demand_scenarios,
    recommend_replenishment_plans,
)
from .grouping import aggregate_hurdle_outputs, build_item_demand_gap_scores
from .io import build_profit_input_frame


@dataclass
class SKCDecisionBatch:
    skc_recommendations: pd.DataFrame
    sku_allocations: pd.DataFrame
    recommendation_details: list[dict[str, Any]]
    quality_report: dict[str, Any]


class ProfitAnalysisQualityError(ValueError):
    def __init__(self, message: str, quality_report: dict[str, Any]):
        super().__init__(message)
        self.quality_report = quality_report


def _require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _duplicate_count(frame: pd.DataFrame, keys: list[str]) -> int:
    return int(frame.duplicated(keys, keep=False).sum())


def _missing_join_count(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
) -> int:
    right_keys = right.loc[:, keys].drop_duplicates()
    joined = left.loc[:, keys].merge(right_keys, on=keys, how="left", indicator=True)
    return int((joined["_merge"] == "left_only").sum())


def _numeric_invalid_count(
    frame: pd.DataFrame,
    column: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> int:
    values = pd.to_numeric(frame[column], errors="coerce")
    invalid = values.isna() | ~np.isfinite(values)
    if strictly_positive:
        invalid |= values <= 0
    elif minimum is not None:
        invalid |= values < minimum
    if maximum is not None:
        invalid |= values > maximum
    return int(invalid.sum())


def _first_non_null(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return None
    text = values.astype(str).str.strip()
    text = text[~text.isin(["", "<NA>", "nan", "NaT", "None"])]
    return text.iloc[0] if not text.empty else None


def _median(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in frame.columns:
        return float(default)
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else float(default)


def _sum(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if column not in frame.columns:
        return float(default)
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).sum())


def _minimum_positive(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce")
    values = values[values > 0]
    return float(values.min()) if not values.empty else None


def _earliest_date(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns:
        return None
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    return values.min().date().isoformat() if not values.empty else None


def _fallback_cost_mask(frame: pd.DataFrame) -> pd.Series:
    if "cost_source" not in frame.columns:
        return pd.Series(True, index=frame.index)
    source = frame["cost_source"].fillna("").astype(str).str.lower()
    return (
        source.eq("")
        | source.str.contains("fallback", regex=False)
        | source.str.contains("price_tag_div_7", regex=False)
        | source.str.contains("price_tag / 7", regex=False)
    )


def _conflicting_style_id_skus(*frames: pd.DataFrame) -> int:
    mappings = []
    for frame in frames:
        if "style_id" not in frame.columns:
            continue
        current = frame.loc[:, ["sku_id", "style_id"]].copy()
        current["style_id"] = current["style_id"].fillna("").astype(str).str.strip()
        current = current[current["style_id"].ne("")]
        mappings.append(current)
    if not mappings:
        return 0
    combined = pd.concat(mappings, ignore_index=True).drop_duplicates()
    return int(combined.groupby("sku_id")["style_id"].nunique().gt(1).sum())


def _quality_report(
    prediction: pd.DataFrame,
    inventory: pd.DataFrame,
    economics: pd.DataFrame,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": "pending",
        "prediction_rows": int(len(prediction)),
        "inventory_rows": int(len(inventory)),
        "economics_rows": int(len(economics)),
        "errors": [],
        "warnings": [],
    }
    report["duplicate_prediction_rows"] = _duplicate_count(
        prediction, ["sku_id", "snapshot_date"]
    )
    report["duplicate_inventory_rows"] = _duplicate_count(
        inventory, ["sku_id", "snapshot_date"]
    )
    report["duplicate_economics_rows"] = _duplicate_count(economics, ["sku_id"])
    report["missing_inventory_rows"] = _missing_join_count(
        prediction, inventory, ["sku_id", "snapshot_date"]
    )
    report["missing_economics_rows"] = _missing_join_count(
        prediction, economics, ["sku_id"]
    )
    report["invalid_prediction_snapshot_date_rows"] = int(
        pd.to_datetime(prediction["snapshot_date"], errors="coerce").isna().sum()
    )
    report["invalid_inventory_snapshot_date_rows"] = int(
        pd.to_datetime(inventory["snapshot_date"], errors="coerce").isna().sum()
    )
    report["missing_inventory_snapshot_rows"] = (
        int(
            pd.to_numeric(
                inventory["inventory_snapshot_present"],
                errors="coerce",
            )
            .fillna(0)
            .le(0)
            .sum()
        )
        if "inventory_snapshot_present" in inventory.columns
        else 0
    )
    report["conflicting_style_id_skus"] = _conflicting_style_id_skus(
        prediction,
        inventory,
        economics,
    )

    checks = {
        "invalid_probability_rows": _numeric_invalid_count(
            prediction, "pred_prob_positive", minimum=0.0, maximum=1.0
        ),
        "invalid_prediction_qty_rows": _numeric_invalid_count(
            prediction, "pred_qty_30d", minimum=0.0
        ),
        "invalid_inventory_qty_rows": _numeric_invalid_count(
            inventory, "current_inventory", minimum=0.0
        ),
        "invalid_inbound_qty_rows": _numeric_invalid_count(
            inventory, "inbound_within_30d", minimum=0.0
        )
        if "inbound_within_30d" in inventory.columns
        else 0,
        "invalid_lead_time_rows": _numeric_invalid_count(
            inventory, "lead_time_days", minimum=0.0
        )
        if "lead_time_days" in inventory.columns
        else 0,
        "invalid_unit_cost_rows": _numeric_invalid_count(
            economics, "unit_cost", strictly_positive=True
        ),
        "invalid_unit_price_rows": _numeric_invalid_count(
            economics, "unit_price", strictly_positive=True
        ),
        "invalid_holding_cost_rows": _numeric_invalid_count(
            economics, "holding_cost_per_unit_per_day", minimum=0.0
        ),
        "invalid_salvage_value_rows": _numeric_invalid_count(
            economics, "salvage_value_per_unit", minimum=0.0
        ),
        "invalid_target_sell_through_rows": _numeric_invalid_count(
            economics,
            "target_sell_through_rate",
            minimum=0.0,
            maximum=1.0,
        )
        if "target_sell_through_rate" in economics.columns
        else 0,
    }
    report.update(checks)
    for field, value in report.items():
        if (
            field.startswith("duplicate_")
            or field.startswith("missing_")
            or field.startswith("invalid_")
            or field.startswith("missing_inventory_snapshot_")
            or field.startswith("conflicting_style_id_")
        ) and value:
            report["errors"].append(f"{field}={value}")
    return report


def build_skc_decision_batch(
    prediction_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    economics_df: pd.DataFrame,
    calibration: DemandScenarioCalibration,
    *,
    policy: str = "balanced",
    horizon_days: int = 45,
    min_batch_qty: float = 100.0,
    increment_batch_qty: float = 10.0,
    max_fallback_cost_rate: float | None = None,
) -> SKCDecisionBatch:
    if policy not in {"conservative", "balanced", "aggressive"}:
        raise ValueError(f"unsupported policy: {policy}")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if min_batch_qty <= 0 or increment_batch_qty <= 0:
        raise ValueError("batch quantities must be positive.")

    prediction = prediction_df.copy()
    inventory = inventory_df.copy()
    economics = economics_df.copy()
    _require_columns(
        prediction,
        ["sku_id", "snapshot_date", "pred_prob_positive", "pred_qty_30d"],
        "prediction",
    )
    _require_columns(
        inventory,
        ["sku_id", "snapshot_date", "current_inventory"],
        "inventory",
    )
    _require_columns(
        economics,
        [
            "sku_id",
            "unit_cost",
            "unit_price",
            "holding_cost_per_unit_per_day",
            "salvage_value_per_unit",
        ],
        "economics",
    )

    for frame in [prediction, inventory, economics]:
        frame["sku_id"] = frame["sku_id"].astype(str).str.strip()
    for frame in [prediction, inventory]:
        parsed = pd.to_datetime(frame["snapshot_date"], errors="coerce")
        frame["snapshot_date"] = parsed.dt.strftime("%Y-%m-%d")

    quality = _quality_report(prediction, inventory, economics)
    if quality["errors"]:
        quality["status"] = "failed"
        raise ProfitAnalysisQualityError(
            "profit-analysis input quality gate failed: "
            + ", ".join(quality["errors"]),
            quality,
        )

    work = build_profit_input_frame(prediction, inventory, economics)
    if "style_id" not in work.columns:
        quality["errors"].append("missing_style_id_column=1")
    else:
        work["style_id"] = work["style_id"].fillna("").astype(str).str.strip()
        missing_style = int(work["style_id"].eq("").sum())
        quality["missing_style_id_rows"] = missing_style
        if missing_style:
            quality["errors"].append(f"missing_style_id_rows={missing_style}")
    if quality["errors"]:
        quality["status"] = "failed"
        raise ProfitAnalysisQualityError(
            "profit-analysis input quality gate failed: "
            + ", ".join(quality["errors"]),
            quality,
        )

    optional_defaults = {
        "inbound_within_30d": 0.0,
        "lead_time_days": 0,
        "max_replenish_qty": np.nan,
        "safety_stock_qty": np.nan,
        "stockout_penalty_per_unit": 0.0,
        "other_fixed_cost": 0.0,
        "target_sell_through_rate": 0.85,
        "lifecycle_days": horizon_days,
        "prediction_version": None,
        "cost_source": None,
        "cost_conflict_flag": 0,
        "category": None,
    }
    for column, default in optional_defaults.items():
        if column not in work.columns:
            work[column] = default

    fallback_mask = _fallback_cost_mask(work)
    quality["joined_rows"] = int(len(work))
    quality["fallback_cost_rows"] = int(fallback_mask.sum())
    quality["fallback_cost_rate"] = float(fallback_mask.mean()) if len(work) else 0.0
    quality["cost_conflict_rows"] = int(
        pd.to_numeric(work["cost_conflict_flag"], errors="coerce")
        .fillna(0)
        .gt(0)
        .sum()
    )
    if (
        max_fallback_cost_rate is not None
        and quality["fallback_cost_rate"] > max_fallback_cost_rate
    ):
        quality["errors"].append(
            "fallback_cost_rate="
            f"{quality['fallback_cost_rate']:.6f}>"
            f"{max_fallback_cost_rate:.6f}"
        )
        quality["status"] = "failed"
        raise ProfitAnalysisQualityError(
            "profit-analysis cost coverage gate failed: "
            + ", ".join(quality["errors"]),
            quality,
        )
    if quality["fallback_cost_rows"]:
        quality["warnings"].append(
            f"{quality['fallback_cost_rows']} joined rows use fallback or unknown cost source."
        )

    skc_rows: list[dict[str, Any]] = []
    sku_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    mean_multiplier = float(
        np.average(
            calibration.positive_multipliers,
            weights=calibration.positive_weights,
        )
    )
    for (snapshot_date, style_id), group in work.groupby(
        ["snapshot_date", "style_id"],
        sort=True,
    ):
        raw_probabilities = pd.to_numeric(
            group["pred_prob_positive"], errors="coerce"
        ).to_numpy(dtype=float)
        probabilities = np.asarray(
            [
                calibration.calibrate_probability(value)
                for value in raw_probabilities
            ],
            dtype=float,
        )
        conditional_qty = pd.to_numeric(
            group["pred_qty_30d"], errors="coerce"
        ).to_numpy(dtype=float)
        group_output = aggregate_hurdle_outputs(probabilities, conditional_qty)

        model_output = ModelOutput(
            sku_id=str(style_id),
            snapshot_date=str(snapshot_date),
            pred_prob_positive=group_output.probability_positive,
            pred_qty_30d=group_output.conditional_qty,
            prediction_version=_first_non_null(group["prediction_version"]),
        )
        inventory_state = InventoryState(
            sku_id=str(style_id),
            snapshot_date=str(snapshot_date),
            current_inventory=_sum(group, "current_inventory"),
            inbound_within_30d=_sum(group, "inbound_within_30d"),
            lead_time_days=int(round(_median(group, "lead_time_days", 0.0))),
            min_batch_qty=float(min_batch_qty),
            increment_batch_qty=float(increment_batch_qty),
            max_replenish_qty=_minimum_positive(group, "max_replenish_qty"),
            safety_stock_qty=_sum(group, "safety_stock_qty")
            if pd.to_numeric(
                group["safety_stock_qty"], errors="coerce"
            ).notna().any()
            else None,
        )
        economics = Economics(
            sku_id=str(style_id),
            unit_cost=_median(group, "unit_cost"),
            unit_price=_median(group, "unit_price"),
            holding_cost_per_unit_per_day=_median(
                group, "holding_cost_per_unit_per_day"
            ),
            salvage_value_per_unit=_median(group, "salvage_value_per_unit"),
            stockout_penalty_per_unit=_median(
                group, "stockout_penalty_per_unit"
            ),
            other_fixed_cost=_median(group, "other_fixed_cost"),
            lifecycle_end_date=_earliest_date(group, "lifecycle_end_date"),
            target_sell_through_rate=_median(
                group, "target_sell_through_rate", 0.85
            ),
            lifecycle_days=horizon_days,
        )
        scenarios = build_default_demand_scenarios(
            model_output,
            positive_multipliers=calibration.positive_multipliers,
            positive_weights=calibration.positive_weights,
            horizon_days=horizon_days,
        )
        recommendation = recommend_replenishment_plans(
            model_output=model_output,
            inventory_state=inventory_state,
            economics=economics,
            policy=policy,
            demand_scenarios=scenarios,
            horizon_days=horizon_days,
        )
        best = recommendation["best_recommended_plan"] or {}
        plan_qty = float(best.get("plan_qty", 0.0))
        demand_scores = build_item_demand_gap_scores(
            probabilities=probabilities,
            conditional_quantities=conditional_qty
            * (float(horizon_days) / 30.0),
            current_inventory=pd.to_numeric(
                group["current_inventory"], errors="coerce"
            ).to_numpy(dtype=float),
            positive_multiplier_mean=mean_multiplier,
        )
        allocations = allocate_integer_plan(
            plan_qty=plan_qty,
            item_ids=group["sku_id"].astype(str).tolist(),
            demand_scores=demand_scores,
        )

        cost_sources = sorted(
            {
                str(value)
                for value in group["cost_source"].dropna()
                if str(value).strip()
            }
        )
        skc_rows.append(
            {
                "snapshot_date": snapshot_date,
                "style_id": style_id,
                "category": _first_non_null(group["category"]),
                "policy": policy,
                "horizon_days": horizon_days,
                "sku_count": int(group["sku_id"].nunique()),
                "group_probability_positive": group_output.probability_positive,
                "group_conditional_qty_30d": group_output.conditional_qty,
                "group_expected_qty_30d": group_output.expected_qty,
                "current_inventory": inventory_state.current_inventory,
                "inbound_within_30d": inventory_state.inbound_within_30d,
                "unit_cost": economics.unit_cost,
                "unit_price": economics.unit_price,
                "cost_sources": "|".join(cost_sources),
                "recommended_plan_qty": plan_qty,
                "expected_profit": best.get("expected_profit"),
                "recommendation_score": best.get("recommendation_score"),
                "profit_positive_probability": best.get(
                    "profit_positive_probability"
                ),
                "sell_through_rate": best.get("sell_through_rate"),
                "sell_through_target_probability": best.get(
                    "sell_through_target_probability"
                ),
                "stockout_rate": best.get("stockout_rate"),
                "expected_sold_qty": best.get("expected_sold_qty"),
                "expected_leftover_qty": best.get("expected_leftover_qty"),
                "expected_lost_sales_qty": best.get("expected_lost_sales_qty"),
                "expected_sales_revenue": best.get("expected_sales_revenue"),
                "expected_replenish_cost": best.get(
                    "expected_replenish_cost"
                ),
                "expected_holding_cost": best.get("expected_holding_cost"),
                "expected_stockout_cost": best.get("expected_stockout_cost"),
                "expected_terminal_value": best.get("expected_terminal_value"),
                "effective_horizon_days": best.get("effective_horizon_days"),
                "remaining_lifecycle_days": best.get(
                    "remaining_lifecycle_days"
                ),
                "late_arrival_risk": best.get("late_arrival_risk"),
            }
        )
        for position, row in enumerate(group.to_dict(orient="records")):
            sku_id = str(row["sku_id"])
            sku_rows.append(
                {
                    "snapshot_date": snapshot_date,
                    "style_id": style_id,
                    "sku_id": sku_id,
                    "category": row.get("category"),
                    "raw_probability_positive": float(
                        raw_probabilities[position]
                    ),
                    "calibrated_probability_positive": float(
                        probabilities[position]
                    ),
                    "conditional_qty_30d": float(
                        conditional_qty[position]
                    ),
                    "current_inventory": float(row["current_inventory"]),
                    "demand_gap_score": float(demand_scores[position]),
                    "skc_recommended_plan_qty": plan_qty,
                    "allocated_plan_qty": int(allocations[sku_id]),
                    "unit_cost": float(row["unit_cost"]),
                    "unit_price": float(row["unit_price"]),
                    "cost_source": row.get("cost_source"),
                }
            )
        details.append(
            {
                "snapshot_date": snapshot_date,
                "style_id": style_id,
                "scenarios": [
                    {
                        "name": scenario.name,
                        "demand_qty": scenario.demand_qty,
                        "probability": scenario.probability,
                    }
                    for scenario in scenarios
                ],
                "recommendation": recommendation,
                "sku_allocation": allocations,
            }
        )

    skc = pd.DataFrame(skc_rows)
    sku = pd.DataFrame(sku_rows)
    allocation_check = (
        sku.groupby(["snapshot_date", "style_id"], as_index=False)[
            "allocated_plan_qty"
        ]
        .sum()
        .merge(
            skc.loc[
                :, ["snapshot_date", "style_id", "recommended_plan_qty"]
            ],
            on=["snapshot_date", "style_id"],
            how="left",
        )
    )
    allocation_check["difference"] = (
        allocation_check["allocated_plan_qty"]
        - allocation_check["recommended_plan_qty"]
    ).abs()
    mismatch_rows = int((allocation_check["difference"] > 1e-6).sum())
    quality.update(
        {
            "status": "passed" if mismatch_rows == 0 else "failed",
            "skc_rows": int(len(skc)),
            "sku_allocation_rows": int(len(sku)),
            "positive_plan_skc_rows": int(
                skc["recommended_plan_qty"].gt(0).sum()
            ),
            "total_recommended_plan_qty": float(
                skc["recommended_plan_qty"].sum()
            ),
            "allocation_mismatch_rows": mismatch_rows,
        }
    )
    if mismatch_rows:
        quality["errors"].append(f"allocation_mismatch_rows={mismatch_rows}")
        raise ProfitAnalysisQualityError(
            "profit-analysis allocation quality gate failed.",
            quality,
        )
    return SKCDecisionBatch(
        skc_recommendations=skc,
        sku_allocations=sku,
        recommendation_details=details,
        quality_report=quality,
    )
