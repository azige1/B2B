from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parents[1]
sys.path.extend([str(MODULE_ROOT / "src"), str(MODULE_ROOT / "scripts")])

import run_real_data_experiment as base  # noqa: E402
from profit_analysis import (  # noqa: E402
    CandidatePlan,
    Economics,
    InventoryState,
    ModelOutput,
    aggregate_hurdle_outputs,
    allocate_integer_plan,
    build_daily_demand_curves,
    build_item_demand_gap_scores,
    build_default_demand_scenarios,
    fit_demand_scenario_calibration,
    load_policy_defaults,
    probability_calibration_metrics,
    realize_replenishment_plan,
    recommend_replenishment_plans,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SKC-level production decisions with SKU allocation on real data."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--inventory-source",
        default=str(PROJECT_ROOT / "data" / "phase8a_prep" / "inventory_daily_features.csv"),
    )
    parser.add_argument(
        "--products-source",
        default=str(PROJECT_ROOT / "data" / "silver" / "clean_products.csv"),
    )
    parser.add_argument(
        "--lifecycle-source",
        default=str(PROJECT_ROOT / "data" / "phase8a_prep" / "lifecycle_launch_date_features.csv"),
    )
    parser.add_argument(
        "--style-cost-source",
        default=str(
            PROJECT_ROOT
            / "data"
            / "incoming"
            / "profit_analysis"
            / "style_costs_2024_2026.csv"
        ),
    )
    parser.add_argument(
        "--defaults-csv",
        default=str(
            MODULE_ROOT
            / "config"
            / "profit_analysis_business_defaults_client_feedback_20260515.csv"
        ),
    )
    parser.add_argument(
        "--wide-table-source",
        default=str(PROJECT_ROOT / "data" / "gold" / "wide_table_sku.csv"),
    )
    parser.add_argument("--horizon-days", type=int, default=45)
    parser.add_argument("--anchors", nargs="+", default=list(base.ANCHORS))
    parser.add_argument("--max-groups", type=int, default=None)
    return parser.parse_args()


def _complete_skc_cohort(
    work: pd.DataFrame,
    prediction: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    universe = prediction.loc[:, ["sku_id"]].merge(
        products.loc[:, ["sku_id", "style_id"]].drop_duplicates("sku_id"),
        on="sku_id",
        how="left",
    )
    universe_counts = universe.groupby("style_id")["sku_id"].nunique()
    exact_counts = work.groupby("style_id")["sku_id"].nunique()
    counts = pd.concat(
        [universe_counts.rename("universe_skus"), exact_counts.rename("exact_skus")],
        axis=1,
    ).fillna(0)
    complete_styles = counts.index[counts["universe_skus"] == counts["exact_skus"]]
    return work[work["style_id"].isin(complete_styles)].copy()


def _round_group_plan(qty: float, minimum: float = 100.0, increment: float = 10.0) -> float:
    qty = max(float(qty), 0.0)
    if qty <= 0:
        return 0.0
    return float(math.ceil(max(qty, minimum) / increment) * increment)


def _group_recommendation(
    group: pd.DataFrame,
    calibration,
    economic_case: str,
    policy: str,
    horizon_days: int,
) -> tuple[dict, dict[str, float]]:
    probabilities = np.array(
        [
            calibration.calibrate_probability(value)
            for value in group["pred_prob_positive"]
        ],
        dtype=float,
    )
    conditional_qty = group["conditional_qty_30d"].to_numpy(dtype=float)
    group_output = aggregate_hurdle_outputs(
        probabilities=probabilities,
        conditional_quantities=conditional_qty,
    )
    style_id = str(group["style_id"].iloc[0])
    anchor = str(group["snapshot_date"].iloc[0])
    model_output = ModelOutput(
        sku_id=style_id,
        snapshot_date=anchor,
        pred_prob_positive=group_output.probability_positive,
        pred_qty_30d=group_output.conditional_qty,
        prediction_version=str(group["prediction_version"].iloc[0]),
    )
    inventory = InventoryState(
        sku_id=style_id,
        snapshot_date=anchor,
        current_inventory=float(group["current_inventory"].sum()),
        inbound_within_30d=float(group["inbound_within_30d"].sum()),
        lead_time_days=int(group["lead_time_days"].max()),
        min_batch_qty=100.0,
        increment_batch_qty=10.0,
    )
    case = base.ECONOMIC_CASES[economic_case]
    unit_cost = float(group["unit_cost"].median())
    economics = Economics(
        sku_id=style_id,
        unit_cost=unit_cost,
        unit_price=float(group["unit_price"].median()) * case["unit_price_ratio"],
        holding_cost_per_unit_per_day=unit_cost * case["holding_cost_ratio"],
        salvage_value_per_unit=float(group["salvage_value_per_unit"].median()),
        stockout_penalty_per_unit=float(group["stockout_penalty_per_unit"].median()),
        other_fixed_cost=float(group["other_fixed_cost"].median()),
        target_sell_through_rate=float(group["target_sell_through_rate"].median()),
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
        inventory_state=inventory,
        economics=economics,
        policy=policy,
        demand_scenarios=scenarios,
        horizon_days=horizon_days,
    )
    mean_multiplier = float(
        np.average(
            calibration.positive_multipliers,
            weights=calibration.positive_weights,
        )
    )
    sku_demand_gap_scores = build_item_demand_gap_scores(
        probabilities=probabilities,
        conditional_quantities=(
            conditional_qty * (float(horizon_days) / 30.0)
        ),
        current_inventory=group["current_inventory"].to_numpy(dtype=float),
        positive_multiplier_mean=mean_multiplier,
    )
    allocation_inputs = {
        str(sku_id): float(score)
        for sku_id, score in zip(group["sku_id"], sku_demand_gap_scores)
    }
    return recommendation, allocation_inputs


def _sku_economics(
    row: dict,
    economic_case: str,
    horizon_days: int,
) -> Economics:
    case = base.ECONOMIC_CASES[economic_case]
    unit_cost = float(row["unit_cost"])
    return Economics(
        sku_id=row["sku_id"],
        unit_cost=unit_cost,
        unit_price=float(row["unit_price"]) * case["unit_price_ratio"],
        holding_cost_per_unit_per_day=unit_cost * case["holding_cost_ratio"],
        salvage_value_per_unit=float(row["salvage_value_per_unit"]),
        stockout_penalty_per_unit=float(row["stockout_penalty_per_unit"]),
        other_fixed_cost=0.0,
        target_sell_through_rate=float(row["target_sell_through_rate"]),
        lifecycle_days=horizon_days,
    )


def _summarize(frame: pd.DataFrame) -> dict:
    supply = frame["available_qty"].sum()
    actual = frame["actual_demand_qty"].sum()
    return {
        "rows": int(len(frame)),
        "skc_rows": int(frame[["anchor_date", "style_id"]].drop_duplicates().shape[0]),
        "positive_skc_plan_rate": float(
            frame.groupby(["anchor_date", "style_id"])["group_plan_qty"].max().gt(0).mean()
        ),
        "total_plan_qty": float(
            frame.groupby(["anchor_date", "style_id"])["group_plan_qty"].max().sum()
        ),
        "total_realized_profit": float(frame["realized_profit"].sum()),
        "stockout_rate": float(frame["stockout_flag"].mean()),
        "sell_through_rate": float(frame["sold_qty"].sum() / max(supply, 1e-9)),
        "lost_sales_rate": float(frame["lost_sales_qty"].sum() / max(actual, 1e-9)),
        "leftover_share_of_supply": float(frame["leftover_qty"].sum() / max(supply, 1e-9)),
    }


def main() -> None:
    args = parse_args()
    if args.horizon_days <= 0:
        raise ValueError("--horizon-days must be positive.")
    anchors = tuple(pd.Timestamp(value).strftime("%Y-%m-%d") for value in args.anchors)
    output_dir = Path(
        args.output_dir
        or (
            PROJECT_ROOT
            / "reports"
            / f"profit_analysis_skc_real_cost_h{args.horizon_days}_20260612"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_demand_source = base._load_daily_demand_source(args.wide_table_source)
    calibration_frame = base._load_calibration_frame(
        horizon_days=args.horizon_days,
        daily_demand_df=daily_demand_source,
    )
    calibration = fit_demand_scenario_calibration(calibration_frame)
    calibration_metrics = probability_calibration_metrics(
        calibration_frame,
        calibration,
    )
    inventory_daily = pd.read_csv(args.inventory_source)
    products = pd.read_csv(args.products_source)
    lifecycle = pd.read_csv(args.lifecycle_source)
    style_cost_path = Path(args.style_cost_source)
    style_costs = pd.read_csv(style_cost_path) if style_cost_path.exists() else None
    defaults = load_policy_defaults(args.defaults_csv)
    daily_curves = build_daily_demand_curves(
        daily_demand_source,
        anchors=anchors,
        horizon_days=args.horizon_days,
    )

    detail_rows: list[dict] = []
    group_rows: list[dict] = []
    cohort_rows: list[dict] = []

    for track in base.TRACKS:
        for anchor in anchors:
            prediction = base._prediction_context(
                track,
                anchor,
                horizon_days=args.horizon_days,
                daily_demand_df=daily_demand_source,
            )
            work = base._build_work_frame(
                track=track,
                anchor=anchor,
                inventory_daily=inventory_daily,
                products=products,
                lifecycle=lifecycle,
                style_costs=style_costs,
                horizon_days=args.horizon_days,
                daily_demand_df=daily_demand_source,
                defaults=defaults,
                max_rows=None,
            )
            work = _complete_skc_cohort(work, prediction, products)
            grouped = list(work.groupby("style_id", sort=False))
            if args.max_groups is not None:
                grouped = grouped[: args.max_groups]
                selected = {style_id for style_id, _ in grouped}
                work = work[work["style_id"].isin(selected)]
            cohort_rows.append(
                {
                    "track": track,
                    "anchor_date": anchor,
                    "complete_skc_rows": len(grouped),
                    "complete_sku_rows": int(len(work)),
                    "actual_positive_sku_rows": int((work["actual_demand_qty"] > 0).sum()),
                    "actual_demand_qty": float(work["actual_demand_qty"].sum()),
                    "inventory_qty": float(work["current_inventory"].sum()),
                    "real_cost_sku_rows": int(
                        work["cost_source"]
                        .astype(str)
                        .str.startswith("client_style_cost")
                        .sum()
                    ),
                    "real_cost_skc_rows": int(
                        work.loc[
                            work["cost_source"]
                            .astype(str)
                            .str.startswith("client_style_cost"),
                            "style_id",
                        ].nunique()
                    ),
                }
            )

            for economic_case, policy in base._experiment_combinations(track):
                for style_id, group in grouped:
                    recommendation, expected_demand_scores = _group_recommendation(
                        group,
                        calibration,
                        economic_case,
                        policy,
                        horizon_days=args.horizon_days,
                    )
                    available_before_plan = float(
                        group["current_inventory"].sum()
                        + group["inbound_within_30d"].sum()
                    )
                    group_plans = {
                        "no_replenishment": 0.0,
                        "model_direct": _round_group_plan(
                            float(group["final_pred_qty_horizon"].sum())
                            - available_before_plan
                        ),
                        "profit_module": float(
                            (recommendation["best_recommended_plan"] or {}).get("plan_qty", 0.0)
                        ),
                        "hindsight_qty": _round_group_plan(
                            float(group["actual_demand_qty"].sum()) - available_before_plan
                        ),
                    }
                    allocation_scores = {
                        "no_replenishment": {
                            str(row.sku_id): 0.0 for row in group.itertuples()
                        },
                        "model_direct": {
                            str(row.sku_id): max(
                                float(row.final_pred_qty_horizon)
                                - float(row.current_inventory),
                                0.0,
                            )
                            for row in group.itertuples()
                        },
                        "profit_module": {
                            str(row.sku_id): expected_demand_scores[str(row.sku_id)]
                            for row in group.itertuples()
                        },
                        "hindsight_qty": {
                            str(row.sku_id): max(
                                float(row.actual_demand_qty) - float(row.current_inventory),
                                0.0,
                            )
                            for row in group.itertuples()
                        },
                    }

                    for strategy, group_plan_qty in group_plans.items():
                        sku_ids = [str(value) for value in group["sku_id"]]
                        allocated = allocate_integer_plan(
                            plan_qty=group_plan_qty,
                            item_ids=sku_ids,
                            demand_scores=[
                                allocation_scores[strategy][sku_id]
                                for sku_id in sku_ids
                            ],
                        )
                        group_rows.append(
                            {
                                "track": track,
                                "anchor_date": anchor,
                                "economic_case": economic_case,
                                "policy": policy,
                                "strategy": strategy,
                                "style_id": style_id,
                                "sku_count": len(group),
                                "group_plan_qty": group_plan_qty,
                                "group_actual_demand_qty": float(
                                    group["actual_demand_qty"].sum()
                                ),
                                "group_current_inventory": float(
                                    group["current_inventory"].sum()
                                ),
                                "group_final_pred_qty": float(
                                    group["final_pred_qty_horizon"].sum()
                                ),
                                "horizon_days": args.horizon_days,
                                "group_unit_cost": float(group["unit_cost"].median()),
                                "cost_source": str(group["cost_source"].iloc[0]),
                                "cost_conflict_flag": int(
                                    group["cost_conflict_flag"].max()
                                ),
                                "group_expected_profit": (
                                    float(
                                        (recommendation["best_recommended_plan"] or {}).get(
                                            "expected_profit", 0.0
                                        )
                                    )
                                    if strategy == "profit_module"
                                    else math.nan
                                ),
                            }
                        )
                        for row in group.to_dict(orient="records"):
                            model_output = ModelOutput(
                                sku_id=row["sku_id"],
                                snapshot_date=row["snapshot_date"],
                                pred_prob_positive=calibration.calibrate_probability(
                                    row["pred_prob_positive"]
                                ),
                                pred_qty_30d=row["conditional_qty_30d"],
                                prediction_version=row["prediction_version"],
                            )
                            sku_inventory = InventoryState(
                                sku_id=row["sku_id"],
                                snapshot_date=row["snapshot_date"],
                                current_inventory=row["current_inventory"],
                                inbound_within_30d=row["inbound_within_30d"],
                                lead_time_days=row["lead_time_days"],
                                min_batch_qty=1,
                                increment_batch_qty=1,
                            )
                            economics = _sku_economics(
                                row,
                                economic_case,
                                horizon_days=args.horizon_days,
                            )
                            realized = realize_replenishment_plan(
                                model_output=model_output,
                                inventory_state=sku_inventory,
                                economics=economics,
                                plan=CandidatePlan(
                                    plan_qty=allocated[str(row["sku_id"])],
                                    policy=strategy,
                                ),
                                actual_demand_qty=row["actual_demand_qty"],
                                horizon_days=args.horizon_days,
                                actual_daily_demand_curve=daily_curves.get(
                                    (anchor, str(row["sku_id"])),
                                    [0.0] * args.horizon_days,
                                ),
                            )
                            detail_rows.append(
                                {
                                    "track": track,
                                    "anchor_date": anchor,
                                    "economic_case": economic_case,
                                    "policy": policy,
                                    "strategy": strategy,
                                    "style_id": style_id,
                                    "sku_id": row["sku_id"],
                                    "category": row.get("category"),
                                    "group_plan_qty": group_plan_qty,
                                    "allocated_plan_qty": allocated[str(row["sku_id"])],
                                    "actual_demand_qty": row["actual_demand_qty"],
                                    "current_inventory": row["current_inventory"],
                                    "available_qty": (
                                        float(row["current_inventory"])
                                        + float(row["inbound_within_30d"])
                                        + float(realized.plan_qty)
                                    ),
                                    "unit_price": economics.unit_price,
                                    "unit_cost": economics.unit_cost,
                                    "cost_source": row.get("cost_source"),
                                    "cost_record_count": row.get("cost_record_count", 0),
                                    "cost_conflict_flag": row.get("cost_conflict_flag", 0),
                                    **realized.to_dict(),
                                }
                            )

    detail = pd.DataFrame(detail_rows)
    groups = pd.DataFrame(group_rows)
    cohorts = pd.DataFrame(cohort_rows)
    detail.to_csv(output_dir / "skc_allocation_detail.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(output_dir / "skc_plan_detail.csv", index=False, encoding="utf-8-sig")
    cohorts.to_csv(output_dir / "skc_real_inventory_cohorts.csv", index=False, encoding="utf-8-sig")

    group_cols = ["track", "anchor_date", "economic_case", "policy", "strategy"]
    summary_rows = []
    for keys, frame in detail.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row.update(_summarize(frame))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    no_plan = summary[summary["strategy"] == "no_replenishment"][
        group_cols[:-1] + ["total_realized_profit"]
    ].rename(columns={"total_realized_profit": "no_plan_total_profit"})
    direct = summary[summary["strategy"] == "model_direct"][
        group_cols[:-1] + ["total_realized_profit"]
    ].rename(columns={"total_realized_profit": "direct_total_profit"})
    summary = summary.merge(no_plan, on=group_cols[:-1], how="left").merge(
        direct, on=group_cols[:-1], how="left"
    )
    summary["incremental_profit_vs_no_plan"] = (
        summary["total_realized_profit"] - summary["no_plan_total_profit"]
    )
    summary["incremental_profit_vs_direct"] = (
        summary["total_realized_profit"] - summary["direct_total_profit"]
    )
    summary.to_csv(output_dir / "skc_experiment_summary.csv", index=False, encoding="utf-8-sig")

    aggregate_group_cols = ["track", "economic_case", "policy", "strategy"]
    aggregate_rows = []
    for keys, frame in detail.groupby(aggregate_group_cols):
        row = dict(zip(aggregate_group_cols, keys))
        row.update(_summarize(frame))
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    no_plan_agg = aggregate[aggregate["strategy"] == "no_replenishment"][
        aggregate_group_cols[:-1] + ["total_realized_profit"]
    ].rename(columns={"total_realized_profit": "no_plan_total_profit"})
    direct_agg = aggregate[aggregate["strategy"] == "model_direct"][
        aggregate_group_cols[:-1] + ["total_realized_profit"]
    ].rename(columns={"total_realized_profit": "direct_total_profit"})
    aggregate = aggregate.merge(
        no_plan_agg, on=aggregate_group_cols[:-1], how="left"
    ).merge(
        direct_agg, on=aggregate_group_cols[:-1], how="left"
    )
    aggregate["incremental_profit_vs_no_plan"] = (
        aggregate["total_realized_profit"] - aggregate["no_plan_total_profit"]
    )
    aggregate["incremental_profit_vs_direct"] = (
        aggregate["total_realized_profit"] - aggregate["direct_total_profit"]
    )
    aggregate.to_csv(output_dir / "skc_experiment_aggregate.csv", index=False, encoding="utf-8-sig")

    paired_group_cols = [
        "track",
        "anchor_date",
        "economic_case",
        "policy",
        "style_id",
        "strategy",
    ]
    paired = (
        detail.groupby(paired_group_cols, as_index=False)
        .agg(
            realized_profit=("realized_profit", "sum"),
            group_plan_qty=("group_plan_qty", "max"),
        )
    )
    no_plan_group = paired[paired["strategy"] == "no_replenishment"][
        paired_group_cols[:-1] + ["realized_profit"]
    ].rename(columns={"realized_profit": "no_plan_realized_profit"})
    paired = paired.merge(no_plan_group, on=paired_group_cols[:-1], how="left")
    paired["incremental_profit_vs_no_plan"] = (
        paired["realized_profit"] - paired["no_plan_realized_profit"]
    )
    paired["outcome_vs_no_plan"] = np.select(
        [
            paired["incremental_profit_vs_no_plan"] > 1e-4,
            paired["incremental_profit_vs_no_plan"] < -1e-4,
        ],
        ["beneficial", "harmful"],
        default="unchanged",
    )
    paired.to_csv(
        output_dir / "skc_paired_decision_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )

    paired_summary_rows = []
    paired_summary_cols = ["track", "economic_case", "policy", "strategy"]
    for keys, frame in paired.groupby(paired_summary_cols):
        row = dict(zip(paired_summary_cols, keys))
        row.update(
            {
                "skc_rows": int(len(frame)),
                "positive_plan_rows": int((frame["group_plan_qty"] > 0).sum()),
                "beneficial_rows": int((frame["outcome_vs_no_plan"] == "beneficial").sum()),
                "harmful_rows": int((frame["outcome_vs_no_plan"] == "harmful").sum()),
                "unchanged_rows": int((frame["outcome_vs_no_plan"] == "unchanged").sum()),
                "incremental_profit_vs_no_plan": float(
                    frame["incremental_profit_vs_no_plan"].sum()
                ),
            }
        )
        paired_summary_rows.append(row)
    paired_summary = pd.DataFrame(paired_summary_rows)
    paired_summary.to_csv(
        output_dir / "skc_paired_decision_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary_groups = groups[
        (groups["track"] == "phase7_base")
        & (groups["economic_case"] == "tag_price_100pct")
        & (groups["policy"] == "balanced")
        & (groups["strategy"] == "no_replenishment")
    ].copy()
    primary_gap = (
        primary_groups["group_actual_demand_qty"]
        - primary_groups["group_current_inventory"]
    ).clip(lower=0.0)
    gap_summary = pd.DataFrame(
        [
            {
                "skc_rows": int(len(primary_groups)),
                "positive_actual_gap_rows": int((primary_gap > 0).sum()),
                "actual_gap_ge_100_rows": int((primary_gap >= 100).sum()),
                "actual_demand_ge_100_rows": int(
                    (primary_groups["group_actual_demand_qty"] >= 100).sum()
                ),
                "total_positive_actual_gap_qty": float(primary_gap.sum()),
                "max_positive_actual_gap_qty": float(primary_gap.max()),
            }
        ]
    )
    gap_summary.to_csv(
        output_dir / "skc_actual_gap_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    calibration_payload = {
        "horizon_days": args.horizon_days,
        "source": "historical_phase7_calibration_anchors",
        "metrics": calibration_metrics,
        "calibration": calibration.to_dict(),
    }
    (output_dir / "demand_scenario_calibration.json").write_text(
        json.dumps(calibration_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    primary = aggregate[
        (aggregate["track"] == "phase7_base")
        & (aggregate["policy"] == "balanced")
        & (
            aggregate["strategy"].isin(
                ["no_replenishment", "model_direct", "profit_module", "hindsight_qty"]
            )
        )
    ]
    policy_compare = aggregate[
        (aggregate["track"] == "phase7_base")
        & (aggregate["economic_case"] == "tag_price_100pct")
        & (aggregate["strategy"] == "profit_module")
    ]
    track_compare = aggregate[
        (aggregate["economic_case"] == "tag_price_100pct")
        & (aggregate["policy"] == "balanced")
        & (aggregate["strategy"] == "profit_module")
    ]
    paired_compare = paired_summary[
        (paired_summary["track"] == "phase7_base")
        & (paired_summary["economic_case"] == "tag_price_100pct")
        & (paired_summary["policy"] == "balanced")
    ]
    report = [
        "# SKC级盈亏分析真实数据实验",
        "",
        f"- 运行时间：`{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- 分析窗口：锚点后{args.horizon_days}天",
        "- 决策粒度：`style_id`，甲方确认的SKC主键",
        "- 批量约束：每个SKC最低100件，10件递增",
        "- 分配粒度：SKC总量按SKU校准需求缺口做最大余数整数分配",
        "- 主样本：锚点当天该SKC下全部预测SKU均有真实库存快照",
        "",
        "## 真实样本",
        "",
        base._markdown_table(
            cohorts,
            [
                "track",
                "anchor_date",
                "complete_skc_rows",
                "complete_sku_rows",
                "actual_positive_sku_rows",
                "actual_demand_qty",
                "inventory_qty",
                "real_cost_sku_rows",
                "real_cost_skc_rows",
            ],
        ),
        "",
        "## Phase7平衡策略敏感性",
        "",
        base._markdown_table(
            primary,
            [
                "economic_case",
                "strategy",
                "positive_skc_plan_rate",
                "total_plan_qty",
                "lost_sales_rate",
                "leftover_share_of_supply",
                "incremental_profit_vs_no_plan",
                "incremental_profit_vs_direct",
            ],
        ),
        "",
        "## 风险策略对比",
        "",
        base._markdown_table(
            policy_compare,
            [
                "policy",
                "positive_skc_plan_rate",
                "total_plan_qty",
                "lost_sales_rate",
                "leftover_share_of_supply",
                "incremental_profit_vs_no_plan",
                "incremental_profit_vs_direct",
            ],
        ),
        "",
        "## 预测输入对比",
        "",
        base._markdown_table(
            track_compare,
            [
                "track",
                "positive_skc_plan_rate",
                "total_plan_qty",
                "lost_sales_rate",
                "leftover_share_of_supply",
                "incremental_profit_vs_no_plan",
                "incremental_profit_vs_direct",
            ],
        ),
        "",
        "## 逐SKC配对结果",
        "",
        base._markdown_table(
            paired_compare,
            [
                "strategy",
                "positive_plan_rows",
                "beneficial_rows",
                "harmful_rows",
                "unchanged_rows",
                "incremental_profit_vs_no_plan",
            ],
        ),
        "",
        "## 真实需求缺口分布",
        "",
        base._markdown_table(
            gap_summary,
            [
                "skc_rows",
                "positive_actual_gap_rows",
                "actual_gap_ge_100_rows",
                "actual_demand_ge_100_rows",
                "total_positive_actual_gap_qty",
                "max_positive_actual_gap_qty",
            ],
        ),
        "",
        "## 解释边界",
        "",
        "1. 这是历史决策回测，库存、真实补货和逐日节奏均来自项目真实数据。",
        "2. 吊牌价是真实字段，但实际成交价缺失，因此通过20%/30%/50%/100%吊牌价做敏感性。",
        "3. 成本表按style_id精确关联；同一style_id多成本时保守取最大值，未匹配项使用吊牌价/7兜底。",
        "4. style_id已由甲方确认为SKC主键。",
        "5. SKC推荐阶段将库存视为组内汇总，实际收益阶段按尺码SKU分配后逐SKU模拟。",
    ]
    (output_dir / "skc_real_data_experiment_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    (output_dir / "skc_experiment_manifest.json").write_text(
        json.dumps(
            {
                "anchors": list(anchors),
                "horizon_days": args.horizon_days,
                "tracks": list(base.TRACKS),
                "economic_cases": base.ECONOMIC_CASES,
                "style_cost_source": str(style_cost_path.resolve())
                if style_cost_path.exists()
                else None,
                "style_cost_rows": int(len(style_costs))
                if style_costs is not None
                else 0,
                "calibration": calibration.to_dict(),
                "calibration_metrics": calibration_metrics,
                "detail_rows": len(detail),
                "group_rows": len(groups),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] output -> {output_dir}")
    print(f"[OK] allocation detail rows -> {len(detail):,}")
    print(f"[OK] SKC plan rows -> {len(groups):,}")


if __name__ == "__main__":
    main()
