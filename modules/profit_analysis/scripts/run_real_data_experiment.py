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
MODULE_SRC = MODULE_ROOT / "src"
sys.path.append(str(MODULE_SRC))

from profit_analysis import (  # noqa: E402
    CandidatePlan,
    Economics,
    InventoryState,
    ModelOutput,
    build_economics_config,
    build_daily_demand_curves,
    build_horizon_calibration_frame,
    build_horizon_demand,
    build_inventory_snapshot,
    fit_demand_scenario_calibration,
    infer_prediction_column_spec,
    load_policy_defaults,
    normalize_prediction_snapshot,
    realize_replenishment_plan,
    recommend_replenishment_plans,
)


ANCHORS = ("2026-02-15", "2026-02-24")
TRACKS = {
    "phase7_base": (
        "reports/phase8_event_inventory_shadow_2026/{tag}/phase5/"
        "eval_context_p8ei_{tag}_base_tail_full_s2028_hard_g027.csv"
    ),
    "phase8_zero_split": (
        "reports/phase8_inventory_zero_split_shadow_2026/{tag}/phase5/"
        "eval_context_p8ei_{tag}_event_inventory_zero_split_s2028_hard_g027.csv"
    ),
}
ECONOMIC_CASES = {
    "tag_price_100pct": {"unit_price_ratio": 1.00, "holding_cost_ratio": 0.000},
    "tag_price_50pct": {"unit_price_ratio": 0.50, "holding_cost_ratio": 0.000},
    "tag_price_30pct": {"unit_price_ratio": 0.30, "holding_cost_ratio": 0.000},
    "tag_price_20pct": {"unit_price_ratio": 0.20, "holding_cost_ratio": 0.000},
    "tag_price_50pct_holding": {"unit_price_ratio": 0.50, "holding_cost_ratio": 0.001},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe real-data profit-analysis experiments."
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
        "--wide-table-source",
        default=str(PROJECT_ROOT / "data" / "gold" / "wide_table_sku.csv"),
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
    parser.add_argument("--horizon-days", type=int, default=45)
    parser.add_argument("--anchors", nargs="+", default=list(ANCHORS))
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def _load_calibration_context() -> pd.DataFrame:
    paths = sorted(
        (PROJECT_ROOT / "reports" / "phase7_refresh_validation_20260416").rglob(
            "eval_context*.csv"
        )
    )
    if not paths:
        raise FileNotFoundError("No Phase7 historical calibration contexts found.")
    return pd.concat(
        [
            pd.read_csv(
                path,
                usecols=[
                    "sku_id",
                    "anchor_date",
                    "true_replenish_qty",
                    "ai_pred_prob",
                    "ai_pred_qty_open",
                ],
            )
            for path in paths
        ],
        ignore_index=True,
    )


def _load_daily_demand_source(source_path: str | Path) -> pd.DataFrame:
    return pd.read_csv(
        source_path,
        usecols=["date", "sku_id", "qty_replenish"],
    )


def _load_calibration_frame(
    horizon_days: int = 30,
    daily_demand_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    context = _load_calibration_context()
    if horizon_days == 30 and daily_demand_df is None:
        return context
    if daily_demand_df is None:
        raise ValueError("daily_demand_df is required for non-30-day calibration.")
    return build_horizon_calibration_frame(
        prediction_context_df=context,
        daily_demand_df=daily_demand_df,
        horizon_days=horizon_days,
        source_quantity_horizon_days=30,
    )


def _prediction_context(
    track: str,
    anchor: str,
    horizon_days: int = 30,
    daily_demand_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    tag = anchor.replace("-", "")
    path = PROJECT_ROOT / TRACKS[track].format(tag=tag)
    if not path.exists():
        raise FileNotFoundError(path)
    source = pd.read_csv(path)
    prediction = normalize_prediction_snapshot(
        source,
        spec=infer_prediction_column_spec(source),
    )
    prediction["conditional_qty_30d"] = pd.to_numeric(
        source["ai_pred_qty_open"], errors="coerce"
    ).fillna(0.0)
    prediction["final_pred_qty_30d"] = pd.to_numeric(
        source["ai_pred_qty"], errors="coerce"
    ).fillna(0.0)
    if horizon_days == 30 and daily_demand_df is None:
        prediction["actual_demand_qty"] = pd.to_numeric(
            source["true_replenish_qty"], errors="coerce"
        ).fillna(0.0)
    else:
        if daily_demand_df is None:
            raise ValueError("daily_demand_df is required for horizon actuals.")
        actual = build_horizon_demand(
            context_df=source.loc[:, ["anchor_date", "sku_id"]],
            daily_demand_df=daily_demand_df,
            horizon_days=horizon_days,
        )
        actual_lookup = actual.set_index("sku_id")["demand_qty"]
        prediction["actual_demand_qty"] = (
            prediction["sku_id"].astype(str).map(actual_lookup).fillna(0.0)
        )
    prediction["final_pred_qty_horizon"] = (
        prediction["final_pred_qty_30d"] * (float(horizon_days) / 30.0)
    )
    prediction["category_eval"] = source.get("category")
    prediction["style_id_eval"] = source.get("style_id")
    prediction["prediction_version"] = f"{track}_{tag}"
    return prediction


def _build_work_frame(
    track: str,
    anchor: str,
    inventory_daily: pd.DataFrame,
    products: pd.DataFrame,
    lifecycle: pd.DataFrame,
    style_costs: pd.DataFrame | None,
    horizon_days: int,
    daily_demand_df: pd.DataFrame,
    defaults: pd.DataFrame,
    max_rows: int | None,
) -> pd.DataFrame:
    prediction = _prediction_context(
        track,
        anchor,
        horizon_days=horizon_days,
        daily_demand_df=daily_demand_df,
    )
    inventory = build_inventory_snapshot(
        prediction_df=prediction,
        clean_inventory_df=inventory_daily,
        product_df=products,
        defaults_df=defaults,
        wide_table_df=None,
    )
    economics = build_economics_config(
        prediction_df=prediction,
        product_df=products,
        defaults_df=defaults,
        lifecycle_df=lifecycle,
        style_cost_df=style_costs,
    )
    work = prediction.merge(
        inventory,
        on=["sku_id", "snapshot_date"],
        how="inner",
        suffixes=("", "_inventory"),
    ).merge(
        economics,
        on="sku_id",
        how="inner",
        suffixes=("", "_economics"),
    )
    work["inventory_source_date"] = work["inventory_source_date"].fillna("")
    work["is_exact_real_inventory"] = (
        (work["inventory_snapshot_present"] == 1)
        & (work["inventory_source_date"] == work["snapshot_date"])
    )
    work = work[work["is_exact_real_inventory"]].copy()
    if max_rows is not None:
        work = work.head(max_rows).copy()
    return work


def _experiment_combinations(track: str) -> list[tuple[str, str]]:
    if track == "phase8_zero_split":
        return [("tag_price_100pct", "balanced")]
    return [
        ("tag_price_100pct", "conservative"),
        ("tag_price_100pct", "balanced"),
        ("tag_price_100pct", "aggressive"),
        ("tag_price_50pct", "balanced"),
        ("tag_price_30pct", "balanced"),
        ("tag_price_20pct", "balanced"),
        ("tag_price_50pct_holding", "balanced"),
    ]


def _objects(
    row: dict,
    economic_case: str,
    horizon_days: int,
) -> tuple[ModelOutput, InventoryState, Economics]:
    case = ECONOMIC_CASES[economic_case]
    model_output = ModelOutput(
        sku_id=row["sku_id"],
        snapshot_date=row["snapshot_date"],
        pred_prob_positive=row["pred_prob_positive"],
        pred_qty_30d=row["conditional_qty_30d"],
        prediction_version=row["prediction_version"],
    )
    inventory = InventoryState(
        sku_id=row["sku_id"],
        snapshot_date=row["snapshot_date"],
        current_inventory=row["current_inventory"],
        inbound_within_30d=row["inbound_within_30d"],
        lead_time_days=row["lead_time_days"],
        min_batch_qty=row["min_batch_qty"],
        increment_batch_qty=row["increment_batch_qty"],
        max_replenish_qty=row["max_replenish_qty"],
        safety_stock_qty=row["safety_stock_qty"],
    )
    base_economics = Economics(
        sku_id=row["sku_id"],
        unit_cost=row["unit_cost"],
        unit_price=row["unit_price"],
        holding_cost_per_unit_per_day=row["holding_cost_per_unit_per_day"],
        salvage_value_per_unit=row["salvage_value_per_unit"],
        stockout_penalty_per_unit=row["stockout_penalty_per_unit"],
        other_fixed_cost=row["other_fixed_cost"],
        lifecycle_end_date=None,
        target_sell_through_rate=row["target_sell_through_rate"],
        lifecycle_days=horizon_days,
    )
    economics = replace(
        base_economics,
        unit_price=base_economics.unit_price * case["unit_price_ratio"],
        holding_cost_per_unit_per_day=(
            base_economics.unit_cost * case["holding_cost_ratio"]
        ),
    )
    return model_output, inventory, economics


def _strategy_summary(frame: pd.DataFrame) -> dict:
    supply = frame["available_qty"].sum()
    actual = frame["actual_demand_qty"].sum()
    return {
        "rows": int(len(frame)),
        "positive_plan_rate": float((frame["plan_qty"] > 0).mean()),
        "total_plan_qty": float(frame["plan_qty"].sum()),
        "mean_plan_qty": float(frame["plan_qty"].mean()),
        "total_realized_profit": float(frame["realized_profit"].sum()),
        "mean_realized_profit": float(frame["realized_profit"].mean()),
        "stockout_rate": float(frame["stockout_flag"].mean()),
        "sell_through_rate": float(frame["sold_qty"].sum() / max(supply, 1e-9)),
        "lost_sales_rate": float(frame["lost_sales_qty"].sum() / max(actual, 1e-9)),
        "leftover_share_of_supply": float(frame["leftover_qty"].sum() / max(supply, 1e-9)),
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    shown = frame.loc[:, columns].copy()
    for col in shown.select_dtypes(include=[np.number]).columns:
        shown[col] = shown[col].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


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
            / f"profit_analysis_real_data_h{args.horizon_days}_20260612"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_demand_source = _load_daily_demand_source(args.wide_table_source)
    calibration_frame = _load_calibration_frame(
        horizon_days=args.horizon_days,
        daily_demand_df=daily_demand_source,
    )
    calibration = fit_demand_scenario_calibration(calibration_frame)
    calibrated_prob = np.array(
        [
            calibration.calibrate_probability(value)
            for value in calibration_frame["ai_pred_prob"]
        ]
    )
    actual_binary = (calibration_frame["true_replenish_qty"].to_numpy() > 0).astype(float)
    calibration_payload = calibration.to_dict()
    calibration_payload["raw_brier"] = float(
        np.mean((calibration_frame["ai_pred_prob"].to_numpy() - actual_binary) ** 2)
    )
    calibration_payload["calibrated_brier"] = float(
        np.mean((calibrated_prob - actual_binary) ** 2)
    )
    (output_dir / "demand_scenario_calibration.json").write_text(
        json.dumps(calibration_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
    cohort_rows: list[dict] = []
    for track in TRACKS:
        for anchor in anchors:
            work = _build_work_frame(
                track=track,
                anchor=anchor,
                inventory_daily=inventory_daily,
                products=products,
                lifecycle=lifecycle,
                style_costs=style_costs,
                horizon_days=args.horizon_days,
                daily_demand_df=daily_demand_source,
                defaults=defaults,
                max_rows=args.max_rows,
            )
            cohort_rows.append(
                {
                    "track": track,
                    "anchor_date": anchor,
                    "real_inventory_rows": int(len(work)),
                    "actual_positive_rows": int((work["actual_demand_qty"] > 0).sum()),
                    "actual_demand_qty": float(work["actual_demand_qty"].sum()),
                    "inventory_qty": float(work["current_inventory"].sum()),
                }
            )
            records = work.to_dict(orient="records")
            for economic_case, policy in _experiment_combinations(track):
                for row in records:
                    model_output, inventory, economics = _objects(
                        row,
                        economic_case,
                        horizon_days=args.horizon_days,
                    )
                    scenarios = calibration.build_scenarios(
                        model_output,
                        horizon_days=args.horizon_days,
                    )
                    recommendation = recommend_replenishment_plans(
                        model_output=model_output,
                        inventory_state=inventory,
                        economics=economics,
                        policy=policy,
                        demand_scenarios=scenarios,
                        horizon_days=args.horizon_days,
                    )
                    recommended_qty = float(
                        (recommendation["best_recommended_plan"] or {}).get("plan_qty", 0.0)
                    )
                    available_before_plan = (
                        float(row["current_inventory"])
                        + float(row["inbound_within_30d"])
                    )
                    plans = {
                        "no_replenishment": 0.0,
                        "model_direct": max(
                            float(row["final_pred_qty_horizon"]) - available_before_plan,
                            0.0,
                        ),
                        "profit_module": recommended_qty,
                        "hindsight_qty": max(
                            float(row["actual_demand_qty"]) - available_before_plan,
                            0.0,
                        ),
                    }
                    actual_curve = daily_curves.get(
                        (anchor, str(row["sku_id"])),
                        [0.0] * args.horizon_days,
                    )
                    calibrated_probability = calibration.calibrate_probability(
                        row["pred_prob_positive"]
                    )
                    for strategy, plan_qty in plans.items():
                        realized = realize_replenishment_plan(
                            model_output=model_output,
                            inventory_state=inventory,
                            economics=economics,
                            plan=CandidatePlan(plan_qty=plan_qty, policy=strategy),
                            actual_demand_qty=row["actual_demand_qty"],
                            horizon_days=args.horizon_days,
                            actual_daily_demand_curve=actual_curve,
                        )
                        detail_rows.append(
                            {
                                "track": track,
                                "anchor_date": anchor,
                                "economic_case": economic_case,
                                "policy": policy,
                                "strategy": strategy,
                                "sku_id": row["sku_id"],
                                "style_id": row.get("style_id"),
                                "category": row.get("category"),
                                "launch_date": row.get("launch_date"),
                                "inventory_source_date": row["inventory_source_date"],
                                "raw_pred_probability": row["pred_prob_positive"],
                                "calibrated_pred_probability": calibrated_probability,
                                "conditional_pred_qty_30d": row["conditional_qty_30d"],
                                "final_pred_qty_30d": row["final_pred_qty_30d"],
                                "final_pred_qty_horizon": row["final_pred_qty_horizon"],
                                "horizon_days": args.horizon_days,
                                "actual_demand_qty": row["actual_demand_qty"],
                                "current_inventory": row["current_inventory"],
                                "available_qty": (
                                    float(row["current_inventory"])
                                    + float(row["inbound_within_30d"])
                                    + float(realized.plan_qty)
                                ),
                                "unit_price": economics.unit_price,
                                "unit_cost": economics.unit_cost,
                                "expected_profit": (
                                    float(
                                        (recommendation["best_recommended_plan"] or {}).get(
                                            "expected_profit", 0.0
                                        )
                                    )
                                    if strategy == "profit_module"
                                    else math.nan
                                ),
                                **realized.to_dict(),
                            }
                        )

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(
        output_dir / "profit_real_data_experiment_detail.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cohorts = pd.DataFrame(cohort_rows)
    cohorts.to_csv(output_dir / "real_inventory_cohorts.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    group_cols = ["track", "anchor_date", "economic_case", "policy", "strategy"]
    for keys, frame in detail.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row.update(_strategy_summary(frame))
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
    summary.to_csv(
        output_dir / "profit_real_data_experiment_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    aggregate_rows = []
    aggregate_group_cols = ["track", "economic_case", "policy", "strategy"]
    for keys, frame in detail.groupby(aggregate_group_cols):
        row = dict(zip(aggregate_group_cols, keys))
        row.update(_strategy_summary(frame))
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
    aggregate.to_csv(
        output_dir / "profit_real_data_experiment_aggregate.csv",
        index=False,
        encoding="utf-8-sig",
    )

    primary = aggregate[
        (aggregate["track"] == "phase7_base")
        & (aggregate["policy"] == "balanced")
        & (aggregate["strategy"].isin(["no_replenishment", "model_direct", "profit_module"]))
    ].copy()
    default_policy = aggregate[
        (aggregate["track"] == "phase7_base")
        & (aggregate["economic_case"] == "tag_price_100pct")
        & (aggregate["strategy"] == "profit_module")
    ].copy()
    model_compare = aggregate[
        (aggregate["economic_case"] == "tag_price_100pct")
        & (aggregate["policy"] == "balanced")
        & (aggregate["strategy"] == "profit_module")
    ].copy()

    report = [
        "# 盈亏分析模块真实数据实验",
        "",
        f"- 运行时间：`{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "- 主实验模型：Phase7 LightGBM hurdle",
        "- 真实库存锚点：2026-02-15、2026-02-24",
        f"- 分析窗口：锚点后{args.horizon_days}天",
        f"- 实际结果：锚点后{args.horizon_days}天真实补货量及真实逐日补货节奏",
        "- 商品经济数据：真实SKU、SKC、吊牌价；真实成本优先，未匹配成本使用吊牌价/7",
        "- 限制：缺少真实成交价和确认在途订单，因此金额不是财务记账利润",
        "",
        "## 历史场景校准",
        "",
        f"- 校准样本：{calibration.calibration_rows:,}行",
        f"- 正需求量校准样本：{calibration.positive_calibration_rows:,}行",
        f"- 低/中/高倍数：{', '.join(f'{value:.4f}' for value in calibration.positive_multipliers)}",
        f"- 原始概率Brier：{calibration_payload['raw_brier']:.4f}",
        f"- 校准后概率Brier：{calibration_payload['calibrated_brier']:.4f}",
        "",
        "## 真实库存样本",
        "",
        _markdown_table(
            cohorts,
            [
                "track",
                "anchor_date",
                "real_inventory_rows",
                "actual_positive_rows",
                "actual_demand_qty",
                "inventory_qty",
            ],
        ),
        "",
        "## Phase7平衡策略成交价敏感性",
        "",
        _markdown_table(
            primary,
            [
                "economic_case",
                "strategy",
                "positive_plan_rate",
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
        _markdown_table(
            default_policy,
            [
                "policy",
                "positive_plan_rate",
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
        _markdown_table(
            model_compare,
            [
                "track",
                "positive_plan_rate",
                "total_plan_qty",
                "lost_sales_rate",
                "leftover_share_of_supply",
                "incremental_profit_vs_no_plan",
                "incremental_profit_vs_direct",
            ],
        ),
        "",
        "## 口径说明",
        "",
        "1. 所有主结果只使用锚点当天存在真实库存快照的SKU。",
        "2. 历史库存按SKU和锚点日期做as-of连接，禁止使用锚点之后的库存。",
        "3. 场景概率和需求倍数只使用2025年四个历史验证月校准，2026年锚点用于测试。",
        "4. 盈亏模块使用分类器概率和回归器条件数量；直接模型基线使用Phase7最终门控数量。",
        f"5. 真实逐日补货曲线和总量均直接来自宽表的未来{args.horizon_days}天窗口。",
    ]
    (output_dir / "profit_real_data_experiment_report.md").write_text(
        "\n".join(report),
        encoding="utf-8",
    )
    print(f"[OK] output -> {output_dir}")
    print(f"[OK] detail rows -> {len(detail):,}")
    print(f"[OK] exact real-inventory cohorts -> {cohorts['real_inventory_rows'].sum():,}")


if __name__ == "__main__":
    main()
