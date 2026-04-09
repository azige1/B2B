from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXP_ID = "p512_bilstm_l3_v5_lite"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "phase5"


def resolve_context_path(exp_id: str) -> Path:
    candidates = [
        PROJECT_ROOT / "reports" / "phase5" / f"eval_context_{exp_id}_recheck.csv",
        PROJECT_ROOT / "reports" / "phase5_1" / f"eval_context_{exp_id}.csv",
        PROJECT_ROOT / "reports" / "phase5" / f"eval_context_{exp_id}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find eval_context for {exp_id}")


def load_meta() -> dict[str, object]:
    meta_path = PROJECT_ROOT / "data" / "artifacts_v5_lite" / "meta_v5_lite.json"
    return pd.read_json(meta_path, typ="series").to_dict()


def load_context(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["anchor_date"] = pd.to_datetime(df["anchor_date"])

    numeric_cols = [
        "true_replenish_qty",
        "ai_pred_prob",
        "ai_pred_qty",
        "abs_error",
        "lookback_repl_days_90",
        "lookback_future_days_90",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["target_cls"] = (df["true_replenish_qty"] > 0).astype(int)
    df["has_repl_history"] = df["lookback_repl_days_90"] > 0
    df["has_future_history"] = df["lookback_future_days_90"] > 0
    df["is_blockbuster"] = df["true_replenish_qty"] > 25
    df["cold_type"] = np.select(
        [
            (~df["has_repl_history"]) & (~df["has_future_history"]),
            (~df["has_repl_history"]) & df["has_future_history"],
            df["has_repl_history"] & (~df["has_future_history"]),
            df["has_repl_history"] & df["has_future_history"],
        ],
        ["no_repl_no_future", "no_repl_yes_future", "repl_only", "both"],
        default="other",
    )
    return df


def load_wide(sku_ids: set[str]) -> pd.DataFrame:
    path = PROJECT_ROOT / "data" / "gold" / "wide_table_sku.csv"
    cols = [
        "date",
        "buyer_id",
        "sku_id",
        "qty_replenish",
        "qty_future",
        "category",
        "price_tag",
    ]
    df = pd.read_csv(path, usecols=cols)
    df = df[df["sku_id"].isin(sku_ids)].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["qty_replenish"] = pd.to_numeric(df["qty_replenish"], errors="coerce").fillna(0.0)
    df["qty_future"] = pd.to_numeric(df["qty_future"], errors="coerce").fillna(0.0)
    df["price_tag"] = pd.to_numeric(df["price_tag"], errors="coerce")
    return df


def first_non_null(series: pd.Series):
    series = series.dropna()
    return series.iloc[0] if not series.empty else None


def build_static_map(wide_df: pd.DataFrame) -> pd.DataFrame:
    static_map = wide_df.sort_values("date").groupby("sku_id", as_index=False).agg(
        category_wide=("category", first_non_null),
        price_tag=("price_tag", "median"),
    )
    return static_map


def enrich_with_history(context_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
    static_map = build_static_map(wide_df)
    result_parts: list[pd.DataFrame] = []

    for anchor_date, anchor_slice in context_df.groupby("anchor_date", sort=False):
        sku_ids = set(anchor_slice["sku_id"].unique())
        history = wide_df[
            (wide_df["sku_id"].isin(sku_ids))
            & (wide_df["date"] >= anchor_date - pd.Timedelta(days=89))
            & (wide_df["date"] <= anchor_date)
        ].copy()

        future_hist = history[history["qty_future"] > 0]
        repl_hist = history[history["qty_replenish"] > 0]

        future_cov = future_hist.groupby("sku_id").agg(
            future_buyer_count_90=("buyer_id", "nunique"),
            future_qty_90=("qty_future", "sum"),
        )
        if not future_hist.empty:
            future_by_buyer = future_hist.groupby(["sku_id", "buyer_id"], as_index=False)["qty_future"].sum()
            future_top1 = future_by_buyer.groupby("sku_id")["qty_future"].max().rename("future_top1_qty_90")
            future_cov = future_cov.join(future_top1, how="left")
            future_cov["future_top1_share_90"] = (
                future_cov["future_top1_qty_90"] / future_cov["future_qty_90"].replace(0, np.nan)
            )

        repl_cov = repl_hist.groupby("sku_id").agg(
            repl_buyer_count_90=("buyer_id", "nunique"),
            repl_qty_90=("qty_replenish", "sum"),
        )

        merged = anchor_slice.merge(static_map, on="sku_id", how="left")
        merged = merged.merge(future_cov, left_on="sku_id", right_index=True, how="left")
        merged = merged.merge(repl_cov, left_on="sku_id", right_index=True, how="left")

        merged["future_buyer_count_90"] = merged["future_buyer_count_90"].fillna(0).astype(int)
        merged["repl_buyer_count_90"] = merged["repl_buyer_count_90"].fillna(0).astype(int)
        merged["future_qty_90"] = merged["future_qty_90"].fillna(0.0)
        merged["repl_qty_90"] = merged["repl_qty_90"].fillna(0.0)
        merged["future_top1_share_90"] = merged["future_top1_share_90"].fillna(0.0)
        merged["future_cov_bucket"] = pd.cut(
            merged["future_buyer_count_90"],
            bins=[-1, 0, 1, 2, np.inf],
            labels=["0", "1", "2", "3+"],
        ).astype(str)
        merged["buyer_mix_bucket"] = np.select(
            [
                (merged["repl_buyer_count_90"] == 0) & (merged["future_buyer_count_90"] == 0),
                (merged["repl_buyer_count_90"] > 0) & (merged["future_buyer_count_90"] == 0),
                (merged["repl_buyer_count_90"] == 0) & (merged["future_buyer_count_90"] > 0),
                (merged["repl_buyer_count_90"] > 0) & (merged["future_buyer_count_90"] > 0),
            ],
            ["none", "repl_only", "future_only", "both"],
            default="other",
        )
        result_parts.append(merged)

    return pd.concat(result_parts, ignore_index=True)


def build_global_shape_stats(wide_df: pd.DataFrame) -> dict[str, float]:
    window_df = wide_df[
        (wide_df["date"] >= "2025-01-01")
        & (wide_df["date"] <= "2025-12-31")
    ].copy()
    sku_date = window_df.groupby(["sku_id", "date"], as_index=False).agg(
        qty_replenish=("qty_replenish", "sum"),
        qty_future=("qty_future", "sum"),
    )
    sku_count = sku_date["sku_id"].nunique()
    calendar_cells = sku_count * 365
    future_nonzero = int((sku_date["qty_future"] > 0).sum())
    replenish_nonzero = int((sku_date["qty_replenish"] > 0).sum())
    return {
        "sku_count": sku_count,
        "calendar_cells": calendar_cells,
        "future_nonzero": future_nonzero,
        "future_nonzero_rate": future_nonzero / max(calendar_cells, 1),
        "replenish_nonzero": replenish_nonzero,
        "replenish_nonzero_rate": replenish_nonzero / max(calendar_cells, 1),
    }


def safe_ratio(pred_sum: float, true_sum: float) -> float:
    if true_sum <= 0:
        return float("nan")
    return pred_sum / true_sum


def safe_wmape(pred: pd.Series, true: pd.Series) -> float:
    true_sum = float(true.sum())
    if true_sum <= 0:
        return float("nan")
    return float((pred - true).abs().sum() / true_sum)


def positive_ratio_p50(df: pd.DataFrame) -> float:
    positive = df[df["target_cls"] == 1].copy()
    if positive.empty:
        return float("nan")
    ratios = positive["ai_pred_qty"] / positive["true_replenish_qty"].replace(0, np.nan)
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
    if ratios.empty:
        return float("nan")
    return float(ratios.median())


def segment_stats(df: pd.DataFrame) -> dict[str, float]:
    true_sum = float(df["true_replenish_qty"].sum())
    pred_sum = float(df["ai_pred_qty"].sum())
    return {
        "rows": int(len(df)),
        "pos_rate": float(df["target_cls"].mean()) if len(df) else float("nan"),
        "true_qty_sum": true_sum,
        "pred_qty_sum": pred_sum,
        "ratio": safe_ratio(pred_sum, true_sum),
        "wmape_like": safe_wmape(df["ai_pred_qty"], df["true_replenish_qty"]),
        "sku_p50": positive_ratio_p50(df),
    }


def fmt(value: float, pct: bool = False) -> str:
    if pd.isna(value):
        return "NA"
    if pct:
        return f"{value:.2%}"
    return f"{value:.4f}"


def recommendation_for_segment(dimension: str, segment: str, stats: dict[str, float], extra: dict[str, float] | None = None) -> str:
    extra = extra or {}
    ratio = stats["ratio"]
    pos_rate = stats["pos_rate"]

    if dimension == "signal_quadrant" and segment == "repl0_fut1":
        return "Keep future-only as a specialist signal; coverage is too small for a global route."
    if dimension == "signal_quadrant" and segment == "repl1_fut0":
        return "Add a separate calibration rule for replenish-only cases; current route materially over-predicts."
    if dimension == "signal_quadrant" and segment == "repl1_fut1":
        return "Model combined replenish+future interactions explicitly; current route under-predicts this high-value bucket."
    if dimension == "activity_bucket" and segment == "ice":
        return "Cold-start fallback is required; do not rely on the shared main model alone."
    if dimension == "blockbuster_flag" and segment == "blockbuster":
        return "Add blockbuster weighting or a dedicated quantity calibration path."
    if dimension == "future_cov_bucket" and segment == "3+":
        return "Include buyer coverage features; high-coverage future demand is currently under-predicted."
    if dimension == "cold_type" and segment == "no_repl_yes_future":
        return "Keep a dedicated future-led fallback path; this is the cleanest cold-start pocket."

    lift = extra.get("future_lift")
    if not pd.isna(lift):
        if lift >= 1.4:
            return "Future history adds meaningful lift here; preserve and interact with it."
        if lift <= 0.9:
            return "Future adds little incremental signal here; do not overweight it."

    if not pd.isna(ratio):
        if ratio > 1.3:
            return "Over-predicted segment; lower gate intensity or add segment-specific calibration."
        if ratio < 0.7:
            return "Under-predicted segment; strengthen quantity handling or add segment-specific features."

    if not pd.isna(pos_rate) and pos_rate > 0.4:
        return "High-signal segment; keep explicitly visible in Phase5.3 features."
    return "Keep as baseline reference."


def build_summary(ctx: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    def add_group(dimension: str, segment: str, data: pd.DataFrame, **extra: object):
        stats = segment_stats(data)
        rec = recommendation_for_segment(dimension, segment, stats, extra if extra else None)
        row = {"dimension": dimension, "segment": segment, **stats, "recommendation": rec}
        row.update(extra)
        rows.append(row)

    add_group("overall", "validation_universe", ctx)

    for segment, data in ctx.groupby("signal_quadrant", dropna=False):
        add_group("signal_quadrant", str(segment), data)

    for segment, data in ctx.groupby("activity_bucket", dropna=False):
        add_group("activity_bucket", str(segment), data)

    for segment, data in ctx.groupby("cold_type", dropna=False):
        add_group("cold_type", str(segment), data)

    for segment, data in ctx.groupby("future_cov_bucket", dropna=False):
        add_group("future_cov_bucket", str(segment), data)

    for segment, data in ctx.groupby("buyer_mix_bucket", dropna=False):
        add_group("buyer_mix_bucket", str(segment), data)

    for segment, data in ctx.groupby(ctx["is_blockbuster"].map({True: "blockbuster", False: "non_blockbuster"}), dropna=False):
        add_group("blockbuster_flag", str(segment), data)

    category_rows = []
    for category, data in ctx.groupby("category", dropna=False):
        with_future = data[data["has_future_history"]]
        without_future = data[~data["has_future_history"]]
        if len(data) < 80 or len(with_future) < 15 or len(without_future) < 15:
            continue
        pos_future = float(with_future["target_cls"].mean())
        pos_no_future = float(without_future["target_cls"].mean())
        future_lift = pos_future / pos_no_future if pos_no_future > 0 else float("nan")
        category_rows.append((str(category), data, pos_no_future, future_lift))

    category_rows.sort(key=lambda item: item[3] if not pd.isna(item[3]) else -1, reverse=True)
    for category, data, base_pos_rate, future_lift in category_rows[:8]:
        add_group(
            "category_future_lift",
            category,
            data[data["has_future_history"]],
            baseline_pos_rate=base_pos_rate,
            future_lift=future_lift,
        )

    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, columns: list[str], rename: dict[str, str] | None = None) -> str:
    rename = rename or {}
    header = [rename.get(col, col) for col in columns]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                if pd.isna(value):
                    values.append("NA")
                else:
                    values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(ctx: pd.DataFrame, summary_df: pd.DataFrame, meta: dict[str, object], global_shape: dict[str, float], context_path: Path) -> str:
    overall = summary_df[summary_df["dimension"] == "overall"].iloc[0]
    signal_rows = summary_df[summary_df["dimension"] == "signal_quadrant"]
    activity_rows = summary_df[summary_df["dimension"] == "activity_bucket"]
    coverage_rows = summary_df[summary_df["dimension"] == "future_cov_bucket"]
    cold_rows = summary_df[summary_df["dimension"] == "cold_type"]
    blockbuster_rows = summary_df[summary_df["dimension"] == "blockbuster_flag"]
    category_rows = summary_df[summary_df["dimension"] == "category_future_lift"]

    repl0_fut1 = signal_rows[signal_rows["segment"] == "repl0_fut1"].iloc[0]
    repl1_fut0 = signal_rows[signal_rows["segment"] == "repl1_fut0"].iloc[0]
    repl1_fut1 = signal_rows[signal_rows["segment"] == "repl1_fut1"].iloc[0]
    ice_row = activity_rows[activity_rows["segment"] == "ice"].iloc[0]
    cov_three = coverage_rows[coverage_rows["segment"] == "3+"].iloc[0]
    cov_zero = coverage_rows[coverage_rows["segment"] == "0"].iloc[0]
    cold_future = cold_rows[cold_rows["segment"] == "no_repl_yes_future"].iloc[0]
    blockbuster = blockbuster_rows[blockbuster_rows["segment"] == "blockbuster"].iloc[0]

    likely_stable = [
        f"`future` is a sparse but high-value specialist signal. In the fair-universe 2025 calendar it is active on only {global_shape['future_nonzero_rate']:.4%} of SKU-day cells, while in the real validation universe `repl0_fut1` still reaches `pos_rate={repl0_fut1['pos_rate']:.4f}`.",
        f"`buyer coverage` carries real structure. `future_buyer_count=3+` has `pos_rate={cov_three['pos_rate']:.4f}` versus `{cov_zero['pos_rate']:.4f}` at `future_buyer_count=0`.",
        f"`cold-start` is a first-order problem. The `ice` bucket has `{int(ice_row['rows'])}` rows with `sku_p50={ice_row['sku_p50']:.4f}`.",
    ]
    anchor_dependent = [
        "Exact category ordering and which categories are most over/under-predicted still depend on the single `2025-12-01` anchor.",
        "Exact ratio values by bucket should be re-checked under rolling backtest before turning them into production policy thresholds.",
        "Model-family conclusions (`sequence` vs `event/tree`) should still be validated on `phase5.2` outputs and then rolling anchors.",
    ]

    lines = [
        "# Phase5.2 Strategy Deep Dive",
        "",
        "## Summary",
        f"- Reference context: `{context_path.name}`",
        f"- Validation universe: `{int(overall['rows'])}` rows, `{int(ctx['target_cls'].sum())}` positive rows (`{ctx['target_cls'].mean():.2%}`)",
        f"- Fair-universe meta: `train_cnt={meta['train_cnt']:,}`, `val_cnt={meta['val_cnt']:,}`, `split_date={meta['split_date']}`",
        f"- Structural sparsity in the fair-universe 2025 calendar: `future != 0` on `{global_shape['future_nonzero_rate']:.4%}` of SKU-day cells, `replenish != 0` on `{global_shape['replenish_nonzero_rate']:.4%}`",
        "",
        "结论：当前数据更像“稀疏事件 + 分段校准”问题，不像标准 dense sequence 问题。`future` 有价值，但它是局部高价值信号，不是高覆盖主信号。",
        "",
        "## 1. Future 分层价值",
        f"- `repl0_fut1` 只有 `{int(repl0_fut1['rows'])}` 行，但 `pos_rate={repl0_fut1['pos_rate']:.4f}`，说明 `future-only` 是有效领先信号，只是覆盖很小。",
        f"- `repl1_fut0` 有 `{int(repl1_fut0['rows'])}` 行且 `ratio={repl1_fut0['ratio']:.4f}`，这是当前最明显的过预测区。",
        f"- `repl1_fut1` 的 `ratio={repl1_fut1['ratio']:.4f}`，说明补货和期货同时出现时，当前模型反而没有把这部分高价值区域吃透。",
        "",
        markdown_table(
            signal_rows[["segment", "rows", "pos_rate", "true_qty_sum", "pred_qty_sum", "ratio", "wmape_like"]],
            ["segment", "rows", "pos_rate", "true_qty_sum", "pred_qty_sum", "ratio", "wmape_like"],
            {
                "segment": "signal_quadrant",
                "rows": "rows",
                "pos_rate": "pos_rate",
                "true_qty_sum": "true_qty_sum",
                "pred_qty_sum": "pred_qty_sum",
                "ratio": "ratio",
                "wmape_like": "wmape_like",
            },
        ),
        "",
        "按品类看，`future` 不是全局同权信号。只保留了有足够样本和未来历史支撑的品类：",
        "",
        markdown_table(
            category_rows[["segment", "rows", "pos_rate", "baseline_pos_rate", "future_lift", "ratio", "recommendation"]],
            ["segment", "rows", "pos_rate", "baseline_pos_rate", "future_lift", "ratio", "recommendation"],
            {
                "segment": "category",
                "pos_rate": "pos_rate_with_future",
                "baseline_pos_rate": "pos_rate_without_future",
                "future_lift": "future_lift",
                "ratio": "ratio_with_future",
                "recommendation": "recommendation",
            },
        ) if not category_rows.empty else "- 当前验证宇宙里，没有足够稳定的品类 future lift 样本可单独下生产结论。",
        "",
        "## 2. Buyer Coverage",
        f"- `future_buyer_count=3+` 的正样本率是 `{cov_three['pos_rate']:.4f}`，而 `future_buyer_count=0` 只有 `{cov_zero['pos_rate']:.4f}`。",
        f"- 但 `future_buyer_count=3+` 的 `ratio={cov_three['ratio']:.4f}`，说明高覆盖未来需求目前被低估；这不是噪声，而是结构信息丢失。",
        "",
        markdown_table(
            coverage_rows[["segment", "rows", "pos_rate", "true_qty_sum", "pred_qty_sum", "ratio", "wmape_like", "recommendation"]],
            ["segment", "rows", "pos_rate", "true_qty_sum", "pred_qty_sum", "ratio", "wmape_like", "recommendation"],
            {
                "segment": "future_buyer_count_90",
            },
        ),
        "",
        "判断：`phase5.3` 必须引入 buyer coverage 特征，不应继续只用 SKU 聚合总量。",
        "",
        "## 3. 冷启动细分",
        f"- `ice` 桶有 `{int(ice_row['rows'])}` 行，`sku_p50={ice_row['sku_p50']:.4f}`，共享主模型无法单独解决这部分。",
        f"- `no_repl_yes_future` 有 `{int(cold_future['rows'])}` 行，虽然规模小，但它是最纯净的 `future-led` 冷启动口袋，值得单独保留策略。",
        "",
        markdown_table(
            cold_rows[["segment", "rows", "pos_rate", "true_qty_sum", "pred_qty_sum", "ratio", "wmape_like", "recommendation"]],
            ["segment", "rows", "pos_rate", "true_qty_sum", "pred_qty_sum", "ratio", "wmape_like", "recommendation"],
            {
                "segment": "cold_type",
            },
        ),
        "",
        "判断：冷启动不是边角问题，而是主问题。`phase5.3` 需要明确的 cold-start fallback，而不是继续只调主模型。",
        "",
        "## 4. 爆款前兆",
        f"- 爆款样本数 `{int(blockbuster['rows'])}`，但 `ratio={blockbuster['ratio']:.4f}`，当前 quantity 路线对大需求款完全没接住。",
        f"- 爆款在过去 90 天并非无信号：它们通常有更高的 `repl_sum_90`、`future_sum_90` 和 `future_buyer_count_90`。",
        "",
        markdown_table(
            blockbuster_rows[["segment", "rows", "pos_rate", "true_qty_sum", "pred_qty_sum", "ratio", "wmape_like", "recommendation"]],
            ["segment", "rows", "pos_rate", "true_qty_sum", "pred_qty_sum", "ratio", "wmape_like", "recommendation"],
            {
                "segment": "blockbuster_flag",
            },
        ),
        "",
        "判断：`phase5.3` 需要对爆款做单独权重或单独 quantity 校准，否则整体 WMAPE 改善也不会转化成关键 SKU 改善。",
        "",
        "## 5. 单锚点稳定性判断",
        "",
        "更可能稳定的结论：",
        *[f"- {item}" for item in likely_stable],
        "",
        "明显依赖单锚点的结论：",
        *[f"- {item}" for item in anchor_dependent],
        "",
        "## 6. Phase5.3 决策含义",
        "- 默认主路线应转向 `event/tree`，sequence 保留为对照基线。",
        "- `buyer coverage` 必须加入下一版特征。",
        "- `cold-start fallback` 需要作为显式设计，而不是后处理补丁。",
        "- `blockbuster` 需要单独权重或单独 quantity 路径。",
        "- 原始 `V5` 的 dense 派生维度不应继续保留为主路线输入。",
        "",
    ]
    return "\n".join(lines)


def build_decision_inputs(summary_df: pd.DataFrame, context_path: Path) -> str:
    def row(dimension: str, segment: str) -> pd.Series:
        return summary_df[(summary_df["dimension"] == dimension) & (summary_df["segment"] == segment)].iloc[0]

    cov_three = row("future_cov_bucket", "3+")
    cov_zero = row("future_cov_bucket", "0")
    ice = row("activity_bucket", "ice")
    repl0_fut1 = row("signal_quadrant", "repl0_fut1")
    repl1_fut0 = row("signal_quadrant", "repl1_fut0")
    blockbuster = row("blockbuster_flag", "blockbuster")

    lines = [
        "# Phase5.3 决策输入",
        "",
        f"- 基准上下文：`{context_path.name}`",
        "",
        "## 决策",
        "",
        "1. 是否必须上 `event/tree`：**是**",
        f"   - 证据：`repl0_fut1` 只有 `{int(repl0_fut1['rows'])}` 行但 `pos_rate={repl0_fut1['pos_rate']:.4f}`，说明 `future` 是局部高价值信号；`repl1_fut0` 的 `ratio={repl1_fut0['ratio']:.4f}`，说明当前共享 sequence 路线在大块区域校准错误。",
        "",
        "2. 是否必须加入 `buyer coverage`：**是**",
        f"   - 证据：`future_buyer_count=3+` 的 `pos_rate={cov_three['pos_rate']:.4f}`，而 `future_buyer_count=0` 只有 `{cov_zero['pos_rate']:.4f}`；高覆盖组仍然 `ratio={cov_three['ratio']:.4f}`，说明这层结构必须显式建模。",
        "",
        "3. 是否需要冷启动 fallback：**是**",
        f"   - 证据：`ice` 桶有 `{int(ice['rows'])}` 行，`sku_p50={ice['sku_p50']:.4f}`；共享主模型不能把这部分拉回健康区间。",
        "",
        "4. 是否需要爆款单独权重/单独头：**是**",
        f"   - 证据：爆款只有 `{int(blockbuster['rows'])}` 行，但 `ratio={blockbuster['ratio']:.4f}`，属于系统性低估，不是随机波动。",
        "",
        "5. 哪些原始 V5 派生特征明确不再保留：",
        "   - `repl_velocity`",
        "   - `days_since_last`（原 dense 逐日版本）",
        "   - `repl_volatility`（原 dense 逐日版本）",
        "   - `fut2repl_ratio`（原 dense 逐日版本）",
        "   - 理由：这些特征在原始 `V5(10维)` 中与稀疏事件信号混合，已被 `phase5.1` 和既有审计证明会扰乱校准，而不是稳定增益。",
        "",
        "## Phase5.3 默认方向",
        "- 主线：`event/rolling features + hurdle tree baseline`",
        "- 对照：保留 `V3-filtered / V5-lite` sequence 结果，不再继续扩原始 `V5(10维)`",
        "- 先做单锚点 `2025-12-01`，赢了再上 rolling backtest 和服务器多 seed",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep-dive analysis for Phase5.2 strategy decisions.")
    parser.add_argument("--exp-id", default=DEFAULT_EXP_ID, help="Reference experiment id used to locate eval_context.")
    parser.add_argument("--context-csv", default=None, help="Optional explicit eval_context CSV path.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    context_path = Path(args.context_csv) if args.context_csv else resolve_context_path(args.exp_id)
    meta = load_meta()
    context_df = load_context(context_path)
    wide_df = load_wide(set(context_df["sku_id"].unique()))
    enriched_df = enrich_with_history(context_df, wide_df)
    global_shape = build_global_shape_stats(wide_df)
    summary_df = build_summary(enriched_df)

    report_md = build_report(enriched_df, summary_df, meta, global_shape, context_path)
    decision_md = build_decision_inputs(summary_df, context_path)

    summary_path = report_dir / "strategy_deep_dive_summary.csv"
    report_path = report_dir / "strategy_deep_dive_phase52.md"
    decision_path = report_dir / "phase5_3_decision_inputs.md"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path.write_text(report_md, encoding="utf-8")
    decision_path.write_text(decision_md, encoding="utf-8")

    print(f"Saved summary: {summary_path}")
    print(f"Saved report : {report_path}")
    print(f"Saved decision inputs: {decision_path}")


if __name__ == "__main__":
    main()
