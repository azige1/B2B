import json
import math
import os
from datetime import timedelta

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE54_PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_4", "phase5")
PHASE55_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_5")
PHASE57_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_7")
PHASE6E_WINNER = os.path.join(PROJECT_ROOT, "reports", "phase6e_tree_validation_pack", "phase6e_tree_validation_pack_winner.json")
PHASE6H_READABLE = os.path.join(PROJECT_ROOT, "reports", "phase6h_december_readable_report", "dec_20251201_sku_compare_readable.csv")
STATIC_PATH = os.path.join(PROJECT_ROOT, "data", "gold", "wide_table_sku.csv")
SILVER_PATH = os.path.join(PROJECT_ROOT, "data", "silver", "clean_orders.csv")

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase7_tail_allocation_optimization")
OUT_TABLE = os.path.join(OUT_DIR, "phase7a_tail_gap_pack_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase7a_tail_gap_pack_summary.md")
OUT_CANDIDATES = os.path.join(OUT_DIR, "phase7a_tail_gap_candidates.json")

ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
BASELINE_EXP = "p527_lstm_l3_v5_lite_s2027"
CONFIRMED_TREE = "p535_tree_hard_cov_activity"
RAW_TEMPLATE = "p57a_{anchor_tag}_covact_lr005_l63_s2026_hard_g025"
CALIBRATION_SCALES = {"2025-09-01": 0.98, "2025-10-01": 0.93, "2025-11-01": 1.00, "2025-12-01": 1.00}


def ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def phase55_context_path(anchor_date, base_exp):
    return os.path.join(
        PHASE55_DIR,
        anchor_tag(anchor_date),
        "phase5",
        f"eval_context_p55_{anchor_tag(anchor_date)}_{base_exp}_s2026.csv",
    )


def phase54_dec_baseline_path():
    return os.path.join(PHASE54_PHASE_DIR, f"eval_context_p54_{BASELINE_EXP}_s2026.csv")


def phase57_context_path(anchor_date):
    exp_id = RAW_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
    return os.path.join(PHASE57_DIR, anchor_tag(anchor_date), "phase5", f"eval_context_{exp_id}.csv")


def apply_calibration(df, anchor_date):
    out = df.copy()
    scale = float(CALIBRATION_SCALES.get(anchor_date, 1.0))
    if scale != 1.0:
        for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
            out[col] = out[col].astype(float) * scale
    out["ai_pred_positive_qty"] = (out["ai_pred_qty"].astype(float) > 0).astype(int)
    out["abs_error"] = (out["ai_pred_qty"].astype(float) - out["true_replenish_qty"].astype(float)).abs()
    return out


def load_static_table():
    usecols = [
        "sku_id",
        "product_name",
        "category",
        "sub_category",
        "style_id",
        "season",
        "series",
        "band",
        "size_id",
        "color_id",
        "qty_first_order",
        "price_tag",
    ]
    static_df = pd.read_csv(STATIC_PATH, usecols=usecols)
    static_df["sku_id"] = static_df["sku_id"].astype(str)
    return static_df.drop_duplicates("sku_id")


def qfo_bucket(value):
    value = float(value)
    if value <= 0:
        return "0"
    if value <= 1:
        return "1"
    if value <= 10:
        return "2-10"
    if value <= 30:
        return "11-30"
    if value <= 100:
        return "31-100"
    return "100+"


def top1_bucket(value):
    if pd.isna(value) or float(value) <= 0:
        return "0"
    value = float(value)
    if value <= 0.34:
        return "(0,0.34]"
    if value <= 0.67:
        return "(0.34,0.67]"
    return "(0.67,1]"


def load_mainline_frames():
    frames = []
    for anchor_date in ANCHORS:
        raw = pd.read_csv(phase57_context_path(anchor_date))
        adj = apply_calibration(raw, anchor_date)
        adj["anchor_date"] = anchor_date
        frames.append(adj)
    return pd.concat(frames, ignore_index=True, sort=False)


def build_future_buyer_features(mainline_df):
    usecols = ["sku_id", "buyer_id", "order_date", "qty_future"]
    silver = pd.read_csv(SILVER_PATH, usecols=usecols)
    silver["sku_id"] = silver["sku_id"].astype(str)
    silver["buyer_id"] = silver["buyer_id"].astype(str)
    silver["order_date"] = pd.to_datetime(silver["order_date"]).dt.date

    rows = []
    sku_sets = {
        anchor_date: set(mainline_df.loc[mainline_df["anchor_date"] == anchor_date, "sku_id"].astype(str).unique())
        for anchor_date in ANCHORS
    }
    for anchor_date in ANCHORS:
        anchor = pd.to_datetime(anchor_date).date()
        start = anchor - timedelta(days=89)
        sub = silver[
            (silver["order_date"] >= start)
            & (silver["order_date"] <= anchor)
            & (silver["qty_future"] > 0)
            & (silver["sku_id"].isin(sku_sets[anchor_date]))
        ].copy()
        if sub.empty:
            continue
        buyer_level = sub.groupby(["sku_id", "buyer_id"], as_index=False)["qty_future"].sum()
        for sku_id, sku_df in buyer_level.groupby("sku_id"):
            qtys = sku_df["qty_future"].astype(float).sort_values(ascending=False).to_numpy()
            total = float(qtys.sum())
            if total <= 0:
                continue
            rows.append(
                {
                    "sku_id": str(sku_id),
                    "anchor_date": anchor_date,
                    "future_buyer_count_90": int(len(qtys)),
                    "future_top1_share_90": float(qtys[0] / total),
                    "future_top3_share_90": float(qtys[:3].sum() / total),
                    "future_hhi_90": float(np.sum((qtys / total) ** 2)),
                }
            )
    return pd.DataFrame(rows)


def build_current_mainline_frame():
    mainline = load_mainline_frames()
    static_df = load_static_table()
    buyer_df = build_future_buyer_features(mainline)
    out = mainline.merge(
        static_df,
        on=["sku_id"],
        how="left",
        suffixes=("", "_static"),
    )
    if not buyer_df.empty:
        out = out.merge(buyer_df, on=["sku_id", "anchor_date"], how="left")
    for col in ("future_buyer_count_90", "future_top1_share_90", "future_top3_share_90", "future_hhi_90"):
        if col not in out.columns:
            out[col] = np.nan
    out["qty_first_order_bucket"] = out["qty_first_order"].fillna(0.0).map(qfo_bucket)
    out["future_top1_bucket"] = out["future_top1_share_90"].map(top1_bucket)
    out["is_true_blockbuster"] = (out["true_replenish_qty"].astype(float) > 25).astype(int)
    out["ratio_mainline"] = out["ai_pred_qty"].astype(float) / out["true_replenish_qty"].replace(0, np.nan).astype(float)
    out["is_repl0_fut0"] = (out["signal_quadrant"].astype(str) == "repl0_fut0").astype(int)
    out["is_cold_start"] = (out["activity_bucket"].astype(str) == "ice").astype(int)
    return out


def load_phase6e_summary():
    with open(PHASE6E_WINNER, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    keep = []
    for row in payload["candidate_summary"]:
        if row["candidate_key"] in {"sep098_oct093", "sep095_oct090", "p535", "p527"}:
            keep.append(row)
    return pd.DataFrame(keep)


def aggregate_rows(df, group_cols, view_type):
    work = df.copy()
    if work.empty:
        return pd.DataFrame(columns=["view_type", "segment_key", "rows", "total_true", "total_pred", "ratio", "under_wape", "mean_true", "mean_pred", "mean_abs_error", "mean_qty_first_order"])
    grouped = work.groupby(group_cols, dropna=False, as_index=False).agg(
        rows=("sku_id", "count"),
        total_true=("true_replenish_qty", "sum"),
        total_pred=("ai_pred_qty", "sum"),
        mean_true=("true_replenish_qty", "mean"),
        mean_pred=("ai_pred_qty", "mean"),
        mean_abs_error=("abs_error", "mean"),
        mean_qty_first_order=("qty_first_order", "mean"),
    )
    grouped["ratio"] = grouped["total_pred"] / grouped["total_true"].replace(0, np.nan)
    grouped["under_wape"] = np.clip(grouped["total_true"] - grouped["total_pred"], a_min=0, a_max=None) / grouped["total_true"].replace(0, np.nan)
    grouped["view_type"] = view_type
    grouped["segment_key"] = grouped[group_cols].astype(str).agg(" | ".join, axis=1)
    return grouped


def build_top20_missed_skus(df):
    sku_agg = (
        df.groupby("sku_id", as_index=False)
        .agg(
            total_true=("true_replenish_qty", "sum"),
            total_pred=("ai_pred_qty", "sum"),
            product_name=("product_name", "first"),
            category=("category", "first"),
            sub_category=("sub_category", "first"),
            style_id=("style_id", "first"),
            qty_first_order=("qty_first_order", "first"),
        )
    )
    sku_agg = sku_agg[sku_agg["total_true"] > 0].copy()
    sku_agg["ratio"] = sku_agg["total_pred"] / sku_agg["total_true"]
    sku_agg["abs_error"] = (sku_agg["total_true"] - sku_agg["total_pred"]).abs()
    top_n = max(1, int(math.ceil(len(sku_agg) * 0.20)))
    true_top = set(sku_agg.sort_values(["total_true", "total_pred"], ascending=[False, False]).head(top_n)["sku_id"])
    pred_top = set(sku_agg.sort_values(["total_pred", "total_true"], ascending=[False, False]).head(top_n)["sku_id"])
    missed = sku_agg[sku_agg["sku_id"].isin(true_top - pred_top)].copy()
    missed["view_type"] = "top20_missed_skus"
    missed["segment_key"] = missed["sku_id"]
    return missed.sort_values(["total_true", "abs_error"], ascending=[False, False]).head(30)


def build_blockbuster_severe_under(df):
    sku_agg = (
        df.groupby("sku_id", as_index=False)
        .agg(
            total_true=("true_replenish_qty", "sum"),
            total_pred=("ai_pred_qty", "sum"),
            product_name=("product_name", "first"),
            category=("category", "first"),
            sub_category=("sub_category", "first"),
            style_id=("style_id", "first"),
            qty_first_order=("qty_first_order", "first"),
            signal_quadrant=("signal_quadrant", "first"),
            activity_bucket=("activity_bucket", "first"),
        )
    )
    sku_agg = sku_agg[sku_agg["total_true"] > 25].copy()
    sku_agg["ratio"] = sku_agg["total_pred"] / sku_agg["total_true"]
    sku_agg["abs_error"] = (sku_agg["total_true"] - sku_agg["total_pred"]).abs()
    sku_agg["view_type"] = "blockbuster_severe_under_skus"
    sku_agg["segment_key"] = sku_agg["sku_id"]
    return sku_agg.sort_values(["ratio", "abs_error"], ascending=[True, False]).head(30)


def candidate_json_payload():
    return {
        "phase": "phase7a_tail_gap_pack",
        "baseline_mainline": "p57_covact_lr005_l63_hard_g025 + sep098_oct093",
        "recommended_variants": [
            {
                "variant_key": "qfo_plus",
                "problem_target": "repl0_fut0 + ice + 高首单量的 tail 漏补",
                "signal_source": "qty_first_order bucket, cold-start blockbuster under, signal_quadrant mismatch",
                "why_now": "当前 blockbuster 最大漏补集中在弱动态信号但首单量很大的冷启动 SKU，上游证据最强。",
                "expected_metric_targets": ["blockbuster_under_wape", "blockbuster_sku_p50", "ice_4_25_sku_p50"],
            },
            {
                "variant_key": "tail_peak",
                "problem_target": ">25 尾部事件记忆不足",
                "signal_source": "blockbuster under, high-qty tail miss, top20 missed SKU concentration",
                "why_now": "当前模型对高需求 SKU 的 tail 放大量不足，需要补历史高量事件与极值记忆。",
                "expected_metric_targets": ["blockbuster_under_wape", "blockbuster_sku_p50", "top20_true_volume_capture"],
            },
            {
                "variant_key": "style_category_priors",
                "problem_target": "category/style 级 tail under",
                "signal_source": "blockbuster category/sub_category under summary",
                "why_now": "tail under 已经表现出明显的品类和款式聚类，适合引入历史强度先验。",
                "expected_metric_targets": ["blockbuster_under_wape", "blockbuster_sku_p50", "rank_corr_positive_skus"],
            },
            {
                "variant_key": "tail_full",
                "problem_target": "同时补 tail 与 allocation",
                "signal_source": "qfo, category priors, buyer concentration, top20 miss",
                "why_now": "当前 phase7 目标是同时改善 tail 和 allocation，最终候选必须验证组合方案。",
                "expected_metric_targets": ["blockbuster_under_wape", "blockbuster_sku_p50", "top20_true_volume_capture", "rank_corr_positive_skus"],
            },
        ],
    }


def render_summary(mainline_df, summary_df, tables):
    winner_row = summary_df.loc[summary_df["candidate_key"] == "sep098_oct093"].iloc[0]
    qfo_bucket = tables["blockbuster_qfo_bucket"].sort_values("mean_qty_first_order", ascending=False)
    quadrant = tables["blockbuster_signal_quadrant"].sort_values("under_wape", ascending=False)
    cat_under = tables["blockbuster_category_under"].sort_values("under_wape", ascending=False).head(8)
    top_missed = tables["top20_missed_skus"].head(10)
    severe_under = tables["blockbuster_severe_under_skus"].head(10)
    buyer_bucket = tables["blockbuster_buyer_bucket"].sort_values("under_wape", ascending=False)

    lines = [
        "# Phase7a Tail Gap Pack",
        "",
        "- baseline_mainline: `p57_covact_lr005_l63_hard_g025 + sep098_oct093`",
        f"- `4_25_under_wape`: `{float(winner_row['4_25_under_wape']):.4f}`",
        f"- `4_25_sku_p50`: `{float(winner_row['4_25_sku_p50']):.4f}`",
        f"- `ice_4_25_sku_p50`: `{float(winner_row['ice_4_25_sku_p50']):.4f}`",
        f"- `blockbuster_under_wape`: `{float(winner_row['blockbuster_under_wape']):.4f}`",
        f"- `blockbuster_sku_p50`: `{float(winner_row['blockbuster_sku_p50']):.4f}`",
        f"- `top20_true_volume_capture`: `{float(winner_row['top20_true_volume_capture']):.4f}`",
        f"- `rank_corr_positive_skus`: `{float(winner_row['rank_corr_positive_skus']):.4f}`",
        "",
        "## Key Findings",
        "",
        f"- `true_blockbuster_rows`: `{int((mainline_df['true_replenish_qty'] > 25).sum())}`",
        f"- `repl0_fut0 + ice + true_blockbuster` rows: `{int(((mainline_df['signal_quadrant'] == 'repl0_fut0') & (mainline_df['activity_bucket'] == 'ice') & (mainline_df['true_replenish_qty'] > 25)).sum())}`",
        "- 当前最大的 tail 漏补集中在 `repl0_fut0 + ice + 高首单量`。",
        "- 当前最大的 allocation 缺口不是整体排序崩，而是高价值 tail SKU 抓取不够集中。",
        "",
        "## Blockbuster by Signal Quadrant",
        "",
        quadrant[["segment_key", "rows", "total_true", "total_pred", "ratio", "under_wape", "mean_qty_first_order"]].to_markdown(index=False),
        "",
        "## Blockbuster by Qty First Order Bucket",
        "",
        qfo_bucket[["segment_key", "rows", "total_true", "total_pred", "ratio", "under_wape", "mean_qty_first_order"]].to_markdown(index=False),
        "",
        "## Blockbuster by Buyer Concentration Bucket",
        "",
        buyer_bucket[["segment_key", "rows", "total_true", "total_pred", "ratio", "under_wape"]].to_markdown(index=False),
        "",
        "## Worst Blockbuster Categories",
        "",
        cat_under[["segment_key", "rows", "total_true", "total_pred", "ratio", "under_wape"]].to_markdown(index=False),
        "",
        "## Top20 Missed SKUs",
        "",
        top_missed[["sku_id", "product_name", "category", "style_id", "qty_first_order", "total_true", "total_pred", "ratio"]].to_markdown(index=False),
        "",
        "## Severe Blockbuster Under SKUs",
        "",
        severe_under[["sku_id", "product_name", "category", "style_id", "qty_first_order", "total_true", "total_pred", "ratio", "activity_bucket", "signal_quadrant"]].to_markdown(index=False),
        "",
        "## Recommended Phase7b Variants",
        "",
        "- `qfo_plus`",
        "- `tail_peak`",
        "- `style_category_priors`",
        "- `tail_full`",
    ]
    return "\n".join(lines)


def main():
    ensure_dir()

    mainline_df = build_current_mainline_frame()
    summary_df = load_phase6e_summary()

    blockbuster_df = mainline_df[mainline_df["true_replenish_qty"].astype(float) > 25].copy()
    tables = {
        "blockbuster_signal_quadrant": aggregate_rows(blockbuster_df, ["signal_quadrant"], "blockbuster_signal_quadrant"),
        "blockbuster_qfo_bucket": aggregate_rows(blockbuster_df, ["qty_first_order_bucket"], "blockbuster_qfo_bucket"),
        "blockbuster_category_under": aggregate_rows(blockbuster_df, ["category"], "blockbuster_category_under"),
        "blockbuster_subcategory_under": aggregate_rows(blockbuster_df, ["sub_category"], "blockbuster_subcategory_under"),
        "blockbuster_buyer_bucket": aggregate_rows(blockbuster_df, ["future_top1_bucket"], "blockbuster_buyer_bucket"),
        "top20_missed_skus": build_top20_missed_skus(mainline_df),
        "blockbuster_severe_under_skus": build_blockbuster_severe_under(mainline_df),
    }

    summary_long = summary_df.copy()
    summary_long["view_type"] = "candidate_compare"
    summary_long["segment_key"] = summary_long["candidate_key"]

    export_frames = [summary_long]
    for df in tables.values():
        export_frames.append(df.copy())
    export_df = pd.concat(export_frames, ignore_index=True, sort=False)
    export_df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    with open(OUT_SUMMARY, "w", encoding="utf-8-sig") as fh:
        fh.write(render_summary(mainline_df, summary_df, tables))

    with open(OUT_CANDIDATES, "w", encoding="utf-8") as fh:
        json.dump(candidate_json_payload(), fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase7a tail gap pack table -> {OUT_TABLE}")
    print(f"[OK] phase7a tail gap pack summary -> {OUT_SUMMARY}")
    print(f"[OK] phase7a tail gap candidates -> {OUT_CANDIDATES}")


if __name__ == "__main__":
    main()
