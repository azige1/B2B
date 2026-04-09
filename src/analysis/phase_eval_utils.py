import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def parse_exp_id_from_context(path):
    name = os.path.basename(path)
    return name[len("eval_context_") :].rsplit(".", 1)[0]


def aggregate_sku(df):
    agg = (
        df.groupby("sku_id", as_index=False)
        .agg(
            total_true=("true_replenish_qty", "sum"),
            total_pred=("ai_pred_qty", "sum"),
            lookback_repl_days_90=("lookback_repl_days_90", "max"),
        )
    )
    agg = agg[agg["total_true"] > 0].copy()
    if agg.empty:
        agg["sku_ratio"] = np.nan
        return agg
    agg["sku_ratio"] = agg["total_pred"] / agg["total_true"]
    return agg


def trimmed_mean(values, trim_ratio=0.10):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if arr.size < 3:
        return float(arr.mean())

    arr.sort()
    trim_n = int(np.floor(arr.size * trim_ratio))
    if trim_n == 0 or (trim_n * 2) >= arr.size:
        return float(arr.mean())
    return float(arr[trim_n : arr.size - trim_n].mean())


def safe_spearman(a, b):
    sa = pd.Series(a, dtype=float)
    sb = pd.Series(b, dtype=float)
    if len(sa) < 2 or sa.nunique(dropna=True) < 2 or sb.nunique(dropna=True) < 2:
        return np.nan
    return float(sa.corr(sb, method="spearman"))


def top_true_volume_capture(sku_df, frac):
    if sku_df.empty:
        return np.nan
    true_sum = float(sku_df["total_true"].sum())
    if true_sum <= 0:
        return np.nan
    top_n = max(1, int(np.ceil(len(sku_df) * frac)))
    top = sku_df.sort_values(["total_pred", "total_true"], ascending=[False, False]).head(top_n)
    return float(top["total_true"].sum() / true_sum)


def decision_error_metrics(df):
    true_qty = df["true_replenish_qty"].astype(float)
    pred_qty = df["ai_pred_qty"].astype(float)

    true_pos = true_qty > 0
    true_zero = true_qty <= 0
    true_gt_10 = true_qty > 10

    return {
        "false_zero_rate": float((pred_qty.loc[true_pos] <= 0).mean()) if true_pos.any() else np.nan,
        "false_positive_rate_zero_true": float((pred_qty.loc[true_zero] > 0).mean()) if true_zero.any() else np.nan,
        "zero_true_pred_ge_3_rate": float((pred_qty.loc[true_zero] >= 3).mean()) if true_zero.any() else np.nan,
        "true_gt_10_pred_le_1_rate": float((pred_qty.loc[true_gt_10] <= 1).mean()) if true_gt_10.any() else np.nan,
    }


def category_diagnostics(df):
    work = df.copy()
    if "category" not in work.columns:
        return {"category_worst5_ratio": "[]", "category_worst5_under_wape": "[]"}

    work["category"] = work["category"].fillna("Unknown").astype(str)
    grouped = (
        work.groupby("category", as_index=False)
        .agg(
            total_true=("true_replenish_qty", "sum"),
            total_pred=("ai_pred_qty", "sum"),
        )
    )
    grouped["under_sum"] = (
        work.assign(under_qty=np.clip(work["true_replenish_qty"].astype(float) - work["ai_pred_qty"].astype(float), a_min=0, a_max=None))
        .groupby("category")["under_qty"]
        .sum()
        .reindex(grouped["category"])
        .to_numpy()
    )
    grouped = grouped[grouped["total_true"] > 0].copy()
    if grouped.empty:
        return {"category_worst5_ratio": "[]", "category_worst5_under_wape": "[]"}

    grouped["ratio"] = grouped["total_pred"] / grouped["total_true"]
    grouped["under_wape"] = grouped["under_sum"] / grouped["total_true"]
    grouped["ratio_score"] = np.abs(np.log(np.clip(grouped["ratio"].astype(float), 1e-9, None)))

    worst_ratio = grouped.sort_values(["ratio_score", "total_true"], ascending=[False, False]).head(5)
    worst_under = grouped.sort_values(["under_wape", "total_true"], ascending=[False, False]).head(5)

    return {
        "category_worst5_ratio": json.dumps(
            worst_ratio[["category", "ratio", "total_true", "total_pred"]].to_dict(orient="records"),
            ensure_ascii=False,
        ),
        "category_worst5_under_wape": json.dumps(
            worst_under[["category", "under_wape", "total_true", "total_pred"]].to_dict(orient="records"),
            ensure_ascii=False,
        ),
    }


def quadrant_metrics(df):
    work = df.copy()
    if "signal_quadrant" not in work.columns:
        return {}

    quadrants = ["repl0_fut0", "repl0_fut1", "repl1_fut0", "repl1_fut1"]
    work["signal_quadrant"] = work["signal_quadrant"].fillna("unknown").astype(str)
    out = {}
    for quad in quadrants:
        sub = work[work["signal_quadrant"] == quad].copy()
        true_sum = float(sub["true_replenish_qty"].sum())
        pred_sum = float(sub["ai_pred_qty"].sum())
        under_sum = float(np.clip(sub["true_replenish_qty"].astype(float) - sub["ai_pred_qty"].astype(float), a_min=0, a_max=None).sum())
        out[f"{quad}_rows"] = int(len(sub))
        out[f"{quad}_ratio"] = pred_sum / true_sum if true_sum > 0 else np.nan
        out[f"{quad}_under_wape"] = under_sum / true_sum if true_sum > 0 else np.nan
    return out


def compute_slice_metrics(sku_df, mask):
    sub = sku_df.loc[mask].copy()
    if sub.empty:
        return {
            "rows": 0,
            "ratio": np.nan,
            "wmape_like": np.nan,
            "sku_p50": np.nan,
            "sku_trimmed_mean_10": np.nan,
            "within_20pct_rate": np.nan,
            "within_30pct_rate": np.nan,
            "within_50pct_rate": np.nan,
            "catastrophic_under_rate": np.nan,
            "catastrophic_over_rate": np.nan,
            "under_wape": np.nan,
            "over_wape": np.nan,
        }

    true_sum = float(sub["total_true"].sum())
    pred_sum = float(sub["total_pred"].sum())
    abs_sum = float(np.abs(sub["total_pred"] - sub["total_true"]).sum())
    under_sum = float(np.clip(sub["total_true"] - sub["total_pred"], a_min=0, a_max=None).sum())
    over_sum = float(np.clip(sub["total_pred"] - sub["total_true"], a_min=0, a_max=None).sum())
    ratios = sub["sku_ratio"].astype(float)
    return {
        "rows": int(len(sub)),
        "ratio": pred_sum / true_sum if true_sum > 0 else np.nan,
        "wmape_like": abs_sum / true_sum if true_sum > 0 else np.nan,
        "sku_p50": float(ratios.median()),
        "sku_trimmed_mean_10": trimmed_mean(ratios.values, trim_ratio=0.10),
        "within_20pct_rate": float(((ratios >= 0.8) & (ratios <= 1.2)).mean()),
        "within_30pct_rate": float(((ratios >= 0.7) & (ratios <= 1.3)).mean()),
        "within_50pct_rate": float(((ratios >= 0.5) & (ratios <= 1.5)).mean()),
        "catastrophic_under_rate": float((ratios < 0.3).mean()),
        "catastrophic_over_rate": float((ratios > 2.0).mean()),
        "under_wape": under_sum / true_sum if true_sum > 0 else np.nan,
        "over_wape": over_sum / true_sum if true_sum > 0 else np.nan,
    }


def bucket_masks(sku_df):
    return {
        "1_3": (sku_df["total_true"] >= 1) & (sku_df["total_true"] <= 3),
        "4_10": (sku_df["total_true"] >= 4) & (sku_df["total_true"] <= 10),
        "11_25": (sku_df["total_true"] >= 11) & (sku_df["total_true"] <= 25),
        "blockbuster": sku_df["total_true"] > 25,
        "ice": sku_df["lookback_repl_days_90"] == 0,
        "4_25": (sku_df["total_true"] >= 4) & (sku_df["total_true"] <= 25),
    }


def evaluate_context_frame(df, exp_id):
    if df.empty:
        return None

    y_true = (df["true_replenish_qty"] > 0).astype(int)
    y_prob = df["ai_pred_prob"].astype(float)
    y_pred = df["cls_pred_best_f1"].astype(int)

    true_sum = float(df["true_replenish_qty"].sum())
    pred_sum = float(df["ai_pred_qty"].sum())
    abs_sum = float(np.abs(df["ai_pred_qty"] - df["true_replenish_qty"]).sum())
    under_sum = float(np.clip(df["true_replenish_qty"] - df["ai_pred_qty"], a_min=0, a_max=None).sum())
    over_sum = float(np.clip(df["ai_pred_qty"] - df["true_replenish_qty"], a_min=0, a_max=None).sum())

    sku_df = aggregate_sku(df)
    masks = bucket_masks(sku_df)
    global_metrics = compute_slice_metrics(sku_df, np.ones(len(sku_df), dtype=bool))
    row_decisions = decision_error_metrics(df)
    row_categories = category_diagnostics(df)
    row_quadrants = quadrant_metrics(df)

    row = {
        "exp_id": exp_id,
        "rows": int(len(df)),
        "auc": float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "global_ratio": pred_sum / true_sum if true_sum > 0 else np.nan,
        "global_wmape": abs_sum / true_sum if true_sum > 0 else np.nan,
        "sku_p50": float(sku_df["sku_ratio"].median()) if not sku_df.empty else np.nan,
        "positive_sku_rows": int(len(sku_df)),
        "sku_trimmed_mean_10": global_metrics["sku_trimmed_mean_10"],
        "within_20pct_rate": global_metrics["within_20pct_rate"],
        "within_30pct_rate": global_metrics["within_30pct_rate"],
        "within_50pct_rate": global_metrics["within_50pct_rate"],
        "catastrophic_under_rate": global_metrics["catastrophic_under_rate"],
        "catastrophic_over_rate": global_metrics["catastrophic_over_rate"],
        "under_wape": under_sum / true_sum if true_sum > 0 else np.nan,
        "over_wape": over_sum / true_sum if true_sum > 0 else np.nan,
        "rank_corr_positive_skus": safe_spearman(sku_df["total_pred"], sku_df["total_true"]) if not sku_df.empty else np.nan,
        "top10_true_volume_capture": top_true_volume_capture(sku_df, 0.10),
        "top20_true_volume_capture": top_true_volume_capture(sku_df, 0.20),
    }
    row.update(row_decisions)
    row.update(row_categories)
    row.update(row_quadrants)

    slice_defs = {
        "ice": masks["ice"],
        "4_25": masks["4_25"],
        "ice_4_25": masks["ice"] & masks["4_25"],
        "1_3": masks["1_3"],
        "4_10": masks["4_10"],
        "11_25": masks["11_25"],
        "blockbuster": masks["blockbuster"],
    }
    for prefix, mask in slice_defs.items():
        metrics = compute_slice_metrics(sku_df, mask)
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    blockbusters = sku_df.loc[masks["blockbuster"]].copy()
    row["blockbuster_top10_true_volume_capture"] = top_true_volume_capture(blockbusters, 0.10)
    row["blockbuster_top20_true_volume_capture"] = top_true_volume_capture(blockbusters, 0.20)
    return row


def evaluate_context_csv(path):
    exp_id = parse_exp_id_from_context(path)
    df = pd.read_csv(path)
    return evaluate_context_frame(df, exp_id)


def numeric_cols_for_rounding(df):
    return [
        col
        for col in df.columns
        if any(
            token in col
            for token in (
                "ratio",
                "wmape",
                "p50",
                "auc",
                "f1",
                "elapsed",
                "std",
                "trimmed",
                "within",
                "catastrophic",
                "under",
                "over",
                "capture",
                "corr",
                "false_",
                "_rate",
            )
        )
    ]
