import os

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8_event_shadow")
CURRENT_EVAL_ALL = os.path.join(
    PROJECT_ROOT,
    "reports",
    "phase8a_prep",
    "phase8_current_mainline_eval_context_all_anchors.csv",
)

OUT_WEAK = os.path.join(OUT_DIR, "phase8_event_shadow_weak_signal_table.csv")
OUT_ZERO = os.path.join(OUT_DIR, "phase8_event_shadow_zero_true_table.csv")
OUT_CAT = os.path.join(OUT_DIR, "phase8_event_shadow_category_month_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase8_event_shadow_slice_summary.md")

ANCHORS = ["2025-10-01", "2025-11-01", "2025-12-01"]
CALIBRATION_SCALES = {"2025-10-01": 0.93, "2025-11-01": 1.00, "2025-12-01": 1.00}
QUADRANTS = ["repl0_fut0", "repl1_fut0"]
RAW_EXP_TEMPLATE = "p8event_{anchor_tag}_event_intent_plus_s2028_hard_g027"


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def shadow_context_path(anchor_date):
    exp_id = RAW_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
    return os.path.join(OUT_DIR, anchor_tag(anchor_date), "phase5", f"eval_context_{exp_id}.csv")


def apply_calibration(df, anchor_date):
    out = df.copy()
    scale = float(CALIBRATION_SCALES.get(anchor_date, 1.0))
    for col in ("ai_pred_qty_open", "ai_pred_qty"):
        if col in out.columns:
            out[col] = out[col].astype(float) * scale
    out["ai_pred_positive_qty"] = (out["ai_pred_qty"].astype(float) > 0).astype(int)
    out["abs_error"] = (out["ai_pred_qty"].astype(float) - out["true_replenish_qty"].astype(float)).abs()
    out["zero_true_fp"] = ((out["true_replenish_qty"].astype(float) <= 0) & (out["ai_pred_qty"].astype(float) > 0)).astype(int)
    out["is_true_blockbuster"] = (out["true_replenish_qty"].astype(float) > 25).astype(int)
    return out


def safe_div(num, den):
    den = float(den)
    if den <= 0:
        return np.nan
    return float(num) / den


def markdown_table(df, columns):
    headers = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                vals.append("" if np.isnan(value) else f"{value:.4f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([headers, sep, *rows])


def load_frames():
    current_all = pd.read_csv(CURRENT_EVAL_ALL)
    current_all["anchor_date"] = pd.to_datetime(current_all["anchor_date"]).dt.strftime("%Y-%m-%d")
    frames = []
    for anchor_date in ANCHORS:
        current = current_all[current_all["anchor_date"] == anchor_date].copy()
        current["track"] = "current_phase7"
        frames.append(current)

        shadow = pd.read_csv(shadow_context_path(anchor_date))
        shadow = apply_calibration(shadow, anchor_date)
        shadow["anchor_date"] = anchor_date
        shadow["track"] = "event_shadow"
        frames.append(shadow)
    return pd.concat(frames, ignore_index=True, sort=False)


def build_weak_signal_table(df):
    block = df[df["true_replenish_qty"].astype(float) > 25].copy()
    block = block[block["signal_quadrant"].isin(QUADRANTS)].copy()
    grouped = (
        block.groupby(["anchor_date", "track", "signal_quadrant"], as_index=False)
        .agg(
            rows=("sku_id", "size"),
            total_true=("true_replenish_qty", "sum"),
            total_pred=("ai_pred_qty", "sum"),
        )
    )
    if grouped.empty:
        return grouped
    grouped["ratio"] = grouped.apply(lambda r: safe_div(r["total_pred"], r["total_true"]), axis=1)
    grouped["under_wape"] = grouped.apply(
        lambda r: safe_div(max(float(r["total_true"]) - float(r["total_pred"]), 0.0), r["total_true"]), axis=1
    )
    return grouped.sort_values(["anchor_date", "signal_quadrant", "track"]).reset_index(drop=True)


def build_zero_true_table(df):
    rows = []
    for (anchor_date, track), sub in df.groupby(["anchor_date", "track"]):
        zero_true = sub[sub["true_replenish_qty"].astype(float) <= 0].copy()
        fp = zero_true[zero_true["ai_pred_qty"].astype(float) > 0].copy()
        rows.append(
            {
                "anchor_date": anchor_date,
                "track": track,
                "zero_true_rows": int(len(zero_true)),
                "false_positive_rows": int(len(fp)),
                "false_positive_rate_zero_true": safe_div(len(fp), len(zero_true)),
                "pred_sum_on_zero_true": float(fp["ai_pred_qty"].sum()) if not fp.empty else 0.0,
                "pred_ge_3_rows": int((zero_true["ai_pred_qty"].astype(float) >= 3).sum()),
                "pred_ge_5_rows": int((zero_true["ai_pred_qty"].astype(float) >= 5).sum()),
                "pred_ge_10_rows": int((zero_true["ai_pred_qty"].astype(float) >= 10).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["anchor_date", "track"]).reset_index(drop=True)


def aggregate_category(sub):
    grouped = (
        sub.groupby("category", as_index=False)
        .agg(
            rows=("sku_id", "size"),
            true_sum=("true_replenish_qty", "sum"),
            pred_sum=("ai_pred_qty", "sum"),
        )
    )
    grouped = grouped[grouped["true_sum"] > 0].copy()
    if grouped.empty:
        return grouped
    under_map = (
        sub.assign(under_qty=np.clip(sub["true_replenish_qty"].astype(float) - sub["ai_pred_qty"].astype(float), a_min=0, a_max=None))
        .groupby("category")["under_qty"]
        .sum()
    )
    over_map = (
        sub.assign(over_qty=np.clip(sub["ai_pred_qty"].astype(float) - sub["true_replenish_qty"].astype(float), a_min=0, a_max=None))
        .groupby("category")["over_qty"]
        .sum()
    )
    grouped["under_sum"] = grouped["category"].map(under_map).fillna(0.0)
    grouped["over_sum"] = grouped["category"].map(over_map).fillna(0.0)
    grouped["ratio"] = grouped["pred_sum"] / grouped["true_sum"]
    grouped["under_wape"] = grouped["under_sum"] / grouped["true_sum"]
    grouped["over_wape"] = grouped["over_sum"] / grouped["true_sum"]
    return grouped


def build_category_month_table(df):
    rows = []
    for (anchor_date, track), sub in df.groupby(["anchor_date", "track"]):
        pos = sub[sub["true_replenish_qty"].astype(float) > 0].copy()
        block = sub[sub["true_replenish_qty"].astype(float) > 25].copy()
        for slice_name, slice_df in [("positive_all", pos), ("blockbuster", block)]:
            if slice_df.empty:
                continue
            cat = aggregate_category(slice_df)
            if cat.empty:
                continue
            cat["anchor_date"] = anchor_date
            cat["track"] = track
            cat["slice_name"] = slice_name
            rows.append(cat)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    cols = [
        "anchor_date",
        "track",
        "slice_name",
        "category",
        "rows",
        "true_sum",
        "pred_sum",
        "ratio",
        "under_wape",
        "over_wape",
    ]
    return out[cols].sort_values(["anchor_date", "slice_name", "track", "under_wape", "true_sum"], ascending=[True, True, True, False, False]).reset_index(drop=True)


def compare_pivot(df, key_cols, value_cols, left_name="current_phase7", right_name="event_shadow"):
    left = df[df["track"] == left_name][key_cols + value_cols].copy()
    right = df[df["track"] == right_name][key_cols + value_cols].copy()
    merged = left.merge(right, on=key_cols, how="outer", suffixes=("_current", "_shadow"))
    for col in value_cols:
        merged[f"{col}_delta"] = merged[f"{col}_shadow"] - merged[f"{col}_current"]
    return merged


def top_category_delta(category_df, slice_name):
    block = category_df[category_df["slice_name"] == slice_name].copy()
    if block.empty:
        return pd.DataFrame()
    cmp_df = compare_pivot(
        block,
        ["anchor_date", "slice_name", "category"],
        ["rows", "true_sum", "pred_sum", "ratio", "under_wape", "over_wape"],
    )
    cmp_df["true_sum_shadow"] = cmp_df["true_sum_shadow"].fillna(cmp_df["true_sum_current"])
    return cmp_df.sort_values(
        ["anchor_date", "under_wape_delta", "true_sum_shadow"],
        ascending=[True, True, False],
    )


def render_summary(weak_cmp, zero_cmp, block_cat_cmp):
    weak_best = weak_cmp.sort_values(["under_wape_delta", "total_true_shadow"], ascending=[True, False]).head(12)
    zero_best = zero_cmp.sort_values(["anchor_date"]).copy()
    block_focus = block_cat_cmp[
        (block_cat_cmp["rows_shadow"].fillna(0) >= 3) | (block_cat_cmp["rows_current"].fillna(0) >= 3)
    ].copy()
    block_focus = block_focus.sort_values(["anchor_date", "under_wape_delta", "true_sum_shadow"], ascending=[True, True, False]).groupby("anchor_date").head(8)

    lines = [
        "# Phase8 Event Shadow Slice Summary",
        "",
        "- Status: `shadow_only_no_replace`",
        "- Scope: `2025-10/11/12` only",
        "- Compare: current calibrated phase7 vs event-intent shadow",
        "",
        "## Weak-Signal Blockbuster",
        "",
        markdown_table(
            weak_best,
            [
                "anchor_date",
                "signal_quadrant",
                "rows_current",
                "rows_shadow",
                "total_true_current",
                "total_pred_current",
                "ratio_current",
                "under_wape_current",
                "total_pred_shadow",
                "ratio_shadow",
                "under_wape_shadow",
                "under_wape_delta",
            ],
        ) if not weak_best.empty else "No weak-signal blockbuster rows.",
        "",
        "## Zero-True False Positive",
        "",
        markdown_table(
            zero_best,
            [
                "anchor_date",
                "false_positive_rows_current",
                "false_positive_rate_zero_true_current",
                "pred_sum_on_zero_true_current",
                "pred_ge_5_rows_current",
                "false_positive_rows_shadow",
                "false_positive_rate_zero_true_shadow",
                "pred_sum_on_zero_true_shadow",
                "pred_ge_5_rows_shadow",
                "false_positive_rate_zero_true_delta",
            ],
        ) if not zero_best.empty else "No zero-true rows.",
        "",
        "## Blockbuster Category x Month",
        "",
        markdown_table(
            block_focus,
            [
                "anchor_date",
                "category",
                "rows_current",
                "rows_shadow",
                "true_sum_current",
                "pred_sum_current",
                "ratio_current",
                "under_wape_current",
                "pred_sum_shadow",
                "ratio_shadow",
                "under_wape_shadow",
                "under_wape_delta",
            ],
        ) if not block_focus.empty else "No blockbuster category rows.",
        "",
        "## Interpretation",
        "",
        "- Negative `under_wape_delta` means the event shadow reduced under-forecasting.",
        "- Negative `false_positive_rate_zero_true_delta` means the event shadow reduced zero-true false positives.",
        "- This remains a partial-coverage shadow result; it does not replace the official phase7 mainline.",
        "",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_frames()

    weak_df = build_weak_signal_table(df)
    zero_df = build_zero_true_table(df)
    cat_df = build_category_month_table(df)

    weak_df.to_csv(OUT_WEAK, index=False, encoding="utf-8-sig")
    zero_df.to_csv(OUT_ZERO, index=False, encoding="utf-8-sig")
    cat_df.to_csv(OUT_CAT, index=False, encoding="utf-8-sig")

    weak_cmp = compare_pivot(
        weak_df,
        ["anchor_date", "signal_quadrant"],
        ["rows", "total_true", "total_pred", "ratio", "under_wape"],
    )
    zero_cmp = compare_pivot(
        zero_df,
        ["anchor_date"],
        [
            "zero_true_rows",
            "false_positive_rows",
            "false_positive_rate_zero_true",
            "pred_sum_on_zero_true",
            "pred_ge_3_rows",
            "pred_ge_5_rows",
            "pred_ge_10_rows",
        ],
    )
    block_cat_cmp = top_category_delta(cat_df, "blockbuster")

    with open(OUT_SUMMARY, "w", encoding="utf-8-sig") as fh:
        fh.write(render_summary(weak_cmp, zero_cmp, block_cat_cmp))

    print(f"[OK] weak-signal shadow table -> {OUT_WEAK}")
    print(f"[OK] zero-true shadow table -> {OUT_ZERO}")
    print(f"[OK] category-month shadow table -> {OUT_CAT}")
    print(f"[OK] shadow slice summary -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
