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

OUT_ROW_COMPARE = os.path.join(OUT_DIR, "phase8_event_shadow_row_compare.csv")
OUT_CASES = os.path.join(OUT_DIR, "phase8_event_shadow_focus_cases.csv")
OUT_CATEGORY_DELTA = os.path.join(OUT_DIR, "phase8_event_shadow_category_delta_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase8_event_shadow_detail_summary.md")

ANCHORS = ["2025-10-01", "2025-11-01", "2025-12-01"]
CALIBRATION_SCALES = {"2025-10-01": 0.93, "2025-11-01": 1.00, "2025-12-01": 1.00}
WEAK_QUADRANTS = {"repl0_fut0", "repl1_fut0"}
RAW_EXP_TEMPLATE = "p8event_{anchor_tag}_event_intent_plus_s2028_hard_g027"


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def shadow_context_path(anchor_date):
    exp_id = RAW_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
    return os.path.join(OUT_DIR, anchor_tag(anchor_date), "phase5", f"eval_context_{exp_id}.csv")


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


def apply_shadow_calibration(df, anchor_date):
    out = df.copy()
    scale = float(CALIBRATION_SCALES.get(anchor_date, 1.0))
    out["anchor_date"] = anchor_date
    for col in ("ai_pred_qty_open", "ai_pred_qty"):
        out[col] = out[col].astype(float) * scale
    out["ai_pred_positive_qty"] = (out["ai_pred_qty"].astype(float) > 0).astype(int)
    out["abs_error"] = (out["ai_pred_qty"].astype(float) - out["true_replenish_qty"].astype(float)).abs()
    return out


def load_current_shadow_rows():
    current_all = pd.read_csv(CURRENT_EVAL_ALL)
    current_all["anchor_date"] = pd.to_datetime(current_all["anchor_date"]).dt.strftime("%Y-%m-%d")
    current_all = current_all[current_all["anchor_date"].isin(ANCHORS)].copy()

    current_cols = [
        "anchor_date",
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
        "true_replenish_qty",
        "ai_pred_prob",
        "ai_pred_qty",
        "abs_error",
        "ai_pred_positive_qty",
        "signal_quadrant",
        "activity_bucket",
        "lookback_repl_days_90",
        "lookback_future_days_90",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
    ]
    current = current_all[current_cols].copy()
    current = current.rename(
        columns={
            "ai_pred_prob": "current_pred_prob",
            "ai_pred_qty": "current_pred_qty",
            "abs_error": "current_abs_error",
            "ai_pred_positive_qty": "current_pred_positive",
        }
    )

    shadow_frames = []
    for anchor_date in ANCHORS:
        shadow = pd.read_csv(shadow_context_path(anchor_date))
        shadow = apply_shadow_calibration(shadow, anchor_date)
        shadow_frames.append(shadow)
    shadow = pd.concat(shadow_frames, ignore_index=True, sort=False)
    shadow_cols = [
        "anchor_date",
        "sku_id",
        "true_replenish_qty",
        "ai_pred_prob",
        "ai_pred_qty",
        "abs_error",
        "ai_pred_positive_qty",
        "signal_quadrant",
        "activity_bucket",
        "lookback_repl_days_90",
        "lookback_future_days_90",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
        "category",
        "style_id",
        "season",
        "series",
        "band",
    ]
    shadow = shadow[shadow_cols].copy()
    shadow = shadow.rename(
        columns={
            "ai_pred_prob": "shadow_pred_prob",
            "ai_pred_qty": "shadow_pred_qty",
            "abs_error": "shadow_abs_error",
            "ai_pred_positive_qty": "shadow_pred_positive",
        }
    )

    merged = current.merge(
        shadow,
        on=["anchor_date", "sku_id", "true_replenish_qty"],
        how="inner",
        suffixes=("", "_shadowdup"),
    )
    merged["pred_delta"] = merged["shadow_pred_qty"] - merged["current_pred_qty"]
    merged["abs_error_delta"] = merged["shadow_abs_error"] - merged["current_abs_error"]
    merged["abs_error_gain"] = merged["current_abs_error"] - merged["shadow_abs_error"]
    merged["ratio_current"] = merged.apply(lambda r: safe_div(r["current_pred_qty"], r["true_replenish_qty"]), axis=1)
    merged["ratio_shadow"] = merged.apply(lambda r: safe_div(r["shadow_pred_qty"], r["true_replenish_qty"]), axis=1)
    merged["is_true_blockbuster"] = (merged["true_replenish_qty"].astype(float) > 25).astype(int)
    merged["is_zero_true"] = (merged["true_replenish_qty"].astype(float) <= 0).astype(int)
    merged["current_zero_true_fp"] = (
        (merged["true_replenish_qty"].astype(float) <= 0) & (merged["current_pred_qty"].astype(float) > 0)
    ).astype(int)
    merged["shadow_zero_true_fp"] = (
        (merged["true_replenish_qty"].astype(float) <= 0) & (merged["shadow_pred_qty"].astype(float) > 0)
    ).astype(int)
    return merged


def slice_metrics(sub, pred_col):
    true_sum = float(sub["true_replenish_qty"].sum())
    pred_sum = float(sub[pred_col].sum())
    under_sum = float(np.clip(sub["true_replenish_qty"].astype(float) - sub[pred_col].astype(float), a_min=0, a_max=None).sum())
    over_sum = float(np.clip(sub[pred_col].astype(float) - sub["true_replenish_qty"].astype(float), a_min=0, a_max=None).sum())
    return {
        "rows": int(len(sub)),
        "true_sum": true_sum,
        "pred_sum": pred_sum,
        "ratio": safe_div(pred_sum, true_sum),
        "under_wape": safe_div(under_sum, true_sum),
        "over_wape": safe_div(over_sum, true_sum),
    }


def build_focus_cases(row_df):
    frames = []

    weak = row_df[(row_df["is_true_blockbuster"] == 1) & (row_df["signal_quadrant"].isin(WEAK_QUADRANTS))].copy()
    if not weak.empty:
        improved = weak.sort_values(["abs_error_gain", "true_replenish_qty"], ascending=[False, False]).head(40).copy()
        improved["case_type"] = "weak_signal_blockbuster_improved"
        worsened = weak.sort_values(["abs_error_gain", "true_replenish_qty"], ascending=[True, False]).head(20).copy()
        worsened["case_type"] = "weak_signal_blockbuster_worsened"
        frames.extend([improved, worsened])

    zero = row_df[row_df["is_zero_true"] == 1].copy()
    if not zero.empty:
        zero_improved = zero.sort_values(["pred_delta", "current_pred_qty"], ascending=[True, False]).head(40).copy()
        zero_improved["case_type"] = "zero_true_fp_reduced"
        zero_worsened = zero.sort_values(["pred_delta", "shadow_pred_qty"], ascending=[False, False]).head(20).copy()
        zero_worsened["case_type"] = "zero_true_fp_worsened"
        frames.extend([zero_improved, zero_worsened])

    if not frames:
        return pd.DataFrame()

    cases = pd.concat(frames, ignore_index=True, sort=False)
    keep_cols = [
        "case_type",
        "anchor_date",
        "sku_id",
        "product_name",
        "category",
        "sub_category",
        "style_id",
        "signal_quadrant",
        "activity_bucket",
        "true_replenish_qty",
        "current_pred_prob",
        "current_pred_qty",
        "ratio_current",
        "current_abs_error",
        "shadow_pred_prob",
        "shadow_pred_qty",
        "ratio_shadow",
        "shadow_abs_error",
        "pred_delta",
        "abs_error_delta",
        "abs_error_gain",
        "qty_first_order",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
    ]
    return cases[keep_cols].reset_index(drop=True)


def build_category_delta(row_df):
    rows = []
    for anchor_date in ANCHORS:
        anchor_df = row_df[row_df["anchor_date"] == anchor_date].copy()
        for slice_name, slice_mask in {
            "positive_all": anchor_df["true_replenish_qty"].astype(float) > 0,
            "blockbuster": anchor_df["true_replenish_qty"].astype(float) > 25,
        }.items():
            sub = anchor_df[slice_mask].copy()
            if sub.empty:
                continue
            for category, cat_df in sub.groupby("category", dropna=False):
                category = str(category)
                cur = slice_metrics(cat_df, "current_pred_qty")
                sh = slice_metrics(cat_df, "shadow_pred_qty")
                rows.append(
                    {
                        "anchor_date": anchor_date,
                        "slice_name": slice_name,
                        "category": category,
                        "rows_current": cur["rows"],
                        "rows_shadow": sh["rows"],
                        "true_sum_current": cur["true_sum"],
                        "true_sum_shadow": sh["true_sum"],
                        "pred_sum_current": cur["pred_sum"],
                        "pred_sum_shadow": sh["pred_sum"],
                        "ratio_current": cur["ratio"],
                        "ratio_shadow": sh["ratio"],
                        "under_wape_current": cur["under_wape"],
                        "under_wape_shadow": sh["under_wape"],
                        "over_wape_current": cur["over_wape"],
                        "over_wape_shadow": sh["over_wape"],
                        "ratio_delta": sh["ratio"] - cur["ratio"] if pd.notna(cur["ratio"]) and pd.notna(sh["ratio"]) else np.nan,
                        "under_wape_delta": sh["under_wape"] - cur["under_wape"] if pd.notna(cur["under_wape"]) and pd.notna(sh["under_wape"]) else np.nan,
                        "over_wape_delta": sh["over_wape"] - cur["over_wape"] if pd.notna(cur["over_wape"]) and pd.notna(sh["over_wape"]) else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["anchor_date", "slice_name", "under_wape_delta", "true_sum_current"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def build_summary(row_df, cases_df, category_df):
    weak_improved = cases_df[cases_df["case_type"] == "weak_signal_blockbuster_improved"].head(12).copy()
    weak_worsened = cases_df[cases_df["case_type"] == "weak_signal_blockbuster_worsened"].head(8).copy()
    zero_improved = cases_df[cases_df["case_type"] == "zero_true_fp_reduced"].head(12).copy()
    zero_worsened = cases_df[cases_df["case_type"] == "zero_true_fp_worsened"].head(8).copy()

    blockbuster_categories = category_df[category_df["slice_name"] == "blockbuster"].copy()
    blockbuster_focus = blockbuster_categories[
        (blockbuster_categories["rows_current"] >= 3) | (blockbuster_categories["rows_shadow"] >= 3)
    ].copy()
    category_best = blockbuster_focus.sort_values(
        ["anchor_date", "under_wape_delta", "true_sum_current"], ascending=[True, True, False]
    ).groupby("anchor_date").head(8)
    category_worst = blockbuster_focus.sort_values(
        ["anchor_date", "under_wape_delta", "true_sum_current"], ascending=[True, False, False]
    ).groupby("anchor_date").head(5)

    lines = [
        "# Phase8 Event Shadow Detail Summary",
        "",
        "- Status: `shadow_only_no_replace`",
        "- Scope: `2025-10/11/12` only",
        "- Purpose: inspect which rows and categories were helped or hurt by event-intent features",
        "",
        "## Weak-Signal Blockbuster Improved Cases",
        "",
        markdown_table(
            weak_improved,
            [
                "anchor_date",
                "sku_id",
                "category",
                "signal_quadrant",
                "true_replenish_qty",
                "current_pred_qty",
                "shadow_pred_qty",
                "current_abs_error",
                "shadow_abs_error",
                "abs_error_gain",
            ],
        ) if not weak_improved.empty else "No improved weak-signal blockbuster cases.",
        "",
        "## Weak-Signal Blockbuster Worsened Cases",
        "",
        markdown_table(
            weak_worsened,
            [
                "anchor_date",
                "sku_id",
                "category",
                "signal_quadrant",
                "true_replenish_qty",
                "current_pred_qty",
                "shadow_pred_qty",
                "current_abs_error",
                "shadow_abs_error",
                "abs_error_gain",
            ],
        ) if not weak_worsened.empty else "No worsened weak-signal blockbuster cases.",
        "",
        "## Zero-True Reduced False Positives",
        "",
        markdown_table(
            zero_improved,
            [
                "anchor_date",
                "sku_id",
                "category",
                "current_pred_qty",
                "shadow_pred_qty",
                "pred_delta",
                "current_abs_error",
                "shadow_abs_error",
            ],
        ) if not zero_improved.empty else "No reduced zero-true false-positive cases.",
        "",
        "## Zero-True Worsened Cases",
        "",
        markdown_table(
            zero_worsened,
            [
                "anchor_date",
                "sku_id",
                "category",
                "current_pred_qty",
                "shadow_pred_qty",
                "pred_delta",
                "current_abs_error",
                "shadow_abs_error",
            ],
        ) if not zero_worsened.empty else "No worsened zero-true cases.",
        "",
        "## Blockbuster Category Improvements",
        "",
        markdown_table(
            category_best,
            [
                "anchor_date",
                "category",
                "rows_current",
                "true_sum_current",
                "pred_sum_current",
                "pred_sum_shadow",
                "under_wape_current",
                "under_wape_shadow",
                "under_wape_delta",
            ],
        ) if not category_best.empty else "No blockbuster category improvements.",
        "",
        "## Blockbuster Category Regressions",
        "",
        markdown_table(
            category_worst,
            [
                "anchor_date",
                "category",
                "rows_current",
                "true_sum_current",
                "pred_sum_current",
                "pred_sum_shadow",
                "under_wape_current",
                "under_wape_shadow",
                "under_wape_delta",
            ],
        ) if not category_worst.empty else "No blockbuster category regressions.",
        "",
        "## Output Files",
        "",
        "- `reports/phase8_event_shadow/phase8_event_shadow_row_compare.csv`",
        "- `reports/phase8_event_shadow/phase8_event_shadow_focus_cases.csv`",
        "- `reports/phase8_event_shadow/phase8_event_shadow_category_delta_table.csv`",
        "",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    row_df = load_current_shadow_rows()
    row_df.to_csv(OUT_ROW_COMPARE, index=False, encoding="utf-8-sig")

    cases_df = build_focus_cases(row_df)
    cases_df.to_csv(OUT_CASES, index=False, encoding="utf-8-sig")

    category_df = build_category_delta(row_df)
    category_df.to_csv(OUT_CATEGORY_DELTA, index=False, encoding="utf-8-sig")

    with open(OUT_SUMMARY, "w", encoding="utf-8-sig") as fh:
        fh.write(build_summary(row_df, cases_df, category_df))

    print(f"[OK] row compare -> {OUT_ROW_COMPARE}")
    print(f"[OK] focus cases -> {OUT_CASES}")
    print(f"[OK] category delta -> {OUT_CATEGORY_DELTA}")
    print(f"[OK] detail summary -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
