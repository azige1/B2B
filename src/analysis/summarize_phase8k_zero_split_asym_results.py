import json
import os

import numpy as np
import pandas as pd

from phase_eval_utils import evaluate_context_frame


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8_inventory_zero_split_shadow_2026")
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8_zero_split_asym_train_2026")
ROW_COMPARE_PATH = os.path.join(
    PROJECT_ROOT,
    "reports",
    "phase8_event_inventory_shadow_2026",
    "phase8_event_inventory_shadow_row_compare.csv",
)
OUT_ANCHOR = os.path.join(OUT_DIR, "phase8_zero_split_asym_anchor_table.csv")
OUT_STATE = os.path.join(OUT_DIR, "phase8_zero_split_asym_state_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase8_zero_split_asym_summary.md")
OUT_RESULT = os.path.join(OUT_DIR, "phase8_zero_split_asym_result.json")

ANCHORS = ["2026-02-15", "2026-02-24"]
BASELINE_TEMPLATE = "p8ei_{anchor_tag}_event_inventory_zero_split_s2028_hard_g027"
CANDIDATES = [
    "zero_split_asym_mild",
    "zero_split_asym_balanced",
    "zero_split_asym_strong",
]
TARGETS = {
    "mean_global_wmape": 0.6667,
    "mean_blockbuster_under_wape": 0.3635,
    "long_zero_zero_true_fp_rate": 0.7361,
    "long_zero_positive_true_under_wape": 0.1662,
    "under_tolerance": 0.0050,
}


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def baseline_context_path(anchor_date):
    exp_id = BASELINE_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
    return os.path.join(BASE_DIR, anchor_tag(anchor_date), "phase5", f"eval_context_{exp_id}.csv")


def candidate_context_path(anchor_date, candidate_key):
    exp_id = f"p8k_{anchor_tag(anchor_date)}_{candidate_key}_s2028_hard_g027"
    return os.path.join(OUT_DIR, anchor_tag(anchor_date), "phase5", f"eval_context_{exp_id}.csv")


def load_rule_features():
    df = pd.read_csv(ROW_COMPARE_PATH, encoding="utf-8-sig")
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.strftime("%Y-%m-%d")
    df["sku_id"] = df["sku_id"].astype(str)
    keep_cols = [
        "sku_id",
        "anchor_date",
        "true_replenish_qty",
        "inv_short_zero",
        "inv_long_zero",
        "stock_zero",
        "inv_stock_zero_streak",
    ]
    for col in ["true_replenish_qty", "stock_zero", "inv_stock_zero_streak"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "inv_short_zero" in df.columns:
        df["inv_short_zero"] = pd.to_numeric(df["inv_short_zero"], errors="coerce").fillna(0.0)
    else:
        df["inv_short_zero"] = (
            (df["stock_zero"] > 0) & (df["inv_stock_zero_streak"] <= 7)
        ).astype(float)
    if "inv_long_zero" in df.columns:
        df["inv_long_zero"] = pd.to_numeric(df["inv_long_zero"], errors="coerce").fillna(0.0)
    else:
        df["inv_long_zero"] = (
            (df["stock_zero"] > 0) & (df["inv_stock_zero_streak"] > 7)
        ).astype(float)
    return df[keep_cols].drop_duplicates(["sku_id", "anchor_date", "true_replenish_qty"]).copy()


def merge_state_features(df, feature_df):
    merged = df.copy()
    merged["anchor_date"] = pd.to_datetime(merged["anchor_date"]).dt.strftime("%Y-%m-%d")
    merged["sku_id"] = merged["sku_id"].astype(str)
    merged["true_replenish_qty"] = pd.to_numeric(merged["true_replenish_qty"], errors="coerce").fillna(0.0)
    merged = merged.merge(
        feature_df,
        on=["sku_id", "anchor_date", "true_replenish_qty"],
        how="left",
        suffixes=("", "_feat"),
    )
    for col in ["inv_short_zero", "inv_long_zero"]:
        feat_col = f"{col}_feat"
        if col not in merged.columns and feat_col in merged.columns:
            merged[col] = merged[feat_col]
        elif feat_col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(
                pd.to_numeric(merged[feat_col], errors="coerce")
            )
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged


def state_metrics(df):
    work = df.copy()
    work["stock_state"] = "other"
    work.loc[work["inv_short_zero"] > 0, "stock_state"] = "short_zero"
    work.loc[work["inv_long_zero"] > 0, "stock_state"] = "long_zero"
    rows = []
    for stock_state, sub in work.groupby("stock_state", dropna=False):
        zero = sub[sub["true_replenish_qty"] <= 0].copy()
        pos = sub[sub["true_replenish_qty"] > 0].copy()
        row = {
            "stock_state": str(stock_state),
            "rows": int(len(sub)),
            "zero_true_rows": int(len(zero)),
            "positive_true_rows": int(len(pos)),
        }
        if len(zero) > 0:
            row["zero_true_fp_rate"] = float((zero["ai_pred_qty"].astype(float) > 0).mean())
        else:
            row["zero_true_fp_rate"] = np.nan
        if len(pos) > 0:
            true_sum = float(pos["true_replenish_qty"].sum())
            row["positive_true_under_wape"] = float(
                np.clip(pos["true_replenish_qty"].astype(float) - pos["ai_pred_qty"].astype(float), a_min=0, a_max=None).sum()
                / max(true_sum, 1e-9)
            )
        else:
            row["positive_true_under_wape"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("stock_state").reset_index(drop=True)


def summarize_anchor_rows(df, candidate_key):
    metric_cols = [
        "global_wmape",
        "blockbuster_under_wape",
        "4_25_under_wape",
        "rank_corr_positive_skus",
        "top20_true_volume_capture",
    ]
    out = {"candidate_key": candidate_key}
    for col in metric_cols:
        out[f"mean_{col}"] = float(df[col].astype(float).mean())
    return out


def summarize_state_rows(df):
    out = {}
    for stock_state in ["short_zero", "long_zero"]:
        sub = df[df["stock_state"] == stock_state]
        if sub.empty:
            out[f"{stock_state}_zero_true_fp_rate"] = np.nan
            out[f"{stock_state}_positive_true_under_wape"] = np.nan
            continue
        row = sub.iloc[0]
        out[f"{stock_state}_zero_true_fp_rate"] = float(row["zero_true_fp_rate"])
        out[f"{stock_state}_positive_true_under_wape"] = float(row["positive_true_under_wape"])
    return out


def build_candidate_summary(candidate_key, feature_df, context_paths):
    anchor_rows = []
    state_frames = []
    for anchor_date, path in context_paths.items():
        df = pd.read_csv(path)
        eval_row = evaluate_context_frame(df, f"{candidate_key}_{anchor_tag(anchor_date)}")
        eval_row["anchor_date"] = anchor_date
        eval_row["candidate_key"] = candidate_key
        anchor_rows.append(eval_row)

        merged = merge_state_features(df, feature_df)
        state_df = state_metrics(merged)
        state_df["anchor_date"] = anchor_date
        state_df["candidate_key"] = candidate_key
        state_frames.append(state_df)

    anchor_df = pd.DataFrame(anchor_rows)
    state_df = pd.concat(state_frames, ignore_index=True, sort=False)
    mean_state_df = (
        state_df.groupby(["candidate_key", "stock_state"], as_index=False)
        .agg(
            rows=("rows", "sum"),
            zero_true_rows=("zero_true_rows", "sum"),
            positive_true_rows=("positive_true_rows", "sum"),
            zero_true_fp_rate=("zero_true_fp_rate", "mean"),
            positive_true_under_wape=("positive_true_under_wape", "mean"),
        )
    )
    summary = summarize_anchor_rows(anchor_df, candidate_key)
    summary.update(summarize_state_rows(mean_state_df))
    return anchor_df, state_df, mean_state_df, summary


def choose_best(summary_df):
    work = summary_df.copy()
    work["passes_all"] = (
        (work["mean_global_wmape"] <= TARGETS["mean_global_wmape"])
        & (work["mean_blockbuster_under_wape"] <= TARGETS["mean_blockbuster_under_wape"])
        & (work["long_zero_zero_true_fp_rate"] < TARGETS["long_zero_zero_true_fp_rate"])
        & (work["long_zero_positive_true_under_wape"] <= TARGETS["long_zero_positive_true_under_wape"] + TARGETS["under_tolerance"])
    )
    work["selection_score"] = (
        work["mean_global_wmape"]
        + 0.20 * work["mean_blockbuster_under_wape"]
        + 0.25 * work["long_zero_zero_true_fp_rate"]
        + 0.20 * work["long_zero_positive_true_under_wape"]
    )
    passed = work[work["passes_all"]].copy()
    if not passed.empty:
        return passed.sort_values(
            [
                "mean_global_wmape",
                "mean_blockbuster_under_wape",
                "long_zero_zero_true_fp_rate",
                "long_zero_positive_true_under_wape",
            ],
            ascending=[True, True, True, True],
        ).iloc[0].to_dict()
    return work.sort_values("selection_score", ascending=True).iloc[0].to_dict()


def render_summary(summary_df, best_row):
    lines = [
        "# Phase8 Zero-Split Asymmetric Training Summary",
        "",
        "- Status: `analysis_only_shadow`",
        "- Baseline: `event_inventory_zero_split`",
        "- Scope: `2026-02-15 / 2026-02-24`",
        "- Method: model-internal asymmetric weighting for `short_zero / long_zero` only",
        "- No new data, no label change, no additional post-processing",
        "",
        "## Decision Rule",
        "",
        f"- mean `global_wmape <= {TARGETS['mean_global_wmape']:.4f}`",
        f"- mean `blockbuster_under_wape <= {TARGETS['mean_blockbuster_under_wape']:.4f}`",
        f"- `long_zero zero_true_fp_rate < {TARGETS['long_zero_zero_true_fp_rate']:.4f}`",
        f"- `long_zero positive_true_under_wape <= {TARGETS['long_zero_positive_true_under_wape'] + TARGETS['under_tolerance']:.4f}`",
        "",
        "## Candidate Table",
        "",
        "| candidate | mean_global_wmape | mean_blockbuster_under_wape | long_zero_zero_true_fp_rate | long_zero_positive_true_under_wape | passes_all |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['candidate_key']} | {row['mean_global_wmape']:.4f} | "
            f"{row['mean_blockbuster_under_wape']:.4f} | {row['long_zero_zero_true_fp_rate']:.4f} | "
            f"{row['long_zero_positive_true_under_wape']:.4f} | {bool(row['passes_all'])} |"
        )
    lines.extend(
        [
            "",
            "## Best Candidate",
            "",
            f"- `candidate = {best_row['candidate_key']}`",
            f"- `passes_all = {bool(best_row['passes_all'])}`",
            f"- `mean_global_wmape = {best_row['mean_global_wmape']:.4f}`",
            f"- `mean_blockbuster_under_wape = {best_row['mean_blockbuster_under_wape']:.4f}`",
            f"- `long_zero_zero_true_fp_rate = {best_row['long_zero_zero_true_fp_rate']:.4f}`",
            f"- `long_zero_positive_true_under_wape = {best_row['long_zero_positive_true_under_wape']:.4f}`",
            "",
        ]
    )
    if bool(best_row["passes_all"]):
        lines.extend(
            [
                "## Recommendation",
                "",
                "- Promote this candidate as the new best exploratory phase8 branch.",
                "- Use it as the default comparison line for the next phase8 exploratory round.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Recommendation",
                "",
                "- Do not promote the asymmetric variant yet.",
                "- Keep `event_inventory_zero_split` as the current best exploratory baseline.",
                "- Pause further optimization on this line until client-side data semantics are clarified.",
                "",
            ]
        )
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    feature_df = load_rule_features()

    baseline_paths = {anchor: baseline_context_path(anchor) for anchor in ANCHORS}
    baseline_anchor, baseline_state_raw, baseline_state_mean, baseline_summary = build_candidate_summary(
        "event_inventory_zero_split",
        feature_df,
        baseline_paths,
    )

    anchor_frames = [baseline_anchor]
    state_frames = [baseline_state_mean]
    summary_rows = [baseline_summary]

    for candidate_key in CANDIDATES:
        context_paths = {anchor: candidate_context_path(anchor, candidate_key) for anchor in ANCHORS}
        anchor_df, _, state_mean_df, summary = build_candidate_summary(candidate_key, feature_df, context_paths)
        anchor_frames.append(anchor_df)
        state_frames.append(state_mean_df)
        summary_rows.append(summary)

    anchor_table = pd.concat(anchor_frames, ignore_index=True, sort=False)
    state_table = pd.concat(state_frames, ignore_index=True, sort=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df["passes_all"] = (
        (summary_df["mean_global_wmape"] <= TARGETS["mean_global_wmape"])
        & (summary_df["mean_blockbuster_under_wape"] <= TARGETS["mean_blockbuster_under_wape"])
        & (summary_df["long_zero_zero_true_fp_rate"] < TARGETS["long_zero_zero_true_fp_rate"])
        & (summary_df["long_zero_positive_true_under_wape"] <= TARGETS["long_zero_positive_true_under_wape"] + TARGETS["under_tolerance"])
    )

    best_row = choose_best(summary_df)

    anchor_table.to_csv(OUT_ANCHOR, index=False, encoding="utf-8-sig")
    state_table.to_csv(OUT_STATE, index=False, encoding="utf-8-sig")
    with open(OUT_SUMMARY, "w", encoding="utf-8-sig") as fh:
        fh.write(render_summary(summary_df, best_row))

    payload = {
        "status": "analysis_only_shadow",
        "baseline": "event_inventory_zero_split",
        "anchors": ANCHORS,
        "targets": TARGETS,
        "best_candidate": best_row,
        "candidate_table": summary_df.to_dict(orient="records"),
        "outputs": {
            "anchor_table": os.path.relpath(OUT_ANCHOR, PROJECT_ROOT),
            "state_table": os.path.relpath(OUT_STATE, PROJECT_ROOT),
            "summary_md": os.path.relpath(OUT_SUMMARY, PROJECT_ROOT),
        },
    }
    with open(OUT_RESULT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase8 zero_split asym anchor table -> {OUT_ANCHOR}")
    print(f"[OK] phase8 zero_split asym state table -> {OUT_STATE}")
    print(f"[OK] phase8 zero_split asym summary -> {OUT_SUMMARY}")
    print(f"[OK] phase8 zero_split asym result -> {OUT_RESULT}")


if __name__ == "__main__":
    main()
