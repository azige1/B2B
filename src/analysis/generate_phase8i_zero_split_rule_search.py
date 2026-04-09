import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase_eval_utils import evaluate_context_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ZERO_SPLIT_DIR = PROJECT_ROOT / "reports" / "phase8_inventory_zero_split_shadow_2026"
ROW_COMPARE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "phase8_event_inventory_shadow_2026"
    / "phase8_event_inventory_shadow_row_compare.csv"
)
OUT_DIR = PROJECT_ROOT / "reports" / "phase8_zero_split_rule_search_2026"
ANCHORS = ["2026-02-15", "2026-02-24"]
ZERO_SPLIT_TEMPLATE = "p8ei_{anchor_tag}_event_inventory_zero_split_s2028_hard_g027"


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def safe_rate(num, den):
    if den in (0, 0.0, None) or pd.isna(den):
        return np.nan
    return float(num) / float(den)


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


def context_path(anchor_date):
    exp_id = ZERO_SPLIT_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
    return ZERO_SPLIT_DIR / anchor_tag(anchor_date) / "phase5" / f"eval_context_{exp_id}.csv"


def load_zero_split_predictions():
    frames = []
    for anchor_date in ANCHORS:
        df = pd.read_csv(context_path(anchor_date))
        df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.strftime("%Y-%m-%d")
        df["sku_id"] = df["sku_id"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_rule_features():
    df = pd.read_csv(ROW_COMPARE_PATH, encoding="utf-8-sig")
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.strftime("%Y-%m-%d")
    df["sku_id"] = df["sku_id"].astype(str)
    numeric_cols = [
        "true_replenish_qty",
        "event_strong_30",
        "lookback_repl_sum_90",
        "qty_first_order",
        "inv_short_zero",
        "inv_long_zero",
        "inv_stock_zero_streak",
        "inv_positive_to_zero_switch",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "inv_short_zero" not in df.columns:
        df["inv_short_zero"] = (
            (pd.to_numeric(df.get("stock_zero", 0), errors="coerce").fillna(0.0) > 0)
            & (pd.to_numeric(df.get("inv_stock_zero_streak", 0), errors="coerce").fillna(0.0) <= 7)
        ).astype(float)
    if "inv_long_zero" not in df.columns:
        df["inv_long_zero"] = (
            (pd.to_numeric(df.get("stock_zero", 0), errors="coerce").fillna(0.0) > 0)
            & (pd.to_numeric(df.get("inv_stock_zero_streak", 0), errors="coerce").fillna(0.0) > 7)
        ).astype(float)
    keep_cols = [
        "sku_id",
        "anchor_date",
        "true_replenish_qty",
        "event_strong_30",
        "lookback_repl_sum_90",
        "qty_first_order",
        "inv_short_zero",
        "inv_long_zero",
        "inv_stock_zero_streak",
        "inv_positive_to_zero_switch",
    ]
    return df[keep_cols].drop_duplicates(["sku_id", "anchor_date", "true_replenish_qty"]).copy()


def merge_rule_frame():
    pred_df = load_zero_split_predictions()
    feat_df = load_rule_features()
    keys = ["sku_id", "anchor_date", "true_replenish_qty"]
    merged = pred_df.merge(feat_df, on=keys, how="left", suffixes=("", "_feat"))
    for col in [
        "event_strong_30",
        "lookback_repl_sum_90",
        "qty_first_order",
        "inv_short_zero",
        "inv_long_zero",
        "inv_stock_zero_streak",
        "inv_positive_to_zero_switch",
    ]:
        src_col = col
        feat_col = f"{col}_feat"
        if src_col not in merged.columns and feat_col in merged.columns:
            merged[src_col] = merged[feat_col]
        elif src_col in merged.columns and feat_col in merged.columns:
            merged[src_col] = pd.to_numeric(merged[src_col], errors="coerce").fillna(
                pd.to_numeric(merged[feat_col], errors="coerce")
            )
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    return merged


def apply_rule(df, params):
    work = df.copy()
    multiplier = np.ones(len(work), dtype=np.float32)
    has_signal = (
        (work["event_strong_30"] > 0) | (work["lookback_repl_sum_90"] > 0)
    ).to_numpy()
    short_zero = (work["inv_short_zero"] > 0).to_numpy()
    long_zero = (work["inv_long_zero"] > 0).to_numpy()

    multiplier[short_zero & ~has_signal] = float(params["short_zero_scale"])
    multiplier[short_zero & has_signal] = float(params["short_zero_signal_scale"])
    multiplier[long_zero & ~has_signal] = float(params["long_zero_scale"])
    multiplier[long_zero & has_signal] = float(params["long_zero_signal_scale"])

    work["rule_multiplier"] = multiplier
    work["ai_pred_qty"] = work["ai_pred_qty"].astype(float) * work["rule_multiplier"]
    work["ai_pred_positive_qty"] = work["ai_pred_positive_qty"].astype(float) * work["rule_multiplier"]
    work["ai_pred_qty_open"] = work["ai_pred_qty_open"].astype(float) * work["rule_multiplier"]
    work["abs_error"] = np.abs(work["ai_pred_qty"].astype(float) - work["true_replenish_qty"].astype(float))
    return work


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
        if not zero.empty:
            row["zero_true_fp_rate"] = safe_rate((zero["ai_pred_qty"] > 0).sum(), len(zero))
            row["zero_true_pred_mean"] = float(zero["ai_pred_qty"].mean())
        else:
            row["zero_true_fp_rate"] = np.nan
            row["zero_true_pred_mean"] = np.nan
        if not pos.empty:
            true_sum = float(pos["true_replenish_qty"].sum())
            row["positive_true_under_wape"] = safe_rate(
                np.clip(pos["true_replenish_qty"].astype(float) - pos["ai_pred_qty"].astype(float), a_min=0, a_max=None).sum(),
                true_sum,
            )
            row["positive_true_pred_mean"] = float(pos["ai_pred_qty"].mean())
        else:
            row["positive_true_under_wape"] = np.nan
            row["positive_true_pred_mean"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("stock_state").reset_index(drop=True)


def evaluate_candidate(df, params, candidate_name):
    adjusted = apply_rule(df, params)
    anchor_rows = []
    for anchor_date in ANCHORS:
        anchor_df = adjusted[adjusted["anchor_date"] == anchor_date].copy()
        row = evaluate_context_frame(anchor_df, f"{candidate_name}_{anchor_tag(anchor_date)}")
        row["anchor_date"] = anchor_date
        anchor_rows.append(row)
    anchor_eval = pd.DataFrame(anchor_rows)
    state_eval = state_metrics(adjusted)
    return adjusted, anchor_eval, state_eval


def candidate_grid():
    for short_zero_scale, short_zero_signal_scale, long_zero_scale, long_zero_signal_scale in itertools.product(
        [0.80, 0.85, 0.90],
        [0.90, 0.95, 1.00],
        [0.20, 0.30, 0.40],
        [0.45, 0.60, 0.75],
    ):
        if short_zero_signal_scale < short_zero_scale:
            continue
        if long_zero_signal_scale < long_zero_scale:
            continue
        yield {
            "short_zero_scale": short_zero_scale,
            "short_zero_signal_scale": short_zero_signal_scale,
            "long_zero_scale": long_zero_scale,
            "long_zero_signal_scale": long_zero_signal_scale,
        }


def summarize_candidate(anchor_eval, state_eval, params):
    summary = {
        **params,
        "mean_global_wmape": float(anchor_eval["global_wmape"].mean()),
        "mean_4_25_under_wape": float(anchor_eval["4_25_under_wape"].mean()),
        "mean_blockbuster_under_wape": float(anchor_eval["blockbuster_under_wape"].mean()),
        "mean_rank_corr_positive_skus": float(anchor_eval["rank_corr_positive_skus"].mean()),
        "mean_top20_true_volume_capture": float(anchor_eval["top20_true_volume_capture"].mean()),
    }
    for state_name in ["short_zero", "long_zero", "other"]:
        sub = state_eval[state_eval["stock_state"] == state_name]
        if sub.empty:
            continue
        row = sub.iloc[0]
        summary[f"{state_name}_zero_true_fp_rate"] = row.get("zero_true_fp_rate", np.nan)
        summary[f"{state_name}_positive_true_under_wape"] = row.get("positive_true_under_wape", np.nan)
    return summary


def choose_best(candidates_df, baseline_summary):
    improved = candidates_df[
        (candidates_df["mean_global_wmape"] <= baseline_summary["mean_global_wmape"])
        & (
            candidates_df["long_zero_zero_true_fp_rate"]
            <= baseline_summary["long_zero_zero_true_fp_rate"]
        )
        & (
            candidates_df["long_zero_positive_true_under_wape"]
            <= baseline_summary["long_zero_positive_true_under_wape"]
        )
    ].copy()
    if improved.empty:
        improved = candidates_df.copy()
        improved["selection_score"] = (
            improved["mean_global_wmape"]
            + 0.25 * improved["long_zero_zero_true_fp_rate"]
            + 0.15 * improved["long_zero_positive_true_under_wape"]
            + 0.10 * improved["mean_blockbuster_under_wape"]
        )
        improved = improved.sort_values("selection_score", ascending=True)
    else:
        improved = improved.sort_values(
            [
                "mean_global_wmape",
                "long_zero_zero_true_fp_rate",
                "long_zero_positive_true_under_wape",
                "mean_blockbuster_under_wape",
            ],
            ascending=[True, True, True, True],
        )
    return improved.iloc[0].to_dict()


def build_best_cases(df):
    work = df.copy()
    work["stock_state"] = "other"
    work.loc[work["inv_short_zero"] > 0, "stock_state"] = "short_zero"
    work.loc[work["inv_long_zero"] > 0, "stock_state"] = "long_zero"

    frames = []
    zero_fp = work[(work["stock_state"] == "long_zero") & (work["true_replenish_qty"] <= 0) & (work["ai_pred_qty"] > 0)].copy()
    if not zero_fp.empty:
        zero_fp = zero_fp.sort_values(
            ["ai_pred_qty", "qty_first_order", "lookback_repl_sum_90"],
            ascending=[False, False, False],
        ).head(30)
        zero_fp["case_type"] = "long_zero_zero_true_false_positive"
        frames.append(zero_fp)

    pos_under = work[(work["stock_state"] == "long_zero") & (work["true_replenish_qty"] > 0)].copy()
    if not pos_under.empty:
        pos_under["under_gap"] = (
            pos_under["true_replenish_qty"].astype(float) - pos_under["ai_pred_qty"].astype(float)
        ).clip(lower=0.0)
        pos_under = pos_under.sort_values(
            ["under_gap", "true_replenish_qty"],
            ascending=[False, False],
        ).head(30)
        pos_under["case_type"] = "long_zero_positive_true_under"
        frames.append(pos_under)

    short_zero = work[(work["stock_state"] == "short_zero") & (work["true_replenish_qty"] > 0)].copy()
    if not short_zero.empty:
        short_zero["under_gap"] = (
            short_zero["true_replenish_qty"].astype(float) - short_zero["ai_pred_qty"].astype(float)
        ).clip(lower=0.0)
        short_zero = short_zero.sort_values(
            ["under_gap", "true_replenish_qty"],
            ascending=[False, False],
        ).head(30)
        short_zero["case_type"] = "short_zero_positive_true_under"
        frames.append(short_zero)

    cases = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    keep_cols = [
        "case_type",
        "anchor_date",
        "sku_id",
        "true_replenish_qty",
        "ai_pred_qty",
        "rule_multiplier",
        "qty_first_order",
        "lookback_repl_sum_90",
        "event_strong_30",
        "inv_short_zero",
        "inv_long_zero",
        "inv_stock_zero_streak",
        "inv_positive_to_zero_switch",
    ]
    keep_cols = [col for col in keep_cols if col in cases.columns]
    return cases[keep_cols].reset_index(drop=True) if not cases.empty else cases


def write_summary(baseline_summary, best_summary, anchor_eval, state_eval):
    lines = [
        "# Phase8 Zero-Split Rule Search Summary",
        "",
        "- Status: `analysis_only_shadow`",
        "- Scope: `2026-02-15 / 2026-02-24`",
        "- Base variant for this search: `event_inventory_zero_split`",
        "- Method: asymmetric suppressive post-processing on `short_zero / long_zero` rows",
        "",
        "## Best Rule",
        "",
        f"- `short_zero_scale = {best_summary['short_zero_scale']:.2f}`",
        f"- `short_zero_signal_scale = {best_summary['short_zero_signal_scale']:.2f}`",
        f"- `long_zero_scale = {best_summary['long_zero_scale']:.2f}`",
        f"- `long_zero_signal_scale = {best_summary['long_zero_signal_scale']:.2f}`",
        "",
        "## Baseline vs Best Rule",
        "",
        "| metric | zero_split_baseline | best_rule | delta |",
        "| --- | --- | --- | --- |",
    ]
    metric_pairs = [
        ("mean_global_wmape", "mean_global_wmape"),
        ("mean_4_25_under_wape", "mean_4_25_under_wape"),
        ("mean_blockbuster_under_wape", "mean_blockbuster_under_wape"),
        ("mean_rank_corr_positive_skus", "mean_rank_corr_positive_skus"),
        ("mean_top20_true_volume_capture", "mean_top20_true_volume_capture"),
        ("long_zero_zero_true_fp_rate", "long_zero_zero_true_fp_rate"),
        ("long_zero_positive_true_under_wape", "long_zero_positive_true_under_wape"),
        ("short_zero_zero_true_fp_rate", "short_zero_zero_true_fp_rate"),
        ("short_zero_positive_true_under_wape", "short_zero_positive_true_under_wape"),
    ]
    for label, key in metric_pairs:
        base_value = float(baseline_summary.get(key, np.nan))
        best_value = float(best_summary.get(key, np.nan))
        lines.append(f"| {label} | {base_value:.4f} | {best_value:.4f} | {best_value - base_value:+.4f} |")

    lines.extend(
        [
            "",
            "## Best Rule Anchor Metrics",
            "",
            markdown_table(
                anchor_eval,
                [
                    "anchor_date",
                    "global_wmape",
                    "4_25_under_wape",
                    "blockbuster_under_wape",
                    "rank_corr_positive_skus",
                    "top20_true_volume_capture",
                ],
            ),
            "",
            "## Best Rule State Metrics",
            "",
            markdown_table(
                state_eval,
                [
                    "stock_state",
                    "rows",
                    "zero_true_rows",
                    "zero_true_fp_rate",
                    "positive_true_rows",
                    "positive_true_under_wape",
                ],
            ),
            "",
            "## Interpretation",
            "",
            "- This experiment only tests whether a lightweight rule can improve the best current zero-split shadow.",
            "- If the best rule improves both long-zero false positives and long-zero positive-true under-predict, the next step should be to operationalize it more cleanly.",
            "- If gains are tiny or unstable, keep the rule as a diagnostic only.",
            "",
        ]
    )
    (OUT_DIR / "phase8_zero_split_rule_search_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = merge_rule_frame()

    baseline_params = {
        "short_zero_scale": 1.0,
        "short_zero_signal_scale": 1.0,
        "long_zero_scale": 1.0,
        "long_zero_signal_scale": 1.0,
    }
    baseline_adjusted, baseline_anchor, baseline_state = evaluate_candidate(
        merged, baseline_params, "phase8i_baseline"
    )
    baseline_summary = summarize_candidate(baseline_anchor, baseline_state, baseline_params)

    candidate_rows = []
    best_adjusted = None
    best_anchor = None
    best_state = None
    best_meta = None
    for idx, params in enumerate(candidate_grid(), start=1):
        adjusted, anchor_eval, state_eval = evaluate_candidate(merged, params, f"phase8i_rule_{idx}")
        summary = summarize_candidate(anchor_eval, state_eval, params)
        summary["candidate_id"] = f"rule_{idx:03d}"
        candidate_rows.append(summary)

    candidates_df = pd.DataFrame(candidate_rows).sort_values("mean_global_wmape").reset_index(drop=True)
    best_meta = choose_best(candidates_df, baseline_summary)
    best_params = {
        "short_zero_scale": float(best_meta["short_zero_scale"]),
        "short_zero_signal_scale": float(best_meta["short_zero_signal_scale"]),
        "long_zero_scale": float(best_meta["long_zero_scale"]),
        "long_zero_signal_scale": float(best_meta["long_zero_signal_scale"]),
    }
    best_adjusted, best_anchor, best_state = evaluate_candidate(merged, best_params, "phase8i_best_rule")
    best_summary = summarize_candidate(best_anchor, best_state, best_params)

    candidates_df.to_csv(
        OUT_DIR / "phase8_zero_split_rule_search_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_anchor.to_csv(
        OUT_DIR / "phase8_zero_split_rule_search_anchor_eval.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_state.to_csv(
        OUT_DIR / "phase8_zero_split_rule_search_state_eval.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_best_cases(best_adjusted).to_csv(
        OUT_DIR / "phase8_zero_split_rule_search_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    payload = {
        "status": "analysis_only_shadow",
        "baseline_summary": baseline_summary,
        "best_rule_summary": best_summary,
        "best_rule_params": best_params,
        "best_rule_selector": {
            "candidate_id": best_meta.get("candidate_id", ""),
            "mean_global_wmape": float(best_meta["mean_global_wmape"]),
            "long_zero_zero_true_fp_rate": float(best_meta["long_zero_zero_true_fp_rate"]),
            "long_zero_positive_true_under_wape": float(best_meta["long_zero_positive_true_under_wape"]),
        },
        "outputs": {
            "candidates": "reports/phase8_zero_split_rule_search_2026/phase8_zero_split_rule_search_candidates.csv",
            "anchor_eval": "reports/phase8_zero_split_rule_search_2026/phase8_zero_split_rule_search_anchor_eval.csv",
            "state_eval": "reports/phase8_zero_split_rule_search_2026/phase8_zero_split_rule_search_state_eval.csv",
            "cases": "reports/phase8_zero_split_rule_search_2026/phase8_zero_split_rule_search_cases.csv",
            "summary": "reports/phase8_zero_split_rule_search_2026/phase8_zero_split_rule_search_summary.md",
        },
    }
    (OUT_DIR / "phase8_zero_split_rule_search_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_summary(baseline_summary, best_summary, best_anchor, best_state)
    print(f"[OK] zero-split rule search summary -> {OUT_DIR / 'phase8_zero_split_rule_search_summary.md'}")


if __name__ == "__main__":
    main()
