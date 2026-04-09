import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase_eval_utils import evaluate_context_frame
from generate_phase8i_zero_split_rule_search import markdown_table, merge_rule_frame, state_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "reports" / "phase8_zero_split_hard_gate_search_2026"
ANCHORS = ["2026-02-15", "2026-02-24"]


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def apply_gate_rule(df, params):
    work = df.copy()
    has_signal = (
        (work["event_strong_30"] > 0) | (work["lookback_repl_sum_90"] > 0)
    ).to_numpy()
    short_zero = (work["inv_short_zero"] > 0).to_numpy()
    long_zero = (
        (work["inv_long_zero"] > 0)
        & (work["inv_stock_zero_streak"] >= float(params["long_zero_min_streak"]))
    ).to_numpy()

    gate_mask = np.zeros(len(work), dtype=bool)
    gate_mask |= short_zero & ~has_signal & (work["ai_pred_qty"].astype(float).to_numpy() <= float(params["short_zero_gate_qty"]))
    gate_mask |= long_zero & ~has_signal & (work["ai_pred_qty"].astype(float).to_numpy() <= float(params["long_zero_gate_qty"]))
    gate_mask |= long_zero & has_signal & (work["ai_pred_qty"].astype(float).to_numpy() <= float(params["long_zero_signal_gate_qty"]))

    work["gate_rule_applied"] = gate_mask.astype(int)
    work.loc[gate_mask, "ai_pred_qty"] = 0.0
    work.loc[gate_mask, "ai_pred_positive_qty"] = 0.0
    work.loc[gate_mask, "ai_pred_qty_open"] = 0.0
    work["abs_error"] = np.abs(work["ai_pred_qty"].astype(float) - work["true_replenish_qty"].astype(float))
    return work


def evaluate_candidate(df, params, candidate_name):
    adjusted = apply_gate_rule(df, params)
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
    for short_zero_gate_qty, long_zero_gate_qty, long_zero_signal_gate_qty, long_zero_min_streak in itertools.product(
        [0.0, 0.5, 1.0],
        [0.5, 1.0, 1.5, 2.0],
        [0.0, 0.5, 1.0],
        [8.0, 14.0, 21.0],
    ):
        if long_zero_signal_gate_qty > long_zero_gate_qty:
            continue
        yield {
            "short_zero_gate_qty": short_zero_gate_qty,
            "long_zero_gate_qty": long_zero_gate_qty,
            "long_zero_signal_gate_qty": long_zero_signal_gate_qty,
            "long_zero_min_streak": long_zero_min_streak,
        }


def summarize_candidate(anchor_eval, state_eval, params, adjusted):
    summary = {
        **params,
        "mean_global_wmape": float(anchor_eval["global_wmape"].mean()),
        "mean_4_25_under_wape": float(anchor_eval["4_25_under_wape"].mean()),
        "mean_blockbuster_under_wape": float(anchor_eval["blockbuster_under_wape"].mean()),
        "mean_rank_corr_positive_skus": float(anchor_eval["rank_corr_positive_skus"].mean()),
        "mean_top20_true_volume_capture": float(anchor_eval["top20_true_volume_capture"].mean()),
        "gate_rule_rate": float(adjusted["gate_rule_applied"].mean()),
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
    feasible = candidates_df[
        (candidates_df["long_zero_zero_true_fp_rate"] < baseline_summary["long_zero_zero_true_fp_rate"])
        & (
            candidates_df["long_zero_positive_true_under_wape"]
            <= baseline_summary["long_zero_positive_true_under_wape"] + 0.03
        )
        & (candidates_df["mean_global_wmape"] <= baseline_summary["mean_global_wmape"] + 0.01)
    ].copy()
    if feasible.empty:
        candidates_df = candidates_df.copy()
        candidates_df["selection_score"] = (
            candidates_df["mean_global_wmape"]
            + 0.30 * candidates_df["long_zero_zero_true_fp_rate"]
            + 0.20 * candidates_df["long_zero_positive_true_under_wape"]
            + 0.05 * candidates_df["mean_blockbuster_under_wape"]
        )
        return candidates_df.sort_values("selection_score", ascending=True).iloc[0].to_dict()
    return feasible.sort_values(
        [
            "mean_global_wmape",
            "long_zero_zero_true_fp_rate",
            "long_zero_positive_true_under_wape",
        ],
        ascending=[True, True, True],
    ).iloc[0].to_dict()


def build_best_cases(df):
    work = df.copy()
    work["stock_state"] = "other"
    work.loc[work["inv_short_zero"] > 0, "stock_state"] = "short_zero"
    work.loc[work["inv_long_zero"] > 0, "stock_state"] = "long_zero"

    frames = []
    gated = work[work["gate_rule_applied"] == 1].copy()
    if not gated.empty:
        gated["case_type"] = "gated_rows"
        frames.append(gated.sort_values(["anchor_date", "stock_state", "ai_pred_qty"], ascending=[True, True, True]).head(60))

    surviving_fp = work[
        (work["stock_state"] == "long_zero")
        & (work["true_replenish_qty"] <= 0)
        & (work["ai_pred_qty"] > 0)
    ].copy()
    if not surviving_fp.empty:
        surviving_fp["case_type"] = "long_zero_zero_true_still_positive"
        frames.append(
            surviving_fp.sort_values(
                ["ai_pred_qty", "qty_first_order", "lookback_repl_sum_90"],
                ascending=[False, False, False],
            ).head(30)
        )

    risky_pos = work[
        (work["stock_state"] == "long_zero")
        & (work["true_replenish_qty"] > 0)
    ].copy()
    if not risky_pos.empty:
        risky_pos["under_gap"] = (
            risky_pos["true_replenish_qty"].astype(float) - risky_pos["ai_pred_qty"].astype(float)
        ).clip(lower=0.0)
        risky_pos["case_type"] = "long_zero_positive_true_under"
        frames.append(
            risky_pos.sort_values(["under_gap", "true_replenish_qty"], ascending=[False, False]).head(30)
        )

    cases = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    keep_cols = [
        "case_type",
        "anchor_date",
        "sku_id",
        "true_replenish_qty",
        "ai_pred_qty",
        "gate_rule_applied",
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
        "# Phase8 Zero-Split Hard Gate Search Summary",
        "",
        "- Status: `analysis_only_shadow`",
        "- Scope: `2026-02-15 / 2026-02-24`",
        "- Base variant for this search: `event_inventory_zero_split`",
        "- Method: asymmetric hard-gate post-processing on `short_zero / long_zero` rows",
        "",
        "## Best Rule",
        "",
        f"- `short_zero_gate_qty = {best_summary['short_zero_gate_qty']:.2f}`",
        f"- `long_zero_gate_qty = {best_summary['long_zero_gate_qty']:.2f}`",
        f"- `long_zero_signal_gate_qty = {best_summary['long_zero_signal_gate_qty']:.2f}`",
        f"- `long_zero_min_streak = {best_summary['long_zero_min_streak']:.0f}`",
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
        ("gate_rule_rate", "gate_rule_rate"),
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
            "- This experiment is useful only if it actually changes the zero-true false-positive rate on `long_zero` rows.",
            "- If it reduces false positives but destroys positive-true rows, it should remain diagnostic only.",
            "",
        ]
    )
    (OUT_DIR / "phase8_zero_split_hard_gate_search_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = merge_rule_frame()

    baseline_params = {
        "short_zero_gate_qty": 0.0,
        "long_zero_gate_qty": 0.0,
        "long_zero_signal_gate_qty": 0.0,
        "long_zero_min_streak": 8.0,
    }
    baseline_adjusted, baseline_anchor, baseline_state = evaluate_candidate(
        merged, baseline_params, "phase8j_baseline"
    )
    baseline_summary = summarize_candidate(
        baseline_anchor, baseline_state, baseline_params, baseline_adjusted
    )

    candidate_rows = []
    for idx, params in enumerate(candidate_grid(), start=1):
        adjusted, anchor_eval, state_eval = evaluate_candidate(merged, params, f"phase8j_rule_{idx}")
        summary = summarize_candidate(anchor_eval, state_eval, params, adjusted)
        summary["candidate_id"] = f"hard_gate_{idx:03d}"
        candidate_rows.append(summary)

    candidates_df = pd.DataFrame(candidate_rows).sort_values("mean_global_wmape").reset_index(drop=True)
    best_meta = choose_best(candidates_df, baseline_summary)
    best_params = {
        "short_zero_gate_qty": float(best_meta["short_zero_gate_qty"]),
        "long_zero_gate_qty": float(best_meta["long_zero_gate_qty"]),
        "long_zero_signal_gate_qty": float(best_meta["long_zero_signal_gate_qty"]),
        "long_zero_min_streak": float(best_meta["long_zero_min_streak"]),
    }
    best_adjusted, best_anchor, best_state = evaluate_candidate(merged, best_params, "phase8j_best_rule")
    best_summary = summarize_candidate(best_anchor, best_state, best_params, best_adjusted)

    candidates_df.to_csv(
        OUT_DIR / "phase8_zero_split_hard_gate_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_anchor.to_csv(
        OUT_DIR / "phase8_zero_split_hard_gate_anchor_eval.csv",
        index=False,
        encoding="utf-8-sig",
    )
    best_state.to_csv(
        OUT_DIR / "phase8_zero_split_hard_gate_state_eval.csv",
        index=False,
        encoding="utf-8-sig",
    )
    build_best_cases(best_adjusted).to_csv(
        OUT_DIR / "phase8_zero_split_hard_gate_cases.csv",
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
            "candidates": "reports/phase8_zero_split_hard_gate_search_2026/phase8_zero_split_hard_gate_candidates.csv",
            "anchor_eval": "reports/phase8_zero_split_hard_gate_search_2026/phase8_zero_split_hard_gate_anchor_eval.csv",
            "state_eval": "reports/phase8_zero_split_hard_gate_search_2026/phase8_zero_split_hard_gate_state_eval.csv",
            "cases": "reports/phase8_zero_split_hard_gate_search_2026/phase8_zero_split_hard_gate_cases.csv",
            "summary": "reports/phase8_zero_split_hard_gate_search_2026/phase8_zero_split_hard_gate_search_summary.md",
        },
    }
    (OUT_DIR / "phase8_zero_split_hard_gate_search_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_summary(baseline_summary, best_summary, best_anchor, best_state)
    print(f"[OK] zero-split hard gate summary -> {OUT_DIR / 'phase8_zero_split_hard_gate_search_summary.md'}")


if __name__ == "__main__":
    main()
