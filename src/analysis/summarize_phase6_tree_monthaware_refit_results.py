import json
import os

import pandas as pd

from phase_eval_utils import evaluate_context_frame, numeric_cols_for_rounding


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE54_PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_4", "phase5")
PHASE55_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_5")
PHASE5_TREE_CAL_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_tree_anchor_calibration")
PHASE6_DIR = os.path.join(PROJECT_ROOT, "reports", "phase6_tree_monthaware_refit")

OUT_TABLE = os.path.join(PHASE6_DIR, "phase6_tree_monthaware_refit_table.csv")
OUT_SUMMARY = os.path.join(PHASE6_DIR, "phase6_tree_monthaware_refit_summary.md")
OUT_WINNER = os.path.join(PHASE6_DIR, "phase6_tree_monthaware_refit_winner.json")

ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
BASELINE_EXP = "p527_lstm_l3_v5_lite_s2027"
CONFIRMED_TREE = "p535_tree_hard_cov_activity"
BACKUP_TREE = "p531_tree_hard_core"

REF_KEEP_CANDIDATES = [
    "sep098_oct093",
    "sep095_oct090",
    "base_g025_ld080_q3",
]
MONTHAWARE_BASE_KEY = "p6c_monthaware_g025"
MONTHAWARE_LD_KEY = "p6c_monthaware_g025_ld080_q3"
MONTHAWARE_EXP_TEMPLATE = "p6c_{anchor_tag}_covact_mrefit_lr003_l31_c80r60_n600_s2026_hard_g025"


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def ensure_dir():
    os.makedirs(PHASE6_DIR, exist_ok=True)


def load_eval_context(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def phase55_context_path(anchor_date, base_exp):
    return os.path.join(
        PHASE55_DIR,
        anchor_tag(anchor_date),
        "phase5",
        f"eval_context_p55_{anchor_tag(anchor_date)}_{base_exp}_s2026.csv",
    )


def phase54_dec_baseline_path():
    return os.path.join(PHASE54_PHASE_DIR, f"eval_context_p54_{BASELINE_EXP}_s2026.csv")


def monthaware_context_path(anchor_date):
    exp_id = MONTHAWARE_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
    return os.path.join(
        PHASE6_DIR,
        anchor_tag(anchor_date),
        "phase5",
        f"eval_context_{exp_id}.csv",
    )


def build_baseline_rows():
    rows = []
    for anchor_date in ANCHORS:
        if anchor_date == "2025-12-01":
            path = phase54_dec_baseline_path()
            source_phase = "phase5_4_fallback"
            exp_id = f"p54_{BASELINE_EXP}_s2026"
        else:
            path = phase55_context_path(anchor_date, BASELINE_EXP)
            source_phase = "phase5_5"
            exp_id = f"p55_{anchor_tag(anchor_date)}_{BASELINE_EXP}_s2026"
        df = load_eval_context(path)
        row = evaluate_context_frame(df, exp_id)
        row["anchor_date"] = anchor_date
        row["candidate_key"] = "p527"
        row["track"] = "sequence_baseline"
        row["source_phase"] = source_phase
        row["context_path"] = path
        rows.append(row)
    return pd.DataFrame(rows)


def build_reference_rows():
    refs = [
        ("p535", CONFIRMED_TREE, "event_tree_confirmed"),
        ("p531", BACKUP_TREE, "event_tree_backup"),
    ]
    rows = []
    for candidate_key, base_exp, track in refs:
        for anchor_date in ANCHORS:
            path = phase55_context_path(anchor_date, base_exp)
            df = load_eval_context(path)
            row = evaluate_context_frame(df, f"p55_{anchor_tag(anchor_date)}_{base_exp}_s2026")
            row["anchor_date"] = anchor_date
            row["candidate_key"] = candidate_key
            row["track"] = track
            row["source_phase"] = "phase5_5"
            row["context_path"] = path
            rows.append(row)
    return pd.DataFrame(rows)


def build_phase6b_reference_rows():
    path = os.path.join(PHASE5_TREE_CAL_DIR, "phase5_tree_anchor_calibration_table.csv")
    df = pd.read_csv(path)
    keep = df[df["candidate_key"].isin(REF_KEEP_CANDIDATES)].copy()
    keep["track"] = "event_tree_anchor_calibration"
    return keep


def apply_ld080_q3(df):
    adj = df.copy()
    for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
        adj[col] = adj[col].astype(float)
    mask = adj["ai_pred_qty"] <= 3.0
    for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
        adj.loc[mask, col] = adj.loc[mask, col] * 0.80
    adj["ai_pred_positive_qty"] = (adj["ai_pred_qty"] > 0).astype(int)
    adj["abs_error"] = (adj["ai_pred_qty"] - adj["true_replenish_qty"]).abs()
    return adj


def build_monthaware_rows():
    rows = []
    for anchor_date in ANCHORS:
        path = monthaware_context_path(anchor_date)
        raw = load_eval_context(path)
        exp_id = MONTHAWARE_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))

        base_row = evaluate_context_frame(raw, exp_id)
        base_row["anchor_date"] = anchor_date
        base_row["candidate_key"] = MONTHAWARE_BASE_KEY
        base_row["track"] = "event_tree_monthaware_refit"
        base_row["source_phase"] = "phase6_tree_monthaware_refit"
        base_row["context_path"] = path
        rows.append(base_row)

        ld_row = evaluate_context_frame(apply_ld080_q3(raw), f"{exp_id}_ld080_q3")
        ld_row["anchor_date"] = anchor_date
        ld_row["candidate_key"] = MONTHAWARE_LD_KEY
        ld_row["track"] = "event_tree_monthaware_refit"
        ld_row["source_phase"] = "phase6_tree_monthaware_refit"
        ld_row["context_path"] = path
        rows.append(ld_row)
    return pd.DataFrame(rows)


def add_anchor_gate(candidate_df, baseline_df):
    merged = candidate_df.merge(
        baseline_df[
            ["anchor_date", "4_25_sku_p50", "4_25_wmape_like", "ice_4_25_sku_p50"]
        ],
        on="anchor_date",
        how="left",
        suffixes=("", "_baseline"),
    )
    merged["anchor_pass"] = (
        (merged["4_25_sku_p50"] > merged["4_25_sku_p50_baseline"])
        & (merged["4_25_wmape_like"] <= merged["4_25_wmape_like_baseline"])
        & (merged["ice_4_25_sku_p50"] > merged["ice_4_25_sku_p50_baseline"])
        & merged["global_ratio"].between(0.90, 1.10)
    )
    return merged


def summarize_candidates(candidate_df):
    metric_cols = [
        "global_ratio",
        "global_wmape",
        "4_25_ratio",
        "4_25_wmape_like",
        "4_25_sku_p50",
        "4_25_under_wape",
        "ice_sku_p50",
        "ice_4_25_sku_p50",
        "1_3_ratio",
        "1_3_over_wape",
        "blockbuster_ratio",
        "blockbuster_wmape_like",
        "blockbuster_sku_p50",
        "blockbuster_under_wape",
        "blockbuster_within_50pct_rate",
        "top20_true_volume_capture",
        "rank_corr_positive_skus",
        "auc",
        "f1",
    ]
    summary = (
        candidate_df.groupby(["candidate_key", "track"], as_index=False)
        .agg(**{col: (col, "mean") for col in metric_cols}, anchor_passes=("anchor_pass", "sum"))
    )
    summary["strict_rnd_ready"] = (
        (summary["anchor_passes"] >= 4)
        & (summary["4_25_sku_p50"] >= 0.55)
        & (summary["4_25_wmape_like"] <= 0.55)
        & (summary["ice_4_25_sku_p50"] >= 0.35)
        & (summary["1_3_ratio"] <= 1.35)
    )
    summary["relaxed_rnd_ready"] = (
        (summary["anchor_passes"] >= 4)
        & (summary["4_25_sku_p50"] >= 0.55)
        & (summary["4_25_wmape_like"] <= 0.55)
        & (summary["ice_4_25_sku_p50"] >= 0.35)
        & (summary["1_3_ratio"] <= 1.40)
    )
    return summary


def summarize_references(ref_df):
    metric_cols = [
        "global_ratio",
        "global_wmape",
        "4_25_ratio",
        "4_25_wmape_like",
        "4_25_sku_p50",
        "4_25_under_wape",
        "ice_sku_p50",
        "ice_4_25_sku_p50",
        "1_3_ratio",
        "1_3_over_wape",
        "blockbuster_ratio",
        "blockbuster_wmape_like",
        "blockbuster_sku_p50",
        "blockbuster_under_wape",
        "blockbuster_within_50pct_rate",
        "top20_true_volume_capture",
        "rank_corr_positive_skus",
        "auc",
        "f1",
    ]
    return (
        ref_df.groupby(["candidate_key", "track"], as_index=False)
        .agg(**{col: (col, "mean") for col in metric_cols})
        .sort_values(["track", "candidate_key"])
    )


def pick_winner(summary_df):
    candidates = summary_df[summary_df["track"].isin(["event_tree_anchor_calibration", "event_tree_monthaware_refit"])].copy()
    return candidates.sort_values(
        [
            "anchor_passes",
            "4_25_sku_p50",
            "global_wmape",
            "1_3_ratio",
            "blockbuster_sku_p50",
            "blockbuster_under_wape",
            "top20_true_volume_capture",
            "rank_corr_positive_skus",
        ],
        ascending=[False, False, True, True, False, True, False, False],
    ).iloc[0]


def render_summary(summary_df, ref_summary_df, winner):
    disp = summary_df.copy()
    ref_disp = ref_summary_df.copy()
    for frame in (disp, ref_disp):
        for col in numeric_cols_for_rounding(frame):
            frame[col] = frame[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")

    lines = [
        "# Phase6 Tree Month-Aware Refit",
        "",
        f"- selected_candidate: `{winner['candidate_key']}`",
        f"- strict_rnd_ready: `{bool(winner['strict_rnd_ready'])}`",
        f"- relaxed_rnd_ready: `{bool(winner['relaxed_rnd_ready'])}`",
        "",
        "## Candidate Summary",
        "",
        "| candidate_key | track | anchor_passes | global_ratio | global_wmape | 4_25_sku_p50 | 4_25_under_wape | ice_4_25_sku_p50 | 1_3_ratio | blockbuster_sku_p50 | blockbuster_under_wape | top20_true_volume_capture | rank_corr_positive_skus | strict_rnd_ready |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in disp.sort_values(
        ["track", "anchor_passes", "4_25_sku_p50", "global_wmape"],
        ascending=[True, False, False, True],
    ).iterrows():
        anchor_passes = row["anchor_passes"] if "anchor_passes" in row.index and row["track"] in {"event_tree_anchor_calibration", "event_tree_monthaware_refit"} else ""
        strict_flag = bool(row["strict_rnd_ready"]) if "strict_rnd_ready" in row.index else ""
        lines.append(
            f"| {row['candidate_key']} | {row['track']} | {anchor_passes} | {row['global_ratio']} | {row['global_wmape']} | "
            f"{row['4_25_sku_p50']} | {row['4_25_under_wape']} | {row['ice_4_25_sku_p50']} | {row['1_3_ratio']} | "
            f"{row['blockbuster_sku_p50']} | {row['blockbuster_under_wape']} | {row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} | {strict_flag} |"
        )

    lines.extend(
        [
            "",
            "## Reference Summary",
            "",
            "| candidate_key | track | global_ratio | global_wmape | 4_25_sku_p50 | 4_25_under_wape | ice_4_25_sku_p50 | 1_3_ratio | blockbuster_sku_p50 | blockbuster_under_wape | top20_true_volume_capture | rank_corr_positive_skus |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in ref_disp.iterrows():
        lines.append(
            f"| {row['candidate_key']} | {row['track']} | {row['global_ratio']} | {row['global_wmape']} | "
            f"{row['4_25_sku_p50']} | {row['4_25_under_wape']} | {row['ice_4_25_sku_p50']} | {row['1_3_ratio']} | "
            f"{row['blockbuster_sku_p50']} | {row['blockbuster_under_wape']} | {row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} |"
        )
    return "\n".join(lines)


def main():
    ensure_dir()

    baseline_df = build_baseline_rows()
    reference_df = build_reference_rows()
    prior_candidates = build_phase6b_reference_rows()
    monthaware_df = build_monthaware_rows()
    monthaware_df = add_anchor_gate(monthaware_df, baseline_df)

    all_candidates = pd.concat([prior_candidates, monthaware_df], ignore_index=True, sort=False)
    all_candidates.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    summary_df = summarize_candidates(all_candidates)
    ref_summary_df = summarize_references(pd.concat([baseline_df, reference_df], ignore_index=True, sort=False))
    winner = pick_winner(summary_df)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(render_summary(summary_df, ref_summary_df, winner))

    payload = {
        "selected_candidate": str(winner["candidate_key"]),
        "strict_rnd_ready": bool(winner["strict_rnd_ready"]),
        "relaxed_rnd_ready": bool(winner["relaxed_rnd_ready"]),
        "next_stage": "phase6d_tree_family_compare" if bool(winner["strict_rnd_ready"]) else "manual_review_required",
        "candidate_summary": summary_df.sort_values(["track", "candidate_key"]).to_dict(orient="records"),
        "reference_summary": ref_summary_df.to_dict(orient="records"),
    }
    with open(OUT_WINNER, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase6 tree monthaware refit table -> {OUT_TABLE}")
    print(f"[OK] phase6 tree monthaware refit summary -> {OUT_SUMMARY}")
    print(f"[OK] phase6 tree monthaware refit winner -> {OUT_WINNER}")


if __name__ == "__main__":
    main()
