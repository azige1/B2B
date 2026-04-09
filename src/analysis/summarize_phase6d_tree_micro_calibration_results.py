import json
import os
from typing import Dict, List

import pandas as pd

from phase_eval_utils import evaluate_context_frame, numeric_cols_for_rounding


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE54_PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_4", "phase5")
PHASE55_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_5")
PHASE57_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_7")
PHASE6_STAB_DIR = os.path.join(PROJECT_ROOT, "reports", "phase6_tree_stabilization")

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase6d_tree_micro_calibration")
OUT_TABLE = os.path.join(OUT_DIR, "phase6d_tree_micro_calibration_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase6d_tree_micro_calibration_summary.md")
OUT_WINNER = os.path.join(OUT_DIR, "phase6d_tree_micro_calibration_winner.json")

ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
BASELINE_EXP = "p527_lstm_l3_v5_lite_s2027"
CONFIRMED_TREE = "p535_tree_hard_cov_activity"
BACKUP_TREE = "p531_tree_hard_core"
BASE_TUNED_EXP_TEMPLATE = "p57a_{anchor_tag}_covact_lr005_l63_s2026_hard_g025"

PRIMARY_BASELINES = {"sep095": {"sep_scale": 0.95, "oct_scale": 0.90}, "sep098": {"sep_scale": 0.98, "oct_scale": 0.93}}

VARIANTS = [
    {"candidate_key": "sep095_base", "base_family": "sep095", "sep_ld_factor": None, "sep_ld_max_qty": None},
    {"candidate_key": "sep095_sep_ld095_q3", "base_family": "sep095", "sep_ld_factor": 0.95, "sep_ld_max_qty": 3.0},
    {"candidate_key": "sep095_sep_ld090_q3", "base_family": "sep095", "sep_ld_factor": 0.90, "sep_ld_max_qty": 3.0},
    {"candidate_key": "sep095_sep_ld095_q4", "base_family": "sep095", "sep_ld_factor": 0.95, "sep_ld_max_qty": 4.0},
    {"candidate_key": "sep098_base", "base_family": "sep098", "sep_ld_factor": None, "sep_ld_max_qty": None},
    {"candidate_key": "sep098_sep_ld095_q3", "base_family": "sep098", "sep_ld_factor": 0.95, "sep_ld_max_qty": 3.0},
    {"candidate_key": "sep098_sep_ld090_q3", "base_family": "sep098", "sep_ld_factor": 0.90, "sep_ld_max_qty": 3.0},
    {"candidate_key": "sep098_sep_ld095_q4", "base_family": "sep098", "sep_ld_factor": 0.95, "sep_ld_max_qty": 4.0},
]

REFERENCE_CANDIDATES = ["p527", "p535", "p531"]


def ensure_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def anchor_tag(anchor_date: str) -> str:
    return anchor_date.replace("-", "")


def phase55_context_path(anchor_date: str, base_exp: str) -> str:
    return os.path.join(
        PHASE55_DIR,
        anchor_tag(anchor_date),
        "phase5",
        f"eval_context_p55_{anchor_tag(anchor_date)}_{base_exp}_s2026.csv",
    )


def phase54_dec_baseline_path() -> str:
    return os.path.join(PHASE54_PHASE_DIR, f"eval_context_p54_{BASELINE_EXP}_s2026.csv")


def phase57_context_path(anchor_date: str) -> str:
    exp_id = BASE_TUNED_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
    return os.path.join(PHASE57_DIR, anchor_tag(anchor_date), "phase5", f"eval_context_{exp_id}.csv")


def load_eval_context(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_baseline_rows() -> pd.DataFrame:
    rows: List[Dict] = []
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


def build_reference_rows() -> pd.DataFrame:
    refs = [
        ("p535", CONFIRMED_TREE, "event_tree_confirmed"),
        ("p531", BACKUP_TREE, "event_tree_backup"),
    ]
    rows: List[Dict] = []
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


def current_calibration_baselines() -> pd.DataFrame:
    path = os.path.join(PHASE6_STAB_DIR, "phase6_tree_stabilization_table.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    keep = df[df["candidate_key"].isin(REFERENCE_CANDIDATES)].copy()
    return keep


def apply_variant(df: pd.DataFrame, anchor_date: str, variant: Dict) -> pd.DataFrame:
    out = df.copy()
    base_cfg = PRIMARY_BASELINES[variant["base_family"]]
    for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
        out[col] = out[col].astype(float)

    qty_scale = 1.0
    if anchor_date == "2025-09-01":
        qty_scale = float(base_cfg["sep_scale"])
    elif anchor_date == "2025-10-01":
        qty_scale = float(base_cfg["oct_scale"])

    if qty_scale != 1.0:
        for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
            out[col] = out[col] * qty_scale

    if anchor_date == "2025-09-01" and variant["sep_ld_factor"] is not None:
        mask = out["ai_pred_qty"].astype(float) <= float(variant["sep_ld_max_qty"])
        for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
            out.loc[mask, col] = out.loc[mask, col] * float(variant["sep_ld_factor"])

    out["ai_pred_positive_qty"] = (out["ai_pred_qty"].astype(float) > 0).astype(int)
    out["abs_error"] = (out["ai_pred_qty"].astype(float) - out["true_replenish_qty"].astype(float)).abs()
    return out


def build_candidate_rows() -> pd.DataFrame:
    rows: List[Dict] = []
    for anchor_date in ANCHORS:
        raw = load_eval_context(phase57_context_path(anchor_date))
        base_exp = BASE_TUNED_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
        for variant in VARIANTS:
            adj = apply_variant(raw, anchor_date, variant)
            row = evaluate_context_frame(adj, f"{base_exp}_{variant['candidate_key']}")
            row["anchor_date"] = anchor_date
            row["candidate_key"] = variant["candidate_key"]
            row["track"] = "event_tree_micro_calibration"
            row["source_phase"] = "phase6d_tree_micro_calibration"
            row["base_family"] = variant["base_family"]
            row["sep_ld_factor"] = variant["sep_ld_factor"]
            row["sep_ld_max_qty"] = variant["sep_ld_max_qty"]
            row["context_path"] = phase57_context_path(anchor_date)
            rows.append(row)
    return pd.DataFrame(rows)


def add_anchor_gate(candidate_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    merged = candidate_df.merge(
        baseline_df[["anchor_date", "4_25_sku_p50", "4_25_wmape_like", "ice_4_25_sku_p50"]],
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


def summarize_candidates(candidate_df: pd.DataFrame) -> pd.DataFrame:
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
        candidate_df.groupby(["candidate_key", "track"], as_index=False)
        .agg(
            base_family=("base_family", "first"),
            sep_ld_factor=("sep_ld_factor", "first"),
            sep_ld_max_qty=("sep_ld_max_qty", "first"),
            **{col: (col, "mean") for col in metric_cols},
            anchor_passes=("anchor_pass", "sum"),
        )
    )


def summarize_references(ref_df: pd.DataFrame) -> pd.DataFrame:
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


def rank_candidates(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.copy()
    ranked["guardrail_ok"] = ranked["global_ratio"].between(0.90, 1.10)
    ranked["primary_gate_ok"] = (
        (ranked["anchor_passes"] >= 4)
        & (ranked["4_25_sku_p50"] >= 0.55)
        & (ranked["ice_4_25_sku_p50"] >= 0.35)
        & ranked["guardrail_ok"]
    )
    ranked["secondary_ok"] = (
        ranked["blockbuster_under_wape"].notna()
        & ranked["blockbuster_sku_p50"].notna()
    )
    return ranked.sort_values(
        [
            "anchor_passes",
            "4_25_under_wape",
            "4_25_sku_p50",
            "ice_4_25_sku_p50",
            "blockbuster_under_wape",
            "blockbuster_sku_p50",
            "top20_true_volume_capture",
            "rank_corr_positive_skus",
            "global_wmape",
            "1_3_ratio",
        ],
        ascending=[False, True, False, False, True, False, False, False, True, True],
    )


def pick_winner(summary_df: pd.DataFrame) -> pd.Series:
    return rank_candidates(summary_df).iloc[0]


def render_summary(summary_df: pd.DataFrame, ref_summary_df: pd.DataFrame, winner: pd.Series) -> str:
    disp = rank_candidates(summary_df).copy()
    ref_disp = ref_summary_df.copy()
    for frame in (disp, ref_disp):
        for col in numeric_cols_for_rounding(frame):
            frame[col] = frame[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")

    lines = [
        "# Phase6d Tree Micro Calibration",
        "",
        f"- selected_candidate: `{winner['candidate_key']}`",
        f"- primary_gate_ok: `{bool(winner['primary_gate_ok'])}`",
        f"- guardrail_ok: `{bool(winner['guardrail_ok'])}`",
        "",
        "## Candidate Summary",
        "",
        "| candidate_key | anchor_passes | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus | global_ratio | global_wmape | 1_3_ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in disp.iterrows():
        lines.append(
            f"| {row['candidate_key']} | {int(row['anchor_passes'])} | {row['4_25_under_wape']} | {row['4_25_sku_p50']} | "
            f"{row['ice_4_25_sku_p50']} | {row['blockbuster_under_wape']} | {row['blockbuster_sku_p50']} | "
            f"{row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} | {row['global_ratio']} | {row['global_wmape']} | {row['1_3_ratio']} |"
        )

    lines.extend(
        [
            "",
            "## Reference Summary",
            "",
            "| candidate_key | track | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus | global_ratio | global_wmape | 1_3_ratio |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in ref_disp.iterrows():
        lines.append(
            f"| {row['candidate_key']} | {row['track']} | {row['4_25_under_wape']} | {row['4_25_sku_p50']} | "
            f"{row['ice_4_25_sku_p50']} | {row['blockbuster_under_wape']} | {row['blockbuster_sku_p50']} | "
            f"{row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} | {row['global_ratio']} | {row['global_wmape']} | {row['1_3_ratio']} |"
        )
    return "\n".join(lines)


def main() -> None:
    ensure_dir()

    baseline_df = build_baseline_rows()
    reference_df = build_reference_rows()
    candidate_df = build_candidate_rows()
    candidate_df = add_anchor_gate(candidate_df, baseline_df)

    candidate_df.sort_values(["candidate_key", "anchor_date"]).to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    summary_df = summarize_candidates(candidate_df)
    ref_summary_df = summarize_references(pd.concat([baseline_df, reference_df], ignore_index=True, sort=False))
    winner = pick_winner(summary_df)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(render_summary(summary_df, ref_summary_df, winner))

    payload = {
        "selected_candidate": str(winner["candidate_key"]),
        "primary_gate_ok": bool(winner["primary_gate_ok"]),
        "guardrail_ok": bool(winner["guardrail_ok"]),
        "candidate_summary": rank_candidates(summary_df).to_dict(orient="records"),
        "reference_summary": ref_summary_df.to_dict(orient="records"),
        "next_stage": "freeze_current_tree_mainline" if bool(winner["primary_gate_ok"]) else "phase6e_tree_family_compare",
    }
    with open(OUT_WINNER, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase6d tree micro calibration table -> {OUT_TABLE}")
    print(f"[OK] phase6d tree micro calibration summary -> {OUT_SUMMARY}")
    print(f"[OK] phase6d tree micro calibration winner -> {OUT_WINNER}")


if __name__ == "__main__":
    main()
