import json
import os
from typing import Dict, List

import pandas as pd

from phase_eval_utils import evaluate_context_frame, numeric_cols_for_rounding


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE54_PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_4", "phase5")
PHASE55_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_5")
PHASE57_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_7")

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_tree_anchor_calibration")
OUT_TABLE = os.path.join(OUT_DIR, "phase5_tree_anchor_calibration_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase5_tree_anchor_calibration_summary.md")
OUT_WINNER = os.path.join(OUT_DIR, "phase5_tree_anchor_calibration_winner.json")

ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
BASELINE_EXP = "p527_lstm_l3_v5_lite_s2027"
CONFIRMED_TREE = "p535_tree_hard_cov_activity"
BACKUP_TREE = "p531_tree_hard_core"
BASE_TUNED_KEY = "p57_covact_lr005_l63_hard_g025"
BASE_TUNED_EXP_TEMPLATE = "p57a_{anchor_tag}_covact_lr005_l63_s2026_hard_g025"

VARIANTS = [
    {
        "candidate_key": "base_g025",
        "base_mode": "base",
        "sep_scale": 1.00,
        "oct_scale": 1.00,
        "use_ld080_q3": False,
    },
    {
        "candidate_key": "base_g025_ld080_q3",
        "base_mode": "ld080_q3",
        "sep_scale": 1.00,
        "oct_scale": 1.00,
        "use_ld080_q3": True,
    },
    {
        "candidate_key": "sep100_oct095",
        "base_mode": "base",
        "sep_scale": 1.00,
        "oct_scale": 0.95,
        "use_ld080_q3": False,
    },
    {
        "candidate_key": "sep098_oct093",
        "base_mode": "base",
        "sep_scale": 0.98,
        "oct_scale": 0.93,
        "use_ld080_q3": False,
    },
    {
        "candidate_key": "sep095_oct090",
        "base_mode": "base",
        "sep_scale": 0.95,
        "oct_scale": 0.90,
        "use_ld080_q3": False,
    },
    {
        "candidate_key": "sep100_oct095_ld080_q3",
        "base_mode": "ld080_q3",
        "sep_scale": 1.00,
        "oct_scale": 0.95,
        "use_ld080_q3": True,
    },
    {
        "candidate_key": "sep098_oct093_ld080_q3",
        "base_mode": "ld080_q3",
        "sep_scale": 0.98,
        "oct_scale": 0.93,
        "use_ld080_q3": True,
    },
    {
        "candidate_key": "sep095_oct090_ld080_q3",
        "base_mode": "ld080_q3",
        "sep_scale": 0.95,
        "oct_scale": 0.90,
        "use_ld080_q3": True,
    },
]


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
    return os.path.join(
        PHASE54_PHASE_DIR,
        f"eval_context_p54_{BASELINE_EXP}_s2026.csv",
    )


def phase57_context_path(anchor_date: str) -> str:
    exp_id = BASE_TUNED_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
    return os.path.join(
        PHASE57_DIR,
        anchor_tag(anchor_date),
        "phase5",
        f"eval_context_{exp_id}.csv",
    )


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


def apply_variant(df: pd.DataFrame, anchor_date: str, variant: Dict) -> pd.DataFrame:
    out = df.copy()
    for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
        out[col] = out[col].astype(float)
    qty_scale = 1.0
    if anchor_date == "2025-09-01":
        qty_scale = float(variant["sep_scale"])
    elif anchor_date == "2025-10-01":
        qty_scale = float(variant["oct_scale"])

    if qty_scale != 1.0:
        for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
            out[col] = out[col] * qty_scale

    if variant["use_ld080_q3"]:
        mask = out["ai_pred_qty"].astype(float) <= 3.0
        for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
            out.loc[mask, col] = out.loc[mask, col] * 0.80

    out["ai_pred_positive_qty"] = (out["ai_pred_qty"].astype(float) > 0).astype(int)
    out["abs_error"] = (out["ai_pred_qty"].astype(float) - out["true_replenish_qty"].astype(float)).abs()
    return out


def build_candidate_rows() -> pd.DataFrame:
    rows: List[Dict] = []
    for anchor_date in ANCHORS:
        raw = load_eval_context(phase57_context_path(anchor_date))
        exp_id = BASE_TUNED_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
        for variant in VARIANTS:
            adj = apply_variant(raw, anchor_date, variant)
            row = evaluate_context_frame(adj, f"{exp_id}_{variant['candidate_key']}")
            row["anchor_date"] = anchor_date
            row["candidate_key"] = variant["candidate_key"]
            row["track"] = "event_tree_anchor_calibration"
            row["source_phase"] = "phase5_tree_anchor_calibration"
            row["source_candidate_key"] = BASE_TUNED_KEY
            row["base_mode"] = variant["base_mode"]
            row["sep_scale"] = float(variant["sep_scale"])
            row["oct_scale"] = float(variant["oct_scale"])
            row["use_ld080_q3"] = bool(variant["use_ld080_q3"])
            row["context_path"] = phase57_context_path(anchor_date)
            rows.append(row)
    return pd.DataFrame(rows)


def add_anchor_gate(candidate_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    merged = candidate_df.merge(
        baseline_df[
            [
                "anchor_date",
                "4_25_sku_p50",
                "4_25_wmape_like",
                "ice_4_25_sku_p50",
            ]
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
    summary = (
        candidate_df.groupby("candidate_key", as_index=False)
        .agg(
            base_mode=("base_mode", "first"),
            sep_scale=("sep_scale", "first"),
            oct_scale=("oct_scale", "first"),
            use_ld080_q3=("use_ld080_q3", "first"),
            **{col: (col, "mean") for col in metric_cols},
            anchor_passes=("anchor_pass", "sum"),
        )
    )
    summary["delivery_ready"] = (
        (summary["anchor_passes"] >= 3)
        & (summary["4_25_sku_p50"] >= 0.55)
        & (summary["4_25_wmape_like"] <= 0.55)
        & (summary["ice_4_25_sku_p50"] >= 0.35)
        & (summary["1_3_ratio"] <= 1.35)
    )
    return summary


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


def pick_winner(summary_df: pd.DataFrame) -> pd.Series:
    return summary_df.sort_values(
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


def render_summary(summary_df: pd.DataFrame, ref_summary_df: pd.DataFrame, winner: pd.Series) -> str:
    disp = summary_df.copy()
    ref_disp = ref_summary_df.copy()
    for frame in (disp, ref_disp):
        for col in numeric_cols_for_rounding(frame):
            frame[col] = frame[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")

    lines = [
        "# Phase 5 Tree Anchor Calibration",
        "",
        f"- source_candidate: `{BASE_TUNED_KEY}`",
        f"- selected_candidate: `{winner['candidate_key']}`",
        f"- delivery_ready: `{bool(winner['delivery_ready'])}`",
        "- next_stage: `phase6_tree_stabilization`",
        "",
        "## Candidate Summary",
        "",
        "| candidate_key | anchor_passes | global_ratio | global_wmape | 4_25_sku_p50 | 4_25_under_wape | ice_4_25_sku_p50 | 1_3_ratio | blockbuster_sku_p50 | blockbuster_under_wape | top20_true_volume_capture | rank_corr_positive_skus | delivery_ready |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in disp.sort_values(
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
    ).iterrows():
        lines.append(
            f"| {row['candidate_key']} | {int(row['anchor_passes'])} | {row['global_ratio']} | {row['global_wmape']} | "
            f"{row['4_25_sku_p50']} | {row['4_25_under_wape']} | {row['ice_4_25_sku_p50']} | "
            f"{row['1_3_ratio']} | {row['blockbuster_sku_p50']} | {row['blockbuster_under_wape']} | "
            f"{row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} | {bool(row['delivery_ready'])} |"
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
            f"{row['4_25_sku_p50']} | {row['4_25_under_wape']} | {row['ice_4_25_sku_p50']} | "
            f"{row['1_3_ratio']} | {row['blockbuster_sku_p50']} | {row['blockbuster_under_wape']} | "
            f"{row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} |"
        )
    return "\n".join(lines)


def main() -> None:
    ensure_dir()

    baseline_df = build_baseline_rows()
    reference_df = build_reference_rows()
    candidate_df = build_candidate_rows()
    candidate_df = add_anchor_gate(candidate_df, baseline_df)

    table_df = candidate_df.sort_values(["candidate_key", "anchor_date"]).copy()
    table_df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    summary_df = summarize_candidates(candidate_df)
    ref_summary_df = summarize_references(pd.concat([baseline_df, reference_df], ignore_index=True))
    winner = pick_winner(summary_df)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(render_summary(summary_df, ref_summary_df, winner))

    winner_payload = {
        "source_candidate_key": BASE_TUNED_KEY,
        "selected_candidate": str(winner["candidate_key"]),
        "delivery_ready": bool(winner["delivery_ready"]),
        "next_stage": "phase6_tree_stabilization",
        "needs_monthaware_refit": not bool(winner["delivery_ready"]),
        "candidate_summary": summary_df.sort_values("candidate_key").to_dict(orient="records"),
        "reference_summary": ref_summary_df.to_dict(orient="records"),
    }
    with open(OUT_WINNER, "w", encoding="utf-8") as fh:
        json.dump(winner_payload, fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase5_tree_anchor_calibration table -> {OUT_TABLE}")
    print(f"[OK] phase5_tree_anchor_calibration summary -> {OUT_SUMMARY}")
    print(f"[OK] phase5_tree_anchor_calibration winner -> {OUT_WINNER}")


if __name__ == "__main__":
    main()
