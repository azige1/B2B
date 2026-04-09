import json
import os

import pandas as pd

from phase_eval_utils import evaluate_context_frame, numeric_cols_for_rounding


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE57_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_7")
PHASE54_PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_4", "phase5")
PHASE55_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_5")
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase7_tail_allocation_optimization")
OUT_RAW = os.path.join(OUT_DIR, "phase7b_raw_anchor_table.csv")
OUT_CAL = os.path.join(OUT_DIR, "phase7b_calibrated_anchor_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase7b_summary.md")
OUT_WINNER = os.path.join(OUT_DIR, "phase7b_winner.json")

ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
BASELINE_EXP = "p527_lstm_l3_v5_lite_s2027"
RAW_TEMPLATE = "p57a_{anchor_tag}_covact_lr005_l63_s2026_hard_g025"
VARIANTS = [
    {"candidate_key": "qfo_plus", "feature_set": "cov_activity_qfo"},
    {"candidate_key": "tail_peak", "feature_set": "cov_activity_tail"},
    {"candidate_key": "style_category_priors", "feature_set": "cov_activity_priors"},
    {"candidate_key": "tail_full", "feature_set": "cov_activity_tail_full"},
]
CALIBRATION_SCALES = {"2025-09-01": 0.98, "2025-10-01": 0.93, "2025-11-01": 1.00, "2025-12-01": 1.00}


def ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def phase57_context_path(anchor_date):
    exp_id = RAW_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
    return os.path.join(PHASE57_DIR, anchor_tag(anchor_date), "phase5", f"eval_context_{exp_id}.csv")


def phase54_dec_baseline_path():
    return os.path.join(PHASE54_PHASE_DIR, f"eval_context_p54_{BASELINE_EXP}_s2026.csv")


def phase55_context_path(anchor_date, base_exp):
    return os.path.join(
        PHASE55_DIR,
        anchor_tag(anchor_date),
        "phase5",
        f"eval_context_p55_{anchor_tag(anchor_date)}_{base_exp}_s2026.csv",
    )


def phase7_context_path(anchor_date, candidate_key):
    exp_id = f"p7b_{anchor_tag(anchor_date)}_{candidate_key}_s2026_hard_g025"
    return os.path.join(OUT_DIR, anchor_tag(anchor_date), "phase5", f"eval_context_{exp_id}.csv")


def load_context(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def apply_calibration(df, anchor_date):
    out = df.copy()
    scale = float(CALIBRATION_SCALES.get(anchor_date, 1.0))
    if scale != 1.0:
        for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
            out[col] = out[col].astype(float) * scale
    out["ai_pred_positive_qty"] = (out["ai_pred_qty"].astype(float) > 0).astype(int)
    out["abs_error"] = (out["ai_pred_qty"].astype(float) - out["true_replenish_qty"].astype(float)).abs()
    return out


def add_anchor_pass(anchor_df, baseline_df):
    merged = anchor_df.merge(
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


def build_baseline_rows():
    rows = []
    for anchor_date in ANCHORS:
        if anchor_date == "2025-12-01":
            path = phase54_dec_baseline_path()
            exp_id = f"p54_{BASELINE_EXP}_s2026"
        else:
            path = phase55_context_path(anchor_date, BASELINE_EXP)
            exp_id = f"p55_{anchor_tag(anchor_date)}_{BASELINE_EXP}_s2026"
        row = evaluate_context_frame(load_context(path), exp_id)
        row["anchor_date"] = anchor_date
        row["candidate_key"] = "p527"
        row["track"] = "sequence_baseline"
        rows.append(row)
    return pd.DataFrame(rows)


def build_current_raw_rows():
    rows = []
    for anchor_date in ANCHORS:
        df = load_context(phase57_context_path(anchor_date))
        row = evaluate_context_frame(df, f"{RAW_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))}_raw")
        row["anchor_date"] = anchor_date
        row["candidate_key"] = "lightgbm_raw_g025"
        row["track"] = "current_raw_mainline"
        rows.append(row)
    return pd.DataFrame(rows)


def build_variant_raw_rows():
    rows = []
    for variant in VARIANTS:
        for anchor_date in ANCHORS:
            path = phase7_context_path(anchor_date, variant["candidate_key"])
            if not os.path.exists(path):
                continue
            df = load_context(path)
            row = evaluate_context_frame(df, f"p7b_{anchor_tag(anchor_date)}_{variant['candidate_key']}_raw")
            row["anchor_date"] = anchor_date
            row["candidate_key"] = variant["candidate_key"]
            row["track"] = variant["feature_set"]
            rows.append(row)
    return pd.DataFrame(rows)


def build_current_calibrated_rows():
    rows = []
    for anchor_date in ANCHORS:
        df = apply_calibration(load_context(phase57_context_path(anchor_date)), anchor_date)
        row = evaluate_context_frame(df, f"{RAW_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))}_sep098_oct093")
        row["anchor_date"] = anchor_date
        row["candidate_key"] = "lightgbm_sep098_oct093"
        row["track"] = "current_calibrated_mainline"
        rows.append(row)
    return pd.DataFrame(rows)


def build_variant_calibrated_rows():
    rows = []
    for variant in VARIANTS:
        for anchor_date in ANCHORS:
            path = phase7_context_path(anchor_date, variant["candidate_key"])
            if not os.path.exists(path):
                continue
            df = apply_calibration(load_context(path), anchor_date)
            row = evaluate_context_frame(df, f"p7b_{anchor_tag(anchor_date)}_{variant['candidate_key']}_sep098_oct093")
            row["anchor_date"] = anchor_date
            row["candidate_key"] = variant["candidate_key"]
            row["track"] = variant["feature_set"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_candidates(df, include_anchor_pass=False):
    metric_cols = [
        "global_ratio",
        "global_wmape",
        "4_25_under_wape",
        "4_25_sku_p50",
        "ice_4_25_sku_p50",
        "blockbuster_under_wape",
        "blockbuster_sku_p50",
        "top20_true_volume_capture",
        "rank_corr_positive_skus",
        "1_3_ratio",
    ]
    agg = {col: (col, "mean") for col in metric_cols}
    if include_anchor_pass:
        agg["anchor_passes"] = ("anchor_pass", "sum")
    return df.groupby(["candidate_key", "track"], as_index=False).agg(**agg)


def add_raw_gate(raw_summary):
    base = raw_summary.loc[raw_summary["candidate_key"] == "lightgbm_raw_g025"].iloc[0]
    out = raw_summary.copy()
    gain = (
        (out["blockbuster_under_wape"] <= float(base["blockbuster_under_wape"]) - 0.02)
        | (out["blockbuster_sku_p50"] >= float(base["blockbuster_sku_p50"]) + 0.03)
        | (out["top20_true_volume_capture"] >= float(base["top20_true_volume_capture"]) + 0.02)
        | (out["rank_corr_positive_skus"] >= float(base["rank_corr_positive_skus"]) + 0.02)
    )
    out["raw_stage_pass"] = (
        out["global_ratio"].between(0.90, 1.10)
        & (out["4_25_under_wape"] <= float(base["4_25_under_wape"]) + 0.01)
        & (out["4_25_sku_p50"] >= float(base["4_25_sku_p50"]) - 0.01)
        & (out["ice_4_25_sku_p50"] >= float(base["ice_4_25_sku_p50"]) - 0.01)
        & gain
    )
    return out


def add_replacement_gate(cal_summary):
    base = cal_summary.loc[cal_summary["candidate_key"] == "lightgbm_sep098_oct093"].iloc[0]
    tail_gain = (
        (cal_summary["blockbuster_under_wape"] <= float(base["blockbuster_under_wape"]) - 0.02)
        | (cal_summary["blockbuster_sku_p50"] >= float(base["blockbuster_sku_p50"]) + 0.03)
    )
    alloc_gain = (
        (cal_summary["top20_true_volume_capture"] >= float(base["top20_true_volume_capture"]) + 0.02)
        | (cal_summary["rank_corr_positive_skus"] >= float(base["rank_corr_positive_skus"]) + 0.02)
    )
    out = cal_summary.copy()
    out["replacement_gate"] = (
        (out["anchor_passes"] >= 4)
        & (out["4_25_under_wape"] <= float(base["4_25_under_wape"]) + 0.01)
        & (out["4_25_sku_p50"] >= float(base["4_25_sku_p50"]) - 0.01)
        & (out["ice_4_25_sku_p50"] >= float(base["ice_4_25_sku_p50"]) - 0.01)
        & tail_gain
        & alloc_gain
    )
    return out


def render_summary(raw_summary, cal_summary, winner_action):
    raw_disp = raw_summary.copy()
    cal_disp = cal_summary.copy()
    for frame in (raw_disp, cal_disp):
        for col in numeric_cols_for_rounding(frame):
            if col in frame.columns:
                frame[col] = frame[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")

    lines = [
        "# Phase7b Tail Feature Sweep",
        "",
        f"- winner_action: `{winner_action}`",
        "",
        "## Raw Four-Anchor Compare",
        "",
        "| candidate_key | track | raw_stage_pass | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in raw_disp.iterrows():
        lines.append(
            f"| {row['candidate_key']} | {row['track']} | {row.get('raw_stage_pass', '')} | {row['global_ratio']} | {row['4_25_under_wape']} | "
            f"{row['4_25_sku_p50']} | {row['ice_4_25_sku_p50']} | {row['blockbuster_under_wape']} | {row['blockbuster_sku_p50']} | "
            f"{row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} |"
        )

    lines.extend(
        [
            "",
            "## Calibrated Final Compare",
            "",
            "| candidate_key | track | replacement_gate | anchor_passes | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in cal_disp.iterrows():
        lines.append(
            f"| {row['candidate_key']} | {row['track']} | {row.get('replacement_gate', '')} | {int(row.get('anchor_passes', 0))} | {row['global_ratio']} | "
            f"{row['4_25_under_wape']} | {row['4_25_sku_p50']} | {row['ice_4_25_sku_p50']} | {row['blockbuster_under_wape']} | "
            f"{row['blockbuster_sku_p50']} | {row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} |"
        )
    return "\n".join(lines)


def main():
    ensure_dir()

    baseline_anchor = build_baseline_rows()
    raw_anchor = pd.concat([build_current_raw_rows(), build_variant_raw_rows()], ignore_index=True, sort=False)
    cal_anchor = pd.concat([build_current_calibrated_rows(), build_variant_calibrated_rows()], ignore_index=True, sort=False)

    if raw_anchor.empty or cal_anchor.empty:
        print("[BLOCKED] phase7b summary missing candidate eval_context files")
        raise SystemExit(1)

    raw_anchor.to_csv(OUT_RAW, index=False, encoding="utf-8-sig")
    cal_anchor = add_anchor_pass(cal_anchor, baseline_anchor)
    cal_anchor.to_csv(OUT_CAL, index=False, encoding="utf-8-sig")

    raw_summary = add_raw_gate(summarize_candidates(raw_anchor, include_anchor_pass=False))
    cal_summary = add_replacement_gate(summarize_candidates(cal_anchor, include_anchor_pass=True))

    promoted = cal_summary[(cal_summary["candidate_key"] != "lightgbm_sep098_oct093") & (cal_summary["replacement_gate"])]
    if promoted.empty:
        winner_action = "freeze_current_tree_mainline"
        selected = "lightgbm_sep098_oct093"
    else:
        promoted = promoted.sort_values(
            ["blockbuster_under_wape", "blockbuster_sku_p50", "top20_true_volume_capture", "rank_corr_positive_skus"],
            ascending=[True, False, False, False],
        )
        winner_action = "promote_phase7_candidate"
        selected = str(promoted.iloc[0]["candidate_key"])

    with open(OUT_SUMMARY, "w", encoding="utf-8-sig") as fh:
        fh.write(render_summary(raw_summary, cal_summary, winner_action))

    payload = {
        "winner_action": winner_action,
        "selected_candidate": selected,
        "raw_summary": raw_summary.to_dict(orient="records"),
        "calibrated_summary": cal_summary.to_dict(orient="records"),
    }
    with open(OUT_WINNER, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase7b raw anchor table -> {OUT_RAW}")
    print(f"[OK] phase7b calibrated anchor table -> {OUT_CAL}")
    print(f"[OK] phase7b summary -> {OUT_SUMMARY}")
    print(f"[OK] phase7b winner -> {OUT_WINNER}")


if __name__ == "__main__":
    main()
