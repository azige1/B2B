import json
import os

import pandas as pd

from phase_eval_utils import evaluate_context_frame


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8_event_shadow")
CURRENT_EVAL_ALL = os.path.join(PROJECT_ROOT, "reports", "phase8a_prep", "phase8_current_mainline_eval_context_all_anchors.csv")
OUT_ANCHOR = os.path.join(OUT_DIR, "phase8_event_shadow_anchor_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase8_event_shadow_summary.md")
OUT_RESULT = os.path.join(OUT_DIR, "phase8_event_shadow_result.json")

ANCHORS = ["2025-10-01", "2025-11-01", "2025-12-01"]
CALIBRATION_SCALES = {"2025-10-01": 0.93, "2025-11-01": 1.00, "2025-12-01": 1.00}
CANDIDATE_KEY = "event_intent_plus"
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
    return out


def summarize_rows(df, candidate_key, track):
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
    row = {
        "candidate_key": candidate_key,
        "track": track,
        **{col: float(df[col].astype(float).mean()) for col in metric_cols},
    }
    return row


def render_summary(current_summary, shadow_summary):
    lines = [
        "# Phase8 Event Shadow Summary",
        "",
        "- Status: `shadow_only`",
        "- Scope: `2025-10/11/12` only",
        "- Official phase7 mainline remains unchanged",
        "- This compare does not participate in mainline replacement",
        "",
        "## Current Official vs Event Shadow",
        "",
        "| metric | current_phase7_3anchors | event_shadow_3anchors | delta |",
        "| --- | --- | --- | --- |",
    ]
    metric_order = [
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
    for key in metric_order:
        current_value = float(current_summary[key])
        shadow_value = float(shadow_summary[key])
        delta = shadow_value - current_value
        lines.append(f"| {key} | {current_value:.4f} | {shadow_value:.4f} | {delta:+.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Positive signal should be read only as directional evidence for event features.",
            "- Because event coverage starts on `2025-09-18`, this result is partial even on `2025-10/11/12`.",
            "- No result here is allowed to replace the official phase7 winner before client reply.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    current_all = pd.read_csv(CURRENT_EVAL_ALL)
    current_all["anchor_date"] = pd.to_datetime(current_all["anchor_date"]).dt.strftime("%Y-%m-%d")
    current_anchor_rows = []
    shadow_anchor_rows = []

    for anchor_date in ANCHORS:
        current_df = current_all[current_all["anchor_date"] == anchor_date].copy()
        current_row = evaluate_context_frame(current_df, f"phase7_current_{anchor_tag(anchor_date)}")
        current_row["anchor_date"] = anchor_date
        current_row["candidate_key"] = "phase7_current"
        current_row["track"] = "current_calibrated_mainline"
        current_anchor_rows.append(current_row)

        shadow_path = shadow_context_path(anchor_date)
        if not os.path.exists(shadow_path):
            raise FileNotFoundError(shadow_path)
        shadow_df = pd.read_csv(shadow_path)
        shadow_df = apply_calibration(shadow_df, anchor_date)
        shadow_row = evaluate_context_frame(shadow_df, f"phase8_shadow_{anchor_tag(anchor_date)}")
        shadow_row["anchor_date"] = anchor_date
        shadow_row["candidate_key"] = CANDIDATE_KEY
        shadow_row["track"] = "cov_activity_tail_full_event"
        shadow_anchor_rows.append(shadow_row)

    current_anchor_df = pd.DataFrame(current_anchor_rows)
    shadow_anchor_df = pd.DataFrame(shadow_anchor_rows)
    anchor_table = pd.concat([current_anchor_df, shadow_anchor_df], ignore_index=True, sort=False)
    anchor_table.to_csv(OUT_ANCHOR, index=False, encoding="utf-8-sig")

    current_summary = summarize_rows(current_anchor_df, "phase7_current", "current_calibrated_mainline")
    shadow_summary = summarize_rows(shadow_anchor_df, CANDIDATE_KEY, "cov_activity_tail_full_event")

    with open(OUT_SUMMARY, "w", encoding="utf-8-sig") as fh:
        fh.write(render_summary(current_summary, shadow_summary))

    payload = {
        "status": "shadow_only_no_replace",
        "anchors": ANCHORS,
        "current_summary": current_summary,
        "shadow_summary": shadow_summary,
        "outputs": {
            "anchor_table": os.path.relpath(OUT_ANCHOR, PROJECT_ROOT),
            "summary_md": os.path.relpath(OUT_SUMMARY, PROJECT_ROOT),
        },
    }
    with open(OUT_RESULT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase8 event shadow anchor table -> {OUT_ANCHOR}")
    print(f"[OK] phase8 event shadow summary -> {OUT_SUMMARY}")
    print(f"[OK] phase8 event shadow result -> {OUT_RESULT}")


if __name__ == "__main__":
    main()
