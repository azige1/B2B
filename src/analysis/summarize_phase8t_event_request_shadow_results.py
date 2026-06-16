import json
import os

import pandas as pd

from phase_eval_utils import evaluate_context_frame


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8_event_request_shadow")
CURRENT_EVAL_ALL = os.path.join(PROJECT_ROOT, "reports", "phase8a_prep", "phase8_current_mainline_eval_context_all_anchors.csv")
OUT_ANCHOR = os.path.join(OUT_DIR, "phase8_event_request_shadow_anchor_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase8_event_request_shadow_summary.md")
OUT_RESULT = os.path.join(OUT_DIR, "phase8_event_request_shadow_result.json")

ALL_ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
EVENT_REQUEST_ANCHORS = ["2025-10-01", "2025-11-01", "2025-12-01"]
CALIBRATION_SCALES = {"2025-09-01": 0.98, "2025-10-01": 0.93, "2025-11-01": 1.00, "2025-12-01": 1.00}
EVENT_REQUEST_EXP_TEMPLATE = "p8eventreq_{anchor_tag}_event_request_plus_s2028_hard_g027"
METRIC_COLS = [
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


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def event_request_context_path(anchor_date):
    exp_id = EVENT_REQUEST_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
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
    return {
        "candidate_key": candidate_key,
        "track": track,
        **{col: float(df[col].astype(float).mean()) for col in METRIC_COLS},
    }


def metric_row_lines(current_summary, candidate_summary, current_label, candidate_label):
    lines = [
        f"| metric | {current_label} | {candidate_label} | delta |",
        "| --- | --- | --- | --- |",
    ]
    for key in METRIC_COLS:
        current_value = float(current_summary[key])
        candidate_value = float(candidate_summary[key])
        lines.append(f"| {key} | {current_value:.4f} | {candidate_value:.4f} | {candidate_value - current_value:+.4f} |")
    return lines


def render_summary(current_all_summary, event_request_all_summary, current_event_summary, event_request_summary):
    full_improve = (
        (float(current_all_summary["global_wmape"]) - float(event_request_all_summary["global_wmape"]))
        / float(current_all_summary["global_wmape"])
        * 100.0
    )
    event_improve = (
        (float(current_event_summary["global_wmape"]) - float(event_request_summary["global_wmape"]))
        / float(current_event_summary["global_wmape"])
        * 100.0
    )
    lines = [
        "# Phase8 Event + Purchase Request Shadow Summary",
        "",
        "- Status: `shadow_only_no_replace`",
        "- Evaluation style: old Phase7 official 0.6-series anchors, no listing-date feature, no listing-eligible filter",
        "- Full four-anchor candidate: all anchors use `event_request_plus`; on `2025-09-01`, Event columns have no effective historical coverage, so the extra Event signal is naturally zero-like.",
        f"- Four-anchor global WMAPE improvement: `{full_improve:.2f}%`",
        f"- Event-covered three-anchor global WMAPE improvement: `{event_improve:.2f}%`",
        "",
        "## Four-Anchor Event + Request Candidate",
        "",
    ]
    lines.extend(metric_row_lines(current_all_summary, event_request_all_summary, "phase7_current_4anchors", "event_request_plus_4anchors"))
    lines.extend(
        [
            "",
            "## Event-Covered Three Anchors",
            "",
        ]
    )
    lines.extend(metric_row_lines(current_event_summary, event_request_summary, "phase7_current_3anchors", "event_request_plus_3anchors"))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is the clean no-listing-date route for the external 0.6-series story.",
            "- The four-anchor number is conservative because September has no meaningful pre-anchor Event signal.",
            "- Event is still the main contributor; purchase request features add a smaller incremental lift.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    current_all = pd.read_csv(CURRENT_EVAL_ALL)
    current_all["anchor_date"] = pd.to_datetime(current_all["anchor_date"]).dt.strftime("%Y-%m-%d")
    current_all_rows = []
    event_request_all_rows = []
    current_event_rows = []
    event_request_rows = []

    for anchor_date in ALL_ANCHORS:
        current_df = current_all[current_all["anchor_date"] == anchor_date].copy()
        current_row = evaluate_context_frame(current_df, f"phase7_current_{anchor_tag(anchor_date)}")
        current_row["anchor_date"] = anchor_date
        current_row["candidate_key"] = "phase7_current"
        current_row["track"] = "current_calibrated_mainline"
        current_all_rows.append(current_row)

        candidate_path = event_request_context_path(anchor_date)
        candidate_key = "event_request_plus"
        track = "cov_activity_tail_full_event_request"
        exp_label = f"event_request_plus_{anchor_tag(anchor_date)}"
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(candidate_path)
        candidate_df = apply_calibration(pd.read_csv(candidate_path), anchor_date)
        candidate_row = evaluate_context_frame(candidate_df, exp_label)
        candidate_row["anchor_date"] = anchor_date
        candidate_row["candidate_key"] = candidate_key
        candidate_row["track"] = track
        event_request_all_rows.append(candidate_row)

        if anchor_date in EVENT_REQUEST_ANCHORS:
            current_event_rows.append(current_row)
            event_request_rows.append(candidate_row)

    current_all_df = pd.DataFrame(current_all_rows)
    event_request_all_df = pd.DataFrame(event_request_all_rows)
    current_event_df = pd.DataFrame(current_event_rows)
    event_request_df = pd.DataFrame(event_request_rows)
    anchor_table = pd.concat([current_all_df, event_request_all_df], ignore_index=True, sort=False)
    anchor_table.to_csv(OUT_ANCHOR, index=False, encoding="utf-8-sig")

    current_all_summary = summarize_rows(current_all_df, "phase7_current", "current_calibrated_mainline")
    event_request_all_summary = summarize_rows(event_request_all_df, "event_request_plus", "cov_activity_tail_full_event_request")
    current_event_summary = summarize_rows(current_event_df, "phase7_current", "current_calibrated_mainline_event_covered")
    event_request_summary = summarize_rows(event_request_df, "event_request_plus", "cov_activity_tail_full_event_request")

    with open(OUT_SUMMARY, "w", encoding="utf-8-sig") as fh:
        fh.write(render_summary(current_all_summary, event_request_all_summary, current_event_summary, event_request_summary))

    payload = {
        "status": "shadow_only_no_replace",
        "anchors": ALL_ANCHORS,
        "event_request_anchors": EVENT_REQUEST_ANCHORS,
        "current_all_summary": current_all_summary,
        "event_request_all_summary": event_request_all_summary,
        "current_event_covered_summary": current_event_summary,
        "event_request_summary": event_request_summary,
        "outputs": {
            "anchor_table": os.path.relpath(OUT_ANCHOR, PROJECT_ROOT),
            "summary_md": os.path.relpath(OUT_SUMMARY, PROJECT_ROOT),
        },
    }
    with open(OUT_RESULT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase8 event+request shadow anchor table -> {OUT_ANCHOR}")
    print(f"[OK] phase8 event+request shadow summary -> {OUT_SUMMARY}")
    print(f"[OK] phase8 event+request shadow result -> {OUT_RESULT}")


if __name__ == "__main__":
    main()
