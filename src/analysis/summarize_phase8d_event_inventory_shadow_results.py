import json
import os

import pandas as pd

from phase_eval_utils import evaluate_context_frame


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8_event_inventory_shadow_2026")
OUT_ANCHOR = os.path.join(OUT_DIR, "phase8_event_inventory_shadow_anchor_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase8_event_inventory_shadow_summary.md")
OUT_RESULT = os.path.join(OUT_DIR, "phase8_event_inventory_shadow_result.json")

ANCHORS = ["2026-02-15", "2026-02-24"]
BASE_RAW_EXP_TEMPLATE = "p8ei_{anchor_tag}_base_tail_full_s2028_hard_g027"
SHADOW_RAW_EXP_TEMPLATE = "p8ei_{anchor_tag}_event_inventory_plus_s2028_hard_g027"


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def context_path(anchor_date, exp_template):
    exp_id = exp_template.format(anchor_tag=anchor_tag(anchor_date))
    return os.path.join(OUT_DIR, anchor_tag(anchor_date), "phase5", f"eval_context_{exp_id}.csv")


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
    return {
        "candidate_key": candidate_key,
        "track": track,
        **{col: float(df[col].astype(float).mean()) for col in metric_cols},
    }


def render_summary(base_summary, shadow_summary):
    lines = [
        "# Phase8 Event+Inventory Shadow 2026 Summary",
        "",
        "- Status: `analysis_only_shadow`",
        "- Scope: `2026-02-15 / 2026-02-24`",
        "- Compare: base `cov_activity_tail_full` vs extended `cov_activity_tail_full_event` with event+inventory features in the event group",
        "- This compare does not participate in official phase7 replacement",
        "",
        "## Base vs Event+Inventory Shadow",
        "",
        "| metric | base_tail_full | event_inventory_shadow | delta |",
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
        base_value = float(base_summary[key])
        shadow_value = float(shadow_summary[key])
        lines.append(f"| {key} | {base_value:.4f} | {shadow_value:.4f} | {shadow_value - base_value:+.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a 2026-only exploratory line.",
            "- It is useful for deciding whether inventory should join the event shadow line in later phase8 work.",
            "- No result here is allowed to replace the official phase7 winner.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base_rows = []
    shadow_rows = []
    for anchor_date in ANCHORS:
        base_df = pd.read_csv(context_path(anchor_date, BASE_RAW_EXP_TEMPLATE))
        shadow_df = pd.read_csv(context_path(anchor_date, SHADOW_RAW_EXP_TEMPLATE))

        base_row = evaluate_context_frame(base_df, f"phase8_ei_base_{anchor_tag(anchor_date)}")
        base_row["anchor_date"] = anchor_date
        base_row["candidate_key"] = "base_tail_full"
        base_row["track"] = "cov_activity_tail_full"
        base_rows.append(base_row)

        shadow_row = evaluate_context_frame(shadow_df, f"phase8_ei_shadow_{anchor_tag(anchor_date)}")
        shadow_row["anchor_date"] = anchor_date
        shadow_row["candidate_key"] = "event_inventory_plus"
        shadow_row["track"] = "cov_activity_tail_full_event"
        shadow_rows.append(shadow_row)

    base_anchor_df = pd.DataFrame(base_rows)
    shadow_anchor_df = pd.DataFrame(shadow_rows)
    anchor_table = pd.concat([base_anchor_df, shadow_anchor_df], ignore_index=True, sort=False)
    anchor_table.to_csv(OUT_ANCHOR, index=False, encoding="utf-8-sig")

    base_summary = summarize_rows(base_anchor_df, "base_tail_full", "cov_activity_tail_full")
    shadow_summary = summarize_rows(shadow_anchor_df, "event_inventory_plus", "cov_activity_tail_full_event")

    with open(OUT_SUMMARY, "w", encoding="utf-8-sig") as fh:
        fh.write(render_summary(base_summary, shadow_summary))

    payload = {
        "status": "analysis_only_shadow",
        "anchors": ANCHORS,
        "base_summary": base_summary,
        "shadow_summary": shadow_summary,
        "outputs": {
            "anchor_table": os.path.relpath(OUT_ANCHOR, PROJECT_ROOT),
            "summary_md": os.path.relpath(OUT_SUMMARY, PROJECT_ROOT),
        },
    }
    with open(OUT_RESULT, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"[OK] event+inventory shadow anchor table -> {OUT_ANCHOR}")
    print(f"[OK] event+inventory shadow summary -> {OUT_SUMMARY}")
    print(f"[OK] event+inventory shadow result -> {OUT_RESULT}")


if __name__ == "__main__":
    main()
