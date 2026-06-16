import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from phase_eval_utils import evaluate_context_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "reports" / "phase8_event_request_router"
CURRENT_EVAL_ALL = PROJECT_ROOT / "reports" / "phase8a_prep" / "phase8_current_mainline_eval_context_all_anchors.csv"
PHASE8_EVENT_REQUEST_DIR = PROJECT_ROOT / "reports" / "phase8_event_request_shadow"
PHASE8_PURCHASE_DIR = PROJECT_ROOT / "reports" / "phase8_purchase_request_shadow"

ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
EVENT_COVERED_ANCHORS = {"2025-10-01", "2025-11-01", "2025-12-01"}
CALIBRATION_SCALES = {"2025-09-01": 0.98, "2025-10-01": 0.93, "2025-11-01": 1.00, "2025-12-01": 1.00}
PRED_COLS = [
    "ai_pred_prob",
    "cls_pred_best_f1",
    "ai_pred_qty_open",
    "ai_pred_qty",
    "ai_pred_positive_qty",
    "qty_gate_mask",
    "dead_blocked",
]
METRIC_COLS = [
    "global_ratio",
    "global_wmape",
    "under_wape",
    "over_wape",
    "false_positive_rate_zero_true",
    "zero_true_pred_ge_3_rate",
    "4_25_under_wape",
    "ice_4_25_sku_p50",
    "blockbuster_under_wape",
    "blockbuster_sku_p50",
    "rank_corr_positive_skus",
    "top20_true_volume_capture",
    "1_3_ratio",
]


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def event_request_context_path(anchor_date):
    tag = anchor_tag(anchor_date)
    exp_id = f"p8eventreq_{tag}_event_request_plus_s2028_hard_g027"
    return PHASE8_EVENT_REQUEST_DIR / tag / "phase5" / f"eval_context_{exp_id}.csv"


def purchase_context_path(anchor_date):
    tag = anchor_tag(anchor_date)
    exp_id = f"p8req_{tag}_purchase_request_plus_s2028_hard_g027"
    return PHASE8_PURCHASE_DIR / tag / "phase5" / f"eval_context_{exp_id}.csv"


def apply_calibration(df, anchor_date):
    out = df.copy()
    scale = float(CALIBRATION_SCALES.get(anchor_date, 1.0))
    for col in ("ai_pred_qty_open", "ai_pred_qty"):
        if col in out.columns:
            out[col] = out[col].astype(float) * scale
    out["ai_pred_positive_qty"] = (out["ai_pred_qty"].astype(float) > 0).astype(int)
    out["abs_error"] = (out["ai_pred_qty"].astype(float) - out["true_replenish_qty"].astype(float)).abs()
    return out


def normalize_keys(df):
    out = df.copy()
    out["sku_id"] = out["sku_id"].astype(str)
    out["anchor_date"] = pd.to_datetime(out["anchor_date"]).dt.strftime("%Y-%m-%d")
    return out


def load_phase7_frames():
    df = pd.read_csv(CURRENT_EVAL_ALL)
    df = normalize_keys(df)
    return {anchor: df[df["anchor_date"] == anchor].copy().reset_index(drop=True) for anchor in ANCHORS}


def overlay_predictions(base_df, pred_df):
    base = normalize_keys(base_df).copy()
    pred = normalize_keys(pred_df)[["sku_id", "anchor_date", *PRED_COLS]].copy()
    merged = base.drop(columns=[col for col in PRED_COLS if col in base.columns]).merge(
        pred,
        on=["sku_id", "anchor_date"],
        how="left",
        validate="one_to_one",
    )
    if merged[PRED_COLS].isna().any().any():
        raise ValueError("Prediction overlay produced missing prediction values.")
    merged["abs_error"] = (merged["ai_pred_qty"].astype(float) - merged["true_replenish_qty"].astype(float)).abs()
    return merged


def load_model_frames(phase7_frames, model):
    frames = {}
    for anchor in ANCHORS:
        if model == "phase7":
            frames[anchor] = phase7_frames[anchor].copy()
            continue
        if model == "purchase":
            pred = apply_calibration(pd.read_csv(purchase_context_path(anchor)), anchor)
        elif model == "event_request":
            pred = apply_calibration(pd.read_csv(event_request_context_path(anchor)), anchor)
        else:
            raise ValueError(model)
        frames[anchor] = overlay_predictions(phase7_frames[anchor], pred)
    return frames


def load_event_coverage(anchor_date):
    tag = anchor_tag(anchor_date)
    artifacts_dir = PROJECT_ROOT / "data" / f"artifacts_v6_event_p8eventreqshadow_{tag}_v6_event"
    processed_dir = PROJECT_ROOT / "data" / f"processed_v6_event_p8eventreqshadow_{tag}_v6_event"
    meta = json.loads((artifacts_dir / "meta_v6_event.json").read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]
    index = {name: idx for idx, name in enumerate(feature_cols)}
    needed = [
        "event_active_buyers_30",
        "event_clicks_30",
        "event_view_order_30",
        "event_cart_adds_30",
        "event_order_success_30",
        "event_pay_success_30",
        "event_order_submit_qty_30",
        "event_pay_qty_30",
        "event_days_since_last_any",
        "event_days_since_last_strong",
    ]
    missing = [name for name in needed if name not in index]
    if missing:
        raise KeyError(f"Missing event feature columns: {missing}")

    x_val = np.load(processed_dir / "X_val.npy", mmap_mode="r")
    val_keys = pd.read_csv(artifacts_dir / "val_keys.csv")
    out = val_keys.copy()
    out["sku_id"] = out["sku_id"].astype(str)
    out["anchor_date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for name in needed:
        out[name] = np.asarray(x_val[:, index[name]], dtype=np.float32)

    out["event_any_30"] = (
        out["event_clicks_30"]
        + out["event_view_order_30"]
        + out["event_cart_adds_30"]
        + out["event_order_success_30"]
        + out["event_pay_success_30"]
        + out["event_order_submit_qty_30"]
        + out["event_pay_qty_30"]
    )
    out["event_strong_30"] = (
        out["event_view_order_30"]
        + out["event_cart_adds_30"]
        + out["event_order_success_30"]
        + out["event_pay_success_30"]
        + out["event_order_submit_qty_30"]
        + out["event_pay_qty_30"]
    )
    return out[
        [
            "sku_id",
            "anchor_date",
            "event_active_buyers_30",
            "event_any_30",
            "event_strong_30",
            "event_days_since_last_any",
            "event_days_since_last_strong",
        ]
    ]


def attach_coverage(frames):
    out = {}
    event_cols = [
        "event_active_buyers_30",
        "event_any_30",
        "event_strong_30",
        "event_days_since_last_any",
        "event_days_since_last_strong",
    ]
    for anchor, frame in frames.items():
        coverage = load_event_coverage(anchor)
        base = normalize_keys(frame).drop(columns=[col for col in event_cols if col in frame.columns])
        merged = base.merge(
            coverage,
            on=["sku_id", "anchor_date"],
            how="left",
            validate="one_to_one",
        )
        if merged[event_cols].isna().any().any():
            raise ValueError(f"Missing event coverage after merge for {anchor}")
        out[anchor] = merged
    return out


def route_frame(fallback_df, event_df, use_event_mask, candidate_key, rule_name, fallback_name):
    event_pred = event_df[PRED_COLS].reset_index(drop=True)
    out = fallback_df.copy().reset_index(drop=True)
    mask = np.asarray(use_event_mask, dtype=bool)
    for col in PRED_COLS:
        out.loc[mask, col] = event_pred.loc[mask, col].values
    out["abs_error"] = (out["ai_pred_qty"].astype(float) - out["true_replenish_qty"].astype(float)).abs()
    out["candidate_key"] = candidate_key
    out["router_rule"] = rule_name
    out["router_fallback"] = fallback_name
    out["router_use_event"] = mask.astype(int)
    return out


def evaluate_candidate(candidate_key, rule_name, fallback_name, anchor_frames):
    rows = []
    contexts = []
    for anchor, frame in anchor_frames.items():
        row = evaluate_context_frame(frame, f"{candidate_key}_{anchor_tag(anchor)}")
        row["anchor_date"] = anchor
        row["candidate_key"] = candidate_key
        row["router_rule"] = rule_name
        row["router_fallback"] = fallback_name
        row["event_use_rate"] = float(frame["router_use_event"].mean()) if "router_use_event" in frame.columns else np.nan
        rows.append(row)
        contexts.append(frame)
    anchor_df = pd.DataFrame(rows)
    summary = {
        "candidate_key": candidate_key,
        "router_rule": rule_name,
        "router_fallback": fallback_name,
        "event_use_rate": float(np.nanmean(anchor_df["event_use_rate"])) if "event_use_rate" in anchor_df else np.nan,
    }
    for col in METRIC_COLS:
        summary[col] = float(anchor_df[col].astype(float).mean())
    return summary, anchor_df, pd.concat(contexts, ignore_index=True, sort=False)


def build_baseline_candidate(name, frames):
    routed = {}
    for anchor, frame in frames.items():
        out = frame.copy()
        out["candidate_key"] = name
        out["router_rule"] = "baseline"
        out["router_fallback"] = ""
        out["router_use_event"] = 1 if name == "event_request_all" else 0
        routed[anchor] = out
    return evaluate_candidate(name, "baseline", "", routed)


def build_anchor_route_candidate(name, fallback_frames, event_frames, fallback_name):
    routed = {}
    for anchor in ANCHORS:
        use_event = anchor in EVENT_COVERED_ANCHORS
        mask = np.full(len(fallback_frames[anchor]), use_event, dtype=bool)
        routed[anchor] = route_frame(
            fallback_frames[anchor],
            event_frames[anchor],
            mask,
            name,
            "anchor_event_covered",
            fallback_name,
        )
    return evaluate_candidate(name, "anchor_event_covered", fallback_name, routed)


def grid_rules():
    rules = []
    for any_min in [1, 5, 10, 25, 50, 100, 250]:
        rules.append((f"event_any_30_ge_{any_min}", lambda df, x=any_min: df["event_any_30"] >= x))
    for active_min in [1, 2, 5, 10, 20]:
        rules.append((f"event_active_buyers_30_ge_{active_min}", lambda df, x=active_min: df["event_active_buyers_30"] >= x))
    for strong_min in [1, 5, 10, 25, 50]:
        rules.append((f"event_strong_30_ge_{strong_min}", lambda df, x=strong_min: df["event_strong_30"] >= x))
    for any_min in [1, 10, 50, 100]:
        for recency_max in [7, 14, 30, 60, 120]:
            rules.append(
                (
                    f"event_any_30_ge_{any_min}_recency_any_le_{recency_max}",
                    lambda df, x=any_min, r=recency_max: (df["event_any_30"] >= x)
                    & (df["event_days_since_last_any"] <= r),
                )
            )
    for active_min in [1, 2, 5, 10]:
        for strong_min in [1, 5, 10, 25]:
            rules.append(
                (
                    f"active_buyers_ge_{active_min}_strong_ge_{strong_min}",
                    lambda df, a=active_min, s=strong_min: (df["event_active_buyers_30"] >= a)
                    & (df["event_strong_30"] >= s),
                )
            )
    return rules


def build_grid_candidate(rule_name, rule_fn, fallback_frames, event_frames, fallback_name):
    routed = {}
    for anchor in ANCHORS:
        event_with_coverage = event_frames[anchor]
        mask = np.asarray(rule_fn(event_with_coverage), dtype=bool)
        routed[anchor] = route_frame(
            fallback_frames[anchor],
            event_with_coverage,
            mask,
            f"router_{fallback_name}_{rule_name}",
            rule_name,
            fallback_name,
        )
    return evaluate_candidate(f"router_{fallback_name}_{rule_name}", rule_name, fallback_name, routed)


def render_summary(summary_df, anchor_df, best_key):
    lookup = summary_df.set_index("candidate_key")
    phase7 = lookup.loc["phase7_all"]
    event_request = lookup.loc["event_request_all"]
    anchor_route = lookup.loc["router_phase7_anchor_event_covered"]
    best = lookup.loc[best_key]

    lines = [
        "# Phase8 Event Coverage Router Summary",
        "",
        "- Status: `shadow_router_exploration`",
        "- Evaluation style: old Phase7 0.6-series four anchors, no listing-date feature, no listing-eligible filter.",
        "- Router only combines existing predictions; it does not retrain the model and does not use future labels as input.",
        "",
        "## Key Candidates",
        "",
        "| candidate | rule | fallback | event_use_rate | global_wmape | global_ratio | blockbuster_under_wape | rank_corr_positive_skus |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    display_keys = [
        "phase7_all",
        "purchase_all",
        "event_request_all",
        "router_phase7_anchor_event_covered",
    ]
    if best_key not in display_keys:
        display_keys.append(best_key)
    for key in display_keys:
        row = lookup.loc[key]
        lines.append(
            f"| {key} | {row['router_rule']} | {row['router_fallback']} | "
            f"{float(row['event_use_rate']):.4f} | {float(row['global_wmape']):.4f} | "
            f"{float(row['global_ratio']):.4f} | {float(row['blockbuster_under_wape']):.4f} | "
            f"{float(row['rank_corr_positive_skus']):.4f} |"
        )

    def improve(candidate):
        return (float(phase7["global_wmape"]) - float(candidate["global_wmape"])) / float(phase7["global_wmape"]) * 100.0

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- `event_request_all` improves WMAPE from `{float(phase7['global_wmape']):.4f}` to `{float(event_request['global_wmape']):.4f}` (`{improve(event_request):.2f}%`).",
            f"- Anchor-level route improves WMAPE to `{float(anchor_route['global_wmape']):.4f}` (`{improve(anchor_route):.2f}%`) by falling back on `2025-09-01` where Event has no useful pre-anchor coverage.",
            f"- Best gated candidate is `{best_key}`, WMAPE `{float(best['global_wmape']):.4f}`, blockbuster under-WAPE `{float(best['blockbuster_under_wape']):.4f}`.",
            "- If a row-level coverage rule does not beat anchor-level routing while preserving blockbuster metrics, prefer anchor-level routing because it is simpler and easier to explain.",
            "",
            "## Outputs",
            "",
            "- Candidate summary: `reports/phase8_event_request_router/phase8_event_request_router_candidate_summary.csv`",
            "- Anchor metrics: `reports/phase8_event_request_router/phase8_event_request_router_anchor_table.csv`",
            "- Best context: `reports/phase8_event_request_router/phase8_event_request_router_best_context.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    phase7_frames = attach_coverage(load_phase7_frames())
    purchase_frames = attach_coverage(load_model_frames(phase7_frames, "purchase"))
    event_frames = attach_coverage(load_model_frames(phase7_frames, "event_request"))

    summaries = []
    anchor_tables = []
    contexts = {}

    for name, frames in [
        ("phase7_all", phase7_frames),
        ("purchase_all", purchase_frames),
        ("event_request_all", event_frames),
    ]:
        summary, anchor_df, context = build_baseline_candidate(name, frames)
        summaries.append(summary)
        anchor_tables.append(anchor_df)
        contexts[name] = context

    for fallback_name, fallback_frames in [("phase7", phase7_frames), ("purchase", purchase_frames)]:
        name = f"router_{fallback_name}_anchor_event_covered"
        summary, anchor_df, context = build_anchor_route_candidate(name, fallback_frames, event_frames, fallback_name)
        summaries.append(summary)
        anchor_tables.append(anchor_df)
        contexts[name] = context

        for rule_name, rule_fn in grid_rules():
            summary, anchor_df, context = build_grid_candidate(rule_name, rule_fn, fallback_frames, event_frames, fallback_name)
            summaries.append(summary)
            anchor_tables.append(anchor_df)
            contexts[summary["candidate_key"]] = context

    summary_df = pd.DataFrame(summaries)
    anchor_df = pd.concat(anchor_tables, ignore_index=True, sort=False)

    phase7_wmape = float(summary_df.loc[summary_df["candidate_key"] == "phase7_all", "global_wmape"].iloc[0])
    event_wmape = float(summary_df.loc[summary_df["candidate_key"] == "event_request_all", "global_wmape"].iloc[0])
    event_blockbuster = float(summary_df.loc[summary_df["candidate_key"] == "event_request_all", "blockbuster_under_wape"].iloc[0])
    gated = summary_df[
        (summary_df["global_ratio"].between(0.98, 1.05))
        & (summary_df["global_wmape"] <= event_wmape)
        & (summary_df["blockbuster_under_wape"] <= event_blockbuster + 1e-9)
    ].copy()
    if gated.empty:
        gated = summary_df[summary_df["global_ratio"].between(0.98, 1.05)].copy()
    gated["improvement_pct_vs_phase7"] = (phase7_wmape - gated["global_wmape"].astype(float)) / phase7_wmape * 100.0
    best_key = str(gated.sort_values(["global_wmape", "blockbuster_under_wape"], ascending=[True, True]).iloc[0]["candidate_key"])

    summary_df["improvement_pct_vs_phase7"] = (phase7_wmape - summary_df["global_wmape"].astype(float)) / phase7_wmape * 100.0
    summary_df = summary_df.sort_values(["global_wmape", "blockbuster_under_wape"], ascending=[True, True])
    summary_df.to_csv(OUT_DIR / "phase8_event_request_router_candidate_summary.csv", index=False, encoding="utf-8-sig")
    anchor_df.to_csv(OUT_DIR / "phase8_event_request_router_anchor_table.csv", index=False, encoding="utf-8-sig")
    contexts[best_key].to_csv(OUT_DIR / "phase8_event_request_router_best_context.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "phase8_event_request_router_summary.md").write_text(
        render_summary(summary_df, anchor_df, best_key),
        encoding="utf-8-sig",
    )
    payload = {
        "status": "shadow_router_exploration",
        "anchors": ANCHORS,
        "best_candidate_key": best_key,
        "best_summary": summary_df[summary_df["candidate_key"] == best_key].iloc[0].to_dict(),
        "outputs": {
            "candidate_summary": str((OUT_DIR / "phase8_event_request_router_candidate_summary.csv").relative_to(PROJECT_ROOT)),
            "anchor_table": str((OUT_DIR / "phase8_event_request_router_anchor_table.csv").relative_to(PROJECT_ROOT)),
            "summary_md": str((OUT_DIR / "phase8_event_request_router_summary.md").relative_to(PROJECT_ROOT)),
            "best_context": str((OUT_DIR / "phase8_event_request_router_best_context.csv").relative_to(PROJECT_ROOT)),
        },
    }
    (OUT_DIR / "phase8_event_request_router_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] candidates -> {OUT_DIR / 'phase8_event_request_router_candidate_summary.csv'}")
    print(f"[OK] anchor table -> {OUT_DIR / 'phase8_event_request_router_anchor_table.csv'}")
    print(f"[OK] summary -> {OUT_DIR / 'phase8_event_request_router_summary.md'}")
    print(f"[OK] best={best_key}")


if __name__ == "__main__":
    main()
