import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_3")
TIMING_LOG = os.path.join(PROJECT_ROOT, "reports", "phase5_3_timing_log.csv")
PHASE52_REFS = os.path.join(PROJECT_ROOT, "reports", "phase5_2", "phase5_2_sequence_refs.json")

OUT_CSV = os.path.join(PHASE_DIR, "phase5_3_decision_table.csv")
OUT_MD = os.path.join(PHASE_DIR, "phase5_3_decision_table.md")
OUT_JSON = os.path.join(PHASE_DIR, "phase5_3_winners.json")

GLOBAL_RATIO_MIN = 0.90
GLOBAL_RATIO_MAX = 1.10


def parse_exp_id(path):
    name = os.path.basename(path)
    return name[len("eval_context_") :].rsplit(".", 1)[0]


def infer_track(exp_id):
    return "event_tree" if "_tree_" in exp_id else "sequence"


def infer_version(exp_id):
    if "_tree_" in exp_id:
        return "v6_event"
    if "v5_lite_cov" in exp_id:
        return "v5_lite_cov"
    if "v5_lite" in exp_id:
        return "v5_lite"
    if "v3_filtered" in exp_id:
        return "v3_filtered"
    return "unknown"


def infer_config(exp_id):
    if "_tree_" in exp_id:
        parts = exp_id.split("_")
        if len(parts) >= 4:
            return "_".join(parts[2:])
    if "lstm_pool" in exp_id:
        return "lstm_pool"
    if "attn" in exp_id:
        return "attn"
    if "bilstm" in exp_id:
        return "bilstm"
    if "lstm" in exp_id:
        return "lstm"
    return "unknown"


def load_phase52_baseline():
    if not os.path.exists(PHASE52_REFS):
        raise FileNotFoundError(f"Missing frozen refs: {PHASE52_REFS}")
    with open(PHASE52_REFS, "r", encoding="utf-8") as fh:
        refs = json.load(fh)
    baseline = refs.get("v5_lite")
    if not baseline:
        raise ValueError("Frozen refs missing v5_lite baseline.")
    return baseline


def load_timing_status():
    if not os.path.exists(TIMING_LOG):
        return {}
    timing = pd.read_csv(TIMING_LOG)
    if timing.empty:
        return {}
    latest = timing.sort_values("end_time").drop_duplicates("exp_id", keep="last")
    return latest.set_index("exp_id")[["status", "elapsed_min"]].to_dict("index")


def aggregate_sku(df):
    agg = (
        df.groupby("sku_id", as_index=False)
        .agg(
            total_true=("true_replenish_qty", "sum"),
            total_pred=("ai_pred_qty", "sum"),
            lookback_repl_days_90=("lookback_repl_days_90", "max"),
        )
    )
    agg = agg[agg["total_true"] > 0].copy()
    if agg.empty:
        agg["sku_ratio"] = np.nan
        return agg
    agg["sku_ratio"] = agg["total_pred"] / agg["total_true"]
    return agg


def compute_slice_metrics(sku_df, mask):
    sub = sku_df.loc[mask].copy()
    if sub.empty:
        return {
            "rows": 0,
            "ratio": np.nan,
            "wmape_like": np.nan,
            "sku_p50": np.nan,
        }
    true_sum = float(sub["total_true"].sum())
    pred_sum = float(sub["total_pred"].sum())
    abs_sum = float(np.abs(sub["total_pred"] - sub["total_true"]).sum())
    return {
        "rows": int(len(sub)),
        "ratio": pred_sum / true_sum if true_sum > 0 else np.nan,
        "wmape_like": abs_sum / true_sum if true_sum > 0 else np.nan,
        "sku_p50": float(sub["sku_ratio"].median()),
    }


def evaluate_file(path):
    exp_id = parse_exp_id(path)
    df = pd.read_csv(path)
    if df.empty:
        return None

    y_true = (df["true_replenish_qty"] > 0).astype(int)
    y_prob = df["ai_pred_prob"].astype(float)
    y_pred = df["cls_pred_best_f1"].astype(int)

    true_sum = float(df["true_replenish_qty"].sum())
    pred_sum = float(df["ai_pred_qty"].sum())
    abs_sum = float(np.abs(df["ai_pred_qty"] - df["true_replenish_qty"]).sum())

    sku_df = aggregate_sku(df)
    ice_mask = sku_df["lookback_repl_days_90"] == 0
    four_25_mask = (sku_df["total_true"] >= 4) & (sku_df["total_true"] <= 25)
    blockbuster_mask = sku_df["total_true"] > 25

    row = {
        "exp_id": exp_id,
        "track": infer_track(exp_id),
        "version": infer_version(exp_id),
        "config": infer_config(exp_id),
        "rows": int(len(df)),
        "auc": float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "global_ratio": pred_sum / true_sum if true_sum > 0 else np.nan,
        "global_wmape": abs_sum / true_sum if true_sum > 0 else np.nan,
        "sku_p50": float(sku_df["sku_ratio"].median()) if not sku_df.empty else np.nan,
    }

    for prefix, metrics in (
        ("ice", compute_slice_metrics(sku_df, ice_mask)),
        ("4_25", compute_slice_metrics(sku_df, four_25_mask)),
        ("ice_4_25", compute_slice_metrics(sku_df, ice_mask & four_25_mask)),
        ("blockbuster", compute_slice_metrics(sku_df, blockbuster_mask)),
    ):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def determine_winners(summary_df, baseline):
    if summary_df.empty:
        return {
            "baseline_sequence": baseline["exp_id"],
            "main_line": None,
            "event_tree_candidates": [],
            "sequence_keep": [baseline["exp_id"]],
            "promote_phase54": [baseline["exp_id"]],
            "decision": "pending",
        }

    baseline_exp = baseline["exp_id"]
    ref_4_25_p50 = float(baseline["four_25_sku_p50"])
    ref_4_25_wmape = float(baseline["four_25_wmape_like"])
    ref_ice_4_25_p50 = float(baseline["ice_4_25_sku_p50"])

    summary_df = summary_df.copy()
    summary_df["beats_ref_4_25_p50"] = summary_df["4_25_sku_p50"] > ref_4_25_p50
    summary_df["beats_ref_4_25_wmape"] = summary_df["4_25_wmape_like"] <= ref_4_25_wmape
    summary_df["beats_ref_ice_4_25_p50"] = summary_df["ice_4_25_sku_p50"] > ref_ice_4_25_p50
    summary_df["within_global_ratio"] = summary_df["global_ratio"].between(GLOBAL_RATIO_MIN, GLOBAL_RATIO_MAX)
    summary_df["eligible_vs_p527"] = (
        summary_df["beats_ref_4_25_p50"]
        & summary_df["beats_ref_4_25_wmape"]
        & summary_df["beats_ref_ice_4_25_p50"]
        & summary_df["within_global_ratio"]
    )

    def rank_frame(frame):
        ranked = frame.copy()
        ranked["ratio_dist"] = (ranked["global_ratio"] - 1.0).abs()
        ranked = ranked.sort_values(
            [
                "4_25_sku_p50",
                "4_25_wmape_like",
                "ice_4_25_sku_p50",
                "ratio_dist",
                "global_wmape",
                "blockbuster_sku_p50",
                "auc",
                "f1",
            ],
            ascending=[False, True, False, True, True, False, False, False],
        )
        return ranked.drop(columns=["ratio_dist"])

    event_tree_eligible = rank_frame(
        summary_df[(summary_df["track"] == "event_tree") & (summary_df["eligible_vs_p527"])]
    )
    sequence_eligible = rank_frame(
        summary_df[(summary_df["track"] == "sequence") & (summary_df["eligible_vs_p527"])]
    )

    main_line = None
    event_tree_candidates = []
    if not event_tree_eligible.empty:
        main_line = str(event_tree_eligible.iloc[0]["exp_id"])
        event_tree_candidates.append(main_line)
        if len(event_tree_eligible) > 1:
            runner_up = event_tree_eligible.iloc[1]
            leader = event_tree_eligible.iloc[0]
            if (
                float(runner_up["ice_4_25_sku_p50"]) > float(leader["ice_4_25_sku_p50"])
                or float(runner_up["4_25_sku_p50"]) > float(leader["4_25_sku_p50"])
            ):
                event_tree_candidates.append(str(runner_up["exp_id"]))

    sequence_keep = [baseline_exp]
    if not sequence_eligible.empty:
        top_sequence = str(sequence_eligible.iloc[0]["exp_id"])
        if top_sequence != baseline_exp:
            sequence_keep.append(top_sequence)

    promote = []
    if main_line:
        promote.extend(event_tree_candidates)
        promote.extend(sequence_keep)
        decision = "shift_to_event_tree"
    else:
        promote.extend(sequence_keep)
        decision = "keep_sequence_mainline"

    promote = list(dict.fromkeys(promote))

    return {
        "baseline_sequence": baseline_exp,
        "main_line": main_line,
        "event_tree_candidates": event_tree_candidates,
        "sequence_keep": sequence_keep,
        "promote_phase54": promote,
        "decision": decision,
    }


def render_markdown(summary_df, winners):
    lines = [
        "# Phase 5.3 Decision Table",
        "",
        "Business priority:",
        "1. `4-25`",
        "2. `Ice / Ice 4-25`",
        "3. `global_ratio / global_wmape`",
        "4. `auc / f1`",
        "",
    ]
    if summary_df.empty:
        lines.append("No successful `eval_context_*` outputs found yet.")
        return "\n".join(lines)

    display = summary_df.copy()
    metric_cols = [
        "auc",
        "f1",
        "global_ratio",
        "global_wmape",
        "sku_p50",
        "ice_ratio",
        "ice_wmape_like",
        "ice_sku_p50",
        "4_25_ratio",
        "4_25_wmape_like",
        "4_25_sku_p50",
        "ice_4_25_ratio",
        "ice_4_25_wmape_like",
        "ice_4_25_sku_p50",
        "blockbuster_ratio",
        "blockbuster_wmape_like",
        "blockbuster_sku_p50",
    ]
    for col in metric_cols:
        display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")

    cols = [
        "exp_id",
        "track",
        "version",
        "config",
        "auc",
        "f1",
        "global_ratio",
        "global_wmape",
        "4_25_ratio",
        "4_25_wmape_like",
        "4_25_sku_p50",
        "ice_sku_p50",
        "ice_4_25_sku_p50",
        "blockbuster_sku_p50",
        "eligible_vs_p527",
        "status",
        "elapsed_min",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in display[cols].itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")

    lines.extend(
        [
            "",
            "## Winners",
            "",
            "```json",
            json.dumps(winners, ensure_ascii=False, indent=2),
            "```",
        ]
    )
    return "\n".join(lines)


def main():
    os.makedirs(PHASE_DIR, exist_ok=True)
    files = sorted(
        os.path.join(PHASE_DIR, name)
        for name in os.listdir(PHASE_DIR)
        if name.startswith("eval_context_") and name.endswith(".csv")
    )

    rows = [evaluate_file(path) for path in files]
    rows = [row for row in rows if row is not None]
    summary_df = pd.DataFrame(rows)

    timing_map = load_timing_status()
    if not summary_df.empty:
        summary_df["status"] = summary_df["exp_id"].map(lambda exp: timing_map.get(exp, {}).get("status", "success"))
        summary_df["elapsed_min"] = summary_df["exp_id"].map(lambda exp: timing_map.get(exp, {}).get("elapsed_min"))
    else:
        summary_df["status"] = []
        summary_df["elapsed_min"] = []

    baseline = load_phase52_baseline()
    winners = determine_winners(summary_df, baseline)

    if not summary_df.empty:
        ref_4_25_p50 = float(baseline["four_25_sku_p50"])
        ref_4_25_wmape = float(baseline["four_25_wmape_like"])
        ref_ice_4_25_p50 = float(baseline["ice_4_25_sku_p50"])

        summary_df["beats_ref_4_25_p50"] = summary_df["4_25_sku_p50"] > ref_4_25_p50
        summary_df["beats_ref_4_25_wmape"] = summary_df["4_25_wmape_like"] <= ref_4_25_wmape
        summary_df["beats_ref_ice_4_25_p50"] = summary_df["ice_4_25_sku_p50"] > ref_ice_4_25_p50
        summary_df["within_global_ratio"] = summary_df["global_ratio"].between(GLOBAL_RATIO_MIN, GLOBAL_RATIO_MAX)
        summary_df["eligible_vs_p527"] = (
            summary_df["beats_ref_4_25_p50"]
            & summary_df["beats_ref_4_25_wmape"]
            & summary_df["beats_ref_ice_4_25_p50"]
            & summary_df["within_global_ratio"]
        )
        summary_df = summary_df.sort_values(["track", "config", "exp_id"]).reset_index(drop=True)
        summary_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(winners, fh, ensure_ascii=False, indent=2)

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(summary_df, winners))

    print(f"[OK] phase5.3 summary -> {OUT_CSV}")
    print(f"[OK] phase5.3 winners -> {OUT_JSON}")


if __name__ == "__main__":
    main()
