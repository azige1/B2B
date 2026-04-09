import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_2")
TIMING_LOG = os.path.join(PROJECT_ROOT, "reports", "phase5_2_timing_log.csv")
OUT_CSV = os.path.join(PHASE_DIR, "phase5_2_decision_table.csv")
OUT_MD = os.path.join(PHASE_DIR, "phase5_2_decision_table.md")
OUT_JSON = os.path.join(PHASE_DIR, "phase5_2_sequence_refs.json")

REFERENCE_OVERRIDES = {
    "v3_filtered": "p521_lstm_l3_v3_filtered_s2026",
    "v5_lite": "p527_lstm_l3_v5_lite_s2027",
}


def parse_exp_id(path):
    name = os.path.basename(path)
    return name[len("eval_context_") :].rsplit(".", 1)[0]


def infer_version(exp_id):
    for version in ("v3_filtered", "v5_lite", "v3", "v5"):
        if version in exp_id:
            return version
    return "unknown"


def infer_model(exp_id):
    if "bilstm" in exp_id:
        return "bilstm"
    if "attn" in exp_id:
        return "attn"
    if "gru" in exp_id:
        return "gru"
    if "lstm" in exp_id:
        return "lstm"
    return "unknown"


def infer_seed(exp_id):
    marker = "_s"
    if marker not in exp_id:
        return ""
    return exp_id.split(marker)[-1]


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


def compute_sku_slice(sku_df, mask):
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
        "version": infer_version(exp_id),
        "model": infer_model(exp_id),
        "seed": infer_seed(exp_id),
        "rows": int(len(df)),
        "auc": float(roc_auc_score(y_true, y_prob)) if y_true.nunique() > 1 else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "global_ratio": pred_sum / true_sum if true_sum > 0 else np.nan,
        "global_wmape": abs_sum / true_sum if true_sum > 0 else np.nan,
        "sku_p50": float(sku_df["sku_ratio"].median()) if not sku_df.empty else np.nan,
    }

    for prefix, metrics in (
        ("ice", compute_sku_slice(sku_df, ice_mask)),
        ("four_25", compute_sku_slice(sku_df, four_25_mask)),
        ("ice_4_25", compute_sku_slice(sku_df, ice_mask & four_25_mask)),
        ("blockbuster", compute_sku_slice(sku_df, blockbuster_mask)),
    ):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def load_timing_status():
    if not os.path.exists(TIMING_LOG):
        return {}
    timing = pd.read_csv(TIMING_LOG)
    if timing.empty:
        return {}
    latest = timing.sort_values("end_time").drop_duplicates("exp_id", keep="last")
    return latest.set_index("exp_id")[["status", "elapsed_min", "actual_epochs"]].to_dict("index")


def build_reference_payload(summary_df):
    refs = {}
    for version in ("v3_filtered", "v5_lite"):
        subset = summary_df[summary_df["version"] == version].copy()
        if subset.empty:
            continue

        preferred_exp = REFERENCE_OVERRIDES.get(version)
        preferred = subset[subset["exp_id"] == preferred_exp]
        if not preferred.empty:
            best = preferred.iloc[0].to_dict()
            selection_mode = "fixed_override"
        else:
            subset["ratio_dist"] = (subset["global_ratio"] - 1.0).abs()
            subset = subset.sort_values(
                [
                    "four_25_sku_p50",
                    "four_25_wmape_like",
                    "ice_4_25_sku_p50",
                    "ratio_dist",
                    "global_wmape",
                    "auc",
                    "f1",
                ],
                ascending=[False, True, False, True, True, False, False],
            )
            best = subset.iloc[0].to_dict()
            selection_mode = "business_priority"

        refs[version] = {
            "exp_id": best["exp_id"],
            "model": best["model"],
            "seed": str(best["seed"]),
            "selection_mode": selection_mode,
            "auc": float(best["auc"]),
            "f1": float(best["f1"]),
            "global_ratio": float(best["global_ratio"]),
            "global_wmape": float(best["global_wmape"]),
            "sku_p50": float(best["sku_p50"]),
            "ice_sku_p50": float(best["ice_sku_p50"]) if pd.notnull(best["ice_sku_p50"]) else None,
            "four_25_ratio": float(best["four_25_ratio"]) if pd.notnull(best["four_25_ratio"]) else None,
            "four_25_wmape_like": float(best["four_25_wmape_like"]) if pd.notnull(best["four_25_wmape_like"]) else None,
            "four_25_sku_p50": float(best["four_25_sku_p50"]) if pd.notnull(best["four_25_sku_p50"]) else None,
            "ice_4_25_sku_p50": float(best["ice_4_25_sku_p50"]) if pd.notnull(best["ice_4_25_sku_p50"]) else None,
            "blockbuster_sku_p50": float(best["blockbuster_sku_p50"]) if pd.notnull(best["blockbuster_sku_p50"]) else None,
        }
    return refs


def render_markdown(summary_df, refs):
    lines = ["# Phase 5.2 Decision Table", ""]
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
        "four_25_ratio",
        "four_25_wmape_like",
        "four_25_sku_p50",
        "ice_sku_p50",
        "ice_4_25_sku_p50",
        "blockbuster_sku_p50",
    ]
    for col in metric_cols:
        display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")

    cols = [
        "exp_id",
        "version",
        "model",
        "seed",
        "auc",
        "f1",
        "global_ratio",
        "global_wmape",
        "sku_p50",
        "four_25_ratio",
        "four_25_wmape_like",
        "four_25_sku_p50",
        "ice_sku_p50",
        "ice_4_25_sku_p50",
        "blockbuster_sku_p50",
        "status",
        "actual_epochs",
        "elapsed_min",
    ]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in display[cols].itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")

    lines.extend(
        [
            "",
            "## Frozen References",
            "",
            "```json",
            json.dumps(refs, ensure_ascii=False, indent=2),
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
        summary_df["actual_epochs"] = summary_df["exp_id"].map(lambda exp: timing_map.get(exp, {}).get("actual_epochs"))
        summary_df = summary_df.sort_values(["version", "model", "seed"]).reset_index(drop=True)
        summary_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    refs = build_reference_payload(summary_df) if not summary_df.empty else {}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(refs, fh, ensure_ascii=False, indent=2)

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(summary_df, refs))

    print(f"[OK] phase5.2 summary -> {OUT_CSV}")
    print(f"[OK] phase5.2 refs    -> {OUT_JSON}")


if __name__ == "__main__":
    main()
