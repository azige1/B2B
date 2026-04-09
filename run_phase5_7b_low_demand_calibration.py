import json
import os

import pandas as pd

from src.analysis.phase_eval_utils import evaluate_context_frame


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PHASE57_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_7")
WINNER_JSON = os.path.join(PHASE57_DIR, "phase5_7_winner.json")
ANCHOR_TABLE = os.path.join(PHASE57_DIR, "phase5_7_anchor_table.csv")

OUT_TABLE = os.path.join(PHASE57_DIR, "phase5_7b_calibration_table.csv")
OUT_SUMMARY = os.path.join(PHASE57_DIR, "phase5_7b_summary.md")
OUT_WINNER = os.path.join(PHASE57_DIR, "phase5_7b_winner.json")

RULES = {
    "ld085_q3": {"factor": 0.85, "max_qty": 3.0},
    "ld080_q3": {"factor": 0.80, "max_qty": 3.0},
    "ld085_q4": {"factor": 0.85, "max_qty": 4.0},
}


def context_path(anchor_date, exp_id):
    return os.path.join(
        PHASE57_DIR,
        anchor_date.replace("-", ""),
        "phase5",
        f"eval_context_{exp_id}.csv",
    )


def apply_rule(df, rule_name, rule_cfg):
    adj = df.copy()
    mask = adj["ai_pred_qty"] <= rule_cfg["max_qty"]
    adj.loc[mask, "ai_pred_qty"] = adj.loc[mask, "ai_pred_qty"] * rule_cfg["factor"]
    adj["ai_pred_positive_qty"] = (adj["ai_pred_qty"] > 0).astype(int)
    adj["abs_error"] = (adj["ai_pred_qty"] - adj["true_replenish_qty"]).abs()
    return adj


def main():
    if not (os.path.exists(WINNER_JSON) and os.path.exists(ANCHOR_TABLE)):
        print("[BLOCKED] phase5.7 outputs missing")
        raise SystemExit(1)

    with open(WINNER_JSON, "r", encoding="utf-8") as fh:
        winner = json.load(fh)
    if not winner.get("needs_phase57b", True) and os.environ.get("PHASE57B_FORCE", "0") != "1":
        print("[SKIP] phase5.7B not needed")
        raise SystemExit(0)

    candidate_key = winner["recommended_candidate_key"]
    exp_map = winner["candidate_anchor_exp_ids"][candidate_key]
    anchor_df = pd.read_csv(ANCHOR_TABLE)
    baseline = anchor_df[anchor_df["candidate_key"] == "p527"].copy()

    rows = []
    for rule_name, rule_cfg in RULES.items():
        anchor_rows = []
        for anchor_date, exp_id in exp_map.items():
            path = context_path(anchor_date, exp_id)
            if not os.path.exists(path):
                continue
            raw = pd.read_csv(path)
            adj = apply_rule(raw, rule_name, rule_cfg)
            row = evaluate_context_frame(adj, f"{exp_id}_{rule_name}")
            row["anchor_date"] = anchor_date
            row["candidate_key"] = candidate_key
            row["rule_name"] = rule_name
            anchor_rows.append(row)

        if not anchor_rows:
            continue

        rule_df = pd.DataFrame(anchor_rows)
        merged = rule_df.merge(
            baseline[["anchor_date", "4_25_sku_p50", "4_25_wmape_like", "ice_4_25_sku_p50"]],
            on="anchor_date",
            how="left",
            suffixes=("_cand", "_base"),
        )
        merged["meets_anchor_gate"] = (
            (merged["4_25_sku_p50_cand"] > merged["4_25_sku_p50_base"])
            & (merged["4_25_wmape_like_cand"] <= merged["4_25_wmape_like_base"])
            & (merged["ice_4_25_sku_p50_cand"] > merged["ice_4_25_sku_p50_base"])
            & merged["global_ratio"].between(0.90, 1.10)
        )
        rows.append(
            {
                "rule_name": rule_name,
                "anchor_passes": int(merged["meets_anchor_gate"].sum()),
                "global_ratio": float(rule_df["global_ratio"].mean()),
                "global_wmape": float(rule_df["global_wmape"].mean()),
                "4_25_sku_p50": float(rule_df["4_25_sku_p50"].mean()),
                "4_25_wmape_like": float(rule_df["4_25_wmape_like"].mean()),
                "ice_4_25_sku_p50": float(rule_df["ice_4_25_sku_p50"].mean()),
                "1_3_ratio": float(rule_df["1_3_ratio"].mean()),
                "blockbuster_sku_p50": float(rule_df["blockbuster_sku_p50"].mean()),
                "auc": float(rule_df["auc"].mean()),
                "f1": float(rule_df["f1"].mean()),
                "delivery_ready": bool(
                    (int(merged["meets_anchor_gate"].sum()) >= 3)
                    and (float(rule_df["4_25_sku_p50"].mean()) >= 0.55)
                    and (float(rule_df["ice_4_25_sku_p50"].mean()) >= 0.35)
                    and (float(rule_df["1_3_ratio"].mean()) <= 1.35)
                ),
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        raise SystemExit("No phase5.7B calibration rows generated.")

    eligible = result_df[
        (result_df["4_25_sku_p50"] >= 0.55)
        & (result_df["ice_4_25_sku_p50"] >= 0.35)
    ].copy()
    if eligible.empty:
        best = result_df.sort_values(
            ["anchor_passes", "1_3_ratio", "global_wmape"],
            ascending=[False, True, True],
        ).iloc[0]
    else:
        best = eligible.sort_values(
            ["delivery_ready", "anchor_passes", "1_3_ratio", "global_wmape"],
            ascending=[False, False, True, True],
        ).iloc[0]

    result_df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")
    with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write("# Phase 5.7B Low-Demand Calibration\n\n")
        fh.write(result_df.to_csv(index=False))
    with open(OUT_WINNER, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "base_candidate_key": candidate_key,
                "selected_rule": str(best["rule_name"]),
                "delivery_ready": bool(best["delivery_ready"]),
                "summary": result_df.to_dict(orient="records"),
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] phase5.7B calibration table -> {OUT_TABLE}")
    print(f"[OK] phase5.7B summary -> {OUT_SUMMARY}")
    print(f"[OK] phase5.7B winner -> {OUT_WINNER}")


if __name__ == "__main__":
    main()
