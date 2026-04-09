import os

import pandas as pd

from phase_eval_utils import evaluate_context_csv, numeric_cols_for_rounding


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_6")
TIMING_LOG = os.path.join(PHASE_DIR, "phase5_6_timing_log.csv")

OUT_DETAIL = os.path.join(PHASE_DIR, "phase5_6_sweep_table.csv")
OUT_TOP = os.path.join(PHASE_DIR, "phase5_6_top_candidates.csv")


def load_eval_meta():
    if not os.path.exists(TIMING_LOG):
        return {}
    timing = pd.read_csv(TIMING_LOG)
    if timing.empty:
        return {}
    eval_rows = timing[timing["stage"] == "eval"].copy()
    if eval_rows.empty:
        return {}
    return eval_rows.set_index("eval_exp").to_dict("index")


def main():
    meta_map = load_eval_meta()
    rows = []
    search_dirs = [PHASE_DIR, os.path.join(PHASE_DIR, "phase5")]
    seen = set()
    for base_dir in search_dirs:
        if not os.path.isdir(base_dir):
            continue
        for name in sorted(os.listdir(base_dir)):
            if not name.startswith("eval_context_") or not name.endswith(".csv"):
                continue
            path = os.path.join(base_dir, name)
            if path in seen:
                continue
            seen.add(path)
            row = evaluate_context_csv(path)
            if row is None:
                continue
            meta = meta_map.get(row["exp_id"], {})
            row["feature_set"] = meta.get("feature_set", "")
            row["gate_mode"] = meta.get("gate_mode", "")
            row["qty_gate"] = meta.get("qty_gate", "")
            row["lr"] = meta.get("lr", "")
            row["num_leaves"] = meta.get("num_leaves", "")
            row["seed"] = meta.get("seed", "")
            row["elapsed_min"] = meta.get("elapsed_min", "")
            rows.append(row)

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise SystemExit("No phase5.6 eval_context files found.")

    detail_df = detail_df.sort_values(
        ["4_25_sku_p50", "4_25_wmape_like", "ice_4_25_sku_p50", "global_wmape"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    detail_df.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

    top_cols = [
        "exp_id",
        "feature_set",
        "gate_mode",
        "qty_gate",
        "lr",
        "num_leaves",
        "global_ratio",
        "global_wmape",
        "4_25_ratio",
        "4_25_wmape_like",
        "4_25_sku_p50",
        "ice_4_25_sku_p50",
        "blockbuster_sku_p50",
        "auc",
        "f1",
    ]
    top_df = detail_df[top_cols].head(10).copy()
    for col in numeric_cols_for_rounding(top_df):
        top_df[col] = top_df[col].map(lambda v: "" if pd.isna(v) else round(float(v), 4))
    top_df.to_csv(OUT_TOP, index=False, encoding="utf-8-sig")

    print(f"[OK] phase5.6 sweep table -> {OUT_DETAIL}")
    print(f"[OK] phase5.6 top candidates -> {OUT_TOP}")


if __name__ == "__main__":
    main()
