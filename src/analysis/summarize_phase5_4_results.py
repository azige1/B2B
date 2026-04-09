import json
import os

import pandas as pd

from phase_eval_utils import evaluate_context_csv, numeric_cols_for_rounding


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_4")
CONTEXT_DIR = os.path.join(PHASE_DIR, "phase5")
TIMING_LOG = os.path.join(PHASE_DIR, "phase5_4_timing_log.csv")

OUT_SEED = os.path.join(PHASE_DIR, "phase5_4_seed_table.csv")
OUT_BASE = os.path.join(PHASE_DIR, "phase5_4_base_summary.csv")
OUT_JSON = os.path.join(PHASE_DIR, "phase5_4_summary.json")


def parse_base_exp(exp_id):
    if exp_id.startswith("p54_"):
        exp_id = exp_id[len("p54_") :]
    if "_s" not in exp_id:
        return exp_id
    base, _ = exp_id.rsplit("_s", 1)
    return base


def parse_seed(exp_id):
    if "_s" not in exp_id:
        return None
    return int(exp_id.rsplit("_s", 1)[1])


def infer_track(base_exp):
    return "event_tree" if "_tree_" in base_exp else "sequence"


def load_timing():
    if not os.path.exists(TIMING_LOG):
        return {}
    timing = pd.read_csv(TIMING_LOG)
    if timing.empty:
        return {}
    latest = timing.sort_values("end_time").drop_duplicates("exp_id", keep="last")
    return latest.set_index("exp_id")[["status", "elapsed_min"]].to_dict("index")


def main():
    rows = []
    for name in sorted(os.listdir(CONTEXT_DIR)):
        if not name.startswith("eval_context_") or not name.endswith(".csv"):
            continue
        path = os.path.join(CONTEXT_DIR, name)
        row = evaluate_context_csv(path)
        if row is None:
            continue
        row["base_exp"] = parse_base_exp(row["exp_id"])
        row["seed"] = parse_seed(row["exp_id"])
        row["track"] = infer_track(row["base_exp"])
        rows.append(row)

    seed_df = pd.DataFrame(rows)
    if seed_df.empty:
        raise SystemExit("No phase5.4 eval_context files found.")

    timing = load_timing()
    seed_df["status"] = seed_df["exp_id"].map(lambda x: timing.get(x, {}).get("status", "unknown"))
    seed_df["elapsed_min"] = seed_df["exp_id"].map(lambda x: timing.get(x, {}).get("elapsed_min"))

    seed_df = seed_df.sort_values(["track", "base_exp", "seed"]).reset_index(drop=True)
    seed_df.to_csv(OUT_SEED, index=False, encoding="utf-8-sig")

    metric_cols = [
        col
        for col in seed_df.columns
        if col not in {"exp_id", "base_exp", "seed", "track", "status"}
    ]
    base_df = (
        seed_df.groupby(["track", "base_exp"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    base_df.columns = [
        "_".join([str(part) for part in col if str(part) != ""]).rstrip("_")
        for col in base_df.columns.to_flat_index()
    ]
    base_df = base_df.sort_values(
        ["4_25_sku_p50_mean", "4_25_wmape_like_mean", "ice_4_25_sku_p50_mean", "global_wmape_mean"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    rounded = base_df.copy()
    for col in numeric_cols_for_rounding(rounded):
        rounded[col] = rounded[col].map(lambda v: "" if pd.isna(v) else round(float(v), 4))
    rounded.to_csv(OUT_BASE, index=False, encoding="utf-8-sig")

    summary = {
        "top_event_tree": None,
        "top_sequence": None,
    }
    event_df = base_df[base_df["track"] == "event_tree"]
    seq_df = base_df[base_df["track"] == "sequence"]
    if not event_df.empty:
        summary["top_event_tree"] = event_df.iloc[0]["base_exp"]
    if not seq_df.empty:
        summary["top_sequence"] = seq_df.iloc[0]["base_exp"]

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase5.4 seed table -> {OUT_SEED}")
    print(f"[OK] phase5.4 base summary -> {OUT_BASE}")
    print(f"[OK] phase5.4 summary json -> {OUT_JSON}")


if __name__ == "__main__":
    main()
