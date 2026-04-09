import json
import os

import pandas as pd

from phase_eval_utils import evaluate_context_csv, numeric_cols_for_rounding


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_5")
TIMING_LOG = os.path.join(PHASE_DIR, "phase5_5_timing_log.csv")
PHASE53_WINNERS = os.path.join(PROJECT_ROOT, "reports", "phase5_3", "phase5_3_winners.json")
PHASE54_PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_4", "phase5")

OUT_DETAIL = os.path.join(PHASE_DIR, "phase5_5_decision_table.csv")
OUT_MEAN = os.path.join(PHASE_DIR, "phase5_5_anchor_mean_summary.csv")
OUT_MD = os.path.join(PHASE_DIR, "phase5_5_decision_table.md")
OUT_JSON = os.path.join(PHASE_DIR, "phase5_5_winners.json")

GLOBAL_RATIO_MIN = 0.90
GLOBAL_RATIO_MAX = 1.10


def parse_exp_id_parts(exp_id):
    parts = exp_id.split("_")
    anchor_tag = parts[1]
    base_plus_seed = exp_id[len(f"p55_{anchor_tag}_") :]
    base_exp, seed_part = base_plus_seed.rsplit("_s", 1)
    anchor_date = f"{anchor_tag[:4]}-{anchor_tag[4:6]}-{anchor_tag[6:]}"
    return anchor_date, base_exp, int(seed_part)


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


def load_baseline():
    with open(PHASE53_WINNERS, "r", encoding="utf-8") as fh:
        winners = json.load(fh)
    return winners["baseline_sequence"], winners


def inject_phase54_baseline(detail_df, baseline_exp):
    expected_anchor = "2025-12-01"
    has_anchor_baseline = (
        (detail_df["anchor_date"] == expected_anchor)
        & (detail_df["base_exp"] == baseline_exp)
    ).any()
    if has_anchor_baseline:
        return detail_df

    fallback_name = f"eval_context_p54_{baseline_exp}_s2026.csv"
    fallback_path = os.path.join(PHASE54_PHASE_DIR, fallback_name)
    if not os.path.exists(fallback_path):
        return detail_df

    row = evaluate_context_csv(fallback_path)
    if row is None:
        return detail_df

    row["anchor_date"] = expected_anchor
    row["base_exp"] = baseline_exp
    row["seed"] = 2026
    row["track"] = infer_track(baseline_exp)
    row["status"] = "fallback_phase54"
    row["elapsed_min"] = pd.NA
    row["eligible_vs_p527"] = False
    return pd.concat([detail_df, pd.DataFrame([row])], ignore_index=True)


def render_markdown(df, winners):
    lines = [
        "# Phase 5.5 Decision Table",
        "",
        "Business priority:",
        "1. `4-25`",
        "2. `Ice / Ice 4-25`",
        "3. `global_ratio / global_wmape`",
        "4. `AUC / F1`",
        "",
        f"- baseline_sequence: `{winners['baseline_sequence']}`",
        f"- main_line: `{winners.get('main_line')}`",
        f"- anchor_wins: `{winners.get('anchor_wins', {})}`",
        "",
    ]
    cols = [
        "anchor_date",
        "exp_id",
        "track",
        "base_exp",
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
    display = df.copy()
    for col in numeric_cols_for_rounding(display):
        display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for row in display[cols].itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def main():
    rows = []
    for anchor_tag in sorted(os.listdir(PHASE_DIR)):
        anchor_dir = os.path.join(PHASE_DIR, anchor_tag, "phase5")
        if not os.path.isdir(anchor_dir):
            continue
        for name in sorted(os.listdir(anchor_dir)):
            if not name.startswith("eval_context_") or not name.endswith(".csv"):
                continue
            row = evaluate_context_csv(os.path.join(anchor_dir, name))
            if row is None:
                continue
            anchor_date, base_exp, seed = parse_exp_id_parts(row["exp_id"])
            row["anchor_date"] = anchor_date
            row["base_exp"] = base_exp
            row["seed"] = seed
            row["track"] = infer_track(base_exp)
            rows.append(row)

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise SystemExit("No phase5.5 eval_context files found.")

    timing = load_timing()
    detail_df["status"] = detail_df["exp_id"].map(lambda x: timing.get(x, {}).get("status", "unknown"))
    detail_df["elapsed_min"] = detail_df["exp_id"].map(lambda x: timing.get(x, {}).get("elapsed_min"))

    baseline_exp, phase53_winners = load_baseline()
    detail_df = inject_phase54_baseline(detail_df, baseline_exp)
    detail_df["eligible_vs_p527"] = False
    anchor_wins = {}
    anchor_records = []

    for anchor_date, anchor_df in detail_df.groupby("anchor_date", sort=True):
        baseline_rows = anchor_df[anchor_df["base_exp"] == baseline_exp]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        ref_4_25_p50 = float(baseline["4_25_sku_p50"])
        ref_4_25_wmape = float(baseline["4_25_wmape_like"])
        ref_ice_4_25_p50 = float(baseline["ice_4_25_sku_p50"])

        idx = anchor_df.index
        eligible = (
            (anchor_df["4_25_sku_p50"] > ref_4_25_p50)
            & (anchor_df["4_25_wmape_like"] <= ref_4_25_wmape)
            & (anchor_df["ice_4_25_sku_p50"] > ref_ice_4_25_p50)
            & anchor_df["global_ratio"].between(GLOBAL_RATIO_MIN, GLOBAL_RATIO_MAX)
        )
        detail_df.loc[idx, "eligible_vs_p527"] = eligible.values

        event_eligible = anchor_df[(anchor_df["track"] == "event_tree") & eligible]
        if not event_eligible.empty:
            ranked = event_eligible.assign(
                ratio_dist=(event_eligible["global_ratio"] - 1.0).abs()
            ).sort_values(
                ["4_25_sku_p50", "4_25_wmape_like", "ice_4_25_sku_p50", "ratio_dist", "global_wmape"],
                ascending=[False, True, False, True, True],
            )
            winner = ranked.iloc[0]
            anchor_wins[str(winner["base_exp"])] = anchor_wins.get(str(winner["base_exp"]), 0) + 1
            anchor_records.append(
                {
                    "anchor_date": anchor_date,
                    "winner_base_exp": str(winner["base_exp"]),
                    "winner_exp_id": str(winner["exp_id"]),
                }
            )

    detail_df = detail_df.sort_values(["anchor_date", "track", "base_exp"]).reset_index(drop=True)
    detail_df.to_csv(OUT_DETAIL, index=False, encoding="utf-8-sig")

    metric_cols = [
        col
        for col in detail_df.columns
        if col not in {"exp_id", "anchor_date", "base_exp", "seed", "track", "status", "eligible_vs_p527"}
    ]
    mean_df = (
        detail_df.groupby(["track", "base_exp"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["4_25_sku_p50", "4_25_wmape_like", "ice_4_25_sku_p50", "global_wmape"], ascending=[False, True, False, True])
        .reset_index(drop=True)
    )
    rounded = mean_df.copy()
    for col in numeric_cols_for_rounding(rounded):
        rounded[col] = rounded[col].map(lambda v: "" if pd.isna(v) else round(float(v), 4))
    rounded.to_csv(OUT_MEAN, index=False, encoding="utf-8-sig")

    best_event = None
    if anchor_wins:
        win_df = pd.DataFrame(
            [{"base_exp": key, "anchor_wins": value} for key, value in anchor_wins.items()]
        ).sort_values(["anchor_wins", "base_exp"], ascending=[False, True])
        top_wins = int(win_df.iloc[0]["anchor_wins"])
        tied = win_df[win_df["anchor_wins"] == top_wins]["base_exp"].tolist()
        if len(tied) == 1:
            best_event = tied[0]
        else:
            tie_break = mean_df[mean_df["base_exp"].isin(tied)].sort_values(
                ["4_25_sku_p50", "4_25_wmape_like", "ice_4_25_sku_p50", "global_wmape"],
                ascending=[False, True, False, True],
            )
            best_event = None if tie_break.empty else str(tie_break.iloc[0]["base_exp"])

    winners = {
        "baseline_sequence": baseline_exp,
        "main_line": best_event,
        "anchor_wins": anchor_wins,
        "anchor_records": anchor_records,
        "event_tree_candidates": phase53_winners.get("event_tree_candidates", []),
        "decision": "shift_to_event_tree" if best_event and anchor_wins.get(best_event, 0) >= 3 else "needs_more_validation",
    }

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(detail_df, winners))
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(winners, fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase5.5 decision table -> {OUT_DETAIL}")
    print(f"[OK] phase5.5 mean summary -> {OUT_MEAN}")
    print(f"[OK] phase5.5 markdown -> {OUT_MD}")
    print(f"[OK] phase5.5 winners -> {OUT_JSON}")


if __name__ == "__main__":
    main()
