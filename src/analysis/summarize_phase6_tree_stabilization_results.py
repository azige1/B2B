import json
import os

import pandas as pd

from phase_eval_utils import numeric_cols_for_rounding


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE57_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_7")
PHASE5_TREE_CAL_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_tree_anchor_calibration")
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase6_tree_stabilization")

SOURCE_TABLE = os.path.join(PHASE5_TREE_CAL_DIR, "phase5_tree_anchor_calibration_table.csv")
OUT_TABLE = os.path.join(OUT_DIR, "phase6_tree_stabilization_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase6_tree_stabilization_summary.md")
OUT_WINNER = os.path.join(OUT_DIR, "phase6_tree_stabilization_winner.json")

KEEP_CANDIDATES = [
    "sep098_oct093",
    "sep095_oct090",
    "base_g025_ld080_q3",
]
REFERENCE_CANDIDATES = ["p527", "p535", "p531"]


def ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def build_reference_table():
    path = os.path.join(PHASE57_DIR, "phase5_7_anchor_table.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df[df["candidate_key"].isin(REFERENCE_CANDIDATES)].copy()


def summarize(df):
    metric_cols = [
        "global_ratio",
        "global_wmape",
        "4_25_ratio",
        "4_25_wmape_like",
        "4_25_sku_p50",
        "4_25_under_wape",
        "ice_sku_p50",
        "ice_4_25_sku_p50",
        "1_3_ratio",
        "1_3_over_wape",
        "blockbuster_ratio",
        "blockbuster_wmape_like",
        "blockbuster_sku_p50",
        "blockbuster_under_wape",
        "blockbuster_within_50pct_rate",
        "top20_true_volume_capture",
        "rank_corr_positive_skus",
        "auc",
        "f1",
    ]
    summary = (
        df.groupby(["candidate_key", "track"], as_index=False)
        .agg(**{col: (col, "mean") for col in metric_cols}, anchor_passes=("anchor_pass", "sum"))
    )
    return summary


def pick_winner(summary_df):
    candidates = summary_df[summary_df["track"] == "event_tree_anchor_calibration"].copy()
    if candidates.empty:
        raise ValueError("No phase6 candidates found")
    candidates["strict_rnd_ready"] = (
        (candidates["anchor_passes"] >= 4)
        & (candidates["4_25_sku_p50"] >= 0.55)
        & (candidates["4_25_wmape_like"] <= 0.55)
        & (candidates["ice_4_25_sku_p50"] >= 0.35)
        & (candidates["1_3_ratio"] <= 1.35)
    )
    candidates["relaxed_rnd_ready"] = (
        (candidates["anchor_passes"] >= 4)
        & (candidates["4_25_sku_p50"] >= 0.55)
        & (candidates["4_25_wmape_like"] <= 0.55)
        & (candidates["ice_4_25_sku_p50"] >= 0.35)
        & (candidates["1_3_ratio"] <= 1.40)
    )
    return candidates.sort_values(
        [
            "anchor_passes",
            "4_25_sku_p50",
            "global_wmape",
            "1_3_ratio",
            "blockbuster_sku_p50",
            "blockbuster_under_wape",
            "top20_true_volume_capture",
            "rank_corr_positive_skus",
        ],
        ascending=[False, False, True, True, False, True, False, False],
    ).iloc[0]


def render_summary(summary_df, winner):
    disp = summary_df.copy()
    for col in numeric_cols_for_rounding(disp):
        disp[col] = disp[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")

    lines = [
        "# Phase6 Tree Stabilization",
        "",
        f"- selected_candidate: `{winner['candidate_key']}`",
        f"- strict_rnd_ready: `{bool(winner['strict_rnd_ready'])}`",
        f"- relaxed_rnd_ready: `{bool(winner['relaxed_rnd_ready'])}`",
        "",
        "| candidate_key | track | anchor_passes | global_ratio | global_wmape | 4_25_sku_p50 | 4_25_under_wape | ice_4_25_sku_p50 | 1_3_ratio | blockbuster_sku_p50 | blockbuster_under_wape | top20_true_volume_capture | rank_corr_positive_skus |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in disp.sort_values(
        ["track", "anchor_passes", "4_25_sku_p50", "global_wmape"],
        ascending=[True, False, False, True],
    ).iterrows():
        anchor_passes = row["anchor_passes"] if row["track"] == "event_tree_anchor_calibration" else ""
        lines.append(
            f"| {row['candidate_key']} | {row['track']} | {anchor_passes} | {row['global_ratio']} | {row['global_wmape']} | "
            f"{row['4_25_sku_p50']} | {row['4_25_under_wape']} | {row['ice_4_25_sku_p50']} | {row['1_3_ratio']} | "
            f"{row['blockbuster_sku_p50']} | {row['blockbuster_under_wape']} | {row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} |"
        )
    return "\n".join(lines)


def main():
    ensure_dir()
    if not os.path.exists(SOURCE_TABLE):
        raise FileNotFoundError(SOURCE_TABLE)

    candidate_df = pd.read_csv(SOURCE_TABLE)
    candidate_df = candidate_df[candidate_df["candidate_key"].isin(KEEP_CANDIDATES)].copy()
    reference_df = build_reference_table()
    all_df = pd.concat([candidate_df, reference_df], ignore_index=True, sort=False)
    all_df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    summary_df = summarize(all_df)
    winner = pick_winner(summary_df)

    with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(render_summary(summary_df, winner))

    payload = {
        "selected_candidate": str(winner["candidate_key"]),
        "strict_rnd_ready": bool(winner["strict_rnd_ready"]),
        "relaxed_rnd_ready": bool(winner["relaxed_rnd_ready"]),
        "next_stage": "phase6c_tree_monthaware_refit" if not bool(winner["strict_rnd_ready"]) else "phase6d_tree_family_compare",
        "candidate_summary": summary_df.sort_values(["track", "candidate_key"]).to_dict(orient="records"),
    }
    with open(OUT_WINNER, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase6 tree stabilization table -> {OUT_TABLE}")
    print(f"[OK] phase6 tree stabilization summary -> {OUT_SUMMARY}")
    print(f"[OK] phase6 tree stabilization winner -> {OUT_WINNER}")


if __name__ == "__main__":
    main()
