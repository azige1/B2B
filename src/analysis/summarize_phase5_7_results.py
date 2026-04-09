import json
import os

import numpy as np
import pandas as pd

from phase_eval_utils import evaluate_context_csv, numeric_cols_for_rounding


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE55_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_5")
PHASE54_PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_4", "phase5")
PHASE57_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_7")

OUT_TABLE = os.path.join(PHASE57_DIR, "phase5_7_anchor_table.csv")
OUT_SUMMARY = os.path.join(PHASE57_DIR, "phase5_7_summary.md")
OUT_WINNER = os.path.join(PHASE57_DIR, "phase5_7_winner.json")

ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
BASELINE_EXP = "p527_lstm_l3_v5_lite_s2027"
CONFIRMED_TREE = "p535_tree_hard_cov_activity"
BACKUP_TREE = "p531_tree_hard_core"

TUNED = [
    {
        "candidate_key": "p57_covact_lr005_l63_hard_g020",
        "tag": "covact_lr005_l63",
        "gate": "020",
        "qty_gate": 0.20,
    },
    {
        "candidate_key": "p57_covact_lr005_l63_hard_g025",
        "tag": "covact_lr005_l63",
        "gate": "025",
        "qty_gate": 0.25,
    },
]


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def path_phase55(anchor_date, base_exp):
    return os.path.join(
        PHASE55_DIR,
        anchor_tag(anchor_date),
        "phase5",
        f"eval_context_p55_{anchor_tag(anchor_date)}_{base_exp}_s2026.csv",
    )


def path_phase54_baseline():
    return os.path.join(PHASE54_PHASE_DIR, f"eval_context_p54_{BASELINE_EXP}_s2026.csv")


def path_phase57(anchor_date, tuned_cfg):
    exp_id = f"p57a_{anchor_tag(anchor_date)}_{tuned_cfg['tag']}_s2026_hard_g{tuned_cfg['gate']}"
    return exp_id, os.path.join(
        PHASE57_DIR,
        anchor_tag(anchor_date),
        "phase5",
        f"eval_context_{exp_id}.csv",
    )


def load_row(path, anchor_date, base_exp, track, candidate_key, source_phase):
    if not os.path.exists(path):
        return None
    row = evaluate_context_csv(path)
    if row is None:
        return None
    row["anchor_date"] = anchor_date
    row["base_exp"] = base_exp
    row["track"] = track
    row["candidate_key"] = candidate_key
    row["source_phase"] = source_phase
    row["context_path"] = path
    return row


def baseline_gate(anchor_df):
    baseline = anchor_df[anchor_df["candidate_key"] == "p527"]
    if baseline.empty:
        return anchor_df
    baseline = baseline.iloc[0]
    eligible = (
        (anchor_df["4_25_sku_p50"] > float(baseline["4_25_sku_p50"]))
        & (anchor_df["4_25_wmape_like"] <= float(baseline["4_25_wmape_like"]))
        & (anchor_df["ice_4_25_sku_p50"] > float(baseline["ice_4_25_sku_p50"]))
        & anchor_df["global_ratio"].between(0.90, 1.10)
    )
    anchor_df = anchor_df.copy()
    anchor_df["meets_anchor_gate"] = eligible
    return anchor_df


def build_candidate_summary(detail_df):
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
        detail_df.groupby(["candidate_key", "track"], as_index=False)
        .agg(
            base_exp=("base_exp", "first"),
            **{col: (col, "mean") for col in metric_cols},
            anchor_gate_mean=("meets_anchor_gate", "mean"),
        )
    )
    wins = (
        detail_df.groupby("candidate_key", as_index=False)["meets_anchor_gate"]
        .sum()
        .rename(columns={"meets_anchor_gate": "anchor_passes"})
    )
    summary = summary.merge(wins, on="candidate_key", how="left")
    return summary


def pick_recommended(detail_df, summary_df):
    tuned_df = summary_df[summary_df["track"] == "event_tree_tuned"].copy()
    if tuned_df.empty:
        return None, False

    tuned_df["delivery_ready"] = (
        (tuned_df["anchor_passes"] >= 3)
        & (tuned_df["4_25_sku_p50"] >= 0.55)
        & (tuned_df["4_25_wmape_like"] <= 0.55)
        & (tuned_df["ice_4_25_sku_p50"] >= 0.35)
        & (tuned_df["1_3_ratio"] <= 1.35)
    )

    ready = tuned_df[tuned_df["delivery_ready"]].copy()
    if not ready.empty:
        g020 = ready[ready["candidate_key"] == "p57_covact_lr005_l63_hard_g020"]
        g025 = ready[ready["candidate_key"] == "p57_covact_lr005_l63_hard_g025"]
        if not g020.empty and not g025.empty:
            early = detail_df[
                (detail_df["candidate_key"] == "p57_covact_lr005_l63_hard_g020")
                & (detail_df["anchor_date"].isin(["2025-09-01", "2025-10-01"]))
            ]
            if not early.empty and (early["global_ratio"] > 1.10).any():
                return "p57_covact_lr005_l63_hard_g025", True
            return "p57_covact_lr005_l63_hard_g020", True
        ready = ready.sort_values(
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
        )
        return str(ready.iloc[0]["candidate_key"]), True

    tuned_df = tuned_df.sort_values(
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
    )
    return str(tuned_df.iloc[0]["candidate_key"]), False


def render_summary(detail_df, summary_df, recommended_key, delivery_ready):
    lines = [
        "# Phase 5.7 Summary",
        "",
        f"- baseline_sequence: `{BASELINE_EXP}`",
        f"- confirmed_tree_main: `{CONFIRMED_TREE}`",
        f"- confirmed_tree_backup: `{BACKUP_TREE}`",
        f"- recommended_candidate: `{recommended_key}`",
        f"- delivery_ready: `{delivery_ready}`",
        "- next_stage: `phase6_tree_stabilization`",
        "",
        "| candidate_key | track | anchor_passes | global_ratio | global_wmape | 4_25_sku_p50 | 4_25_under_wape | ice_4_25_sku_p50 | 1_3_ratio | blockbuster_sku_p50 | blockbuster_under_wape | top20_true_volume_capture | rank_corr_positive_skus | auc | f1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    display = summary_df.copy()
    for col in numeric_cols_for_rounding(display):
        display[col] = display[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")
    for _, row in display.sort_values(
        ["track", "anchor_passes", "4_25_sku_p50", "global_wmape"],
        ascending=[True, False, False, True],
    ).iterrows():
        lines.append(
            "| {candidate_key} | {track} | {anchor_passes} | {global_ratio} | {global_wmape} | {four_p50} | "
            "{four_under} | {ice_four_p50} | {one_three_ratio} | {blockbuster_p50} | {blockbuster_under} | "
            "{top20_capture} | {rank_corr} | {auc} | {f1} |".format(
                candidate_key=row["candidate_key"],
                track=row["track"],
                anchor_passes=int(row["anchor_passes"]),
                global_ratio=row["global_ratio"],
                global_wmape=row["global_wmape"],
                four_p50=row["4_25_sku_p50"],
                four_under=row["4_25_under_wape"],
                ice_four_p50=row["ice_4_25_sku_p50"],
                one_three_ratio=row["1_3_ratio"],
                blockbuster_p50=row["blockbuster_sku_p50"],
                blockbuster_under=row["blockbuster_under_wape"],
                top20_capture=row["top20_true_volume_capture"],
                rank_corr=row["rank_corr_positive_skus"],
                auc=row["auc"],
                f1=row["f1"],
            )
        )
    return "\n".join(lines)


def main():
    rows = []
    for anchor_date in ANCHORS:
        rows.append(load_row(path_phase55(anchor_date, BASELINE_EXP), anchor_date, BASELINE_EXP, "sequence_baseline", "p527", "phase5_5"))
        rows.append(load_row(path_phase55(anchor_date, CONFIRMED_TREE), anchor_date, CONFIRMED_TREE, "event_tree_confirmed", "p535", "phase5_5"))
        rows.append(load_row(path_phase55(anchor_date, BACKUP_TREE), anchor_date, BACKUP_TREE, "event_tree_backup", "p531", "phase5_5"))
        for tuned_cfg in TUNED:
            exp_id, path = path_phase57(anchor_date, tuned_cfg)
            rows.append(load_row(path, anchor_date, exp_id, "event_tree_tuned", tuned_cfg["candidate_key"], "phase5_7"))

    rows = [row for row in rows if row is not None]

    has_dec_2025_baseline = any(row["anchor_date"] == "2025-12-01" and row["candidate_key"] == "p527" for row in rows)
    if not has_dec_2025_baseline:
        rows.append(load_row(path_phase54_baseline(), "2025-12-01", BASELINE_EXP, "sequence_baseline", "p527", "phase5_4_fallback"))
    rows = [row for row in rows if row is not None]

    detail_df = pd.DataFrame(rows)
    if detail_df.empty:
        raise SystemExit("No phase5.7 rows found.")

    detail_df = pd.concat(
        [baseline_gate(anchor_df) for _, anchor_df in detail_df.groupby("anchor_date", sort=True)],
        ignore_index=True,
    )
    detail_df["candidate_sort"] = detail_df["candidate_key"].map({
        "p527": 0,
        "p535": 1,
        "p531": 2,
        "p57_covact_lr005_l63_hard_g020": 3,
        "p57_covact_lr005_l63_hard_g025": 4,
    }).fillna(99)
    detail_df = detail_df.sort_values(["anchor_date", "candidate_sort"]).drop(columns=["candidate_sort"])
    detail_df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    summary_df = build_candidate_summary(detail_df)
    recommended_key, delivery_ready = pick_recommended(detail_df, summary_df)

    candidate_anchor_exp_ids = {}
    for candidate_key, sub in detail_df.groupby("candidate_key"):
        candidate_anchor_exp_ids[candidate_key] = {
            str(anchor): str(exp_id) for anchor, exp_id in zip(sub["anchor_date"], sub["exp_id"])
        }

    winner = {
        "baseline_sequence": BASELINE_EXP,
        "confirmed_tree_main": CONFIRMED_TREE,
        "confirmed_tree_backup": BACKUP_TREE,
        "recommended_candidate_key": recommended_key,
        "delivery_ready": delivery_ready,
        "needs_phase57b": not delivery_ready,
        "next_stage": "phase6_tree_stabilization",
        "candidate_anchor_exp_ids": candidate_anchor_exp_ids,
        "candidate_summary": summary_df.sort_values("candidate_key").to_dict(orient="records"),
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(render_summary(detail_df, summary_df, recommended_key, delivery_ready))
    with open(OUT_WINNER, "w", encoding="utf-8") as fh:
        json.dump(winner, fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase5.7 anchor table -> {OUT_TABLE}")
    print(f"[OK] phase5.7 summary -> {OUT_SUMMARY}")
    print(f"[OK] phase5.7 winner -> {OUT_WINNER}")


if __name__ == "__main__":
    main()
