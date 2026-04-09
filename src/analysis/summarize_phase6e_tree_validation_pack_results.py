import json
import os
from typing import Dict, List, Tuple

import pandas as pd

from phase_eval_utils import category_diagnostics, evaluate_context_frame, numeric_cols_for_rounding


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE54_PHASE_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_4", "phase5")
PHASE55_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_5")
PHASE57_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_7")
PHASE6D_DIR = os.path.join(PROJECT_ROOT, "reports", "phase6d_tree_micro_calibration")

OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase6e_tree_validation_pack")
OUT_TABLE = os.path.join(OUT_DIR, "phase6e_tree_validation_pack_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase6e_tree_validation_pack_summary.md")
OUT_WINNER = os.path.join(OUT_DIR, "phase6e_tree_validation_pack_winner.json")

ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]

BASELINE_EXP = "p527_lstm_l3_v5_lite_s2027"
CONFIRMED_TREE = "p535_tree_hard_cov_activity"
BACKUP_TREE = "p531_tree_hard_core"
BASE_TUNED_EXP_TEMPLATE = "p57a_{anchor_tag}_covact_lr005_l63_s2026_hard_g025"

CANDIDATE_SCALES = {
    "sep098_oct093": {"sep_scale": 0.98, "oct_scale": 0.93, "track": "event_tree_mainline"},
    "sep095_oct090": {"sep_scale": 0.95, "oct_scale": 0.90, "track": "event_tree_conservative"},
}

REFERENCE_SPECS = {
    "p535": {"track": "event_tree_confirmed", "base_exp": CONFIRMED_TREE},
    "p531": {"track": "event_tree_backup", "base_exp": BACKUP_TREE},
    "p527": {"track": "sequence_baseline", "base_exp": BASELINE_EXP},
}

SUMMARY_METRICS = [
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
    "blockbuster_catastrophic_under_rate",
    "blockbuster_top10_true_volume_capture",
    "blockbuster_top20_true_volume_capture",
    "top10_true_volume_capture",
    "top20_true_volume_capture",
    "rank_corr_positive_skus",
    "false_zero_rate",
    "false_positive_rate_zero_true",
    "zero_true_pred_ge_3_rate",
    "true_gt_10_pred_le_1_rate",
    "repl0_fut0_ratio",
    "repl0_fut0_under_wape",
    "repl0_fut1_ratio",
    "repl0_fut1_under_wape",
    "repl1_fut0_ratio",
    "repl1_fut0_under_wape",
    "repl1_fut1_ratio",
    "repl1_fut1_under_wape",
    "auc",
    "f1",
]

QUADRANT_ROWS = ["repl0_fut0_rows", "repl0_fut1_rows", "repl1_fut0_rows", "repl1_fut1_rows"]


def ensure_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def anchor_tag(anchor_date: str) -> str:
    return anchor_date.replace("-", "")


def phase55_context_path(anchor_date: str, base_exp: str) -> str:
    return os.path.join(
        PHASE55_DIR,
        anchor_tag(anchor_date),
        "phase5",
        f"eval_context_p55_{anchor_tag(anchor_date)}_{base_exp}_s2026.csv",
    )


def phase54_dec_baseline_path() -> str:
    return os.path.join(PHASE54_PHASE_DIR, f"eval_context_p54_{BASELINE_EXP}_s2026.csv")


def phase57_context_path(anchor_date: str) -> str:
    exp_id = BASE_TUNED_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))
    return os.path.join(PHASE57_DIR, anchor_tag(anchor_date), "phase5", f"eval_context_{exp_id}.csv")


def load_eval_context(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def apply_anchor_scaling(df: pd.DataFrame, anchor_date: str, sep_scale: float, oct_scale: float) -> pd.DataFrame:
    out = df.copy()
    for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
        out[col] = out[col].astype(float)

    scale = 1.0
    if anchor_date == "2025-09-01":
        scale = float(sep_scale)
    elif anchor_date == "2025-10-01":
        scale = float(oct_scale)

    if scale != 1.0:
        for col in ("ai_pred_qty_open", "ai_pred_qty", "ai_pred_positive_qty"):
            out[col] = out[col] * scale

    out["ai_pred_positive_qty"] = (out["ai_pred_qty"].astype(float) > 0).astype(int)
    out["abs_error"] = (out["ai_pred_qty"].astype(float) - out["true_replenish_qty"].astype(float)).abs()
    return out


def build_baseline_anchor_rows() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows: List[Dict] = []
    pooled_frames: Dict[str, List[pd.DataFrame]] = {"p527": []}
    for anchor_date in ANCHORS:
        if anchor_date == "2025-12-01":
            path = phase54_dec_baseline_path()
            exp_id = f"p54_{BASELINE_EXP}_s2026"
            source_phase = "phase5_4_fallback"
        else:
            path = phase55_context_path(anchor_date, BASELINE_EXP)
            exp_id = f"p55_{anchor_tag(anchor_date)}_{BASELINE_EXP}_s2026"
            source_phase = "phase5_5"

        df = load_eval_context(path)
        pooled_frames["p527"].append(df.assign(anchor_date=anchor_date))
        row = evaluate_context_frame(df, exp_id)
        row["anchor_date"] = anchor_date
        row["candidate_key"] = "p527"
        row["track"] = REFERENCE_SPECS["p527"]["track"]
        row["source_phase"] = source_phase
        row["context_path"] = path
        rows.append(row)

    return pd.DataFrame(rows), {"p527": pd.concat(pooled_frames["p527"], ignore_index=True, sort=False)}


def build_reference_anchor_rows() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows: List[Dict] = []
    pooled_frames: Dict[str, List[pd.DataFrame]] = {"p535": [], "p531": []}
    for candidate_key in ("p535", "p531"):
        spec = REFERENCE_SPECS[candidate_key]
        for anchor_date in ANCHORS:
            path = phase55_context_path(anchor_date, spec["base_exp"])
            df = load_eval_context(path)
            pooled_frames[candidate_key].append(df.assign(anchor_date=anchor_date))
            row = evaluate_context_frame(df, f"p55_{anchor_tag(anchor_date)}_{spec['base_exp']}_s2026")
            row["anchor_date"] = anchor_date
            row["candidate_key"] = candidate_key
            row["track"] = spec["track"]
            row["source_phase"] = "phase5_5"
            row["context_path"] = path
            rows.append(row)

    pooled = {key: pd.concat(frames, ignore_index=True, sort=False) for key, frames in pooled_frames.items()}
    return pd.DataFrame(rows), pooled


def build_calibrated_anchor_rows() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rows: List[Dict] = []
    pooled_frames: Dict[str, List[pd.DataFrame]] = {key: [] for key in CANDIDATE_SCALES}
    for candidate_key, cfg in CANDIDATE_SCALES.items():
        for anchor_date in ANCHORS:
            raw = load_eval_context(phase57_context_path(anchor_date))
            adj = apply_anchor_scaling(raw, anchor_date, cfg["sep_scale"], cfg["oct_scale"])
            pooled_frames[candidate_key].append(adj.assign(anchor_date=anchor_date))
            row = evaluate_context_frame(adj, f"{BASE_TUNED_EXP_TEMPLATE.format(anchor_tag=anchor_tag(anchor_date))}_{candidate_key}")
            row["anchor_date"] = anchor_date
            row["candidate_key"] = candidate_key
            row["track"] = cfg["track"]
            row["source_phase"] = "phase6e_tree_validation_pack"
            row["context_path"] = phase57_context_path(anchor_date)
            row["sep_scale"] = cfg["sep_scale"]
            row["oct_scale"] = cfg["oct_scale"]
            rows.append(row)

    pooled = {key: pd.concat(frames, ignore_index=True, sort=False) for key, frames in pooled_frames.items()}
    return pd.DataFrame(rows), pooled


def add_anchor_pass(anchor_df: pd.DataFrame, baseline_df: pd.DataFrame) -> pd.DataFrame:
    merged = anchor_df.merge(
        baseline_df[["anchor_date", "4_25_sku_p50", "4_25_wmape_like", "ice_4_25_sku_p50"]],
        on="anchor_date",
        how="left",
        suffixes=("", "_baseline"),
    )
    merged["anchor_pass"] = (
        (merged["candidate_key"] != "p527")
        & (merged["4_25_sku_p50"] > merged["4_25_sku_p50_baseline"])
        & (merged["4_25_wmape_like"] <= merged["4_25_wmape_like_baseline"])
        & (merged["ice_4_25_sku_p50"] > merged["ice_4_25_sku_p50_baseline"])
        & merged["global_ratio"].between(0.90, 1.10)
    )
    return merged


def summarize_candidates(anchor_df: pd.DataFrame, pooled_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    group_keys = ["candidate_key", "track"]
    agg_spec = {col: (col, "mean") for col in SUMMARY_METRICS}
    agg_spec.update({col: (col, "sum") for col in QUADRANT_ROWS})
    agg_spec["anchor_passes"] = ("anchor_pass", "sum")
    agg_spec["source_phase"] = ("source_phase", "first")
    summary = anchor_df.groupby(group_keys, as_index=False).agg(**agg_spec)

    category_rows = []
    for candidate_key, pooled_df in pooled_frames.items():
        diag = category_diagnostics(pooled_df)
        category_rows.append(
            {
                "candidate_key": candidate_key,
                "category_worst5_ratio": diag["category_worst5_ratio"],
                "category_worst5_under_wape": diag["category_worst5_under_wape"],
            }
        )
    category_df = pd.DataFrame(category_rows)
    summary = summary.merge(category_df, on="candidate_key", how="left")
    return summary


def rank_candidates(summary_df: pd.DataFrame) -> pd.DataFrame:
    ranked = summary_df.copy()
    ranked["primary_layer_ok"] = (
        (ranked["anchor_passes"] >= 4)
        & (ranked["4_25_sku_p50"] >= 0.55)
        & (ranked["ice_4_25_sku_p50"] >= 0.35)
    )
    ranked["secondary_ok"] = ranked["blockbuster_under_wape"].notna() & ranked["blockbuster_sku_p50"].notna()
    ranked["guardrail_ok"] = ranked["global_ratio"].between(0.90, 1.10)
    ranked["global_ratio_gap"] = (ranked["global_ratio"] - 1.0).abs()
    return ranked.sort_values(
        [
            "anchor_passes",
            "4_25_under_wape",
            "4_25_sku_p50",
            "ice_4_25_sku_p50",
            "blockbuster_under_wape",
            "blockbuster_sku_p50",
            "top20_true_volume_capture",
            "rank_corr_positive_skus",
            "global_ratio_gap",
            "global_wmape",
            "1_3_ratio",
        ],
        ascending=[False, True, False, False, True, False, False, False, True, True, True],
    )


def trigger_phase6f(summary_df: pd.DataFrame, winner: pd.Series) -> Tuple[bool, Dict[str, bool]]:
    structural_shortfall = (
        (float(winner["blockbuster_under_wape"]) > 0.55)
        or (float(winner["blockbuster_sku_p50"]) < 0.45)
        or (float(winner["top20_true_volume_capture"]) < 0.65)
        or (float(winner["rank_corr_positive_skus"]) < 0.75)
    )

    stands = (
        str(winner["candidate_key"]) == "sep098_oct093"
        and int(winner["anchor_passes"]) >= 4
        and bool(winner["primary_layer_ok"])
    )

    peers = summary_df[summary_df["candidate_key"] != winner["candidate_key"]].copy()
    material_gain = False
    for _, peer in peers.iterrows():
        preserves_primary = (
            int(peer["anchor_passes"]) >= int(winner["anchor_passes"])
            and float(peer["4_25_under_wape"]) <= float(winner["4_25_under_wape"])
            and float(peer["4_25_sku_p50"]) >= float(winner["4_25_sku_p50"])
            and float(peer["ice_4_25_sku_p50"]) >= float(winner["ice_4_25_sku_p50"])
        )
        improves_shortfall = (
            (float(peer["blockbuster_under_wape"]) <= float(winner["blockbuster_under_wape"]) - 0.02)
            or (float(peer["blockbuster_sku_p50"]) >= float(winner["blockbuster_sku_p50"]) + 0.03)
            or (float(peer["top20_true_volume_capture"]) >= float(winner["top20_true_volume_capture"]) + 0.02)
            or (float(peer["rank_corr_positive_skus"]) >= float(winner["rank_corr_positive_skus"]) + 0.02)
        )
        if preserves_primary and improves_shortfall:
            material_gain = True
            break

    no_meaningful_lgbm_gain = not material_gain
    return stands and structural_shortfall and no_meaningful_lgbm_gain, {
        "lightgbm_mainline_stands": stands,
        "structural_shortfall_present": structural_shortfall,
        "no_meaningful_lgbm_gain_left": no_meaningful_lgbm_gain,
    }


def render_summary(ranked_df: pd.DataFrame, winner: pd.Series, phase6f_trigger: bool, trigger_flags: Dict[str, bool]) -> str:
    disp = ranked_df.copy()
    for col in numeric_cols_for_rounding(disp):
        if col in disp.columns and pd.api.types.is_numeric_dtype(disp[col]):
            disp[col] = disp[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")

    lines = [
        "# Phase6e Tree Validation Pack",
        "",
        f"- selected_candidate: `{winner['candidate_key']}`",
        f"- freeze_current_tree_mainline: `{str(winner['candidate_key']) == 'sep098_oct093'}`",
        f"- phase6f_trigger: `{phase6f_trigger}`",
        f"- lightgbm_mainline_stands: `{trigger_flags['lightgbm_mainline_stands']}`",
        f"- structural_shortfall_present: `{trigger_flags['structural_shortfall_present']}`",
        f"- no_meaningful_lgbm_gain_left: `{trigger_flags['no_meaningful_lgbm_gain_left']}`",
        "",
        "## Candidate Summary",
        "",
        "| candidate_key | track | anchor_passes | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus | false_zero_rate | true_gt_10_pred_le_1_rate | global_ratio | global_wmape | 1_3_ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in disp.iterrows():
        lines.append(
            f"| {row['candidate_key']} | {row['track']} | {int(row['anchor_passes'])} | {row['4_25_under_wape']} | {row['4_25_sku_p50']} | "
            f"{row['ice_4_25_sku_p50']} | {row['blockbuster_under_wape']} | {row['blockbuster_sku_p50']} | "
            f"{row['top20_true_volume_capture']} | {row['rank_corr_positive_skus']} | {row['false_zero_rate']} | "
            f"{row['true_gt_10_pred_le_1_rate']} | {row['global_ratio']} | {row['global_wmape']} | {row['1_3_ratio']} |"
        )

    winner_ratio = winner.get("category_worst5_ratio", "[]")
    winner_under = winner.get("category_worst5_under_wape", "[]")
    lines.extend(
        [
            "",
            "## Selected Candidate Diagnostics",
            "",
            f"- `category_worst5_ratio`: `{winner_ratio}`",
            f"- `category_worst5_under_wape`: `{winner_under}`",
            f"- `repl0_fut0_ratio / under_wape`: `{float(winner['repl0_fut0_ratio']):.4f}` / `{float(winner['repl0_fut0_under_wape']):.4f}`",
            f"- `repl0_fut1_ratio / under_wape`: `{float(winner['repl0_fut1_ratio']):.4f}` / `{float(winner['repl0_fut1_under_wape']):.4f}`",
            f"- `repl1_fut0_ratio / under_wape`: `{float(winner['repl1_fut0_ratio']):.4f}` / `{float(winner['repl1_fut0_under_wape']):.4f}`",
            f"- `repl1_fut1_ratio / under_wape`: `{float(winner['repl1_fut1_ratio']):.4f}` / `{float(winner['repl1_fut1_under_wape']):.4f}`",
            f"- `blockbuster_top10_true_volume_capture`: `{float(winner['blockbuster_top10_true_volume_capture']):.4f}`",
            f"- `blockbuster_top20_true_volume_capture`: `{float(winner['blockbuster_top20_true_volume_capture']):.4f}`",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    ensure_dir()

    baseline_anchor_df, baseline_pooled = build_baseline_anchor_rows()
    reference_anchor_df, reference_pooled = build_reference_anchor_rows()
    candidate_anchor_df, candidate_pooled = build_calibrated_anchor_rows()

    all_pooled = {}
    all_pooled.update(candidate_pooled)
    all_pooled.update(reference_pooled)
    all_pooled.update(baseline_pooled)

    anchor_df = pd.concat([candidate_anchor_df, reference_anchor_df, baseline_anchor_df], ignore_index=True, sort=False)
    anchor_df = add_anchor_pass(anchor_df, baseline_anchor_df)

    summary_df = summarize_candidates(anchor_df, all_pooled)
    ranked_df = rank_candidates(summary_df)
    winner = ranked_df.iloc[0]
    phase6f_trigger, trigger_flags = trigger_phase6f(summary_df, winner)

    summary_df.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(render_summary(ranked_df, winner, phase6f_trigger, trigger_flags))

    payload = {
        "selected_candidate": str(winner["candidate_key"]),
        "freeze_current_tree_mainline": str(winner["candidate_key"]) == "sep098_oct093",
        "phase6f_trigger": bool(phase6f_trigger),
        "trigger_flags": trigger_flags,
        "next_stage": "phase6f_tree_family_compare" if phase6f_trigger else "freeze_current_tree_mainline",
        "candidate_summary": ranked_df.to_dict(orient="records"),
    }
    with open(OUT_WINNER, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(f"[OK] phase6e tree validation pack table -> {OUT_TABLE}")
    print(f"[OK] phase6e tree validation pack summary -> {OUT_SUMMARY}")
    print(f"[OK] phase6e tree validation pack winner -> {OUT_WINNER}")


if __name__ == "__main__":
    main()
