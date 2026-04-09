import json
import os

import numpy as np
import pandas as pd

from phase_eval_utils import numeric_cols_for_rounding


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase6f_tree_family_compare")

STAGEA_JSON = os.path.join(OUT_DIR, "phase6f_stagea_smoke.json")
STAGE_STATE_JSON = os.path.join(OUT_DIR, "phase6f_stage_state.json")
SMOKE_TABLE = os.path.join(OUT_DIR, "phase6f_single_anchor_smoke_table.csv")
RAW_ANCHOR_TABLE = os.path.join(OUT_DIR, "phase6f_raw_anchor_table.csv")
CALIBRATED_ANCHOR_TABLE = os.path.join(OUT_DIR, "phase6f_calibrated_anchor_table.csv")

OUT_TABLE = os.path.join(OUT_DIR, "phase6f_tree_family_compare_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase6f_tree_family_compare_summary.md")
OUT_WINNER = os.path.join(OUT_DIR, "phase6f_tree_family_compare_winner.json")

LIGHTGBM_RAW_KEY = "lightgbm_raw_g025"
LIGHTGBM_MAINLINE_KEY = "lightgbm_sep098_oct093"


def summarize_anchor_metrics(anchor_df):
    metric_cols = [
        "global_ratio",
        "global_wmape",
        "4_25_under_wape",
        "4_25_sku_p50",
        "ice_4_25_sku_p50",
        "blockbuster_under_wape",
        "blockbuster_sku_p50",
        "top20_true_volume_capture",
        "rank_corr_positive_skus",
    ]
    agg = {col: (col, "mean") for col in metric_cols}
    if "anchor_pass" in anchor_df.columns:
        agg["anchor_passes"] = ("anchor_pass", "sum")
    return anchor_df.groupby(["candidate_key", "backend", "stage_name"], as_index=False).agg(**agg)


def raw_not_weaker(candidate_row, lgbm_row):
    primary_ok = (
        0.90 <= float(candidate_row["global_ratio"]) <= 1.10
        and float(candidate_row["4_25_under_wape"]) <= float(lgbm_row["4_25_under_wape"]) + 0.01
        and float(candidate_row["4_25_sku_p50"]) >= float(lgbm_row["4_25_sku_p50"]) - 0.01
        and float(candidate_row["ice_4_25_sku_p50"]) >= float(lgbm_row["ice_4_25_sku_p50"]) - 0.01
    )
    improvement_ok = (
        float(candidate_row["blockbuster_under_wape"]) <= float(lgbm_row["blockbuster_under_wape"]) - 0.02
        or float(candidate_row["blockbuster_sku_p50"]) >= float(lgbm_row["blockbuster_sku_p50"]) + 0.03
        or float(candidate_row["top20_true_volume_capture"]) >= float(lgbm_row["top20_true_volume_capture"]) + 0.02
        or float(candidate_row["rank_corr_positive_skus"]) >= float(lgbm_row["rank_corr_positive_skus"]) + 0.02
    )
    return primary_ok and improvement_ok


def replacement_gate(candidate_row, mainline_row):
    primary_ok = (
        int(candidate_row["anchor_passes"]) == 4
        and float(candidate_row["4_25_under_wape"]) <= float(mainline_row["4_25_under_wape"]) + 0.01
        and float(candidate_row["4_25_sku_p50"]) >= float(mainline_row["4_25_sku_p50"]) - 0.01
        and float(candidate_row["ice_4_25_sku_p50"]) >= float(mainline_row["ice_4_25_sku_p50"]) - 0.01
    )
    improvement_ok = (
        float(candidate_row["blockbuster_under_wape"]) <= float(mainline_row["blockbuster_under_wape"]) - 0.02
        or float(candidate_row["blockbuster_sku_p50"]) >= float(mainline_row["blockbuster_sku_p50"]) + 0.03
        or float(candidate_row["top20_true_volume_capture"]) >= float(mainline_row["top20_true_volume_capture"]) + 0.02
        or float(candidate_row["rank_corr_positive_skus"]) >= float(mainline_row["rank_corr_positive_skus"]) + 0.02
    )
    return primary_ok and improvement_ok


def render_summary(stagea, smoke_df, raw_summary_df, calibrated_summary_df, winner_payload):
    smoke_disp = smoke_df.copy()
    raw_disp = raw_summary_df.copy()
    cal_disp = calibrated_summary_df.copy()
    for frame in (smoke_disp, raw_disp, cal_disp):
        for col in numeric_cols_for_rounding(frame):
            if col in frame.columns and pd.api.types.is_numeric_dtype(frame[col]):
                frame[col] = frame[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.4f}")

    lines = [
        "# Phase6f Tree Family Compare",
        "",
        f"- current_mainline: `sep098_oct093`",
        f"- freeze_current_tree_mainline: `{winner_payload['freeze_current_tree_mainline']}`",
        f"- promoted_backend: `{winner_payload['promoted_backend']}`",
        "",
        "## Stage A",
        "",
        f"- syntax_ok: `{stagea.get('syntax_ok', False)}`",
    ]
    for backend, ok in stagea.get("imports", {}).items():
        smoke_ok = stagea.get("backend_smoke", {}).get(backend, {}).get("ok", False)
        lines.append(f"- `{backend}` import/backend_smoke: `{ok}` / `{smoke_ok}`")

    lines.extend(
        [
            "",
            "## Stage B Single-Anchor Smoke",
            "",
            "| candidate_key | backend | smoke_gate_pass | forced_to_stage_c | advance_to_stage_c | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in smoke_disp.iterrows():
        lines.append(
            f"| {row['candidate_key']} | {row['backend']} | {row.get('smoke_gate_pass', '')} | {row.get('forced_to_stage_c', '')} | {row.get('advance_to_stage_c', '')} | {row.get('global_ratio', '')} | "
            f"{row.get('4_25_under_wape', '')} | {row.get('4_25_sku_p50', '')} | {row.get('ice_4_25_sku_p50', '')} | "
            f"{row.get('blockbuster_under_wape', '')} | {row.get('blockbuster_sku_p50', '')} | {row.get('top20_true_volume_capture', '')} | "
            f"{row.get('rank_corr_positive_skus', '')} |"
        )

    lines.extend(
        [
            "",
            "## Stage C Raw Four-Anchor Compare",
            "",
            "| candidate_key | backend | raw_stage_pass | forced_to_stage_d | advance_to_stage_d | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in raw_disp.iterrows():
        lines.append(
            f"| {row['candidate_key']} | {row['backend']} | {row.get('raw_stage_pass', '')} | {row.get('forced_to_stage_d', '')} | {row.get('advance_to_stage_d', '')} | {row.get('global_ratio', '')} | "
            f"{row.get('4_25_under_wape', '')} | {row.get('4_25_sku_p50', '')} | {row.get('ice_4_25_sku_p50', '')} | "
            f"{row.get('blockbuster_under_wape', '')} | {row.get('blockbuster_sku_p50', '')} | {row.get('top20_true_volume_capture', '')} | "
            f"{row.get('rank_corr_positive_skus', '')} |"
        )

    lines.extend(
        [
            "",
            "## Stage D Calibrated Final Compare",
            "",
            "| candidate_key | backend | replacement_gate | anchor_passes | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for _, row in cal_disp.iterrows():
        lines.append(
            f"| {row['candidate_key']} | {row['backend']} | {row.get('replacement_gate', '')} | {int(row.get('anchor_passes', 0)) if pd.notna(row.get('anchor_passes', np.nan)) else ''} | "
            f"{row.get('global_ratio', '')} | {row.get('4_25_under_wape', '')} | {row.get('4_25_sku_p50', '')} | "
            f"{row.get('ice_4_25_sku_p50', '')} | {row.get('blockbuster_under_wape', '')} | {row.get('blockbuster_sku_p50', '')} | "
            f"{row.get('top20_true_volume_capture', '')} | {row.get('rank_corr_positive_skus', '')} |"
        )
    return "\n".join(lines)


def main():
    if not os.path.exists(STAGEA_JSON):
        raise SystemExit("Missing phase6f_stagea_smoke.json")
    if not os.path.exists(SMOKE_TABLE):
        raise SystemExit("Missing phase6f_single_anchor_smoke_table.csv")
    if not os.path.exists(RAW_ANCHOR_TABLE):
        raise SystemExit("Missing phase6f_raw_anchor_table.csv")
    if not os.path.exists(CALIBRATED_ANCHOR_TABLE):
        raise SystemExit("Missing phase6f_calibrated_anchor_table.csv")

    with open(STAGEA_JSON, "r", encoding="utf-8") as fh:
        stagea = json.load(fh)
    with open(STAGE_STATE_JSON, "r", encoding="utf-8") as fh:
        stage_state = json.load(fh)

    smoke_df = pd.read_csv(SMOKE_TABLE)
    raw_anchor_df = pd.read_csv(RAW_ANCHOR_TABLE)
    calibrated_anchor_df = pd.read_csv(CALIBRATED_ANCHOR_TABLE)

    raw_summary_df = summarize_anchor_metrics(raw_anchor_df)
    lgbm_raw_row = raw_summary_df[raw_summary_df["candidate_key"] == LIGHTGBM_RAW_KEY].iloc[0]
    raw_summary_df["raw_stage_pass"] = raw_summary_df.apply(
        lambda row: True if row["candidate_key"] == LIGHTGBM_RAW_KEY else raw_not_weaker(row, lgbm_raw_row),
        axis=1,
    )
    raw_gate_passed_backends = set(stage_state.get("raw_gate_passed_backends", []))
    stage_d_backends = set(stage_state.get("stage_d_backends", []))
    raw_summary_df["forced_to_stage_d"] = raw_summary_df["backend"].map(
        lambda backend: backend != "lightgbm" and backend in stage_d_backends and backend not in raw_gate_passed_backends
    )
    raw_summary_df["advance_to_stage_d"] = raw_summary_df["backend"].map(
        lambda backend: backend == "lightgbm" or backend in stage_d_backends
    )

    calibrated_summary_df = summarize_anchor_metrics(calibrated_anchor_df)
    mainline_row = calibrated_summary_df[calibrated_summary_df["candidate_key"] == LIGHTGBM_MAINLINE_KEY].iloc[0]
    calibrated_summary_df["replacement_gate"] = calibrated_summary_df.apply(
        lambda row: False if row["candidate_key"] == LIGHTGBM_MAINLINE_KEY else replacement_gate(row, mainline_row),
        axis=1,
    )

    final_table = pd.concat(
        [
            smoke_df.assign(compare_stage="single_anchor_smoke"),
            raw_summary_df.assign(compare_stage="raw_4anchor"),
            calibrated_summary_df.assign(compare_stage="calibrated_final"),
        ],
        ignore_index=True,
        sort=False,
    )
    final_table.to_csv(OUT_TABLE, index=False, encoding="utf-8-sig")

    promoted = calibrated_summary_df[calibrated_summary_df["replacement_gate"]].copy()
    if promoted.empty:
        winner_payload = {
            "current_mainline": "sep098_oct093",
            "freeze_current_tree_mainline": True,
            "winner_action": "freeze_current_tree_mainline",
            "promoted_backend": None,
            "force_stage_c_backends": stage_state.get("force_stage_c_backends", []),
            "force_stage_d_backends": stage_state.get("force_stage_d_backends", []),
            "smoke_gate_passed_backends": stage_state.get("smoke_gate_passed_backends", []),
            "stage_c_backends": stage_state.get("stage_c_backends", []),
            "raw_gate_passed_backends": stage_state.get("raw_gate_passed_backends", []),
            "stage_d_backends": stage_state.get("stage_d_backends", []),
            "stage_a": stagea,
            "raw_summary": raw_summary_df.to_dict(orient="records"),
            "calibrated_summary": calibrated_summary_df.to_dict(orient="records"),
        }
    else:
        promoted = promoted.sort_values(
            [
                "blockbuster_under_wape",
                "blockbuster_sku_p50",
                "top20_true_volume_capture",
                "rank_corr_positive_skus",
                "global_wmape",
            ],
            ascending=[True, False, False, False, True],
        )
        chosen = promoted.iloc[0]
        winner_payload = {
            "current_mainline": "sep098_oct093",
            "freeze_current_tree_mainline": False,
            "winner_action": f"promote_{chosen['backend']}",
            "promoted_backend": str(chosen["backend"]),
            "promoted_candidate_key": str(chosen["candidate_key"]),
            "force_stage_c_backends": stage_state.get("force_stage_c_backends", []),
            "force_stage_d_backends": stage_state.get("force_stage_d_backends", []),
            "smoke_gate_passed_backends": stage_state.get("smoke_gate_passed_backends", []),
            "stage_c_backends": stage_state.get("stage_c_backends", []),
            "raw_gate_passed_backends": stage_state.get("raw_gate_passed_backends", []),
            "stage_d_backends": stage_state.get("stage_d_backends", []),
            "stage_a": stagea,
            "raw_summary": raw_summary_df.to_dict(orient="records"),
            "calibrated_summary": calibrated_summary_df.to_dict(orient="records"),
        }

    with open(OUT_WINNER, "w", encoding="utf-8") as fh:
        json.dump(winner_payload, fh, ensure_ascii=False, indent=2)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as fh:
        fh.write(render_summary(stagea, smoke_df, raw_summary_df, calibrated_summary_df, winner_payload))

    print(f"[OK] phase6f compare table -> {OUT_TABLE}")
    print(f"[OK] phase6f summary -> {OUT_SUMMARY}")
    print(f"[OK] phase6f winner -> {OUT_WINNER}")


if __name__ == "__main__":
    main()
