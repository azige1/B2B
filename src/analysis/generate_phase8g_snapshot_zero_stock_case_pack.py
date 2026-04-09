import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROW_COMPARE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "phase8_event_inventory_shadow_2026"
    / "phase8_event_inventory_shadow_row_compare.csv"
)
OUT_DIR = PROJECT_ROOT / "reports" / "phase8_snapshot_zero_stock_2026"


def safe_rate(num, den):
    if den in (0, 0.0, None) or pd.isna(den):
        return np.nan
    return float(num) / float(den)


def markdown_table(df, columns):
    headers = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        vals = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                vals.append("" if np.isnan(value) else f"{value:.4f}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([headers, sep, *rows])


def load_row_compare():
    df = pd.read_csv(ROW_COMPARE_PATH, encoding="utf-8-sig")
    numeric_cols = [
        "true_replenish_qty",
        "base_pred_qty",
        "shadow_pred_qty",
        "base_abs_error",
        "shadow_abs_error",
        "abs_error_gain",
        "pred_delta",
        "qty_first_order",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
        "event_strong_30",
        "event_daily_clicks_30",
        "event_daily_cart_adds_30",
        "event_daily_order_success_30",
        "event_daily_pay_success_30",
        "inv_total_stock",
        "snapshot_present",
        "stock_positive",
        "stock_zero",
        "base_zero_true_fp",
        "shadow_zero_true_fp",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["stock_state"] = "no_snapshot"
    df.loc[df["stock_zero"] > 0, "stock_state"] = "snapshot_zero_stock"
    df.loc[df["stock_positive"] > 0, "stock_state"] = "snapshot_positive_stock"
    df["true_bin"] = np.where(df["true_replenish_qty"] > 0, "positive_true", "zero_true")
    df["base_under_gap"] = (
        df["true_replenish_qty"] - df["base_pred_qty"]
    ).clip(lower=0.0)
    df["shadow_under_gap"] = (
        df["true_replenish_qty"] - df["shadow_pred_qty"]
    ).clip(lower=0.0)
    return df


def build_anchor_mix(df):
    rows = []
    for (anchor_date, true_bin, stock_state), sub in df.groupby(
        ["anchor_date", "true_bin", "stock_state"], dropna=False
    ):
        rows.append(
            {
                "anchor_date": anchor_date,
                "true_bin": true_bin,
                "stock_state": stock_state,
                "rows": int(len(sub)),
                "base_pred_mean": float(sub["base_pred_qty"].mean()),
                "shadow_pred_mean": float(sub["shadow_pred_qty"].mean()),
                "event_strong_rate": safe_rate(sub["event_strong_30"].sum(), len(sub)),
                "lookback_repl_pos_rate": safe_rate(
                    (sub["lookback_repl_sum_90"] > 0).sum(), len(sub)
                ),
                "qfo_ge_25_rate": safe_rate((sub["qty_first_order"] >= 25).sum(), len(sub)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["anchor_date", "true_bin", "stock_state"]
    ).reset_index(drop=True)


def build_zero_true_compare(df):
    zero = df[df["true_bin"] == "zero_true"].copy()
    rows = []
    for stock_state, sub in zero.groupby("stock_state", dropna=False):
        rows.append(
            {
                "stock_state": stock_state,
                "rows": int(len(sub)),
                "base_fp_rate": safe_rate(sub["base_zero_true_fp"].sum(), len(sub)),
                "shadow_fp_rate": safe_rate(sub["shadow_zero_true_fp"].sum(), len(sub)),
                "fp_rate_delta": safe_rate(sub["shadow_zero_true_fp"].sum(), len(sub))
                - safe_rate(sub["base_zero_true_fp"].sum(), len(sub)),
                "base_pred_mean": float(sub["base_pred_qty"].mean()),
                "shadow_pred_mean": float(sub["shadow_pred_qty"].mean()),
                "lookback_repl_pos_rate": safe_rate(
                    (sub["lookback_repl_sum_90"] > 0).sum(), len(sub)
                ),
                "qfo_ge_25_rate": safe_rate((sub["qty_first_order"] >= 25).sum(), len(sub)),
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def build_positive_true_compare(df):
    pos = df[df["true_bin"] == "positive_true"].copy()
    rows = []
    for stock_state, sub in pos.groupby("stock_state", dropna=False):
        true_sum = float(sub["true_replenish_qty"].sum())
        rows.append(
            {
                "stock_state": stock_state,
                "rows": int(len(sub)),
                "true_sum": true_sum,
                "base_under_wape": safe_rate(sub["base_under_gap"].sum(), true_sum),
                "shadow_under_wape": safe_rate(sub["shadow_under_gap"].sum(), true_sum),
                "under_wape_delta": safe_rate(sub["shadow_under_gap"].sum(), true_sum)
                - safe_rate(sub["base_under_gap"].sum(), true_sum),
                "base_pred_mean": float(sub["base_pred_qty"].mean()),
                "shadow_pred_mean": float(sub["shadow_pred_qty"].mean()),
                "event_strong_rate": safe_rate(sub["event_strong_30"].sum(), len(sub)),
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def build_focus_cases(df):
    zero_stock = df[df["stock_state"] == "snapshot_zero_stock"].copy()
    frames = []

    zero_true = zero_stock[zero_stock["true_bin"] == "zero_true"].copy()
    if not zero_true.empty:
        worst_fp = zero_true.sort_values(
            ["shadow_pred_qty", "base_pred_qty", "qty_first_order", "lookback_repl_sum_90"],
            ascending=[False, False, False, False],
        ).head(30)
        worst_fp["case_type"] = "snapshot_zero_zero_true_false_positive"
        frames.append(worst_fp)

        reduced_fp = zero_true[zero_true["shadow_pred_qty"] < zero_true["base_pred_qty"]].copy()
        reduced_fp = reduced_fp.sort_values(
            ["pred_delta", "base_pred_qty"],
            ascending=[True, False],
        ).head(30)
        reduced_fp["case_type"] = "snapshot_zero_zero_true_fp_reduced"
        frames.append(reduced_fp)

    positive_true = zero_stock[zero_stock["true_bin"] == "positive_true"].copy()
    if not positive_true.empty:
        worsened = positive_true.sort_values(
            ["abs_error_gain", "true_replenish_qty"],
            ascending=[True, False],
        ).head(30)
        worsened["case_type"] = "snapshot_zero_positive_true_worsened"
        frames.append(worsened)

        improved = positive_true.sort_values(
            ["abs_error_gain", "true_replenish_qty"],
            ascending=[False, False],
        ).head(30)
        improved["case_type"] = "snapshot_zero_positive_true_improved"
        frames.append(improved)

    cases = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    keep_cols = [
        "case_type",
        "anchor_date",
        "sku_id",
        "product_name",
        "category_static",
        "style_id",
        "signal_quadrant",
        "activity_bucket",
        "true_replenish_qty",
        "base_pred_qty",
        "shadow_pred_qty",
        "pred_delta",
        "base_abs_error",
        "shadow_abs_error",
        "abs_error_gain",
        "qty_first_order",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
        "event_strong_30",
        "event_daily_clicks_30",
        "event_daily_cart_adds_30",
        "event_daily_order_success_30",
        "event_daily_pay_success_30",
        "snapshot_present",
        "stock_positive",
        "stock_zero",
        "inv_snapshot_present_7",
        "inv_snapshot_present_14",
        "inv_snapshot_present_30",
        "inv_stock_positive_7",
        "inv_stock_positive_14",
        "inv_stock_positive_30",
        "inv_stock_zero_7",
        "inv_stock_zero_14",
        "inv_stock_zero_30",
        "inv_days_since_last_snapshot",
        "inv_days_since_last_stock_positive",
        "inv_days_since_last_stock_zero",
        "inv_stock_zero_streak",
        "inv_positive_to_zero_switch",
        "inv_short_zero",
        "inv_long_zero",
        "inv_short_zero_7",
        "inv_short_zero_14",
        "inv_short_zero_30",
        "inv_long_zero_7",
        "inv_long_zero_14",
        "inv_long_zero_30",
        "inv_total_stock",
    ]
    keep_cols = [col for col in keep_cols if col in cases.columns]
    return cases[keep_cols].reset_index(drop=True) if not cases.empty else cases


def build_manifest(anchor_mix, zero_compare, positive_compare, cases):
    payload = {
        "status": "analysis_only",
        "focus_state": "snapshot_zero_stock",
        "source": "reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_row_compare.csv",
        "outputs": {
            "anchor_mix": "reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_anchor_mix.csv",
            "zero_true_compare": "reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_zero_true_compare.csv",
            "positive_true_compare": "reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_positive_true_compare.csv",
            "cases": "reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_cases.csv",
            "summary": "reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_summary.md",
        },
        "counts": {
            "anchor_mix_rows": int(len(anchor_mix)),
            "zero_true_compare_rows": int(len(zero_compare)),
            "positive_true_compare_rows": int(len(positive_compare)),
            "case_rows": int(len(cases)),
        },
    }
    (OUT_DIR / "phase8_snapshot_zero_stock_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_summary(anchor_mix, zero_compare, positive_compare, cases):
    zero_stock_zero = zero_compare[zero_compare["stock_state"] == "snapshot_zero_stock"]
    zero_stock_pos = positive_compare[positive_compare["stock_state"] == "snapshot_zero_stock"]
    positive_stock_pos = positive_compare[
        positive_compare["stock_state"] == "snapshot_positive_stock"
    ]

    lines = [
        "# Phase8 Snapshot Zero Stock Case Pack",
        "",
        "- Status: `analysis_only`",
        "- Scope: `2026-02-15 / 2026-02-24`",
        "- Source: `phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_row_compare.csv`",
        "- Purpose: isolate rows with `snapshot_present=1` and `stock_zero=1` to see whether the inventory line is learning more than a pure positive-stock signal.",
        "",
        "## Bottom Line",
        "",
    ]

    if not zero_stock_zero.empty:
        row = zero_stock_zero.iloc[0]
        lines.append(
            f"- `snapshot_zero_stock` zero-true rows: `{int(row['rows'])}`; false-positive rate improves from `{row['base_fp_rate']:.4f}` to `{row['shadow_fp_rate']:.4f}`."
        )
    if not zero_stock_pos.empty:
        row = zero_stock_pos.iloc[0]
        lines.append(
            f"- `snapshot_zero_stock` positive-true rows: `{int(row['rows'])}`; under-WAPE moves from `{row['base_under_wape']:.4f}` to `{row['shadow_under_wape']:.4f}`."
        )
    if not positive_stock_pos.empty:
        row = positive_stock_pos.iloc[0]
        lines.append(
            f"- For comparison, `snapshot_positive_stock` positive-true rows improve from `{row['base_under_wape']:.4f}` to `{row['shadow_under_wape']:.4f}`."
        )

    lines.extend(
        [
            "",
            "## Anchor Mix",
            "",
            markdown_table(
                anchor_mix,
                [
                    "anchor_date",
                    "true_bin",
                    "stock_state",
                    "rows",
                    "base_pred_mean",
                    "shadow_pred_mean",
                    "event_strong_rate",
                    "lookback_repl_pos_rate",
                    "qfo_ge_25_rate",
                ],
            ),
            "",
            "## Zero-True Compare",
            "",
            markdown_table(
                zero_compare,
                [
                    "stock_state",
                    "rows",
                    "base_fp_rate",
                    "shadow_fp_rate",
                    "fp_rate_delta",
                    "base_pred_mean",
                    "shadow_pred_mean",
                    "lookback_repl_pos_rate",
                    "qfo_ge_25_rate",
                ],
            ),
            "",
            "## Positive-True Compare",
            "",
            markdown_table(
                positive_compare,
                [
                    "stock_state",
                    "rows",
                    "true_sum",
                    "base_under_wape",
                    "shadow_under_wape",
                    "under_wape_delta",
                    "base_pred_mean",
                    "shadow_pred_mean",
                    "event_strong_rate",
                ],
            ),
            "",
            "## Interpretation",
            "",
            "- If `snapshot_zero_stock` keeps reducing false positives but worsens positive-true under-predict, it is acting more like a cautious suppression signal than a complete stock-constraint solution.",
            "- If future work wants to exploit this state better, the next features should be time-based inventory state features rather than another new data source.",
            f"- Focus cases exported: `{len(cases)}`.",
            "",
            "## Output Files",
            "",
            "- `reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_anchor_mix.csv`",
            "- `reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_zero_true_compare.csv`",
            "- `reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_positive_true_compare.csv`",
            "- `reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_cases.csv`",
            "",
        ]
    )
    (OUT_DIR / "phase8_snapshot_zero_stock_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8-sig",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_row_compare()
    anchor_mix = build_anchor_mix(df)
    zero_compare = build_zero_true_compare(df)
    positive_compare = build_positive_true_compare(df)
    cases = build_focus_cases(df)

    anchor_mix.to_csv(
        OUT_DIR / "phase8_snapshot_zero_stock_anchor_mix.csv",
        index=False,
        encoding="utf-8-sig",
    )
    zero_compare.to_csv(
        OUT_DIR / "phase8_snapshot_zero_stock_zero_true_compare.csv",
        index=False,
        encoding="utf-8-sig",
    )
    positive_compare.to_csv(
        OUT_DIR / "phase8_snapshot_zero_stock_positive_true_compare.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cases.to_csv(
        OUT_DIR / "phase8_snapshot_zero_stock_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    build_manifest(anchor_mix, zero_compare, positive_compare, cases)
    write_summary(anchor_mix, zero_compare, positive_compare, cases)

    print(f"[OK] snapshot zero stock summary -> {OUT_DIR / 'phase8_snapshot_zero_stock_summary.md'}")


if __name__ == "__main__":
    main()
