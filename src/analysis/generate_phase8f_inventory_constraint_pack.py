import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports" / "phase8_extended_signal_2026"
DATA_DIR = PROJECT_ROOT / "data" / "phase8a_prep"
SHADOW_ROW_COMPARE = (
    PROJECT_ROOT
    / "reports"
    / "phase8_event_inventory_shadow_2026"
    / "phase8_event_inventory_shadow_row_compare.csv"
)
WIDE_PATH = PROJECT_ROOT / "data" / "gold" / "wide_table_sku.csv"


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


def ensure_inventory_state_columns(df):
    if "snapshot_present" not in df.columns:
        df["snapshot_present"] = (
            (df["has_storage_snapshot"] > 0) | (df["has_b2b_snapshot"] > 0)
        ).astype(int)
    if "stock_positive" not in df.columns:
        df["stock_positive"] = (
            (df["snapshot_present"] > 0) & (df["inv_total_stock"] > 0)
        ).astype(int)
    if "stock_zero" not in df.columns:
        df["stock_zero"] = (
            (df["snapshot_present"] > 0) & (df["inv_total_stock"] <= 0)
        ).astype(int)

    df["stock_state"] = "no_snapshot"
    df.loc[df["stock_zero"] > 0, "stock_state"] = "snapshot_zero_stock"
    df.loc[df["stock_positive"] > 0, "stock_state"] = "snapshot_positive_stock"
    return df


def load_row_compare():
    df = pd.read_csv(SHADOW_ROW_COMPARE, encoding="utf-8-sig")
    numeric_cols = [
        "true_replenish_qty",
        "base_pred_qty",
        "shadow_pred_qty",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
        "qty_first_order",
        "event_strong_30",
        "qty_storage_stock",
        "qty_b2b_hq_stock",
        "inv_total_stock",
        "has_storage_snapshot",
        "has_b2b_snapshot",
        "snapshot_present",
        "stock_positive",
        "stock_zero",
        "base_zero_true_fp",
        "shadow_zero_true_fp",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = ensure_inventory_state_columns(df)
    df["base_under_gap"] = (
        df["true_replenish_qty"] - df["base_pred_qty"]
    ).clip(lower=0.0)
    df["shadow_under_gap"] = (
        df["true_replenish_qty"] - df["shadow_pred_qty"]
    ).clip(lower=0.0)
    return df


def build_zero_true_table(df):
    zero = df[df["true_replenish_qty"] == 0].copy()
    rows = []
    for stock_state, sub in zero.groupby("stock_state", dropna=False):
        rows.append(
            {
                "stock_state": str(stock_state),
                "rows": int(len(sub)),
                "base_fp_rate": safe_rate(sub["base_zero_true_fp"].sum(), len(sub)),
                "shadow_fp_rate": safe_rate(sub["shadow_zero_true_fp"].sum(), len(sub)),
                "fp_rate_delta": safe_rate(
                    sub["shadow_zero_true_fp"].sum(), len(sub)
                )
                - safe_rate(sub["base_zero_true_fp"].sum(), len(sub)),
                "event_strong_rate": safe_rate(sub["event_strong_30"].sum(), len(sub)),
                "lookback_repl_pos_rate": safe_rate(
                    (sub["lookback_repl_sum_90"] > 0).sum(), len(sub)
                ),
                "lookback_future_pos_rate": safe_rate(
                    (sub["lookback_future_sum_90"] > 0).sum(), len(sub)
                ),
                "qfo_ge_25_rate": safe_rate((sub["qty_first_order"] >= 25).sum(), len(sub)),
                "mean_base_pred_qty": float(sub["base_pred_qty"].mean()),
                "mean_shadow_pred_qty": float(sub["shadow_pred_qty"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def build_positive_true_table(df):
    pos = df[df["true_replenish_qty"] > 0].copy()
    rows = []
    for stock_state, sub in pos.groupby("stock_state", dropna=False):
        true_sum = float(sub["true_replenish_qty"].sum())
        rows.append(
            {
                "stock_state": str(stock_state),
                "rows": int(len(sub)),
                "true_sum": true_sum,
                "base_under_wape": safe_rate(sub["base_under_gap"].sum(), true_sum),
                "shadow_under_wape": safe_rate(sub["shadow_under_gap"].sum(), true_sum),
                "under_wape_delta": safe_rate(sub["shadow_under_gap"].sum(), true_sum)
                - safe_rate(sub["base_under_gap"].sum(), true_sum),
                "event_strong_rate": safe_rate(sub["event_strong_30"].sum(), len(sub)),
                "mean_total_stock": float(sub["inv_total_stock"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)


def build_stock_constrained_candidates(df):
    zero = df[df["true_replenish_qty"] == 0].copy()
    zero["demand_risk_signal"] = (
        (zero["event_strong_30"] > 0)
        | (zero["lookback_repl_sum_90"] > 0)
        | (zero["lookback_future_sum_90"] > 0)
        | (zero["qty_first_order"] >= 25)
    ).astype(int)
    candidates = zero[
        (zero["stock_state"] == "no_snapshot")
        & (zero["demand_risk_signal"] == 1)
    ].copy()
    candidates = candidates.sort_values(
        ["shadow_pred_qty", "base_pred_qty", "qty_first_order", "lookback_repl_sum_90"],
        ascending=[False, False, False, False],
    )
    cols = [
        "anchor_date",
        "sku_id",
        "style_id",
        "product_name",
        "category_static",
        "signal_quadrant",
        "activity_bucket",
        "true_replenish_qty",
        "base_pred_qty",
        "shadow_pred_qty",
        "qty_first_order",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
        "event_strong_30",
        "event_daily_clicks_30",
        "event_daily_cart_adds_30",
        "event_daily_order_success_30",
        "event_daily_pay_success_30",
        "stock_state",
        "inv_total_stock",
        "snapshot_present",
        "stock_positive",
        "stock_zero",
    ]
    cols = [col for col in cols if col in candidates.columns]
    return candidates[cols].head(200).reset_index(drop=True)


def build_inventory_source_audit():
    inv = pd.read_csv(DATA_DIR / "inventory_daily_features.csv", encoding="utf-8-sig")
    for col in [
        "qty_storage_stock",
        "qty_b2b_hq_stock",
        "qty_total_stock",
        "snapshot_present",
        "stock_positive",
        "stock_zero",
    ]:
        if col in inv.columns:
            inv[col] = pd.to_numeric(inv[col], errors="coerce").fillna(0.0)
    inv["date"] = pd.to_datetime(inv["date"])

    wide = pd.read_csv(
        WIDE_PATH,
        usecols=lambda col: col
        in {
            "date",
            "qty_stock",
            "is_real_stock",
            "snapshot_present",
            "stock_positive",
            "stock_zero",
        },
        encoding="utf-8-sig",
    )
    wide["date"] = pd.to_datetime(wide["date"])
    wide = wide[wide["date"] >= pd.Timestamp("2026-01-23")].copy()
    for col in [
        "qty_stock",
        "is_real_stock",
        "snapshot_present",
        "stock_positive",
        "stock_zero",
    ]:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce").fillna(0.0)

    return {
        "inventory_daily_rows": int(len(inv)),
        "inventory_distinct_days": int(inv["date"].dt.date.nunique()),
        "inventory_dup_date_sku_rows": int(inv.duplicated(["date", "sku_id"]).sum()),
        "inventory_storage_zero_rows": int((inv["qty_storage_stock"] == 0).sum()),
        "inventory_b2b_zero_rows": int((inv["qty_b2b_hq_stock"] == 0).sum()),
        "inventory_snapshot_present_rows": int(inv.get("snapshot_present", pd.Series(dtype=float)).sum()),
        "inventory_stock_zero_rows": int(inv.get("stock_zero", pd.Series(dtype=float)).sum()),
        "wide_rows_after_20260123": int(len(wide)),
        "wide_qty_stock_gt_0_rate": safe_rate((wide["qty_stock"] > 0).sum(), len(wide)),
        "wide_is_real_stock_rate": safe_rate((wide["is_real_stock"] > 0).sum(), len(wide)),
        "wide_snapshot_present_rate": safe_rate(wide.get("snapshot_present", pd.Series(dtype=float)).sum(), len(wide)),
        "wide_stock_zero_rate": safe_rate(wide.get("stock_zero", pd.Series(dtype=float)).sum(), len(wide)),
    }


def build_state_note(zero_df, state_name):
    sub = zero_df.loc[zero_df["stock_state"] == state_name]
    if sub.empty:
        return ""
    row = sub.iloc[0]
    return (
        f"- `true=0` rows with `{state_name}`: `{int(row['rows'])}`, "
        f"`base_fp_rate={row['base_fp_rate']:.4f}`, "
        f"`shadow_fp_rate={row['shadow_fp_rate']:.4f}`."
    )


def write_summary(zero_df, pos_df, candidates, audit):
    lines = [
        "# Phase8 Inventory Constraint 2026 Summary",
        "",
        "- Status: `analysis_only`",
        "- Source: `phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_row_compare.csv`",
        "- Purpose: inspect whether the 2026 event+inventory shadow behaves differently across `no_snapshot`, `snapshot_zero_stock`, and `snapshot_positive_stock`.",
        "",
        "## Bottom Line",
        "",
        "- This pack does not change the official phase7 mainline.",
        "- Inventory states are now evaluated explicitly instead of inferring zero stock from presence flags.",
        "- The practical question is whether `snapshot_zero_stock` behaves differently from `no_snapshot` after the semantic fix.",
        "",
        "## Zero-True by Stock State",
        "",
        markdown_table(
            zero_df,
            [
                "stock_state",
                "rows",
                "base_fp_rate",
                "shadow_fp_rate",
                "fp_rate_delta",
                "event_strong_rate",
                "lookback_repl_pos_rate",
                "qfo_ge_25_rate",
            ],
        ),
        "",
        build_state_note(zero_df, "no_snapshot"),
        build_state_note(zero_df, "snapshot_zero_stock"),
        build_state_note(zero_df, "snapshot_positive_stock"),
        "",
        "## Positive-True Under-Predict by Stock State",
        "",
        markdown_table(
            pos_df,
            [
                "stock_state",
                "rows",
                "true_sum",
                "base_under_wape",
                "shadow_under_wape",
                "under_wape_delta",
                "event_strong_rate",
                "mean_total_stock",
            ],
        ),
        "",
        "## Inventory Source Audit",
        "",
        f"- inventory_daily_rows: `{audit['inventory_daily_rows']}`",
        f"- inventory_distinct_days: `{audit['inventory_distinct_days']}`",
        f"- duplicate `date+sku` rows in inventory_daily_features: `{audit['inventory_dup_date_sku_rows']}`",
        f"- inventory rows with `qty_storage_stock = 0`: `{audit['inventory_storage_zero_rows']}`",
        f"- inventory rows with `qty_b2b_hq_stock = 0`: `{audit['inventory_b2b_zero_rows']}`",
        f"- inventory rows with `snapshot_present = 1`: `{audit['inventory_snapshot_present_rows']}`",
        f"- inventory rows with `stock_zero = 1`: `{audit['inventory_stock_zero_rows']}`",
        f"- wide `qty_stock > 0` rate after `2026-01-23`: `{audit['wide_qty_stock_gt_0_rate']:.4f}`",
        f"- wide `is_real_stock > 0` rate after `2026-01-23`: `{audit['wide_is_real_stock_rate']:.4f}`",
        f"- wide `snapshot_present = 1` rate after `2026-01-23`: `{audit['wide_snapshot_present_rate']:.4f}`",
        f"- wide `stock_zero = 1` rate after `2026-01-23`: `{audit['wide_stock_zero_rate']:.4f}`",
        "",
        "## Interpretation",
        "",
        "- `snapshot_positive_stock` captures rows with explicit inventory support.",
        "- `snapshot_zero_stock` captures rows where a snapshot exists but total stock is zero or below.",
        "- `no_snapshot` now means missing snapshot evidence, not zero stock by default.",
        "",
        "## Practical Answer",
        "",
        "- If `snapshot_zero_stock` now shows a distinct error pattern from `no_snapshot`, the inventory line is carrying more than a pure positive-stock signal.",
        "- If the gains still sit almost entirely in `snapshot_positive_stock`, most of the lift is coming from positive-stock evidence.",
        f"- Candidate `true=0` demand-risk rows exported for follow-up: `{len(candidates)}`.",
        "",
        "## Output Files",
        "",
        "- `reports/phase8_extended_signal_2026/phase8_inventory_constraint_zero_true_table.csv`",
        "- `reports/phase8_extended_signal_2026/phase8_inventory_constraint_positive_true_table.csv`",
        "- `reports/phase8_extended_signal_2026/phase8_inventory_constraint_candidate_cases.csv`",
        "",
    ]
    (REPORT_DIR / "phase8_inventory_constraint_summary.md").write_text(
        "\n".join(line for line in lines if line is not None),
        encoding="utf-8-sig",
    )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_row_compare()
    zero_df = build_zero_true_table(df)
    pos_df = build_positive_true_table(df)
    candidates = build_stock_constrained_candidates(df)
    audit = build_inventory_source_audit()

    zero_df.to_csv(
        REPORT_DIR / "phase8_inventory_constraint_zero_true_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pos_df.to_csv(
        REPORT_DIR / "phase8_inventory_constraint_positive_true_table.csv",
        index=False,
        encoding="utf-8-sig",
    )
    candidates.to_csv(
        REPORT_DIR / "phase8_inventory_constraint_candidate_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "status": "analysis_only",
        "summary": "reports/phase8_extended_signal_2026/phase8_inventory_constraint_summary.md",
        "zero_true_table": "reports/phase8_extended_signal_2026/phase8_inventory_constraint_zero_true_table.csv",
        "positive_true_table": "reports/phase8_extended_signal_2026/phase8_inventory_constraint_positive_true_table.csv",
        "candidate_cases": "reports/phase8_extended_signal_2026/phase8_inventory_constraint_candidate_cases.csv",
    }
    (REPORT_DIR / "phase8_inventory_constraint_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_summary(zero_df, pos_df, candidates, audit)
    print(f"[OK] inventory constraint summary -> {REPORT_DIR / 'phase8_inventory_constraint_summary.md'}")


if __name__ == "__main__":
    main()
