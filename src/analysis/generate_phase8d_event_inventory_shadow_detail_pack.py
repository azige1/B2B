import os

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8_event_inventory_shadow_2026")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

ANCHORS = ["2026-02-15", "2026-02-24"]
WEAK_QUADRANTS = {"repl0_fut0", "repl1_fut0"}

OUT_ROW_COMPARE = os.path.join(OUT_DIR, "phase8_event_inventory_shadow_row_compare.csv")
OUT_CASES = os.path.join(OUT_DIR, "phase8_event_inventory_shadow_focus_cases.csv")
OUT_WEAK = os.path.join(OUT_DIR, "phase8_event_inventory_shadow_weak_signal_table.csv")
OUT_ZERO = os.path.join(OUT_DIR, "phase8_event_inventory_shadow_zero_true_table.csv")
OUT_CATEGORY = os.path.join(OUT_DIR, "phase8_event_inventory_shadow_category_delta_table.csv")
OUT_SUMMARY = os.path.join(OUT_DIR, "phase8_event_inventory_shadow_detail_summary.md")


def safe_div(num, den):
    den = float(den)
    if den <= 0:
        return np.nan
    return float(num) / den


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


def load_static_products():
    path = os.path.join(DATA_DIR, "silver", "clean_products.csv")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["sku_id"] = df["sku_id"].astype(str)
    keep_cols = [
        "sku_id",
        "style_id",
        "product_name",
        "category",
        "sub_category",
        "season",
        "series",
        "band",
        "qty_first_order",
    ]
    return df[keep_cols].drop_duplicates("sku_id").copy()


def load_event_style_daily():
    path = os.path.join(DATA_DIR, "phase8a_prep", "event_intent_daily_features.csv")
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["style_id"] = df["style_id"].astype(str)
    df["buyer_id"] = df["buyer_id"].astype(str)
    metrics = [
        "daily_clicks_30",
        "daily_view_order_30",
        "daily_cart_adds_30",
        "daily_order_success_30",
        "daily_pay_success_30",
        "daily_order_submit_qty_30",
        "daily_pay_qty_30",
    ]
    grouped = (
        df.groupby(["date", "style_id"], as_index=False)
        .agg(
            event_active_buyer_count_30=("buyer_id", "nunique"),
            **{f"event_{col}": (col, "sum") for col in metrics},
            event_days_since_last_any=("days_since_last_any_event", "min"),
            event_days_since_last_strong=("days_since_last_strong_intent", "min"),
            event_click_to_cart_rate_30=("click_to_cart_rate_30", "mean"),
            event_view_to_order_rate_30=("view_to_order_rate_30", "mean"),
            event_cart_to_order_rate_30=("cart_to_order_rate_30", "mean"),
            event_order_to_pay_rate_30=("order_to_pay_rate_30", "mean"),
        )
    )
    return grouped


def load_inventory_daily():
    path = os.path.join(DATA_DIR, "phase8a_prep", "inventory_daily_features.csv")
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"])
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["sku_id"] = df["sku_id"].astype(str)
    # Match the shadow builder's effective behavior: repeated date+sku rows overwrite earlier values.
    df = df.drop_duplicates(["date", "sku_id"], keep="last").copy()
    df["inv_total_stock"] = df["qty_storage_stock"].fillna(0.0) + df["qty_b2b_hq_stock"].fillna(0.0)
    df["inv_b2b_hq_stock_log1p"] = np.log1p(df["qty_b2b_hq_stock"].fillna(0.0).clip(lower=0.0))
    df["inv_total_stock_log1p"] = np.log1p(df["inv_total_stock"].clip(lower=0.0))
    df["inv_stock_bucket"] = pd.cut(
        df["qty_b2b_hq_stock"].fillna(0.0),
        bins=[-1e-9, 0.0, 10.0, 50.0, np.inf],
        labels=["0", "1-10", "11-50", "50+"],
    ).astype(str)
    keep_cols = [
        "date",
        "sku_id",
        "qty_storage_stock",
        "qty_b2b_hq_stock",
        "inv_total_stock",
        "inv_b2b_hq_stock_log1p",
        "inv_total_stock_log1p",
        "has_storage_snapshot",
        "has_b2b_snapshot",
        "inv_stock_bucket",
    ]
    return df[keep_cols].copy()


def eval_context_path(anchor_date, variant_key):
    tag = anchor_date.replace("-", "")
    exp_id = f"p8ei_{tag}_{variant_key}_s2028_hard_g027"
    return os.path.join(OUT_DIR, tag, "phase5", f"eval_context_{exp_id}.csv")


def load_variant_frame(anchor_date, variant_key):
    df = pd.read_csv(eval_context_path(anchor_date, variant_key))
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.strftime("%Y-%m-%d")
    df["sku_id"] = df["sku_id"].astype(str)
    df["style_id"] = df["style_id"].astype(str)
    return df


def load_compare_rows():
    static_df = load_static_products()
    event_df = load_event_style_daily()
    inventory_df = load_inventory_daily()

    frames = []
    for anchor_date in ANCHORS:
        base = load_variant_frame(anchor_date, "base_tail_full").rename(
            columns={
                "ai_pred_prob": "base_pred_prob",
                "ai_pred_qty_open": "base_pred_qty_open",
                "ai_pred_qty": "base_pred_qty",
                "ai_pred_positive_qty": "base_pred_positive",
                "qty_gate_mask": "base_qty_gate_mask",
                "dead_blocked": "base_dead_blocked",
                "abs_error": "base_abs_error",
            }
        )
        shadow = load_variant_frame(anchor_date, "event_inventory_plus").rename(
            columns={
                "ai_pred_prob": "shadow_pred_prob",
                "ai_pred_qty_open": "shadow_pred_qty_open",
                "ai_pred_qty": "shadow_pred_qty",
                "ai_pred_positive_qty": "shadow_pred_positive",
                "qty_gate_mask": "shadow_qty_gate_mask",
                "dead_blocked": "shadow_dead_blocked",
                "abs_error": "shadow_abs_error",
            }
        )
        merge_cols = [
            "sku_id",
            "anchor_date",
            "true_replenish_qty",
        ]
        shadow_cols = [
            "sku_id",
            "anchor_date",
            "true_replenish_qty",
            "shadow_pred_prob",
            "shadow_pred_qty_open",
            "shadow_pred_qty",
            "shadow_pred_positive",
            "shadow_qty_gate_mask",
            "shadow_dead_blocked",
            "shadow_abs_error",
        ]
        merged = base.merge(shadow[shadow_cols], on=merge_cols, how="inner")
        merged = merged.merge(static_df, on="sku_id", how="left", suffixes=("", "_static"))
        merged = merged.merge(
            event_df,
            left_on=["anchor_date", "style_id"],
            right_on=["date", "style_id"],
            how="left",
        ).drop(columns=["date"])
        merged = merged.merge(
            inventory_df,
            left_on=["anchor_date", "sku_id"],
            right_on=["date", "sku_id"],
            how="left",
        ).drop(columns=["date"])
        frames.append(merged)

    row_df = pd.concat(frames, ignore_index=True, sort=False)
    row_df["pred_delta"] = row_df["shadow_pred_qty"] - row_df["base_pred_qty"]
    row_df["abs_error_delta"] = row_df["shadow_abs_error"] - row_df["base_abs_error"]
    row_df["abs_error_gain"] = row_df["base_abs_error"] - row_df["shadow_abs_error"]
    row_df["ratio_base"] = row_df.apply(lambda r: safe_div(r["base_pred_qty"], r["true_replenish_qty"]), axis=1)
    row_df["ratio_shadow"] = row_df.apply(lambda r: safe_div(r["shadow_pred_qty"], r["true_replenish_qty"]), axis=1)
    row_df["is_true_blockbuster"] = (row_df["true_replenish_qty"].astype(float) > 25).astype(int)
    row_df["is_zero_true"] = (row_df["true_replenish_qty"].astype(float) <= 0).astype(int)
    row_df["base_zero_true_fp"] = (
        (row_df["true_replenish_qty"].astype(float) <= 0) & (row_df["base_pred_qty"].astype(float) > 0)
    ).astype(int)
    row_df["shadow_zero_true_fp"] = (
        (row_df["true_replenish_qty"].astype(float) <= 0) & (row_df["shadow_pred_qty"].astype(float) > 0)
    ).astype(int)
    row_df["event_strong_30"] = (
        (row_df["event_daily_cart_adds_30"].fillna(0.0) > 0)
        | (row_df["event_daily_order_success_30"].fillna(0.0) > 0)
        | (row_df["event_daily_pay_success_30"].fillna(0.0) > 0)
    ).astype(int)
    return row_df


def slice_metrics(sub, pred_col):
    true_sum = float(sub["true_replenish_qty"].sum())
    pred_sum = float(sub[pred_col].sum())
    under_sum = float(np.clip(sub["true_replenish_qty"].astype(float) - sub[pred_col].astype(float), a_min=0, a_max=None).sum())
    over_sum = float(np.clip(sub[pred_col].astype(float) - sub["true_replenish_qty"].astype(float), a_min=0, a_max=None).sum())
    return {
        "rows": int(len(sub)),
        "true_sum": true_sum,
        "pred_sum": pred_sum,
        "ratio": safe_div(pred_sum, true_sum),
        "under_wape": safe_div(under_sum, true_sum),
        "over_wape": safe_div(over_sum, true_sum),
    }


def build_weak_signal_table(row_df):
    weak = row_df[
        (row_df["is_true_blockbuster"] == 1) & (row_df["signal_quadrant"].isin(WEAK_QUADRANTS))
    ].copy()
    rows = []
    for (anchor_date, quadrant), sub in weak.groupby(["anchor_date", "signal_quadrant"], dropna=False):
        base = slice_metrics(sub, "base_pred_qty")
        shadow = slice_metrics(sub, "shadow_pred_qty")
        rows.append(
            {
                "anchor_date": anchor_date,
                "signal_quadrant": quadrant,
                "rows_base": base["rows"],
                "rows_shadow": shadow["rows"],
                "true_sum": base["true_sum"],
                "pred_sum_base": base["pred_sum"],
                "pred_sum_shadow": shadow["pred_sum"],
                "ratio_base": base["ratio"],
                "ratio_shadow": shadow["ratio"],
                "under_wape_base": base["under_wape"],
                "under_wape_shadow": shadow["under_wape"],
                "under_wape_delta": shadow["under_wape"] - base["under_wape"],
            }
        )
    return pd.DataFrame(rows).sort_values(["anchor_date", "under_wape_delta", "true_sum"], ascending=[True, True, False])


def build_zero_true_table(row_df):
    rows = []
    for anchor_date, sub in row_df.groupby("anchor_date", dropna=False):
        zero = sub[sub["is_zero_true"] == 1].copy()
        rows.append(
            {
                "anchor_date": anchor_date,
                "zero_true_rows": int(len(zero)),
                "false_positive_rows_base": int(zero["base_zero_true_fp"].sum()),
                "false_positive_rows_shadow": int(zero["shadow_zero_true_fp"].sum()),
                "false_positive_rate_base": safe_div(zero["base_zero_true_fp"].sum(), len(zero)),
                "false_positive_rate_shadow": safe_div(zero["shadow_zero_true_fp"].sum(), len(zero)),
                "false_positive_rate_delta": safe_div(zero["shadow_zero_true_fp"].sum(), len(zero))
                - safe_div(zero["base_zero_true_fp"].sum(), len(zero)),
                "pred_sum_on_zero_true_base": float(zero.loc[zero["base_zero_true_fp"] == 1, "base_pred_qty"].sum()),
                "pred_sum_on_zero_true_shadow": float(zero.loc[zero["shadow_zero_true_fp"] == 1, "shadow_pred_qty"].sum()),
                "pred_ge_5_rows_base": int((zero["base_pred_qty"].astype(float) >= 5).sum()),
                "pred_ge_5_rows_shadow": int((zero["shadow_pred_qty"].astype(float) >= 5).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("anchor_date").reset_index(drop=True)


def build_category_delta(row_df):
    rows = []
    for anchor_date, anchor_df in row_df.groupby("anchor_date", dropna=False):
        for slice_name, mask in {
            "positive_all": anchor_df["true_replenish_qty"].astype(float) > 0,
            "blockbuster": anchor_df["true_replenish_qty"].astype(float) > 25,
        }.items():
            sub = anchor_df[mask].copy()
            if sub.empty:
                continue
            for category, cat_df in sub.groupby("category", dropna=False):
                base = slice_metrics(cat_df, "base_pred_qty")
                shadow = slice_metrics(cat_df, "shadow_pred_qty")
                rows.append(
                    {
                        "anchor_date": anchor_date,
                        "slice_name": slice_name,
                        "category": str(category),
                        "rows": base["rows"],
                        "true_sum": base["true_sum"],
                        "pred_sum_base": base["pred_sum"],
                        "pred_sum_shadow": shadow["pred_sum"],
                        "ratio_base": base["ratio"],
                        "ratio_shadow": shadow["ratio"],
                        "under_wape_base": base["under_wape"],
                        "under_wape_shadow": shadow["under_wape"],
                        "over_wape_base": base["over_wape"],
                        "over_wape_shadow": shadow["over_wape"],
                        "under_wape_delta": shadow["under_wape"] - base["under_wape"],
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["anchor_date", "slice_name", "under_wape_delta", "true_sum"],
        ascending=[True, True, True, False],
    ).reset_index(drop=True)


def build_focus_cases(row_df):
    frames = []

    weak = row_df[
        (row_df["is_true_blockbuster"] == 1) & (row_df["signal_quadrant"].isin(WEAK_QUADRANTS))
    ].copy()
    if not weak.empty:
        improved = weak.sort_values(["abs_error_gain", "true_replenish_qty"], ascending=[False, False]).head(30).copy()
        improved["case_type"] = "weak_signal_blockbuster_improved"
        worsened = weak.sort_values(["abs_error_gain", "true_replenish_qty"], ascending=[True, False]).head(20).copy()
        worsened["case_type"] = "weak_signal_blockbuster_worsened"
        frames.extend([improved, worsened])

    zero = row_df[row_df["is_zero_true"] == 1].copy()
    if not zero.empty:
        zero_reduced = zero.sort_values(["pred_delta", "base_pred_qty"], ascending=[True, False]).head(30).copy()
        zero_reduced["case_type"] = "zero_true_fp_reduced"
        zero_worsened = zero.sort_values(["pred_delta", "shadow_pred_qty"], ascending=[False, False]).head(20).copy()
        zero_worsened["case_type"] = "zero_true_fp_worsened"
        frames.extend([zero_reduced, zero_worsened])

    block = row_df[row_df["is_true_blockbuster"] == 1].copy()
    if not block.empty:
        block_improved = block.sort_values(["abs_error_gain", "true_replenish_qty"], ascending=[False, False]).head(30).copy()
        block_improved["case_type"] = "blockbuster_improved"
        frames.append(block_improved)

    cases = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    keep_cols = [
        "case_type",
        "anchor_date",
        "sku_id",
        "product_name",
        "category",
        "sub_category",
        "style_id",
        "signal_quadrant",
        "activity_bucket",
        "true_replenish_qty",
        "base_pred_prob",
        "base_pred_qty",
        "ratio_base",
        "base_abs_error",
        "shadow_pred_prob",
        "shadow_pred_qty",
        "ratio_shadow",
        "shadow_abs_error",
        "pred_delta",
        "abs_error_gain",
        "qty_first_order",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
        "event_active_buyer_count_30",
        "event_daily_clicks_30",
        "event_daily_view_order_30",
        "event_daily_cart_adds_30",
        "event_daily_order_success_30",
        "event_daily_pay_success_30",
        "event_days_since_last_any",
        "event_days_since_last_strong",
        "qty_b2b_hq_stock",
        "qty_storage_stock",
        "inv_total_stock",
        "inv_stock_bucket",
    ]
    return cases[keep_cols].reset_index(drop=True) if not cases.empty else cases


def build_summary(weak_df, zero_df, category_df, cases_df):
    weak_best = weak_df.sort_values(["under_wape_delta", "true_sum"], ascending=[True, False]).head(8)
    weak_worst = weak_df.sort_values(["under_wape_delta", "true_sum"], ascending=[False, False]).head(6)
    zero_table = zero_df.copy()

    block_cat = category_df[category_df["slice_name"] == "blockbuster"].copy()
    block_cat_focus = block_cat[block_cat["rows"] >= 3].copy()
    cat_best = block_cat_focus.sort_values(["anchor_date", "under_wape_delta", "true_sum"], ascending=[True, True, False]).groupby("anchor_date").head(6)
    cat_worst = block_cat_focus.sort_values(["anchor_date", "under_wape_delta", "true_sum"], ascending=[True, False, False]).groupby("anchor_date").head(4)

    improved_cases = cases_df[cases_df["case_type"] == "weak_signal_blockbuster_improved"].head(10)
    worsened_cases = cases_df[cases_df["case_type"] == "weak_signal_blockbuster_worsened"].head(8)

    lines = [
        "# Phase8 Event+Inventory Shadow 2026 Detail Summary",
        "",
        "- Status: `analysis_only_shadow`",
        "- Scope: `2026-02-15 / 2026-02-24`",
        "- Purpose: inspect where event+inventory helps or hurts relative to the 2026 base line.",
        "- This detail pack does not participate in official phase7 replacement.",
        "",
        "## Weak-Signal Blockbuster Slice",
        "",
        markdown_table(
            weak_best,
            [
                "anchor_date",
                "signal_quadrant",
                "true_sum",
                "pred_sum_base",
                "pred_sum_shadow",
                "ratio_base",
                "ratio_shadow",
                "under_wape_base",
                "under_wape_shadow",
                "under_wape_delta",
            ],
        ) if not weak_best.empty else "No weak-signal blockbuster rows.",
        "",
        "## Weak-Signal Slice Regressions",
        "",
        markdown_table(
            weak_worst,
            [
                "anchor_date",
                "signal_quadrant",
                "true_sum",
                "pred_sum_base",
                "pred_sum_shadow",
                "ratio_base",
                "ratio_shadow",
                "under_wape_base",
                "under_wape_shadow",
                "under_wape_delta",
            ],
        ) if not weak_worst.empty else "No weak-signal regressions.",
        "",
        "## Zero-True False Positive",
        "",
        markdown_table(
            zero_table,
            [
                "anchor_date",
                "zero_true_rows",
                "false_positive_rows_base",
                "false_positive_rate_base",
                "false_positive_rows_shadow",
                "false_positive_rate_shadow",
                "false_positive_rate_delta",
                "pred_ge_5_rows_base",
                "pred_ge_5_rows_shadow",
            ],
        ) if not zero_table.empty else "No zero-true rows.",
        "",
        "## Blockbuster Category Improvements",
        "",
        markdown_table(
            cat_best,
            [
                "anchor_date",
                "category",
                "rows",
                "true_sum",
                "pred_sum_base",
                "pred_sum_shadow",
                "under_wape_base",
                "under_wape_shadow",
                "under_wape_delta",
            ],
        ) if not cat_best.empty else "No blockbuster category improvements.",
        "",
        "## Blockbuster Category Regressions",
        "",
        markdown_table(
            cat_worst,
            [
                "anchor_date",
                "category",
                "rows",
                "true_sum",
                "pred_sum_base",
                "pred_sum_shadow",
                "under_wape_base",
                "under_wape_shadow",
                "under_wape_delta",
            ],
        ) if not cat_worst.empty else "No blockbuster category regressions.",
        "",
        "## Top Improved Weak-Signal Cases",
        "",
        markdown_table(
            improved_cases,
            [
                "anchor_date",
                "sku_id",
                "category",
                "signal_quadrant",
                "true_replenish_qty",
                "base_pred_qty",
                "shadow_pred_qty",
                "event_daily_order_success_30",
                "qty_b2b_hq_stock",
                "abs_error_gain",
            ],
        ) if not improved_cases.empty else "No improved weak-signal cases.",
        "",
        "## Top Worsened Weak-Signal Cases",
        "",
        markdown_table(
            worsened_cases,
            [
                "anchor_date",
                "sku_id",
                "category",
                "signal_quadrant",
                "true_replenish_qty",
                "base_pred_qty",
                "shadow_pred_qty",
                "event_daily_order_success_30",
                "qty_b2b_hq_stock",
                "abs_error_gain",
            ],
        ) if not worsened_cases.empty else "No worsened weak-signal cases.",
        "",
        "## Output Files",
        "",
        "- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_row_compare.csv`",
        "- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_focus_cases.csv`",
        "- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_weak_signal_table.csv`",
        "- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_zero_true_table.csv`",
        "- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_category_delta_table.csv`",
        "",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    row_df = load_compare_rows()
    row_df.to_csv(OUT_ROW_COMPARE, index=False, encoding="utf-8-sig")

    weak_df = build_weak_signal_table(row_df)
    zero_df = build_zero_true_table(row_df)
    category_df = build_category_delta(row_df)
    cases_df = build_focus_cases(row_df)

    weak_df.to_csv(OUT_WEAK, index=False, encoding="utf-8-sig")
    zero_df.to_csv(OUT_ZERO, index=False, encoding="utf-8-sig")
    category_df.to_csv(OUT_CATEGORY, index=False, encoding="utf-8-sig")
    cases_df.to_csv(OUT_CASES, index=False, encoding="utf-8-sig")

    with open(OUT_SUMMARY, "w", encoding="utf-8-sig") as fh:
        fh.write(build_summary(weak_df, zero_df, category_df, cases_df))

    print(f"[OK] row compare -> {OUT_ROW_COMPARE}")
    print(f"[OK] focus cases -> {OUT_CASES}")
    print(f"[OK] weak signal table -> {OUT_WEAK}")
    print(f"[OK] zero true table -> {OUT_ZERO}")
    print(f"[OK] category delta -> {OUT_CATEGORY}")
    print(f"[OK] detail summary -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
