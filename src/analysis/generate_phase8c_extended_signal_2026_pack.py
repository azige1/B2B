import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
PHASE8_DATA_DIR = DATA_DIR / "phase8a_prep"
OUT_DIR = PROJECT_ROOT / "reports" / "phase8_extended_signal_2026"


def safe_div(num, den):
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
            v = row[col]
            if isinstance(v, float):
                vals.append("" if np.isnan(v) else f"{v:.4f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([headers, sep, *rows])


def inventory_bucket(value):
    value = float(value)
    if value <= 0:
        return "0"
    if value <= 10:
        return "1-10"
    if value <= 50:
        return "11-50"
    return "50+"


def event_bucket(row):
    if float(row["event_daily_pay_success_30"]) > 0 or float(row["event_daily_order_success_30"]) > 0:
        return "order_or_pay_30"
    if float(row["event_daily_cart_adds_30"]) > 0:
        return "cart_30"
    if float(row["event_daily_view_order_30"]) > 0:
        return "view_30"
    if float(row["event_daily_clicks_30"]) > 0:
        return "click_only_30"
    return "no_event_30"


def group_metrics(df, group_col):
    rows = []
    for value, sub in df.groupby(group_col, dropna=False):
        rows.append(
            {
                group_col: str(value),
                "rows": int(len(sub)),
                "buyers": int(sub["buyer_id"].nunique()),
                "skus": int(sub["sku_id"].nunique()),
                "styles": int(sub["style_id"].nunique()),
                "replenish_positive_rate": safe_div((sub["qty_replenish"] > 0).sum(), len(sub)),
                "replenish_gt10_rate": safe_div((sub["qty_replenish"] > 10).sum(), len(sub)),
                "replenish_gt25_rate": safe_div((sub["qty_replenish"] > 25).sum(), len(sub)),
                "future_positive_rate": safe_div((sub["qty_future"] > 0).sum(), len(sub)),
                "mean_replenish_qty": float(sub["qty_replenish"].mean()),
                "p50_replenish_qty": float(sub["qty_replenish"].median()),
                "total_replenish_qty": float(sub["qty_replenish"].sum()),
                "total_future_qty": float(sub["qty_future"].sum()),
            }
        )
    return pd.DataFrame(rows)


def load_analysis_base():
    orders = pd.read_csv(SILVER_DIR / "clean_orders.csv", encoding="utf-8-sig")
    products = pd.read_csv(SILVER_DIR / "clean_products.csv", encoding="utf-8-sig")
    inventory = pd.read_csv(PHASE8_DATA_DIR / "inventory_daily_features.csv", encoding="utf-8-sig")
    preorder = pd.read_csv(PHASE8_DATA_DIR / "preorder_daily_features.csv", encoding="utf-8-sig")
    event = pd.read_csv(PHASE8_DATA_DIR / "event_intent_daily_features.csv", encoding="utf-8-sig")

    orders["order_date"] = pd.to_datetime(orders["order_date"])
    products["sku_id"] = products["sku_id"].astype(str)
    products["style_id"] = products["style_id"].astype(str)
    inventory["date"] = pd.to_datetime(inventory["date"])
    preorder["date"] = pd.to_datetime(preorder["date"])
    event["date"] = pd.to_datetime(event["date"])

    base = orders.merge(
        products[
            [
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
        ],
        on="sku_id",
        how="left",
    )
    base["style_id"] = base["style_id"].fillna("").astype(str)
    base = base[base["order_date"] >= pd.Timestamp("2026-01-23")].copy()

    inventory_join = inventory.rename(columns={"date": "order_date"})
    base = base.merge(
        inventory_join[
            [
                "order_date",
                "sku_id",
                "qty_storage_stock",
                "qty_b2b_hq_stock",
                "has_storage_snapshot",
                "has_b2b_snapshot",
            ]
        ],
        on=["order_date", "sku_id"],
        how="left",
    )

    preorder_join = preorder.copy()
    preorder_join["order_date"] = preorder_join["date"] + pd.Timedelta(days=1)
    preorder_cols = [
        "order_date",
        "buyer_id",
        "sku_id",
        "preorder_row_count",
        "preorder_qty_rem_sum",
        "preorder_qty_rem_pos_sum",
        "preorder_nonzero_row_count",
    ]
    base = base.merge(preorder_join[preorder_cols], on=["order_date", "buyer_id", "sku_id"], how="left")

    event_join = event.copy()
    event_join["order_date"] = event_join["date"] + pd.Timedelta(days=1)
    event_metric_cols = [
        "daily_clicks_7",
        "daily_clicks_14",
        "daily_clicks_30",
        "daily_view_order_7",
        "daily_view_order_14",
        "daily_view_order_30",
        "daily_cart_adds_7",
        "daily_cart_adds_14",
        "daily_cart_adds_30",
        "daily_order_success_7",
        "daily_order_success_14",
        "daily_order_success_30",
        "daily_pay_success_7",
        "daily_pay_success_14",
        "daily_pay_success_30",
        "daily_order_submit_qty_7",
        "daily_order_submit_qty_14",
        "daily_order_submit_qty_30",
        "daily_pay_qty_7",
        "daily_pay_qty_14",
        "daily_pay_qty_30",
        "days_since_last_any_event",
        "days_since_last_strong_intent",
        "click_to_cart_rate_30",
        "view_to_order_rate_30",
        "cart_to_order_rate_30",
        "order_to_pay_rate_30",
    ]
    event_join = event_join.rename(columns={col: f"event_{col}" for col in event_metric_cols})
    base = base.merge(
        event_join[["order_date", "buyer_id", "style_id", *[f"event_{c}" for c in event_metric_cols]]],
        on=["order_date", "buyer_id", "style_id"],
        how="left",
    )

    base["month"] = base["order_date"].dt.strftime("%Y-%m")
    base["inventory_match"] = base["qty_b2b_hq_stock"].notna().astype(int)
    base["preorder_match"] = base["preorder_row_count"].notna().astype(int)
    base["event_match"] = base["event_daily_clicks_30"].notna().astype(int)
    base["preorder_positive"] = (base["preorder_qty_rem_pos_sum"].fillna(0.0) > 0).astype(int)
    base["inventory_total_stock"] = base["qty_storage_stock"].fillna(0.0) + base["qty_b2b_hq_stock"].fillna(0.0)
    base["inventory_bucket"] = base["qty_b2b_hq_stock"].fillna(0.0).map(inventory_bucket)
    base["event_bucket"] = base.apply(event_bucket, axis=1)
    base["event_strong_30"] = (
        (base["event_daily_cart_adds_30"].fillna(0.0) > 0)
        | (base["event_daily_order_success_30"].fillna(0.0) > 0)
        | (base["event_daily_pay_success_30"].fillna(0.0) > 0)
    ).astype(int)
    base["joint_signal_bucket"] = np.where(
        base["event_strong_30"] == 1,
        np.where(base["preorder_positive"] == 1, "event_strong+preorder_pos", "event_strong_only"),
        np.where(base["preorder_positive"] == 1, "preorder_pos_only", "neither"),
    )
    return base


def write_summary(base, month_df, event_df, preorder_df, inventory_df, joint_df, focus_cases):
    top_event = event_df.sort_values("replenish_gt10_rate", ascending=False).head(3)
    top_joint = joint_df.sort_values("replenish_gt10_rate", ascending=False).head(4)
    coverage = {
        "rows": int(len(base)),
        "date_min": str(base["order_date"].min().date()) if not base.empty else "",
        "date_max": str(base["order_date"].max().date()) if not base.empty else "",
        "inventory_match_rate": safe_div(base["inventory_match"].sum(), len(base)),
        "preorder_match_rate": safe_div(base["preorder_match"].sum(), len(base)),
        "event_match_rate": safe_div(base["event_match"].sum(), len(base)),
    }
    lines = [
        "# Phase8 Extended Signal 2026 Summary",
        "",
        "## Scope",
        "",
        "- This is an exploratory 2026 analysis pack only.",
        "- It does not participate in official phase7 replacement.",
        "- Inventory is joined same-day on `sku_id + order_date`.",
        "- Event and preorder signals are joined as prior-day features to avoid same-day leakage in the analysis view.",
        "",
        "## Coverage",
        "",
        f"- rows: `{coverage['rows']}`",
        f"- order_date range: `{coverage['date_min']} ~ {coverage['date_max']}`",
        f"- inventory_match_rate: `{coverage['inventory_match_rate']:.4f}`",
        f"- preorder_match_rate: `{coverage['preorder_match_rate']:.4f}`",
        f"- event_match_rate: `{coverage['event_match_rate']:.4f}`",
        "",
        "## Monthly Coverage and Replenish Rates",
        "",
        markdown_table(
            month_df,
            [
                "month",
                "rows",
                "inventory_match_rate",
                "preorder_match_rate",
                "event_match_rate",
                "replenish_positive_rate",
                "replenish_gt10_rate",
                "replenish_gt25_rate",
            ],
        ),
        "",
        "## Event Bucket",
        "",
        markdown_table(
            event_df,
            [
                "event_bucket",
                "rows",
                "replenish_positive_rate",
                "replenish_gt10_rate",
                "replenish_gt25_rate",
                "mean_replenish_qty",
                "total_replenish_qty",
            ],
        ),
        "",
        "## Preorder Bucket",
        "",
        markdown_table(
            preorder_df,
            [
                "preorder_bucket",
                "rows",
                "replenish_positive_rate",
                "replenish_gt10_rate",
                "replenish_gt25_rate",
                "mean_replenish_qty",
                "total_replenish_qty",
            ],
        ),
        "",
        "## Inventory Bucket",
        "",
        markdown_table(
            inventory_df,
            [
                "inventory_bucket",
                "rows",
                "replenish_positive_rate",
                "replenish_gt10_rate",
                "replenish_gt25_rate",
                "mean_replenish_qty",
                "total_replenish_qty",
            ],
        ),
        "",
        "## Joint Signal Bucket",
        "",
        markdown_table(
            joint_df,
            [
                "joint_signal_bucket",
                "rows",
                "replenish_positive_rate",
                "replenish_gt10_rate",
                "replenish_gt25_rate",
                "mean_replenish_qty",
                "total_replenish_qty",
            ],
        ),
        "",
        "## Quick Read",
        "",
        f"- Highest event-led replenish>10 bucket: `{top_event.iloc[0]['event_bucket']}` with `replenish_gt10_rate={top_event.iloc[0]['replenish_gt10_rate']:.4f}`." if not top_event.empty else "- No event rows available.",
        f"- Strongest joint bucket: `{top_joint.iloc[0]['joint_signal_bucket']}` with `replenish_gt10_rate={top_joint.iloc[0]['replenish_gt10_rate']:.4f}`." if not top_joint.empty else "- No joint-signal rows available.",
        f"- Focus cases exported: `{len(focus_cases)}` rows.",
        "",
        "## Output Files",
        "",
        "- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_base.csv`",
        "- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_month_table.csv`",
        "- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_event_bucket_table.csv`",
        "- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_preorder_table.csv`",
        "- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_inventory_table.csv`",
        "- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_joint_signal_table.csv`",
        "- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_focus_cases.csv`",
        "",
    ]
    (OUT_DIR / "phase8_extended_signal_2026_summary.md").write_text("\n".join(lines), encoding="utf-8-sig")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_analysis_base()
    base.to_csv(OUT_DIR / "phase8_extended_signal_2026_base.csv", index=False, encoding="utf-8-sig")

    month_rows = []
    for month, sub in base.groupby("month", dropna=False):
        month_rows.append(
            {
                "month": str(month),
                "rows": int(len(sub)),
                "inventory_match_rate": safe_div(sub["inventory_match"].sum(), len(sub)),
                "preorder_match_rate": safe_div(sub["preorder_match"].sum(), len(sub)),
                "event_match_rate": safe_div(sub["event_match"].sum(), len(sub)),
                "replenish_positive_rate": safe_div((sub["qty_replenish"] > 0).sum(), len(sub)),
                "replenish_gt10_rate": safe_div((sub["qty_replenish"] > 10).sum(), len(sub)),
                "replenish_gt25_rate": safe_div((sub["qty_replenish"] > 25).sum(), len(sub)),
            }
        )
    month_df = pd.DataFrame(month_rows).sort_values("month").reset_index(drop=True)
    month_df.to_csv(OUT_DIR / "phase8_extended_signal_2026_month_table.csv", index=False, encoding="utf-8-sig")

    event_df = group_metrics(base, "event_bucket").sort_values(
        ["replenish_gt10_rate", "rows"], ascending=[False, False]
    ).reset_index(drop=True)
    event_df.to_csv(OUT_DIR / "phase8_extended_signal_2026_event_bucket_table.csv", index=False, encoding="utf-8-sig")

    preorder_base = base.copy()
    preorder_base["preorder_bucket"] = np.where(
        preorder_base["preorder_positive"] == 1,
        "preorder_pos",
        np.where(preorder_base["preorder_match"] == 1, "preorder_zero", "preorder_missing"),
    )
    preorder_df = group_metrics(preorder_base, "preorder_bucket").sort_values(
        ["replenish_gt10_rate", "rows"], ascending=[False, False]
    ).reset_index(drop=True)
    preorder_df.to_csv(OUT_DIR / "phase8_extended_signal_2026_preorder_table.csv", index=False, encoding="utf-8-sig")

    inventory_df = group_metrics(base, "inventory_bucket").sort_values(
        ["inventory_bucket"]
    ).reset_index(drop=True)
    inventory_df.to_csv(OUT_DIR / "phase8_extended_signal_2026_inventory_table.csv", index=False, encoding="utf-8-sig")

    joint_df = group_metrics(base, "joint_signal_bucket").sort_values(
        ["replenish_gt10_rate", "rows"], ascending=[False, False]
    ).reset_index(drop=True)
    joint_df.to_csv(OUT_DIR / "phase8_extended_signal_2026_joint_signal_table.csv", index=False, encoding="utf-8-sig")

    focus_cases = base[
        (base["qty_replenish"] >= 10)
        & (
            (base["event_strong_30"] == 1)
            | (base["preorder_positive"] == 1)
            | (base["inventory_total_stock"] <= 10)
        )
    ].copy()
    focus_cases = focus_cases.sort_values(
        ["qty_replenish", "event_daily_order_success_30", "preorder_qty_rem_pos_sum"],
        ascending=[False, False, False],
    ).head(100)
    focus_cols = [
        "order_date",
        "buyer_id",
        "sku_id",
        "style_id",
        "product_name",
        "category",
        "qty_replenish",
        "qty_future",
        "event_bucket",
        "event_strong_30",
        "event_daily_clicks_30",
        "event_daily_view_order_30",
        "event_daily_cart_adds_30",
        "event_daily_order_success_30",
        "event_daily_pay_success_30",
        "preorder_match",
        "preorder_positive",
        "preorder_qty_rem_sum",
        "preorder_qty_rem_pos_sum",
        "qty_b2b_hq_stock",
        "qty_storage_stock",
        "inventory_bucket",
    ]
    focus_cases[focus_cols].to_csv(
        OUT_DIR / "phase8_extended_signal_2026_focus_cases.csv",
        index=False,
        encoding="utf-8-sig",
    )

    manifest = {
        "status": "analysis_only",
        "date_range": {
            "order_min": str(base["order_date"].min().date()) if not base.empty else "",
            "order_max": str(base["order_date"].max().date()) if not base.empty else "",
        },
        "rows": int(len(base)),
        "outputs": {
            "summary": "reports/phase8_extended_signal_2026/phase8_extended_signal_2026_summary.md",
            "base": "reports/phase8_extended_signal_2026/phase8_extended_signal_2026_base.csv",
            "month_table": "reports/phase8_extended_signal_2026/phase8_extended_signal_2026_month_table.csv",
            "event_bucket_table": "reports/phase8_extended_signal_2026/phase8_extended_signal_2026_event_bucket_table.csv",
            "preorder_table": "reports/phase8_extended_signal_2026/phase8_extended_signal_2026_preorder_table.csv",
            "inventory_table": "reports/phase8_extended_signal_2026/phase8_extended_signal_2026_inventory_table.csv",
            "joint_signal_table": "reports/phase8_extended_signal_2026/phase8_extended_signal_2026_joint_signal_table.csv",
            "focus_cases": "reports/phase8_extended_signal_2026/phase8_extended_signal_2026_focus_cases.csv",
        },
    }
    (OUT_DIR / "phase8_extended_signal_2026_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_summary(base, month_df, event_df, preorder_df, inventory_df, joint_df, focus_cases)
    print(f"[OK] extended 2026 signal summary -> {OUT_DIR / 'phase8_extended_signal_2026_summary.md'}")


if __name__ == "__main__":
    main()
