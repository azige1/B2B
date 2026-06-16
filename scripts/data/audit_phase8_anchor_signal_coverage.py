import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_ROOT = PROJECT_ROOT / "data" / "phase8_20260614"
SIGNAL_DIR = SNAPSHOT_ROOT / "signals"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase8_data_audit"

ANCHORS = [
    "2025-09-01",
    "2025-10-01",
    "2025-11-01",
    "2025-12-01",
    "2026-01-01",
    "2026-01-15",
    "2026-02-01",
    "2026-02-15",
    "2026-03-01",
    "2026-03-15",
    "2026-04-01",
    "2026-04-15",
    "2026-05-01",
    "2026-05-14",
]


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def active_entities(frame, entity_col, anchor, window):
    start = anchor - pd.Timedelta(days=window - 1)
    return set(
        frame.loc[
            frame["date"].between(start, anchor, inclusive="both"),
            entity_col,
        ].astype(str)
    )


def coverage_count(universe, active):
    return int(universe.isin(active).sum())


def main():
    gold = pd.read_csv(
        SNAPSHOT_ROOT / "gold" / "wide_table_sku.csv",
        usecols=["date", "sku_id", "style_id", "qty_replenish"],
        parse_dates=["date"],
        dtype={"sku_id": str, "style_id": str},
    )
    gold["qty_replenish"] = pd.to_numeric(
        gold["qty_replenish"], errors="coerce"
    ).fillna(0.0)

    sku_totals = gold.groupby("sku_id")["qty_replenish"].sum()
    keep_skus = set(sku_totals[sku_totals > 0].index.astype(str))
    sku_static = (
        gold.loc[gold["sku_id"].isin(keep_skus), ["sku_id", "style_id"]]
        .drop_duplicates("sku_id")
        .sort_values("sku_id")
        .reset_index(drop=True)
    )
    sku_static["sku_id"] = sku_static["sku_id"].astype(str)
    sku_static["style_id"] = sku_static["style_id"].astype(str)

    target_daily = (
        gold.loc[gold["sku_id"].isin(keep_skus)]
        .groupby(["date", "sku_id"], as_index=False)["qty_replenish"]
        .sum()
    )

    listing = pd.read_csv(
        SIGNAL_DIR / "lifecycle_launch_date_features.csv",
        usecols=["sku_id", "launch_date"],
        dtype={"sku_id": str},
        parse_dates=["launch_date"],
    ).drop_duplicates("sku_id")
    listing_map = listing.set_index("sku_id")["launch_date"]
    sku_static["launch_date"] = sku_static["sku_id"].map(listing_map)

    event = pd.read_csv(
        SIGNAL_DIR / "event_intent_daily_features.csv",
        usecols=["date", "style_id"],
        dtype={"style_id": str},
        parse_dates=["date"],
    ).drop_duplicates(["date", "style_id"])

    inventory = pd.read_csv(
        SIGNAL_DIR / "inventory_daily_features.csv",
        usecols=["date", "sku_id", "snapshot_present", "stock_zero"],
        dtype={"sku_id": str},
        parse_dates=["date"],
    )
    inventory["snapshot_present"] = pd.to_numeric(
        inventory["snapshot_present"], errors="coerce"
    ).fillna(0.0)
    inventory["stock_zero"] = pd.to_numeric(
        inventory["stock_zero"], errors="coerce"
    ).fillna(0.0)

    request = pd.read_csv(
        SIGNAL_DIR / "purchase_request_daily_features.csv",
        usecols=["date", "sku_id", "style_id"],
        dtype={"sku_id": str, "style_id": str},
        parse_dates=["date"],
    )

    rows = []
    for anchor_text in ANCHORS:
        anchor = pd.Timestamp(anchor_text)
        target_start = anchor + pd.Timedelta(days=1)
        target_end = anchor + pd.Timedelta(days=30)
        target = (
            target_daily.loc[
                target_daily["date"].between(
                    target_start,
                    target_end,
                    inclusive="both",
                )
            ]
            .groupby("sku_id")["qty_replenish"]
            .sum()
        )

        universe = sku_static.copy()
        universe["target_qty"] = universe["sku_id"].map(target).fillna(0.0)
        positive = universe["target_qty"] > 0
        positive_count = int(positive.sum())
        total_count = len(universe)

        launch_known = universe["launch_date"].notna()
        launch_eligible = launch_known & (universe["launch_date"] <= anchor)
        launch_future = launch_known & (universe["launch_date"] > anchor)
        listing_default = universe["launch_date"].eq(pd.Timestamp("2024-12-12"))

        event_style_30 = active_entities(event, "style_id", anchor, 30)
        event_style_all = set(
            event.loc[event["date"] <= anchor, "style_id"].astype(str)
        )
        event_30_count = coverage_count(universe["style_id"], event_style_30)
        event_all_count = coverage_count(universe["style_id"], event_style_all)
        event_30_positive = coverage_count(
            universe.loc[positive, "style_id"],
            event_style_30,
        )

        inventory_sku_30 = active_entities(inventory, "sku_id", anchor, 30)
        inventory_sku_7 = active_entities(inventory, "sku_id", anchor, 7)
        inventory_today = set(
            inventory.loc[
                (inventory["date"] == anchor)
                & (inventory["snapshot_present"] > 0),
                "sku_id",
            ].astype(str)
        )
        inventory_zero_today = set(
            inventory.loc[
                (inventory["date"] == anchor)
                & (inventory["stock_zero"] > 0),
                "sku_id",
            ].astype(str)
        )
        inventory_today_count = coverage_count(
            universe["sku_id"],
            inventory_today,
        )
        inventory_today_positive = coverage_count(
            universe.loc[positive, "sku_id"],
            inventory_today,
        )

        request_sku_30 = active_entities(request, "sku_id", anchor, 30)
        request_sku_90 = active_entities(request, "sku_id", anchor, 90)
        request_style_90 = active_entities(request, "style_id", anchor, 90)
        request_any_90 = universe["sku_id"].isin(request_sku_90) | universe[
            "style_id"
        ].isin(request_style_90)
        request_any_90_positive = int((request_any_90 & positive).sum())

        last_request_date = request.loc[request["date"] <= anchor, "date"].max()
        request_lag = (
            int((anchor - last_request_date).days)
            if pd.notna(last_request_date)
            else None
        )

        rows.append(
            {
                "anchor_date": anchor.date().isoformat(),
                "model_sku_universe": total_count,
                "positive_target_skus": positive_count,
                "positive_target_rate": ratio(positive_count, total_count),
                "target_qty_30d": float(universe["target_qty"].sum()),
                "listing_known_rate": ratio(int(launch_known.sum()), total_count),
                "listing_eligible_rate": ratio(
                    int(launch_eligible.sum()), total_count
                ),
                "prelaunch_sku_count": int(launch_future.sum()),
                "listing_default_date_rate": ratio(
                    int(listing_default.sum()), total_count
                ),
                "event_any_history_rate": ratio(event_all_count, total_count),
                "event_30d_rate": ratio(event_30_count, total_count),
                "event_30d_positive_target_rate": ratio(
                    event_30_positive,
                    positive_count,
                ),
                "inventory_today_rate": ratio(
                    inventory_today_count,
                    total_count,
                ),
                "inventory_today_positive_target_rate": ratio(
                    inventory_today_positive,
                    positive_count,
                ),
                "inventory_7d_rate": ratio(
                    coverage_count(universe["sku_id"], inventory_sku_7),
                    total_count,
                ),
                "inventory_30d_rate": ratio(
                    coverage_count(universe["sku_id"], inventory_sku_30),
                    total_count,
                ),
                "inventory_zero_today_rate": ratio(
                    coverage_count(
                        universe["sku_id"],
                        inventory_zero_today,
                    ),
                    total_count,
                ),
                "request_sku_30d_rate": ratio(
                    coverage_count(universe["sku_id"], request_sku_30),
                    total_count,
                ),
                "request_sku_90d_rate": ratio(
                    coverage_count(universe["sku_id"], request_sku_90),
                    total_count,
                ),
                "request_sku_or_style_90d_rate": ratio(
                    int(request_any_90.sum()),
                    total_count,
                ),
                "request_90d_positive_target_rate": ratio(
                    request_any_90_positive,
                    positive_count,
                ),
                "request_last_observed_date": (
                    last_request_date.date().isoformat()
                    if pd.notna(last_request_date)
                    else ""
                ),
                "request_data_lag_days": request_lag,
            }
        )

    result = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT_DIR / "phase8_anchor_signal_coverage_20260614.csv"
    json_path = REPORT_DIR / "phase8_anchor_signal_coverage_20260614.json"
    md_path = REPORT_DIR / "phase8_anchor_signal_coverage_20260614.md"
    result.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    table_cols = [
        "anchor_date",
        "positive_target_rate",
        "target_qty_30d",
        "event_30d_rate",
        "inventory_today_rate",
        "request_sku_or_style_90d_rate",
        "request_data_lag_days",
        "prelaunch_sku_count",
    ]
    display = result[table_cols].copy()
    for column in [
        "positive_target_rate",
        "event_30d_rate",
        "inventory_today_rate",
        "request_sku_or_style_90d_rate",
    ]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    display["target_qty_30d"] = display["target_qty_30d"].round(1)

    lines = [
        "# Phase8 锚点信号覆盖审计",
        "",
        "口径：模型 SKU 宇宙为最新隔离 gold 中历史补货总量大于 0 的 SKU；"
        "所有窗口只使用锚点日及之前的数据，标签为锚点后 1 至 30 天。",
        "",
        display.to_markdown(index=False),
        "",
        "## 解释边界",
        "",
        "- Event 从 2025-09-18 开始，2025-09-01 锚点完全不可用；"
        "早期锚点只能视为覆盖爬坡期。",
        "- Inventory 从 2026-01-23 开始，2026-02-15 以前没有完整 30 天历史。",
        "- 求购文件截至 2026-04-21，之后的无记录不能直接解释为无需求；"
        "必须结合数据滞后天数。",
        "- `prelaunch_sku_count` 用于识别历史商品宇宙污染。未上市 SKU 不应作为"
        "正式补货评估对象，也不应贡献训练负样本。",
        "- 2024-12-12 是上市日期高频默认/迁移日期，相关切片必须单独报告。",
        "",
        f"- CSV: `{csv_path.relative_to(PROJECT_ROOT)}`",
        f"- JSON: `{json_path.relative_to(PROJECT_ROOT)}`",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result.to_string(index=False))
    print(f"[OK] {md_path}")


if __name__ == "__main__":
    main()
