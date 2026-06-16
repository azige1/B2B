import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = PROJECT_ROOT / "data_warehouse"


EVENT_METRICS = {
    "\u5546\u54c1\u70b9\u51fb": "daily_clicks",
    "\u67e5\u770b\u4e0b\u5355": "daily_view_order",
    "\u52a0\u8d2d\u7269\u8f66": "daily_cart_adds",
    "\u4e0b\u5355\u6210\u529f": "daily_order_success",
    "\u652f\u4ed8\u6210\u529f": "daily_pay_success",
}
SOURCE_CLICK_METRICS = {
    "\u5546\u54c1\u4e13\u533a": "daily_source_zone_clicks",
    "\u641c\u7d22": "daily_source_search_clicks",
    "\u5206\u7c7b": "daily_source_category_clicks",
    "\u6392\u884c\u699c": "daily_source_ranking_clicks",
    "\u518d\u6765\u4e00\u5355": "daily_source_reorder_clicks",
}
ORDER_INTENT_TYPES = {
    "\u73b0\u8d27\u4e0b\u5355": "spot",
    "\u9884\u552e\u4e0b\u5355": "preorder",
}
RICH_EVENT_COUNT_COLS = [
    *SOURCE_CLICK_METRICS.values(),
    "daily_spot_views",
    "daily_preorder_views",
    "daily_spot_carts",
    "daily_preorder_carts",
    "daily_spot_orders",
    "daily_preorder_orders",
    "daily_spot_pays",
    "daily_preorder_pays",
]
RICH_EVENT_QTY_COLS = [
    "daily_spot_cart_qty",
    "daily_preorder_cart_qty",
    "daily_spot_order_qty",
    "daily_preorder_order_qty",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_id(series):
    values = series.fillna("").astype(str).str.upper().str.strip()
    return values.str.extract(r"^([A-Z0-9]+)", expand=False).fillna(values)


def parse_business_date(series):
    raw = series.fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    compact = raw.str.fullmatch(r"\d{8}")
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    parsed.loc[compact] = pd.to_datetime(
        raw.loc[compact],
        format="%Y%m%d",
        errors="coerce",
    )
    parsed.loc[~compact] = pd.to_datetime(raw.loc[~compact], errors="coerce")
    return parsed.dt.normalize()


def dated_paths(folder, suffix, cutoff):
    paths = []
    for path in sorted(folder.glob(f"*_{suffix}.csv")):
        match = re.match(r"^(\d{8})_", path.name)
        if match and pd.Timestamp(match.group(1)) <= cutoff:
            paths.append(path)
    return paths


def read_event_file(path):
    frame = pd.read_csv(
        path,
        usecols=lambda col: col.upper()
        in {
            "USERNAME",
            "PRODUCTNAME",
            "SOURCE_PAGE",
            "CAR_TYPE",
            "CAR_QTY",
            "ORDER_TYPE",
            "ORDER_QTY",
            "CURRENT_STAGE",
            "CREATIONDATE",
        },
    )
    frame.columns = frame.columns.str.upper()
    return frame


def build_event_features(cutoff):
    daily_paths = dated_paths(WAREHOUSE / "fact_events", "user_events", cutoff)
    full_path = (
        WAREHOUSE
        / "fact_events"
        / f"V_IRS_EVENT_{cutoff.strftime('%Y%m%d')}.csv"
    )
    if not full_path.exists():
        raise FileNotFoundError(f"Missing full V_IRS_EVENT extract: {full_path}")
    raw = read_event_file(full_path)
    raw["event_time"] = pd.to_datetime(raw["CREATIONDATE"], errors="coerce")
    raw["date"] = raw["event_time"].dt.normalize()
    raw = raw[
        raw["event_time"].notna()
        & (raw["date"] <= cutoff)
        & (raw["date"] >= pd.Timestamp("2025-01-01"))
    ].copy()
    raw["buyer_id"] = normalize_id(raw["USERNAME"])
    raw["style_id"] = normalize_id(raw["PRODUCTNAME"])
    raw = raw[(raw["buyer_id"].str.len() >= 6) & (raw["style_id"].str.len() >= 6)]
    raw["metric"] = raw["CURRENT_STAGE"].map(EVENT_METRICS)
    raw = raw[raw["metric"].notna()].copy()
    raw["car_qty"] = pd.to_numeric(raw["CAR_QTY"], errors="coerce").fillna(0.0)
    raw["order_qty"] = pd.to_numeric(raw["ORDER_QTY"], errors="coerce").fillna(0.0)

    identity_cols = [
        "USERNAME",
        "PRODUCTNAME",
        "SOURCE_PAGE",
        "CAR_TYPE",
        "CAR_QTY",
        "ORDER_TYPE",
        "CURRENT_STAGE",
        "CREATIONDATE",
        "ORDER_QTY",
    ]
    model_input_duplicates = int(raw.duplicated(identity_cols).sum())

    counts = (
        raw.assign(metric_count=1.0)
        .pivot_table(
            index=["date", "buyer_id", "style_id"],
            columns="metric",
            values="metric_count",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    for metric in EVENT_METRICS.values():
        if metric not in counts.columns:
            counts[metric] = 0.0

    order_qty = (
        raw[raw["metric"] == "daily_order_success"]
        .groupby(["date", "buyer_id", "style_id"], as_index=False)["order_qty"]
        .sum()
        .rename(columns={"order_qty": "daily_order_submit_qty"})
    )
    pay_qty = (
        raw[raw["metric"] == "daily_pay_success"]
        .groupby(["date", "buyer_id", "style_id"], as_index=False)["order_qty"]
        .sum()
        .rename(columns={"order_qty": "daily_pay_qty"})
    )
    features = counts.merge(order_qty, on=["date", "buyer_id", "style_id"], how="left")
    features = features.merge(pay_qty, on=["date", "buyer_id", "style_id"], how="left")
    features["daily_order_submit_qty"] = features["daily_order_submit_qty"].fillna(0.0)
    features["daily_pay_qty"] = features["daily_pay_qty"].fillna(0.0)

    rich_count_parts = []
    source_clicks = raw.loc[raw["CURRENT_STAGE"] == "\u5546\u54c1\u70b9\u51fb"].copy()
    source_clicks["rich_metric"] = source_clicks["SOURCE_PAGE"].map(
        SOURCE_CLICK_METRICS
    )
    rich_count_parts.append(source_clicks[source_clicks["rich_metric"].notna()])

    stage_prefix = {
        "\u67e5\u770b\u4e0b\u5355": "views",
        "\u52a0\u8d2d\u7269\u8f66": "carts",
        "\u4e0b\u5355\u6210\u529f": "orders",
        "\u652f\u4ed8\u6210\u529f": "pays",
    }
    intent = raw.loc[raw["CURRENT_STAGE"].isin(stage_prefix)].copy()
    intent["intent_type"] = intent["CAR_TYPE"].map(ORDER_INTENT_TYPES)
    order_type = intent["ORDER_TYPE"].map(ORDER_INTENT_TYPES)
    intent["intent_type"] = intent["intent_type"].fillna(order_type)
    intent["stage_prefix"] = intent["CURRENT_STAGE"].map(stage_prefix)
    valid_intent = intent["intent_type"].notna() & intent["stage_prefix"].notna()
    intent = intent.loc[valid_intent].copy()
    intent["rich_metric"] = (
        "daily_"
        + intent["intent_type"].astype(str)
        + "_"
        + intent["stage_prefix"].astype(str)
    )
    rich_count_parts.append(intent)

    rich_counts_raw = pd.concat(rich_count_parts, ignore_index=True)
    rich_counts = (
        rich_counts_raw.assign(metric_count=1.0)
        .pivot_table(
            index=["date", "buyer_id", "style_id"],
            columns="rich_metric",
            values="metric_count",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    for column in RICH_EVENT_COUNT_COLS:
        if column not in rich_counts.columns:
            rich_counts[column] = 0.0

    cart_qty = (
        intent.loc[intent["stage_prefix"] == "carts"]
        .assign(
            qty_metric=lambda frame: (
                "daily_"
                + frame["intent_type"].astype(str)
                + "_cart_qty"
            )
        )
        .pivot_table(
            index=["date", "buyer_id", "style_id"],
            columns="qty_metric",
            values="car_qty",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    order_qty_by_type = (
        intent.loc[intent["stage_prefix"] == "orders"]
        .assign(
            qty_metric=lambda frame: (
                "daily_"
                + frame["intent_type"].astype(str)
                + "_order_qty"
            )
        )
        .pivot_table(
            index=["date", "buyer_id", "style_id"],
            columns="qty_metric",
            values="order_qty",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    for rich_frame in [cart_qty, order_qty_by_type]:
        features = features.merge(
            rich_frame,
            on=["date", "buyer_id", "style_id"],
            how="left",
        )
    features = features.merge(
        rich_counts[
            ["date", "buyer_id", "style_id", *RICH_EVENT_COUNT_COLS]
        ],
        on=["date", "buyer_id", "style_id"],
        how="left",
    )
    for column in [*RICH_EVENT_COUNT_COLS, *RICH_EVENT_QTY_COLS]:
        if column not in features.columns:
            features[column] = 0.0
        features[column] = features[column].fillna(0.0)
    features["skc_id"] = features["style_id"]
    features = features.sort_values(["date", "buyer_id", "style_id"]).reset_index(drop=True)

    coverage_dates = pd.date_range(raw["date"].min(), cutoff, freq="D")
    coverage = pd.DataFrame({"date": coverage_dates})
    coverage["event_extract_present"] = 1
    coverage["event_source_mode"] = "full_oracle_export"
    daily_dates = {pd.Timestamp(path.name[:8]) for path in daily_paths}
    coverage["daily_audit_extract_present"] = coverage["date"].isin(daily_dates).astype(int)

    audit = {
        "full_source": str(full_path.relative_to(PROJECT_ROOT)),
        "full_source_rows": int(len(pd.read_csv(full_path, usecols=["CREATIONDATE"]))),
        "daily_files": len(daily_paths),
        "cutoff": cutoff.date().isoformat(),
        "raw_rows_after_stage_filter": int(len(raw)),
        "model_input_duplicate_rows_observed_not_removed": model_input_duplicates,
        "feature_rows": int(len(features)),
        "rich_event_columns": [
            *RICH_EVENT_COUNT_COLS,
            *RICH_EVENT_QTY_COLS,
        ],
        "feature_date_min": features["date"].min().date().isoformat(),
        "feature_date_max": features["date"].max().date().isoformat(),
        "daily_audit_missing_dates": [
            date.date().isoformat()
            for date in coverage.loc[
                (coverage["date"] >= min(daily_dates))
                & (coverage["daily_audit_extract_present"] == 0),
                "date",
            ]
        ],
    }
    return features, coverage, audit


def build_inventory_features(cutoff):
    storage_paths = dated_paths(WAREHOUSE / "snapshot_inventory", "storage_stock", cutoff)
    b2b_paths = dated_paths(WAREHOUSE / "snapshot_inventory", "b2b_stock", cutoff)

    storage_frames = []
    for path in storage_paths:
        frame = pd.read_csv(
            path,
            usecols=lambda col: col.upper() in {"NO", "NAME", "QTYCAN"},
        )
        frame.columns = frame.columns.str.upper()
        frame["date"] = pd.Timestamp(path.name[:8])
        frame["sku_id"] = normalize_id(frame["NO"])
        frame["style_id"] = normalize_id(frame["NAME"])
        frame["qty_storage_stock"] = pd.to_numeric(
            frame["QTYCAN"], errors="coerce"
        ).fillna(0.0)
        storage_frames.append(frame[["date", "sku_id", "style_id", "qty_storage_stock"]])

    b2b_frames = []
    for path in b2b_paths:
        frame = pd.read_csv(
            path,
            usecols=lambda col: col.upper() in {"NO", "NAME", "QTY_HQ"},
        )
        frame.columns = frame.columns.str.upper()
        frame["date"] = pd.Timestamp(path.name[:8])
        frame["sku_id"] = normalize_id(frame["NO"])
        frame["style_id"] = normalize_id(frame["NAME"])
        frame["qty_b2b_hq_stock"] = pd.to_numeric(
            frame["QTY_HQ"], errors="coerce"
        ).fillna(0.0)
        b2b_frames.append(frame[["date", "sku_id", "style_id", "qty_b2b_hq_stock"]])

    storage = (
        pd.concat(storage_frames, ignore_index=True)
        .groupby(["date", "sku_id"], as_index=False)
        .agg(style_id=("style_id", "first"), qty_storage_stock=("qty_storage_stock", "sum"))
    )
    storage["has_storage_snapshot"] = 1
    b2b = (
        pd.concat(b2b_frames, ignore_index=True)
        .groupby(["date", "sku_id"], as_index=False)
        .agg(style_id=("style_id", "first"), qty_b2b_hq_stock=("qty_b2b_hq_stock", "sum"))
    )
    b2b["has_b2b_snapshot"] = 1

    features = storage.merge(
        b2b,
        on=["date", "sku_id"],
        how="outer",
        suffixes=("_storage", "_b2b"),
    )
    features["style_id"] = features["style_id_storage"].fillna(features["style_id_b2b"])
    features = features.drop(columns=["style_id_storage", "style_id_b2b"])
    for column in [
        "qty_storage_stock",
        "qty_b2b_hq_stock",
        "has_storage_snapshot",
        "has_b2b_snapshot",
    ]:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    features["qty_total_stock"] = (
        features["qty_storage_stock"] + features["qty_b2b_hq_stock"]
    )
    features["snapshot_present"] = (
        (features["has_storage_snapshot"] > 0) | (features["has_b2b_snapshot"] > 0)
    ).astype(int)
    features["stock_positive"] = (
        (features["snapshot_present"] > 0) & (features["qty_total_stock"] > 0)
    ).astype(int)
    features["stock_zero"] = (
        (features["snapshot_present"] > 0) & (features["qty_total_stock"] <= 0)
    ).astype(int)
    features = features.sort_values(["date", "sku_id"]).reset_index(drop=True)

    audit = {
        "storage_files": len(storage_paths),
        "b2b_files": len(b2b_paths),
        "rows": len(features),
        "date_min": features["date"].min().date().isoformat(),
        "date_max": features["date"].max().date().isoformat(),
        "duplicate_sku_day_rows": int(features.duplicated(["date", "sku_id"]).sum()),
    }
    return features, audit


def build_listing_features(cutoff):
    path = WAREHOUSE / "dim_product" / f"product_info_{cutoff.strftime('%Y%m%d')}.csv"
    frame = pd.read_csv(path, usecols=["NO", "NAME", "LISTING_DATE"])
    frame["sku_id"] = normalize_id(frame["NO"])
    frame["style_id"] = normalize_id(frame["NAME"])
    frame["launch_date"] = parse_business_date(frame["LISTING_DATE"])
    features = (
        frame[["sku_id", "style_id", "launch_date"]]
        .dropna(subset=["launch_date"])
        .drop_duplicates(["sku_id"], keep="first")
        .sort_values("sku_id")
        .reset_index(drop=True)
    )
    audit = {
        "source": str(path.relative_to(PROJECT_ROOT)),
        "rows": len(features),
        "coverage": float(len(features) / len(frame)),
        "date_min": features["launch_date"].min().date().isoformat(),
        "date_max": features["launch_date"].max().date().isoformat(),
        "mode_date": features["launch_date"].mode().iloc[0].date().isoformat(),
        "mode_rows": int(
            (features["launch_date"] == features["launch_date"].mode().iloc[0]).sum()
        ),
    }
    return features, audit


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "rows": len(frame),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main():
    parser = argparse.ArgumentParser(description="Build frozen Phase8 signal tables.")
    parser.add_argument("--cutoff", default="2026-06-14")
    parser.add_argument("--output", default="data/phase8_20260614/signals")
    args = parser.parse_args()

    cutoff = pd.Timestamp(args.cutoff).normalize()
    output = PROJECT_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    event, event_coverage, event_audit = build_event_features(cutoff)
    inventory, inventory_audit = build_inventory_features(cutoff)
    listing, listing_audit = build_listing_features(cutoff)

    outputs = {
        "event": write_csv(event, output / "event_intent_daily_features.csv"),
        "event_coverage": write_csv(event_coverage, output / "event_coverage_daily.csv"),
        "inventory": write_csv(inventory, output / "inventory_daily_features.csv"),
        "listing": write_csv(listing, output / "lifecycle_launch_date_features.csv"),
    }

    request_source = PROJECT_ROOT / "data" / "phase8a_prep" / "purchase_request_daily_features.csv"
    if request_source.exists():
        request_destination = output / "purchase_request_daily_features.csv"
        shutil.copy2(request_source, request_destination)
        request = pd.read_csv(request_destination, usecols=["date"])
        outputs["purchase_request"] = {
            "path": str(request_destination.relative_to(PROJECT_ROOT)),
            "rows": len(pd.read_csv(request_destination)),
            "date_min": str(pd.to_datetime(request["date"]).min().date()),
            "date_max": str(pd.to_datetime(request["date"]).max().date()),
            "bytes": request_destination.stat().st_size,
            "sha256": sha256(request_destination),
        }

    manifest = {
        "snapshot_id": f"phase8_signals_{cutoff.strftime('%Y%m%d')}",
        "cutoff": cutoff.date().isoformat(),
        "event_policy": "full_oracle_export_daily_extracts_for_audit_only",
        "inventory_policy": "sum_repeated_rows_at_sku_day",
        "listing_policy": "LISTING_DATE_only_PL_CYCLE_excluded",
        "profile_policy": "excluded_from_mainline",
        "store_policy": "mapping_only",
        "audits": {
            "event": event_audit,
            "inventory": inventory_audit,
            "listing": listing_audit,
        },
        "outputs": outputs,
    }
    manifest_path = output / "phase8_signal_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] {manifest_path}")
    for name, details in outputs.items():
        print(f"- {name}: rows={details['rows']} path={details['path']}")


if __name__ == "__main__":
    main()
