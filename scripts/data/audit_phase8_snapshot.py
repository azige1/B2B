import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = PROJECT_ROOT / "data_warehouse"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase8_data_audit"
SNAPSHOT_ID = "20260614"


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
    return parsed


def file_dates(paths):
    dates = []
    for path in paths:
        match = re.match(r"^(\d{8})_", path.name)
        if match:
            dates.append(pd.Timestamp(match.group(1)))
    return sorted(set(dates))


def coverage(paths):
    dates = file_dates(paths)
    if not dates:
        return {"date_min": "", "date_max": "", "days": 0, "missing_dates": []}
    full = pd.date_range(dates[0], dates[-1], freq="D")
    missing = full.difference(pd.DatetimeIndex(dates))
    return {
        "date_min": dates[0].date().isoformat(),
        "date_max": dates[-1].date().isoformat(),
        "days": len(dates),
        "missing_dates": [date.date().isoformat() for date in missing],
    }


def read_many(paths, usecols=None):
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=usecols)
        frame["_source_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def order_audit():
    path = WAREHOUSE / "fact_orders" / "V_IRS_ORDERFTP_6_14.csv"
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["BILLDATE"].astype(str), errors="coerce")
    qty = pd.to_numeric(df["QTY"], errors="coerce")
    return {
        "table": "V_IRS_ORDERFTP",
        "role": "label_and_order_history",
        "files": 1,
        "rows": len(df),
        "columns": list(df.columns),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
        "date_parse_failures": int(dates.isna().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "type_missing_rows": int(df["TYPE"].isna().sum()),
        "negative_qty_rows": int((qty < 0).sum()),
        "positive_qty_rows": int((qty > 0).sum()),
        "notes": "Only canonical order and label source. Negative quantities are reversal entries.",
    }


def inventory_audit(suffix, table, qty_col):
    paths = sorted((WAREHOUSE / "snapshot_inventory").glob(f"*_{suffix}.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path, usecols=lambda col: col.upper() in {"NO", "NAME", qty_col})
        frame.columns = frame.columns.str.upper()
        frame["snapshot_date"] = pd.Timestamp(path.name[:8])
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    key_duplicates = int(df.duplicated(["snapshot_date", "NO"]).sum())
    values = pd.to_numeric(df[qty_col], errors="coerce")
    result = coverage(paths)
    result.update(
        {
            "table": table,
            "role": "phase8_inventory_feature",
            "files": len(paths),
            "rows": len(df),
            "columns": [col for col in df.columns if col != "snapshot_date"],
            "key": "snapshot_date+NO",
            "duplicate_key_rows": key_duplicates,
            "distinct_skus": int(df["NO"].astype(str).nunique()),
            "qty_missing_rows": int(values.isna().sum()),
            "qty_negative_rows": int((values < 0).sum()),
            "notes": (
                "Aggregate repeated SKU rows at SKU-day level."
                if key_duplicates
                else "SKU-day keys are unique."
            ),
        }
    )
    return result


def event_audit():
    path = WAREHOUSE / "fact_events" / f"V_IRS_EVENT_{SNAPSHOT_ID}.csv"
    daily_paths = sorted((WAREHOUSE / "fact_events").glob("*_user_events.csv"))
    df = pd.read_csv(
        path,
        usecols=lambda col: col.upper()
        in {"USERNAME", "PRODUCTNAME", "CURRENT_STAGE", "CREATIONDATE", "ORDER_QTY"},
    )
    event_time = pd.to_datetime(df["CREATIONDATE"], errors="coerce")
    daily_coverage = coverage(daily_paths)
    return {
        "table": "V_IRS_EVENT",
        "role": "phase8_behavior_feature",
        "files": 1,
        "rows": len(df),
        "columns": list(df.columns),
        "date_min": event_time.min().date().isoformat(),
        "date_max": event_time.max().date().isoformat(),
        "event_time_min": event_time.min().isoformat(),
        "event_time_max": event_time.max().isoformat(),
        "event_time_parse_failures": int(event_time.isna().sum()),
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "distinct_buyers": int(normalize_id(df["USERNAME"]).nunique()),
        "distinct_products": int(normalize_id(df["PRODUCTNAME"]).nunique()),
        "event_stage_counts": {
            str(key): int(value)
            for key, value in df["CURRENT_STAGE"].fillna("<NULL>").value_counts().items()
        },
        "daily_audit_files": len(daily_paths),
        "daily_audit_missing_dates": daily_coverage["missing_dates"],
        "notes": "Full Oracle extract is authoritative; daily files are coverage audit only.",
    }


def product_audit():
    path = WAREHOUSE / "dim_product" / f"product_info_{SNAPSHOT_ID}.csv"
    df = pd.read_csv(path)
    listing = parse_business_date(df["LISTING_DATE"])
    return {
        "table": "V_IRS_PRODUCT",
        "role": "product_dimension_and_listing_date",
        "files": 1,
        "rows": len(df),
        "columns": list(df.columns),
        "duplicate_sku_rows": int(df.duplicated(["NO"]).sum()),
        "distinct_skus": int(df["NO"].astype(str).nunique()),
        "distinct_styles": int(df["NAME"].astype(str).nunique()),
        "listing_date_coverage": float(listing.notna().mean()),
        "listing_date_min": listing.min().date().isoformat(),
        "listing_date_max": listing.max().date().isoformat(),
        "listing_date_mode": listing.mode().iloc[0].date().isoformat(),
        "listing_date_mode_rows": int((listing == listing.mode().iloc[0]).sum()),
        "notes": "PL_CYCLE is excluded. LISTING_DATE is used only with time-safe handling.",
    }


def store_audit():
    path = WAREHOUSE / "dim_store" / f"store_info_{SNAPSHOT_ID}.csv"
    df = pd.read_csv(path)
    return {
        "table": "V_IRS_STORE",
        "role": "mapping_and_audit_only",
        "files": 1,
        "rows": len(df),
        "columns": list(df.columns),
        "distinct_buyer_names": int(normalize_id(df["NAME"]).nunique()),
        "duplicate_store_labels": int(df.duplicated(["STORENAME"]).sum()),
        "notes": "Not a Phase8 model feature source.",
    }


def profile_audit():
    paths = sorted((WAREHOUSE / "snapshot_metrics").glob("*_customer_profile.csv"))
    latest = pd.read_csv(paths[-1])
    latest["buyer_id"] = normalize_id(latest["CUSTOMER_NAME"])
    numeric_cols = [
        "COOPERATION_YEARS",
        "MONTHLY_AVERAGE_REPLENISHMENT",
        "AVG_DISCOUNT_RATE",
        "REPLENISHMENT_FREQUENCY",
        "ITEM_COVERAGE_RATE",
    ]
    conflicts = 0
    for _, group in latest.groupby("buyer_id"):
        if any(pd.to_numeric(group[col], errors="coerce").nunique(dropna=False) > 1 for col in numeric_cols):
            conflicts += 1
    result = coverage(paths)
    result.update(
        {
            "table": "V_IRS_CUS_PROFILE",
            "role": "shadow_only",
            "files": len(paths),
            "latest_rows": len(latest),
            "columns": [col for col in latest.columns if col != "buyer_id"],
            "latest_distinct_buyers": int(latest["buyer_id"].nunique()),
            "latest_duplicate_buyer_rows": int(latest.duplicated(["buyer_id"]).sum()),
            "buyers_with_conflicting_metrics": conflicts,
            "notes": "Excluded from Phase8 mainline until field windows, leakage safety, and alias aggregation are proven.",
        }
    )
    return result


def purchase_request_audit():
    path = PROJECT_ROOT / "data" / "silver" / "clean_purchase_requests.csv"
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["request_date"], errors="coerce")
    qty = pd.to_numeric(df["request_qty"], errors="coerce")
    return {
        "table": "client_purchase_request_file",
        "role": "phase8_request_feature",
        "files": 1,
        "rows": len(df),
        "columns": list(df.columns),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
        "date_parse_failures": int(dates.isna().sum()),
        "distinct_skus": int(df["sku_id"].astype(str).nunique()),
        "distinct_styles": int(df["style_id_from_request"].astype(str).nunique()),
        "request_qty_sum": float(qty.fillna(0).sum()),
        "notes": "Client-provided file. Final Phase8 must record its refresh cutoff explicitly.",
    }


def preorder_audit():
    paths = sorted((WAREHOUSE / "fact_orders").glob("*_b2b_preorder.csv"))
    result = coverage(paths)
    result.update(
        {
            "table": "V_IRS_PREORDER",
            "role": "diagnostic_only",
            "files": len(paths),
            "rows": sum(len(pd.read_csv(path)) for path in paths),
            "notes": "Sparse extracts; not a Phase8 mainline feature.",
        }
    )
    return result


def markdown_table(rows):
    columns = ["table", "role", "files", "rows", "date_min", "date_max", "notes"]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(column, "")).replace("|", "/") for column in columns)
            + " |"
        )
    return "\n".join(lines)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        order_audit(),
        inventory_audit("storage_stock", "V_IRS_STORAGE", "QTYCAN"),
        inventory_audit("b2b_stock", "V_IRS_B2BSTORAGE", "QTY_HQ"),
        event_audit(),
        product_audit(),
        purchase_request_audit(),
        store_audit(),
        profile_audit(),
        preorder_audit(),
    ]

    json_path = REPORT_DIR / f"phase8_data_audit_{SNAPSHOT_ID}.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    event = next(row for row in rows if row["table"] == "V_IRS_EVENT")
    profile = next(row for row in rows if row["table"] == "V_IRS_CUS_PROFILE")
    report = [
        "# Phase8 Data Snapshot Audit",
        "",
        f"Snapshot cutoff: `{SNAPSHOT_ID[:4]}-{SNAPSHOT_ID[4:6]}-{SNAPSHOT_ID[6:]}`",
        "",
        "## Source Roles",
        "",
        markdown_table(rows),
        "",
        "## Mainline Policy",
        "",
        "- Mainline: `V_IRS_ORDERFTP`, `V_IRS_STORAGE`, `V_IRS_B2BSTORAGE`, `V_IRS_EVENT`, `V_IRS_PRODUCT`, and the client purchase-request file.",
        "- Mapping/audit only: `V_IRS_STORE`.",
        "- Shadow only: `V_IRS_CUS_PROFILE`.",
        "- Diagnostic only: `V_IRS_PREORDER`.",
        "- Excluded: `V_IRS_ORDER` and all `*_order_history.csv` extracts.",
        "",
        "## Required Guards",
        "",
        f"- Daily event audit missing extract dates: `{', '.join(event['daily_audit_missing_dates'])}`.",
        f"- Customer-profile conflicting buyers in latest snapshot: `{profile['buyers_with_conflicting_metrics']}`.",
        "- The full Oracle event extract is used for features; daily missing dates do not create model-side zeros.",
        "- Inventory repeated SKU rows must be aggregated at SKU-day level.",
        "- Listing dates later than a training row must use a neutral state; future listing timing cannot be exposed.",
        "- Purchase-request status fields are audit-only; model features use submission-time information.",
        "- Customer profile remains excluded until its field calculation windows and alias rules are verified.",
        "",
        "## Detailed Evidence",
        "",
        f"- JSON audit: `{json_path.relative_to(PROJECT_ROOT)}`",
        f"- Snapshot manifest: `data/manifests/phase8_data_snapshot_{SNAPSHOT_ID}.json`",
    ]
    md_path = REPORT_DIR / f"phase8_data_audit_{SNAPSHOT_ID}.md"
    md_path.write_text("\n".join(report), encoding="utf-8-sig")
    print(f"[OK] {md_path}")
    print(f"[OK] {json_path}")


if __name__ == "__main__":
    main()
