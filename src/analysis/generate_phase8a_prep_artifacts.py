import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
RAW_DW_DIR = PROJECT_ROOT / "data_warehouse"
REPORT_DIR = PROJECT_ROOT / "reports" / "phase8a_prep"
PHASE8_DATA_DIR = DATA_DIR / "phase8a_prep"
CURRENT_MODEL_DIR = PROJECT_ROOT / "models" / "current_phase7_mainline"

ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
LOOKBACK = 90
CALIBRATION_SCALES = {
    "2025-09-01": 0.98,
    "2025-10-01": 0.93,
    "2025-11-01": 1.00,
    "2025-12-01": 1.00,
}
CURRENT_MAINLINE_TEMPLATE = (
    "p7ov_20260402_overnight_stage_s3_{anchor_tag}_tail_full_lr005_l63_g027_n800_s2028_hard_g027"
)
EVENT_SOURCE_CANDIDATES = [
    "V_IRS_EVENT1.csv",
    "V_IRS_EVENT.csv",
    "V_IRS_EVENT_new.csv",
]


def normalize_series(series):
    s = series.astype(str).str.upper().str.strip()
    return s.str.extract(r"^([A-Z0-9]+)", expand=False).fillna(s)


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def ensure_dirs():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PHASE8_DATA_DIR.mkdir(parents=True, exist_ok=True)


def resolve_event_source():
    fact_events_dir = RAW_DW_DIR / "fact_events"
    for name in EVENT_SOURCE_CANDIDATES:
        path = fact_events_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No event source found under {fact_events_dir}")


def safe_div(num, den):
    if den in (0, 0.0, None) or pd.isna(den):
        return np.nan
    return float(num) / float(den)


def qfo_bucket(value):
    value = float(value)
    if value <= 0:
        return "0"
    if value <= 1:
        return "1"
    if value <= 10:
        return "2-10"
    if value <= 30:
        return "11-30"
    if value <= 100:
        return "31-100"
    return "100+"


def activity_bucket_from_days(days):
    days = int(days)
    if days > 30:
        return "hot"
    if days >= 10:
        return "warm"
    if days >= 1:
        return "cold"
    return "ice"


def markdown_table(df, columns):
    view = df[columns].copy()
    headers = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in view.iterrows():
        vals = []
        for col in columns:
            v = row[col]
            if isinstance(v, float):
                if np.isnan(v):
                    vals.append("")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([headers, sep, *rows])


def support_status_from_days(days):
    if days <= 0:
        return "unsupported"
    if days < LOOKBACK:
        return "partial"
    return "full"


def build_inventory_daily_features(products):
    snapshot_dir = RAW_DW_DIR / "snapshot_inventory"
    storage_rows = []
    b2b_rows = []

    for path in sorted(snapshot_dir.glob("*_storage_stock.csv")):
        m = re.search(r"(\d{8})", path.name)
        if not m:
            continue
        date = pd.to_datetime(m.group(1), format="%Y%m%d")
        df = pd.read_csv(path)
        df.columns = df.columns.str.upper()
        df["date"] = date
        df["style_id"] = normalize_series(df["NAME"])
        df["sku_id"] = normalize_series(df["NO"])
        df["qty_storage_stock"] = pd.to_numeric(df.get("QTYCAN"), errors="coerce").fillna(0.0)
        # Presence should mean “snapshot row exists”, not “stock > 0”.
        # Otherwise we cannot distinguish zero-stock rows from missing snapshots.
        df["has_storage_snapshot"] = 1
        storage_rows.append(df[["date", "sku_id", "style_id", "qty_storage_stock", "has_storage_snapshot"]])

    for path in sorted(snapshot_dir.glob("*_b2b_stock.csv")):
        m = re.search(r"(\d{8})", path.name)
        if not m:
            continue
        date = pd.to_datetime(m.group(1), format="%Y%m%d")
        df = pd.read_csv(path)
        df.columns = df.columns.str.upper()
        df["date"] = date
        df["style_id"] = normalize_series(df["NAME"])
        df["sku_id"] = normalize_series(df["NO"])
        df["qty_b2b_hq_stock"] = pd.to_numeric(df.get("QTY_HQ"), errors="coerce").fillna(0.0)
        df["has_b2b_snapshot"] = 1
        b2b_rows.append(df[["date", "sku_id", "style_id", "qty_b2b_hq_stock", "has_b2b_snapshot"]])

    storage_df = pd.concat(storage_rows, ignore_index=True) if storage_rows else pd.DataFrame(
        columns=["date", "sku_id", "style_id", "qty_storage_stock", "has_storage_snapshot"]
    )
    b2b_df = pd.concat(b2b_rows, ignore_index=True) if b2b_rows else pd.DataFrame(
        columns=["date", "sku_id", "style_id", "qty_b2b_hq_stock", "has_b2b_snapshot"]
    )

    features = pd.merge(
        storage_df,
        b2b_df,
        on=["date", "sku_id", "style_id"],
        how="outer",
    )
    features["qty_storage_stock"] = features["qty_storage_stock"].fillna(0.0)
    features["qty_b2b_hq_stock"] = features["qty_b2b_hq_stock"].fillna(0.0)
    features["has_storage_snapshot"] = pd.to_numeric(
        features["has_storage_snapshot"], errors="coerce"
    ).fillna(0.0).astype(int)
    features["has_b2b_snapshot"] = pd.to_numeric(
        features["has_b2b_snapshot"], errors="coerce"
    ).fillna(0.0).astype(int)
    features["qty_total_stock"] = features["qty_storage_stock"] + features["qty_b2b_hq_stock"]
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

    product_skus = set(products["sku_id"].astype(str))
    product_styles = set(products["style_id"].astype(str))
    key_dupes = int(features.duplicated(["date", "sku_id"]).sum())
    audit = {
        "source": "inventory_daily_features",
        "source_file": "snapshot_inventory/*.csv",
        "rows": int(len(features)),
        "date_min": str(features["date"].min().date()) if not features.empty else "",
        "date_max": str(features["date"].max().date()) if not features.empty else "",
        "distinct_days": int(features["date"].nunique()) if not features.empty else 0,
        "key_columns": "date+sku_id",
        "duplicate_rows_on_key": key_dupes,
        "sku_mapping_rate": float(features["sku_id"].isin(product_skus).mean()) if not features.empty else np.nan,
        "style_mapping_rate": float(features["style_id"].isin(product_styles).mean()) if not features.empty else np.nan,
        "buyer_mapping_rate": np.nan,
    }
    return features, audit


def build_preorder_daily_features(products, valid_buyers):
    fact_orders_dir = RAW_DW_DIR / "fact_orders"
    frames = []
    full_history = fact_orders_dir / "V_IRS_PREORDER.csv"
    source_file = "fact_orders/*_b2b_preorder.csv"
    if full_history.exists():
        df = pd.read_csv(full_history)
        df.columns = df.columns.str.upper()
        df["date"] = pd.to_datetime(df["MODIFIEDDATE"], errors="coerce").dt.normalize()
        df["style_id"] = normalize_series(df["NAME"])
        df["sku_id"] = normalize_series(df["PRODUCTNO"])
        df["buyer_id"] = normalize_series(df["CUSTOMERNAME"])
        df["store_line_id"] = normalize_series(df["STORENAME"])
        df["qty_rem"] = pd.to_numeric(df["QTY_REM"], errors="coerce").fillna(0.0)
        df["modified_time"] = pd.to_datetime(df["MODIFIEDDATE"], errors="coerce")
        df = df.dropna(subset=["date"])
        frames.append(
            df[
                [
                    "date",
                    "buyer_id",
                    "store_line_id",
                    "style_id",
                    "sku_id",
                    "qty_rem",
                    "modified_time",
                ]
            ]
        )
        source_file = full_history.name
    else:
        for path in sorted(fact_orders_dir.glob("*_b2b_preorder.csv")):
            m = re.search(r"(\d{8})", path.name)
            if not m:
                continue
            snapshot_date = pd.to_datetime(m.group(1), format="%Y%m%d")
            df = pd.read_csv(path)
            df.columns = df.columns.str.upper()
            df["date"] = snapshot_date
            df["style_id"] = normalize_series(df["NAME"])
            df["sku_id"] = normalize_series(df["PRODUCTNO"])
            df["buyer_id"] = normalize_series(df["CUSTOMERNAME"])
            df["store_line_id"] = normalize_series(df["STORENAME"])
            df["qty_rem"] = pd.to_numeric(df["QTY_REM"], errors="coerce").fillna(0.0)
            df["modified_time"] = pd.to_datetime(df["MODIFIEDDATE"], errors="coerce")
            frames.append(
                df[
                    [
                        "date",
                        "buyer_id",
                        "store_line_id",
                        "style_id",
                        "sku_id",
                        "qty_rem",
                        "modified_time",
                    ]
                ]
            )

    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["date", "buyer_id", "store_line_id", "style_id", "sku_id", "qty_rem", "modified_time"]
    )
    if raw.empty:
        features = raw.copy()
    else:
        features = (
            raw.groupby(["date", "buyer_id", "store_line_id", "style_id", "sku_id"], as_index=False)
            .agg(
                preorder_row_count=("qty_rem", "size"),
                preorder_qty_rem_sum=("qty_rem", "sum"),
                preorder_qty_rem_pos_sum=("qty_rem", lambda s: float(s[s > 0].sum())),
                preorder_nonzero_row_count=("qty_rem", lambda s: int((s != 0).sum())),
                preorder_last_modified_time=("modified_time", "max"),
            )
            .sort_values(["date", "buyer_id", "sku_id"])
            .reset_index(drop=True)
        )

    product_skus = set(products["sku_id"].astype(str))
    product_styles = set(products["style_id"].astype(str))
    key_dupes = int(features.duplicated(["date", "buyer_id", "sku_id"]).sum()) if not features.empty else 0
    audit = {
        "source": "preorder_daily_features",
        "source_file": source_file,
        "rows": int(len(features)),
        "date_min": str(features["date"].min().date()) if not features.empty else "",
        "date_max": str(features["date"].max().date()) if not features.empty else "",
        "distinct_days": int(features["date"].nunique()) if not features.empty else 0,
        "key_columns": "date+buyer_id+sku_id",
        "duplicate_rows_on_key": key_dupes,
        "sku_mapping_rate": float(features["sku_id"].isin(product_skus).mean()) if not features.empty else np.nan,
        "style_mapping_rate": float(features["style_id"].isin(product_styles).mean()) if not features.empty else np.nan,
        "buyer_mapping_rate": float(features["buyer_id"].isin(valid_buyers).mean()) if not features.empty else np.nan,
    }
    return features, audit


def add_rolling_features(df, group_cols, base_cols, windows=(7, 14, 30)):
    out = []
    for _, grp in df.groupby(group_cols, sort=False):
        grp = grp.sort_values("date").copy()
        for col in base_cols:
            series = grp[col].astype(float)
            for window in windows:
                grp[f"{col}_{window}"] = series.rolling(window=window, min_periods=1).sum()

        last_any = []
        last_strong = []
        prev_any = None
        prev_strong = None
        for date, any_flag, strong_flag in zip(grp["date"], grp["_any_event_flag"], grp["_strong_intent_flag"]):
            last_any.append(999.0 if prev_any is None else float((date - prev_any).days))
            last_strong.append(999.0 if prev_strong is None else float((date - prev_strong).days))
            if bool(any_flag):
                prev_any = date
            if bool(strong_flag):
                prev_strong = date

        grp["days_since_last_any_event"] = last_any
        grp["days_since_last_strong_intent"] = last_strong
        grp["click_to_cart_rate_30"] = grp["daily_cart_adds_30"] / grp["daily_clicks_30"].replace(0, np.nan)
        grp["view_to_order_rate_30"] = grp["daily_order_success_30"] / grp["daily_view_order_30"].replace(0, np.nan)
        grp["cart_to_order_rate_30"] = grp["daily_order_success_30"] / grp["daily_cart_adds_30"].replace(0, np.nan)
        grp["order_to_pay_rate_30"] = grp["daily_pay_success_30"] / grp["daily_order_success_30"].replace(0, np.nan)
        out.append(grp)

    merged = pd.concat(out, ignore_index=True) if out else df.copy()
    for col in merged.columns:
        if col.endswith("_rate_30"):
            merged[col] = merged[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return merged


def build_event_intent_daily_features(products, valid_buyers):
    path = resolve_event_source()
    df = pd.read_csv(
        path,
        usecols=lambda c: c.upper() in {"USERNAME", "PRODUCTNAME", "CURRENT_STAGE", "CREATIONDATE", "ORDER_QTY"},
    )
    df.columns = df.columns.str.upper()
    df["buyer_id"] = normalize_series(df["USERNAME"])
    df["skc_id"] = normalize_series(df["PRODUCTNAME"])
    df["style_id"] = df["skc_id"]
    df["event_type"] = df["CURRENT_STAGE"].astype(str).str.strip()
    df["event_time"] = pd.to_datetime(df["CREATIONDATE"], errors="coerce")
    df["date"] = df["event_time"].dt.normalize()
    df["order_qty"] = pd.to_numeric(df["ORDER_QTY"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["date"])
    df = df[(df["buyer_id"].str.len() >= 6) & (df["skc_id"].str.len() >= 6)].copy()

    stage_map = {
        "商品点击": "daily_clicks",
        "查看下单": "daily_view_order",
        "加购物车": "daily_cart_adds",
        "下单成功": "daily_order_success",
        "支付成功": "daily_pay_success",
    }
    df["metric"] = df["event_type"].map(stage_map)
    df = df[df["metric"].notna()].copy()
    df["cnt"] = 1.0

    counts = (
        df.pivot_table(
            index=["date", "buyer_id", "skc_id", "style_id"],
            columns="metric",
            values="cnt",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )
    for col in stage_map.values():
        if col not in counts.columns:
            counts[col] = 0.0

    order_qty = (
        df[df["metric"] == "daily_order_success"]
        .groupby(["date", "buyer_id", "skc_id", "style_id"], as_index=False)["order_qty"]
        .sum()
        .rename(columns={"order_qty": "daily_order_submit_qty"})
    )
    pay_qty = (
        df[df["metric"] == "daily_pay_success"]
        .groupby(["date", "buyer_id", "skc_id", "style_id"], as_index=False)["order_qty"]
        .sum()
        .rename(columns={"order_qty": "daily_pay_qty"})
    )
    features = counts.merge(order_qty, on=["date", "buyer_id", "skc_id", "style_id"], how="left")
    features = features.merge(pay_qty, on=["date", "buyer_id", "skc_id", "style_id"], how="left")
    features["daily_order_submit_qty"] = features["daily_order_submit_qty"].fillna(0.0)
    features["daily_pay_qty"] = features["daily_pay_qty"].fillna(0.0)
    features["_any_event_flag"] = (
        features[["daily_clicks", "daily_view_order", "daily_cart_adds", "daily_order_success", "daily_pay_success"]].sum(axis=1)
        > 0
    )
    features["_strong_intent_flag"] = (
        features[["daily_cart_adds", "daily_order_success", "daily_pay_success"]].sum(axis=1)
        > 0
    )
    base_cols = [
        "daily_clicks",
        "daily_view_order",
        "daily_cart_adds",
        "daily_order_success",
        "daily_pay_success",
        "daily_order_submit_qty",
        "daily_pay_qty",
    ]
    features = add_rolling_features(features, ["buyer_id", "skc_id"], base_cols)
    features = features.drop(columns=["_any_event_flag", "_strong_intent_flag"])
    features = features.sort_values(["date", "buyer_id", "skc_id"]).reset_index(drop=True)

    product_styles = set(products["style_id"].astype(str))
    key_dupes = int(features.duplicated(["date", "buyer_id", "skc_id"]).sum()) if not features.empty else 0
    audit = {
        "source": "event_intent_daily_features",
        "source_file": path.name,
        "rows": int(len(features)),
        "date_min": str(features["date"].min().date()) if not features.empty else "",
        "date_max": str(features["date"].max().date()) if not features.empty else "",
        "distinct_days": int(features["date"].nunique()) if not features.empty else 0,
        "key_columns": "date+buyer_id+skc_id",
        "duplicate_rows_on_key": key_dupes,
        "sku_mapping_rate": np.nan,
        "style_mapping_rate": float(features["style_id"].isin(product_styles).mean()) if not features.empty else np.nan,
        "buyer_mapping_rate": float(features["buyer_id"].isin(valid_buyers).mean()) if not features.empty else np.nan,
    }
    return features, audit


def add_anchor_support(audit_df):
    out = audit_df.copy()
    for anchor in ANCHORS:
        anchor_dt = pd.Timestamp(anchor)
        days_col = f"{anchor_tag(anchor)}_available_days"
        support_col = f"{anchor_tag(anchor)}_support"
        days_list = []
        supports = []
        for _, row in out.iterrows():
            if not row["date_min"]:
                days = 0
            else:
                source_min = pd.Timestamp(row["date_min"])
                source_max = pd.Timestamp(row["date_max"])
                if source_min >= anchor_dt:
                    days = 0
                else:
                    capped_max = min(source_max, anchor_dt - pd.Timedelta(days=1))
                    days = int((capped_max - source_min).days) + 1 if capped_max >= source_min else 0
            days_list.append(days)
            supports.append(support_status_from_days(days))
        out[days_col] = days_list
        out[support_col] = supports
    return out


def render_coverage_audit_md(audit_df):
    base_cols = [
        "source",
        "source_file",
        "rows",
        "date_min",
        "date_max",
        "distinct_days",
        "duplicate_rows_on_key",
        "sku_mapping_rate",
        "style_mapping_rate",
        "buyer_mapping_rate",
    ]
    anchor_cols = []
    for anchor in ANCHORS:
        anchor_cols.extend([f"{anchor_tag(anchor)}_available_days", f"{anchor_tag(anchor)}_support"])

    lines = [
        "# Phase8 Data Coverage Audit",
        "",
        "## Summary",
        "",
        "- Official label logic remains frozen before client reply.",
        "- This audit only evaluates data coverage and feature-table feasibility.",
        "- Support status is defined against a 90-day lookback requirement.",
        "",
        "## Source Coverage",
        "",
        markdown_table(audit_df, base_cols),
        "",
        "## Anchor Support",
        "",
        markdown_table(audit_df, ["source", *anchor_cols]),
        "",
        "## Current Default Judgement",
        "",
        "- Inventory snapshots are 2026-only in the current workspace, so they do not support the official 2025 four-anchor replacement compare.",
        "- Preorder snapshots are sparse and 2026-heavy in the current workspace, so they do not support the official 2025 four-anchor replacement compare.",
        f"- Event source currently resolves to `{audit_df.loc[audit_df['source'] == 'event_intent_daily_features', 'source_file'].iloc[0]}`.",
        "- Event data starts on 2025-09-18, so it is only shadow-experiment eligible for `2025-10/11/12`, and still partial against a 90-day lookback.",
    ]
    return "\n".join(lines)


def load_gold_context():
    gold = pd.read_csv(GOLD_DIR / "wide_table_sku.csv")
    gold["date"] = pd.to_datetime(gold["date"])
    gold["sku_id"] = gold["sku_id"].astype(str)
    static_cols = [
        "sku_id",
        "product_name",
        "category",
        "sub_category",
        "style_id",
        "season",
        "series",
        "band",
        "size_id",
        "color_id",
        "qty_first_order",
        "price_tag",
    ]
    static_info = gold[static_cols].drop_duplicates("sku_id")
    daily = (
        gold.groupby(["sku_id", "date"], as_index=False)
        .agg(qty_replenish=("qty_replenish", "sum"), qty_future=("qty_future", "sum"))
    )
    return daily, static_info


def attach_current_context(detail_df, anchor_date, daily_gold, static_info):
    anchor_dt = pd.Timestamp(anchor_date)
    hist_start = anchor_dt - pd.Timedelta(days=LOOKBACK)
    hist = daily_gold[(daily_gold["date"] >= hist_start) & (daily_gold["date"] < anchor_dt)].copy()
    context = (
        hist.groupby("sku_id", as_index=False)
        .agg(
            lookback_repl_days_90=("qty_replenish", lambda s: int((s > 0).sum())),
            lookback_future_days_90=("qty_future", lambda s: int((s > 0).sum())),
            lookback_repl_sum_90=("qty_replenish", "sum"),
            lookback_future_sum_90=("qty_future", "sum"),
        )
    )
    out = detail_df.merge(static_info, on="sku_id", how="left")
    out = out.merge(context, on="sku_id", how="left")
    for col in [
        "lookback_repl_days_90",
        "lookback_future_days_90",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
    ]:
        out[col] = out[col].fillna(0.0)

    out["activity_bucket"] = out["lookback_repl_days_90"].astype(int).map(activity_bucket_from_days)
    out["signal_quadrant"] = "repl0_fut0"
    out.loc[(out["lookback_repl_days_90"] > 0) & (out["lookback_future_days_90"] == 0), "signal_quadrant"] = "repl1_fut0"
    out.loc[(out["lookback_repl_days_90"] == 0) & (out["lookback_future_days_90"] > 0), "signal_quadrant"] = "repl0_fut1"
    out.loc[(out["lookback_repl_days_90"] > 0) & (out["lookback_future_days_90"] > 0), "signal_quadrant"] = "repl1_fut1"
    out["qfo_bucket"] = out["qty_first_order"].fillna(0).astype(float).map(qfo_bucket)
    out["is_true_blockbuster"] = (out["true_replenish_qty"] > 25).astype(int)
    out["is_cold_start"] = (out["lookback_repl_days_90"] <= 0).astype(int)
    out["is_repl0_fut0"] = (out["signal_quadrant"] == "repl0_fut0").astype(int)
    out["zero_true_fp"] = ((out["true_replenish_qty"] <= 0) & (out["ai_pred_qty"] > 0)).astype(int)
    return out


def load_current_phase7_contexts():
    from src.models.tabular_hurdle import TabularHurdleModel
    from src.analysis.phase_eval_utils import evaluate_context_frame

    daily_gold, static_info = load_gold_context()
    all_rows = []
    anchor_eval_rows = []
    feature_names = None
    cls_contrib_list = []
    reg_contrib_list = []

    for anchor in ANCHORS:
        tag = anchor_tag(anchor)
        processed_dir = DATA_DIR / f"processed_v6_event_p7b_{tag}_v6_event"
        artifacts_dir = DATA_DIR / f"artifacts_v6_event_p7b_{tag}_v6_event"
        exp_id = CURRENT_MAINLINE_TEMPLATE.format(anchor_tag=tag)
        cls_path = CURRENT_MODEL_DIR / f"{exp_id}_cls.pkl"
        reg_path = CURRENT_MODEL_DIR / f"{exp_id}_reg.pkl"
        meta_path = CURRENT_MODEL_DIR / f"{exp_id}_meta.json"

        model, _ = TabularHurdleModel.load(str(cls_path), str(reg_path), str(meta_path))
        with open(meta_path, "r", encoding="utf-8") as fh:
            model_meta = json.load(fh)

        x_val = np.load(processed_dir / "X_val.npy", mmap_mode="r")
        y_val_cls = np.load(processed_dir / "y_val_cls.npy", mmap_mode="r")
        y_val_reg = np.load(processed_dir / "y_val_reg.npy", mmap_mode="r")
        val_keys = pd.read_csv(artifacts_dir / "val_keys.csv")

        selected_idx = model_meta["selected_feature_indices"]
        selected_cols = model_meta["selected_feature_cols"]
        if feature_names is None:
            feature_names = selected_cols
        elif feature_names != selected_cols:
            raise ValueError("Selected feature columns differ across anchors.")

        x_sel = np.nan_to_num(np.asarray(x_val[:, selected_idx], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        pred = model.predict_quantity(x_sel, gate_mode="hard", gate_threshold=0.27)
        actual_qty = np.expm1(y_val_reg).astype(np.float32)
        precision, recall, thresholds = precision_recall_curve(y_val_cls, pred["prob"])
        f1_scores = 2 * (precision * recall) / np.maximum(precision + recall, 1e-8)
        best_threshold = 0.5 if len(thresholds) == 0 else float(thresholds[int(np.argmax(f1_scores[:-1]))])
        cls_pred_best_f1 = (pred["prob"] >= best_threshold).astype(int)

        detail = pd.DataFrame(
            {
                "sku_id": val_keys["sku_id"].astype(str),
                "anchor_date": pd.to_datetime(val_keys["date"]),
                "true_replenish_qty": actual_qty,
                "ai_pred_prob": pred["prob"],
                "cls_pred_best_f1": cls_pred_best_f1,
                "ai_pred_qty_open_raw": pred["qty_open"],
                "ai_pred_qty_raw": pred["qty"],
                "ai_pred_positive_qty_raw": (pred["qty"] > 0).astype(int),
                "qty_gate_mask": pred["gate_mask"].astype(int),
                "dead_blocked": np.zeros(len(actual_qty), dtype=int),
            }
        )
        scale = float(CALIBRATION_SCALES[anchor])
        detail["calibration_scale"] = scale
        detail["ai_pred_qty_open"] = detail["ai_pred_qty_open_raw"] * scale
        detail["ai_pred_qty"] = detail["ai_pred_qty_raw"] * scale
        detail["ai_pred_positive_qty"] = (detail["ai_pred_qty"] > 0).astype(int)
        detail["abs_error"] = (detail["ai_pred_qty"] - detail["true_replenish_qty"]).abs()
        detail = attach_current_context(detail, anchor, daily_gold, static_info)
        all_rows.append(detail)

        anchor_eval = evaluate_context_frame(detail.copy(), exp_id)
        anchor_eval["anchor_date"] = anchor
        anchor_eval_rows.append(anchor_eval)

        cls_contrib_list.append(model.classifier.booster_.predict(x_sel, pred_contrib=True))
        reg_contrib_list.append(model.regressor.booster_.predict(x_sel, pred_contrib=True))

    combined = pd.concat(all_rows, ignore_index=True)
    anchor_eval_df = pd.DataFrame(anchor_eval_rows)
    cls_contrib_all = np.vstack(cls_contrib_list)
    reg_contrib_all = np.vstack(reg_contrib_list)
    return combined, anchor_eval_df, feature_names, cls_contrib_all, reg_contrib_all


def build_residual_outputs(eval_all, anchor_eval_df):
    metric_keys = [
        "global_ratio",
        "global_wmape",
        "4_25_under_wape",
        "4_25_sku_p50",
        "ice_4_25_sku_p50",
        "blockbuster_under_wape",
        "blockbuster_sku_p50",
        "top20_true_volume_capture",
        "rank_corr_positive_skus",
        "1_3_ratio",
    ]
    summary_metrics = {
        key: float(anchor_eval_df[key].astype(float).mean()) if key in anchor_eval_df.columns else np.nan
        for key in metric_keys
    }
    summary_row = {
        "anchor_date": "all_anchors_mean",
        **{key: summary_metrics.get(key) for key in metric_keys},
    }
    eval_table = pd.concat([anchor_eval_df, pd.DataFrame([summary_row])], ignore_index=True, sort=False)
    eval_table.to_csv(REPORT_DIR / "phase8_current_mainline_anchor_eval.csv", index=False, encoding="utf-8-sig")
    eval_all.to_csv(REPORT_DIR / "phase8_current_mainline_eval_context_all_anchors.csv", index=False, encoding="utf-8-sig")

    block = eval_all[eval_all["true_replenish_qty"] > 25].copy()
    weak_mask = block["signal_quadrant"].isin(["repl0_fut0", "repl1_fut0"])
    weak_block = block[weak_mask].copy()
    weak_block["under_qty"] = np.clip(weak_block["true_replenish_qty"] - weak_block["ai_pred_qty"], a_min=0, a_max=None)
    weak_block = (
        weak_block.groupby("signal_quadrant", as_index=False)
        .agg(rows=("sku_id", "size"), total_true=("true_replenish_qty", "sum"), total_pred=("ai_pred_qty", "sum"), under_sum=("under_qty", "sum"))
    )
    if not weak_block.empty:
        weak_block["ratio"] = weak_block["total_pred"] / weak_block["total_true"]
        weak_block["under_wape"] = weak_block["under_sum"] / weak_block["total_true"]

    block["under_qty"] = np.clip(block["true_replenish_qty"] - block["ai_pred_qty"], a_min=0, a_max=None)
    category_under = (
        block.groupby("category", as_index=False)
        .agg(rows=("sku_id", "size"), total_true=("true_replenish_qty", "sum"), total_pred=("ai_pred_qty", "sum"), under_sum=("under_qty", "sum"))
    )
    category_under = category_under[category_under["total_true"] > 0].copy()
    if not category_under.empty:
        category_under["under_wape"] = category_under["under_sum"] / category_under["total_true"]
        category_under["ratio"] = category_under["total_pred"] / category_under["total_true"]
        category_under = category_under.sort_values(["under_wape", "total_true"], ascending=[False, False]).head(10)

    zero_true = eval_all[eval_all["true_replenish_qty"] <= 0].copy()
    zero_fp = zero_true[zero_true["ai_pred_qty"] > 0].copy()
    zero_stats = {
        "zero_true_rows": int(len(zero_true)),
        "false_positive_rows": int(len(zero_fp)),
        "false_positive_rate_zero_true": safe_div(len(zero_fp), len(zero_true)),
        "pred_sum_on_zero_true": float(zero_fp["ai_pred_qty"].sum()) if not zero_fp.empty else 0.0,
        "pred_ge_3_rows": int((zero_true["ai_pred_qty"] >= 3).sum()),
        "pred_ge_5_rows": int((zero_true["ai_pred_qty"] >= 5).sum()),
        "pred_ge_10_rows": int((zero_true["ai_pred_qty"] >= 10).sum()),
    }

    key_categories = set(category_under["category"].head(5).astype(str)) if not category_under.empty else set()
    case_frames = []

    weak_cases = block[
        block["signal_quadrant"].isin(["repl0_fut0", "repl1_fut0"]) & (block["ai_pred_qty"] < block["true_replenish_qty"])
    ].copy()
    weak_cases["case_type"] = "weak_signal_blockbuster_under"
    case_frames.append(weak_cases.sort_values(["abs_error", "true_replenish_qty"], ascending=[False, False]).head(30))

    zero_cases = zero_fp.copy()
    zero_cases["case_type"] = "zero_true_false_positive"
    case_frames.append(zero_cases.sort_values(["ai_pred_qty", "ai_pred_prob"], ascending=[False, False]).head(30))

    category_cases = block[block["category"].astype(str).isin(key_categories) & (block["ai_pred_qty"] < block["true_replenish_qty"])].copy()
    category_cases["case_type"] = "key_category_tail_under"
    case_frames.append(category_cases.sort_values(["abs_error", "true_replenish_qty"], ascending=[False, False]).head(30))

    small_over = eval_all[(eval_all["true_replenish_qty"] <= 3) & (eval_all["ai_pred_qty"] >= 5)].copy()
    small_over["case_type"] = "small_demand_overpredict"
    case_frames.append(small_over.sort_values(["ai_pred_qty", "abs_error"], ascending=[False, False]).head(30))

    cases = pd.concat(case_frames, ignore_index=True, sort=False)
    case_cols = [
        "case_type",
        "anchor_date",
        "sku_id",
        "product_name",
        "category",
        "style_id",
        "true_replenish_qty",
        "ai_pred_qty",
        "ai_pred_prob",
        "abs_error",
        "signal_quadrant",
        "activity_bucket",
        "qty_first_order",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
    ]
    cases[case_cols].to_csv(REPORT_DIR / "phase8_residual_gap_cases.csv", index=False, encoding="utf-8-sig")

    with open(PROJECT_ROOT / "reports" / "current" / "current_mainline.json", "r", encoding="utf-8") as fh:
        official = json.load(fh)
    official_metrics = official["key_metrics"]
    recomputed = summary_row

    lines = [
        "# Phase8 Residual Gap Summary",
        "",
        "## Scope",
        "",
        "- Source: current official phase7 mainline",
        "- Evaluation base: recomputed four-anchor official mainline contexts with official Sep/Oct calibration scales",
        "- Metric summary rule: per-anchor evaluation first, then four-anchor mean; combined all-anchor context is used only for residual case mining",
        "- Formal winner status remains unchanged; this is a residual-gap analysis only",
        "",
        "## Official Metrics vs Recomputed Check",
        "",
        "| metric | official_phase7 | recomputed_all_anchors |",
        "| --- | --- | --- |",
    ]
    for key in metric_keys:
        lines.append(f"| {key} | {float(official_metrics[key]):.4f} | {float(recomputed[key]):.4f} |")

    lines.extend(
        [
            "",
            "## Weak-Signal Blockbuster Residuals",
            "",
            markdown_table(
                weak_block if not weak_block.empty else pd.DataFrame([{"signal_quadrant": "none", "rows": 0, "total_true": 0.0, "total_pred": 0.0, "ratio": np.nan, "under_wape": np.nan}]),
                ["signal_quadrant", "rows", "total_true", "total_pred", "ratio", "under_wape"],
            ),
            "",
            "## Blockbuster Worst Categories",
            "",
            markdown_table(
                category_under if not category_under.empty else pd.DataFrame([{"category": "none", "rows": 0, "total_true": 0.0, "total_pred": 0.0, "ratio": np.nan, "under_wape": np.nan}]),
                ["category", "rows", "total_true", "total_pred", "ratio", "under_wape"],
            ),
            "",
            "## Zero-True False Positives",
            "",
            f"- zero_true_rows: `{zero_stats['zero_true_rows']}`",
            f"- false_positive_rows: `{zero_stats['false_positive_rows']}`",
            f"- false_positive_rate_zero_true: `{zero_stats['false_positive_rate_zero_true']:.4f}`",
            f"- pred_sum_on_zero_true: `{zero_stats['pred_sum_on_zero_true']:.4f}`",
            f"- pred_ge_3_rows: `{zero_stats['pred_ge_3_rows']}`",
            f"- pred_ge_5_rows: `{zero_stats['pred_ge_5_rows']}`",
            f"- pred_ge_10_rows: `{zero_stats['pred_ge_10_rows']}`",
            "",
            "## Output Files",
            "",
            "- `reports/phase8a_prep/phase8_current_mainline_eval_context_all_anchors.csv`",
            "- `reports/phase8a_prep/phase8_current_mainline_anchor_eval.csv`",
            "- `reports/phase8a_prep/phase8_residual_gap_cases.csv`",
        ]
    )
    (REPORT_DIR / "phase8_residual_gap_summary.md").write_text("\n".join(lines), encoding="utf-8-sig")


def format_top_features(contrib_row, feature_names, positive=True, topn=5):
    values = np.asarray(contrib_row[:-1], dtype=float)
    order = np.argsort(values)[::-1] if positive else np.argsort(values)
    out = []
    for idx in order:
        value = float(values[idx])
        if positive and value <= 0:
            continue
        if (not positive) and value >= 0:
            continue
        out.append(f"{feature_names[idx]}({value:.4f})")
        if len(out) >= topn:
            break
    return "; ".join(out) if out else ""


def build_shap_outputs(eval_all, feature_names, cls_contrib_all, reg_contrib_all):
    subsets = {
        "all": np.ones(len(eval_all), dtype=bool),
        "4_25": ((eval_all["true_replenish_qty"] >= 4) & (eval_all["true_replenish_qty"] <= 25)).to_numpy(),
        "blockbuster": (eval_all["true_replenish_qty"] > 25).to_numpy(),
        "zero_true_fp": ((eval_all["true_replenish_qty"] <= 0) & (eval_all["ai_pred_qty"] > 0)).to_numpy(),
    }

    rows = []
    for subset_name, mask in subsets.items():
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        for component, contrib in [("classifier_raw_score", cls_contrib_all), ("regressor_log_qty", reg_contrib_all)]:
            vals = contrib[idx, :-1]
            mean_abs = np.mean(np.abs(vals), axis=0)
            mean_signed = np.mean(vals, axis=0)
            order = np.argsort(-mean_abs)[:20]
            for rank, j in enumerate(order, start=1):
                rows.append(
                    {
                        "subset": subset_name,
                        "component": component,
                        "sample_count": int(idx.size),
                        "rank": rank,
                        "feature": feature_names[j],
                        "mean_abs_shap": float(mean_abs[j]),
                        "mean_shap": float(mean_signed[j]),
                    }
                )
    global_df = pd.DataFrame(rows)
    global_df.to_csv(REPORT_DIR / "phase8_shap_global_summary.csv", index=False, encoding="utf-8-sig")

    local_specs = [
        (
            "repl0_fut0_blockbuster_under",
            eval_all[(eval_all["true_replenish_qty"] > 25) & (eval_all["signal_quadrant"] == "repl0_fut0") & (eval_all["ai_pred_qty"] < eval_all["true_replenish_qty"])].sort_values(
                ["abs_error", "true_replenish_qty"], ascending=[False, False]
            ).head(5),
        ),
        (
            "repl1_fut0_blockbuster_under",
            eval_all[(eval_all["true_replenish_qty"] > 25) & (eval_all["signal_quadrant"] == "repl1_fut0") & (eval_all["ai_pred_qty"] < eval_all["true_replenish_qty"])].sort_values(
                ["abs_error", "true_replenish_qty"], ascending=[False, False]
            ).head(5),
        ),
        (
            "zero_true_false_positive",
            eval_all[(eval_all["true_replenish_qty"] <= 0) & (eval_all["ai_pred_qty"] > 0)].sort_values(
                ["ai_pred_qty", "ai_pred_prob"], ascending=[False, False]
            ).head(5),
        ),
        (
            "key_category_tail_under",
            eval_all[
                (eval_all["true_replenish_qty"] > 25)
                & (eval_all["category"].astype(str).isin(["棉服", "毛衣开衫", "外套", "卫衣", "毛衣"]))
                & (eval_all["ai_pred_qty"] < eval_all["true_replenish_qty"])
            ].sort_values(["abs_error", "true_replenish_qty"], ascending=[False, False]).head(5),
        ),
    ]

    local_rows = []
    for case_type, case_df in local_specs:
        for idx in case_df.index.tolist():
            row = eval_all.loc[idx]
            local_rows.append(
                {
                    "case_type": case_type,
                    "anchor_date": str(pd.Timestamp(row["anchor_date"]).date()),
                    "sku_id": row["sku_id"],
                    "product_name": row.get("product_name", ""),
                    "category": row.get("category", ""),
                    "style_id": row.get("style_id", ""),
                    "true_replenish_qty": float(row["true_replenish_qty"]),
                    "ai_pred_qty": float(row["ai_pred_qty"]),
                    "ai_pred_prob": float(row["ai_pred_prob"]),
                    "signal_quadrant": row.get("signal_quadrant", ""),
                    "activity_bucket": row.get("activity_bucket", ""),
                    "qty_first_order": float(row.get("qty_first_order", 0.0) or 0.0),
                    "lookback_repl_sum_90": float(row.get("lookback_repl_sum_90", 0.0) or 0.0),
                    "lookback_future_sum_90": float(row.get("lookback_future_sum_90", 0.0) or 0.0),
                    "top_classifier_positive_features": format_top_features(cls_contrib_all[idx], feature_names, positive=True),
                    "top_classifier_negative_features": format_top_features(cls_contrib_all[idx], feature_names, positive=False),
                    "top_regressor_positive_features": format_top_features(reg_contrib_all[idx], feature_names, positive=True),
                    "top_regressor_negative_features": format_top_features(reg_contrib_all[idx], feature_names, positive=False),
                }
            )
    local_df = pd.DataFrame(local_rows)
    local_df.to_csv(REPORT_DIR / "phase8_shap_local_cases.csv", index=False, encoding="utf-8-sig")

    def top_rows(subset, component):
        return global_df[(global_df["subset"] == subset) & (global_df["component"] == component)].head(10)

    lines = [
        "# Phase8 SHAP Summary",
        "",
        "## Scope",
        "",
        "- Model: current official phase7 LightGBM hurdle mainline",
        "- Method: LightGBM native `pred_contrib=True` contribution outputs",
        "- Classifier contributions are in raw-score space",
        "- Regressor contributions are in log-quantity space",
        "- Sep/Oct external calibration scales are not part of the additive SHAP decomposition",
        "",
    ]
    for subset in ["all", "4_25", "blockbuster", "zero_true_fp"]:
        lines.extend(
            [
                f"## Top Features: {subset}",
                "",
                "### Classifier",
                "",
                markdown_table(
                    top_rows(subset, "classifier_raw_score")
                    if not top_rows(subset, "classifier_raw_score").empty
                    else pd.DataFrame([{"feature": "none", "mean_abs_shap": np.nan, "mean_shap": np.nan, "sample_count": 0}]),
                    ["feature", "mean_abs_shap", "mean_shap", "sample_count"],
                ),
                "",
                "### Regressor",
                "",
                markdown_table(
                    top_rows(subset, "regressor_log_qty")
                    if not top_rows(subset, "regressor_log_qty").empty
                    else pd.DataFrame([{"feature": "none", "mean_abs_shap": np.nan, "mean_shap": np.nan, "sample_count": 0}]),
                    ["feature", "mean_abs_shap", "mean_shap", "sample_count"],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Output Files",
            "",
            "- `reports/phase8a_prep/phase8_shap_global_summary.csv`",
            "- `reports/phase8a_prep/phase8_shap_local_cases.csv`",
        ]
    )
    (REPORT_DIR / "phase8_shap_summary.md").write_text("\n".join(lines), encoding="utf-8-sig")


def write_shadow_policy():
    text = """# Phase8 Shadow Experiment Policy

## Status

- Current official mainline remains `tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`.
- Client-facing order semantic questions are still unresolved.
- No experiment in this stage is allowed to replace the official phase7 mainline.

## Allowed Before Client Reply

- Event-only shadow experiments on `2025-10/11/12`
- Inventory and preorder exploratory analysis on `2026` windows
- SHAP and residual-gap diagnostics on the current official phase7 mainline

## Not Allowed Before Client Reply

- Any formal replacement compare against the current official four-anchor phase7 winner
- Any change to the official label-cleaning logic for `V_IRS_ORDERFTP`
- Any claim that a new candidate should replace the current official mainline
- Any open-ended hyperparameter search

## Blocking Issues

- `TYPE` missing semantics in `V_IRS_ORDERFTP`
- `QTY < 0` semantics in `V_IRS_ORDERFTP`
- Duplicate-row business meaning in `V_IRS_ORDERFTP`
- Lifecycle table not yet provided
"""
    (REPORT_DIR / "phase8_shadow_experiment_policy.md").write_text(text, encoding="utf-8-sig")


def write_manifest(audit_df):
    event_row = audit_df[audit_df["source"] == "event_intent_daily_features"].iloc[0].to_dict()
    payload = {
        "status": "phase7_frozen_phase8a_prep_ready",
        "official_mainline": "tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093",
        "frozen_label_logic": True,
        "event_source": event_row.get("source_file", ""),
        "outputs": {
            "coverage_audit_csv": "reports/phase8a_prep/phase8_data_coverage_audit.csv",
            "coverage_audit_md": "reports/phase8a_prep/phase8_data_coverage_audit.md",
            "inventory_daily_features": "data/phase8a_prep/inventory_daily_features.csv",
            "preorder_daily_features": "data/phase8a_prep/preorder_daily_features.csv",
            "event_intent_daily_features": "data/phase8a_prep/event_intent_daily_features.csv",
            "current_eval_context_all_anchors": "reports/phase8a_prep/phase8_current_mainline_eval_context_all_anchors.csv",
            "current_anchor_eval": "reports/phase8a_prep/phase8_current_mainline_anchor_eval.csv",
            "residual_summary": "reports/phase8a_prep/phase8_residual_gap_summary.md",
            "residual_cases": "reports/phase8a_prep/phase8_residual_gap_cases.csv",
            "shap_global": "reports/phase8a_prep/phase8_shap_global_summary.csv",
            "shap_local": "reports/phase8a_prep/phase8_shap_local_cases.csv",
            "shap_summary": "reports/phase8a_prep/phase8_shap_summary.md",
            "shadow_policy": "reports/phase8a_prep/phase8_shadow_experiment_policy.md",
        },
        "anchor_support": audit_df.to_dict(orient="records"),
    }
    (REPORT_DIR / "phase8a_prep_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ensure_dirs()

    products = pd.read_csv(SILVER_DIR / "clean_products.csv")
    products["sku_id"] = products["sku_id"].astype(str)
    products["style_id"] = products["style_id"].astype(str)
    stores = pd.read_csv(SILVER_DIR / "clean_stores.csv")
    orders = pd.read_csv(SILVER_DIR / "clean_orders.csv")
    valid_buyers = set(pd.concat([stores["buyer_id"], orders["buyer_id"]]).astype(str).unique().tolist())

    inventory_features, inventory_audit = build_inventory_daily_features(products)
    preorder_features, preorder_audit = build_preorder_daily_features(products, valid_buyers)
    event_features, event_audit = build_event_intent_daily_features(products, valid_buyers)

    inventory_features.to_csv(PHASE8_DATA_DIR / "inventory_daily_features.csv", index=False, encoding="utf-8-sig")
    preorder_features.to_csv(PHASE8_DATA_DIR / "preorder_daily_features.csv", index=False, encoding="utf-8-sig")
    event_features.to_csv(PHASE8_DATA_DIR / "event_intent_daily_features.csv", index=False, encoding="utf-8-sig")

    audit_df = pd.DataFrame([inventory_audit, preorder_audit, event_audit])
    audit_df = add_anchor_support(audit_df)
    audit_df.to_csv(REPORT_DIR / "phase8_data_coverage_audit.csv", index=False, encoding="utf-8-sig")
    (REPORT_DIR / "phase8_data_coverage_audit.md").write_text(render_coverage_audit_md(audit_df), encoding="utf-8-sig")

    eval_all, anchor_eval_df, feature_names, cls_contrib_all, reg_contrib_all = load_current_phase7_contexts()
    build_residual_outputs(eval_all, anchor_eval_df)
    build_shap_outputs(eval_all, feature_names, cls_contrib_all, reg_contrib_all)
    write_shadow_policy()
    write_manifest(audit_df)

    print(f"[OK] coverage audit -> {REPORT_DIR / 'phase8_data_coverage_audit.md'}")
    print(f"[OK] inventory features -> {PHASE8_DATA_DIR / 'inventory_daily_features.csv'}")
    print(f"[OK] preorder features -> {PHASE8_DATA_DIR / 'preorder_daily_features.csv'}")
    print(f"[OK] event features -> {PHASE8_DATA_DIR / 'event_intent_daily_features.csv'}")
    print(f"[OK] residual summary -> {REPORT_DIR / 'phase8_residual_gap_summary.md'}")
    print(f"[OK] shap summary -> {REPORT_DIR / 'phase8_shap_summary.md'}")


if __name__ == "__main__":
    main()
