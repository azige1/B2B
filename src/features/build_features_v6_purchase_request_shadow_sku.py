import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

PHASE8_SIGNAL_DIR = Path(
    os.environ.get(
        "PHASE8_SIGNAL_DIR",
        str(PROJECT_ROOT / "data" / "phase8a_prep"),
    )
)

from src.features.phase53_feature_utils import (  # noqa: E402
    FORECAST,
    LOOKBACK,
    NEG_STEP,
    days_since_last_positive,
    load_gold_frame,
    load_keep_skus_from_v5_lite,
    split_flags,
)


REQUEST_WINDOWS = (7, 14, 30, 60, 90)
REQUEST_COLS = [
    *[f"req_sku_qty_{window}" for window in REQUEST_WINDOWS],
    *[f"req_sku_rows_{window}" for window in REQUEST_WINDOWS],
    *[f"req_sku_days_{window}" for window in REQUEST_WINDOWS],
    *[f"req_sku_sources_{window}" for window in REQUEST_WINDOWS],
    *[f"req_style_qty_{window}" for window in REQUEST_WINDOWS],
    *[f"req_style_rows_{window}" for window in REQUEST_WINDOWS],
    *[f"req_style_days_{window}" for window in REQUEST_WINDOWS],
    *[f"req_style_sources_{window}" for window in REQUEST_WINDOWS],
    "req_days_since_last_sku",
    "req_days_since_last_style",
    "req_sku_qty_per_source_30",
    "req_style_qty_per_source_30",
    "req_sku_to_style_qty_share_30",
]
REQUEST_CORE_COLS = [
    "req_days_since_last_sku",
    "req_days_since_last_style",
    "req_sku_qty_90",
    "req_style_qty_60",
    "req_style_qty_90",
    "req_sku_sources_30",
    "req_style_sources_30",
    "req_style_days_30",
    "req_style_qty_per_source_30",
    "req_sku_to_style_qty_share_30",
]


def rolling_sum_matrix(matrix, window):
    if matrix.size == 0:
        return np.zeros_like(matrix, dtype=np.float32)
    padded = np.pad(matrix.astype(np.float64), ((0, 0), (1, 0)), mode="constant")
    csum = np.cumsum(padded, axis=1)
    idx = np.arange(1, matrix.shape[1] + 1)
    starts = np.maximum(0, idx - window)
    return (csum[:, idx] - csum[:, starts]).astype(np.float32)


def safe_rate(num, den):
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    out = np.zeros_like(num, dtype=np.float32)
    mask = den > 1e-6
    out[mask] = num[mask] / den[mask]
    return out


def load_base_paths(split_date):
    tag = split_date.replace("-", "")
    base_prefix = os.environ.get("PHASE8_REQ_BASE_PREFIX", "p7b").strip()
    output_prefix = os.environ.get(
        "PHASE8_REQ_OUTPUT_PREFIX",
        "p8reqshadow",
    ).strip()
    base_tag = f"{base_prefix}_{tag}_v6_event"
    output_tag = f"{output_prefix}_{tag}_v6_event"
    base_processed = PROJECT_ROOT / "data" / f"processed_v6_event_{base_tag}"
    base_artifacts = PROJECT_ROOT / "data" / f"artifacts_v6_event_{base_tag}"
    out_processed = PROJECT_ROOT / "data" / f"processed_v6_event_{output_tag}"
    out_artifacts = PROJECT_ROOT / "data" / f"artifacts_v6_event_{output_tag}"
    keep_tag = os.environ.get("PHASE8_REQ_KEEP_TAG", base_tag).strip() or base_tag
    out_processed.mkdir(parents=True, exist_ok=True)
    out_artifacts.mkdir(parents=True, exist_ok=True)
    return {
        "base_tag": base_tag,
        "keep_tag": keep_tag,
        "output_tag": output_tag,
        "base_processed": base_processed,
        "base_artifacts": base_artifacts,
        "out_processed": out_processed,
        "out_artifacts": out_artifacts,
    }


def preflight(paths):
    required = [
        paths["base_processed"] / "X_train.npy",
        paths["base_processed"] / "X_val.npy",
        paths["base_processed"] / "y_train_cls.npy",
        paths["base_processed"] / "y_train_reg.npy",
        paths["base_processed"] / "y_val_cls.npy",
        paths["base_processed"] / "y_val_reg.npy",
        paths["base_artifacts"] / "meta_v6_event.json",
        paths["base_artifacts"] / "val_keys.csv",
        PHASE8_SIGNAL_DIR / "purchase_request_daily_features.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))


def load_purchase_request_daily():
    path = PHASE8_SIGNAL_DIR / "purchase_request_daily_features.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df["sku_id"] = df["sku_id"].astype(str)
    df["style_id"] = df["style_id"].fillna("Unknown").astype(str)
    df["request_source_id"] = df["request_source_id"].fillna("").astype(str)
    df["request_qty"] = pd.to_numeric(df["request_qty"], errors="coerce").fillna(0.0)
    df["request_rows"] = pd.to_numeric(df["request_rows"], errors="coerce").fillna(0.0)
    return df


def build_rolling_source_counts(daily, entity_values, date_to_idx, n_days, entity_col):
    entity_to_idx = {value: idx for idx, value in enumerate(entity_values)}
    out = {window: np.zeros((len(entity_values), n_days), dtype=np.float32) for window in REQUEST_WINDOWS}
    grouped = (
        daily.groupby([entity_col, "date", "request_source_id"], as_index=False)
        .agg(request_qty=("request_qty", "sum"), request_rows=("request_rows", "sum"))
        .groupby(entity_col)
    )

    for entity, entity_df in tqdm(grouped, desc=f"request_sources_{entity_col}", total=len(grouped)):
        entity_idx = entity_to_idx.get(str(entity))
        if entity_idx is None:
            continue

        day_maps = [dict() for _ in range(n_days)]
        for row in entity_df.itertuples(index=False):
            day_idx = date_to_idx.get(row.date)
            if day_idx is None:
                continue
            source_id = str(row.request_source_id)
            if not source_id:
                continue
            weight = float(row.request_qty) if float(row.request_qty) > 0 else float(row.request_rows)
            if weight <= 0:
                continue
            day_maps[day_idx][source_id] = day_maps[day_idx].get(source_id, 0.0) + weight

        for window in REQUEST_WINDOWS:
            source_day_count = {}
            source_qty_sum = {}
            for day_idx in range(n_days):
                for source_id, qty in day_maps[day_idx].items():
                    source_day_count[source_id] = source_day_count.get(source_id, 0) + 1
                    source_qty_sum[source_id] = source_qty_sum.get(source_id, 0.0) + float(qty)

                old_idx = day_idx - window
                if old_idx >= 0:
                    for source_id, qty in day_maps[old_idx].items():
                        source_day_count[source_id] -= 1
                        if source_day_count[source_id] <= 0:
                            source_day_count.pop(source_id, None)
                        source_qty_sum[source_id] -= float(qty)
                        if source_qty_sum[source_id] <= 1e-9:
                            source_qty_sum.pop(source_id, None)

                out[window][entity_idx, day_idx] = float(len(source_qty_sum))

    return out


def build_entity_request_arrays(daily, entity_values, date_to_idx, n_days, entity_col):
    entity_to_idx = {value: idx for idx, value in enumerate(entity_values)}
    qty_matrix = np.zeros((len(entity_values), n_days), dtype=np.float32)
    row_matrix = np.zeros((len(entity_values), n_days), dtype=np.float32)
    grouped = (
        daily.groupby([entity_col, "date"], as_index=False)
        .agg(request_qty=("request_qty", "sum"), request_rows=("request_rows", "sum"))
    )
    for row in grouped.itertuples(index=False):
        entity_idx = entity_to_idx.get(str(getattr(row, entity_col)))
        day_idx = date_to_idx.get(row.date)
        if entity_idx is None or day_idx is None:
            continue
        qty_matrix[entity_idx, day_idx] = float(row.request_qty)
        row_matrix[entity_idx, day_idx] = float(row.request_rows)

    active_matrix = (row_matrix > 0).astype(np.float32)
    rolling = {
        "qty": {window: rolling_sum_matrix(qty_matrix, window) for window in REQUEST_WINDOWS},
        "rows": {window: rolling_sum_matrix(row_matrix, window) for window in REQUEST_WINDOWS},
        "days": {window: rolling_sum_matrix(active_matrix, window) for window in REQUEST_WINDOWS},
        "sources": build_rolling_source_counts(daily, entity_values, date_to_idx, n_days, entity_col),
    }
    recency = np.zeros((len(entity_values), n_days), dtype=np.float32)
    for idx in range(len(entity_values)):
        recency[idx] = days_since_last_positive(active_matrix[idx])
    rolling["recency"] = recency
    return rolling


def request_row_values(sku_cache, style_cache, sku_idx, style_idx, day_idx):
    values = []
    for window in REQUEST_WINDOWS:
        values.append(float(sku_cache["qty"][window][sku_idx, day_idx]))
    for window in REQUEST_WINDOWS:
        values.append(float(sku_cache["rows"][window][sku_idx, day_idx]))
    for window in REQUEST_WINDOWS:
        values.append(float(sku_cache["days"][window][sku_idx, day_idx]))
    for window in REQUEST_WINDOWS:
        values.append(float(sku_cache["sources"][window][sku_idx, day_idx]))
    for window in REQUEST_WINDOWS:
        values.append(float(style_cache["qty"][window][style_idx, day_idx]))
    for window in REQUEST_WINDOWS:
        values.append(float(style_cache["rows"][window][style_idx, day_idx]))
    for window in REQUEST_WINDOWS:
        values.append(float(style_cache["days"][window][style_idx, day_idx]))
    for window in REQUEST_WINDOWS:
        values.append(float(style_cache["sources"][window][style_idx, day_idx]))

    sku_qty_30 = float(sku_cache["qty"][30][sku_idx, day_idx])
    sku_sources_30 = float(sku_cache["sources"][30][sku_idx, day_idx])
    style_qty_30 = float(style_cache["qty"][30][style_idx, day_idx])
    style_sources_30 = float(style_cache["sources"][30][style_idx, day_idx])
    values.extend(
        [
            float(sku_cache["recency"][sku_idx, day_idx]),
            float(style_cache["recency"][style_idx, day_idx]),
            sku_qty_30 / sku_sources_30 if sku_sources_30 > 1e-6 else 0.0,
            style_qty_30 / style_sources_30 if style_sources_30 > 1e-6 else 0.0,
            sku_qty_30 / style_qty_30 if style_qty_30 > 1e-6 else 0.0,
        ]
    )
    return values


def main():
    split_date = os.environ.get("FEATURE_SPLIT_DATE", "").strip()
    if not split_date:
        raise ValueError("FEATURE_SPLIT_DATE is required.")
    val_mode = os.environ.get("FEATURE_VAL_MODE", "single_anchor").strip().lower()
    if val_mode != "single_anchor":
        raise ValueError("Purchase request shadow builder only supports FEATURE_VAL_MODE=single_anchor.")

    paths = load_base_paths(split_date)
    preflight(paths)

    print("=" * 72)
    print("[Phase8 shadow] build v6_event + purchase_request shadow assets")
    print(f"split_date={split_date} | output_tag={paths['output_tag']}")
    print("=" * 72)

    with open(paths["base_artifacts"] / "meta_v6_event.json", "r", encoding="utf-8") as fh:
        base_meta = json.load(fh)

    base_x_train = np.load(paths["base_processed"] / "X_train.npy", mmap_mode="r")
    base_x_val = np.load(paths["base_processed"] / "X_val.npy", mmap_mode="r")
    train_cnt = int(base_meta["train_cnt"])
    val_cnt = int(base_meta["val_cnt"])
    base_dim = int(base_x_train.shape[1])
    request_dim = len(REQUEST_COLS)

    keep_skus, _ = load_keep_skus_from_v5_lite(paths["keep_tag"])
    gold = load_gold_frame(keep_skus)
    dyn_agg = (
        gold.groupby(["sku_id", "date"], as_index=False)
        .agg(qty_replenish=("qty_replenish", "sum"))
    )
    static_source = gold[["sku_id", "style_id"]].drop_duplicates("sku_id").copy()
    static_source["sku_id"] = static_source["sku_id"].astype(str)
    static_source["style_id"] = static_source["style_id"].fillna("Unknown").astype(str)
    style_by_sku = dict(zip(static_source["sku_id"], static_source["style_id"]))

    calendar_start = base_meta.get("calendar_start")
    calendar_end = base_meta.get("calendar_end")
    if not calendar_start or not calendar_end:
        calendar_mode = os.environ.get("PHASE8_REQ_CALENDAR_MODE", "2025_only").strip().lower()
        if calendar_mode == "extended":
            calendar_start = str(pd.to_datetime(gold["date"]).min().date())
            calendar_end = str(pd.to_datetime(gold["date"]).max().date())
        elif calendar_mode == "2025_only":
            calendar_start = "2025-01-01"
            calendar_end = "2025-12-31"
        else:
            raise ValueError(
                f"Unsupported PHASE8_REQ_CALENDAR_MODE={calendar_mode}. "
                "Expected one of: 2025_only, extended."
            )
    all_dates = pd.date_range(calendar_start, calendar_end, freq="D").date
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    split_date_obj = pd.to_datetime(split_date).date()
    end_idx = len(all_dates) - FORECAST

    sku_list = sorted(dyn_agg["sku_id"].astype(str).unique())
    sku_to_idx = {sku: idx for idx, sku in enumerate(sku_list)}
    repl_matrix = np.zeros((len(sku_list), len(all_dates)), dtype=np.float32)
    for row in dyn_agg.itertuples(index=False):
        sku_idx = sku_to_idx.get(str(row.sku_id))
        day_idx = date_to_idx.get(row.date)
        if sku_idx is None or day_idx is None:
            continue
        repl_matrix[sku_idx, day_idx] = float(row.qty_replenish)

    request_daily = load_purchase_request_daily()
    request_daily = request_daily[request_daily["sku_id"].isin(set(sku_list))].copy()

    style_values = sorted(set(static_source["style_id"].astype(str)) | set(request_daily["style_id"].astype(str)))
    style_to_idx = {style: idx for idx, style in enumerate(style_values)}
    default_style_idx = style_to_idx.get("Unknown", 0)

    print(f"[{time.strftime('%H:%M:%S')}] build request rolling caches")
    sku_cache = build_entity_request_arrays(request_daily, sku_list, date_to_idx, len(all_dates), "sku_id")
    style_cache = build_entity_request_arrays(request_daily, style_values, date_to_idx, len(all_dates), "style_id")

    request_train = np.zeros((train_cnt, request_dim), dtype=np.float32)
    request_val = np.zeros((val_cnt, request_dim), dtype=np.float32)
    train_keys = []
    val_keys = []
    train_idx = 0
    val_idx = 0

    print(f"[{time.strftime('%H:%M:%S')}] build purchase request shadow rows")
    for sku in tqdm(sku_list, desc="request_windows"):
        sku_idx = sku_to_idx[str(sku)]
        style_id = style_by_sku.get(str(sku), "Unknown")
        style_idx = style_to_idx.get(style_id, default_style_idx)
        repl = repl_matrix[sku_idx]

        for i in range(LOOKBACK - 1, end_idx):
            target = float(repl[i + 1 : i + FORECAST + 1].sum())
            anchor_date = all_dates[i]
            is_train, is_val = split_flags(anchor_date, split_date_obj, val_mode)
            if not is_train and not is_val:
                continue
            if target == 0 and is_train and (i % NEG_STEP != 0):
                continue

            row = np.asarray(request_row_values(sku_cache, style_cache, sku_idx, style_idx, i), dtype=np.float32)
            if is_train:
                request_train[train_idx] = row
                train_keys.append({"sku_id": sku, "date": anchor_date})
                train_idx += 1
            else:
                request_val[val_idx] = row
                val_keys.append({"sku_id": sku, "date": anchor_date})
                val_idx += 1

    if train_idx != train_cnt or val_idx != val_cnt:
        raise ValueError(
            f"Purchase request shadow sample count mismatch: "
            f"expected train/val={train_cnt}/{val_cnt}, built={train_idx}/{val_idx}"
        )

    base_val_keys = pd.read_csv(paths["base_artifacts"] / "val_keys.csv")
    built_val_keys = pd.DataFrame(val_keys)
    built_val_keys["date"] = pd.to_datetime(built_val_keys["date"]).dt.strftime("%Y-%m-%d")
    base_val_norm = base_val_keys.copy()
    base_val_norm["date"] = pd.to_datetime(base_val_norm["date"]).dt.strftime("%Y-%m-%d")
    if not built_val_keys.equals(base_val_norm):
        raise ValueError("Purchase request shadow val_keys do not match base val_keys; sample order is not aligned.")

    out_x_train = open_memmap(
        paths["out_processed"] / "X_train.npy",
        mode="w+",
        dtype=np.float32,
        shape=(train_cnt, base_dim + request_dim),
    )
    out_x_val = open_memmap(
        paths["out_processed"] / "X_val.npy",
        mode="w+",
        dtype=np.float32,
        shape=(val_cnt, base_dim + request_dim),
    )
    out_x_train[:, :base_dim] = base_x_train
    out_x_train[:, base_dim:] = request_train
    out_x_val[:, :base_dim] = base_x_val
    out_x_val[:, base_dim:] = request_val
    out_x_train.flush()
    out_x_val.flush()

    for name in ["y_train_cls.npy", "y_train_reg.npy", "y_val_cls.npy", "y_val_reg.npy"]:
        shutil.copy2(paths["base_processed"] / name, paths["out_processed"] / name)

    for name in ["label_encoders_v6_event.pkl", "val_keys.csv"]:
        src = paths["base_artifacts"] / name
        if src.exists():
            shutil.copy2(src, paths["out_artifacts"] / name)
    pd.DataFrame(train_keys).to_csv(paths["out_artifacts"] / "train_keys.csv", index=False)

    meta = dict(base_meta)
    meta["feature_version"] = "v6_purchase_request_shadow"
    meta["feature_cols"] = list(base_meta["feature_cols"]) + REQUEST_COLS
    feature_groups = dict(base_meta["feature_groups"])
    feature_groups["purchase_request"] = REQUEST_COLS
    feature_groups["purchase_request_core"] = REQUEST_CORE_COLS
    meta["feature_groups"] = feature_groups
    meta["purchase_request_cols"] = REQUEST_COLS
    meta["purchase_request_core_cols"] = REQUEST_CORE_COLS
    meta["shadow_base_tag"] = paths["base_tag"]
    meta["shadow_source"] = "data/phase8a_prep/purchase_request_daily_features.csv"
    meta["leakage_guard"] = "Status columns from the request table are not used in model features."
    with open(paths["out_artifacts"] / "meta_v6_event.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"[OK] purchase request shadow processed -> {paths['out_processed']}")
    print(f"[OK] purchase request shadow artifacts -> {paths['out_artifacts']}")
    print(f"[OK] feature dim -> {base_dim + request_dim}")


if __name__ == "__main__":
    main()
