"""
V6-event feature builder.

Purpose:
- Build row-wise event/rolling features for a tabular hurdle baseline.
- Keep the same fair-universe as V5-lite.
- Preserve the same 2025 split logic used by Phase 5.x sequence experiments.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.features.phase53_feature_utils import (
    FORECAST,
    LOOKBACK,
    NEG_STEP,
    STATIC_BASE_NUM_COLS,
    STATIC_CAT_COLS,
    activity_bucket_from_days,
    build_buyer_window_arrays,
    days_since_last_positive,
    encode_static_table,
    get_runtime_dirs,
    load_gold_frame,
    load_keep_skus_from_v5_lite,
    load_silver_frame,
    map_output_tag_to_v5_lite,
    rolling_count,
    rolling_max,
    rolling_sum,
    split_flags,
)


BASE_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed_v6_event")
BASE_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "data", "artifacts_v6_event")

ROLL_WINDOWS = [7, 14, 30, 60, 90]
BUYER_COLS = ["repl_buyer_count_90", "future_buyer_count_90", "future_top1_share_90"]
ACTIVITY_COLS = ["active_days_any_30", "active_days_any_90", "activity_bucket"]
QFO_COLS = [
    "qty_first_order_bucket",
    "qty_first_order_style_z",
    "qty_first_order_category_z",
    "qty_first_order_x_repl0_fut0",
    "qty_first_order_x_repl0_fut1",
    "qty_first_order_x_repl1_fut0",
    "qty_first_order_x_repl1_fut1",
]
TAIL_COLS = [
    *[f"count_repl_gt10_{window}" for window in (30, 60, 90)],
    *[f"count_repl_gt25_{window}" for window in (30, 60, 90)],
    *[f"count_future_gt10_{window}" for window in (30, 60, 90)],
    *[f"count_future_gt25_{window}" for window in (30, 60, 90)],
]
PRIOR_COLS = [
    "style_sum_repl_30",
    "style_sum_repl_90",
    "style_sum_future_30",
    "style_sum_future_90",
    "category_sum_repl_30",
    "category_sum_repl_90",
    "category_sum_future_30",
    "category_sum_future_90",
    "subcat_sum_repl_30",
    "subcat_sum_repl_90",
    "subcat_sum_future_30",
    "subcat_sum_future_90",
    "repl_top1_share_90",
    "repl_top3_share_90",
    "repl_hhi_90",
    "future_top3_share_90",
    "future_hhi_90",
]


def build_feature_columns():
    static_cols = STATIC_CAT_COLS + ["month"] + STATIC_BASE_NUM_COLS
    core_cols = []
    for prefix in ("repl", "future"):
        for stat in ("sum", "count", "max"):
            for window in ROLL_WINDOWS:
                core_cols.append(f"{stat}_{prefix}_{window}")
    core_cols.extend([
        "days_since_last_repl",
        "days_since_last_future",
        "days_since_last_any_order",
    ])
    return static_cols, core_cols, BUYER_COLS, ACTIVITY_COLS, QFO_COLS, TAIL_COLS, PRIOR_COLS


def rolling_sum_matrix(matrix, window):
    if matrix.size == 0:
        return np.zeros_like(matrix, dtype=np.float32)
    padded = np.pad(matrix.astype(np.float64), ((0, 0), (1, 0)), mode="constant")
    csum = np.cumsum(padded, axis=1)
    idx = np.arange(1, matrix.shape[1] + 1)
    starts = np.maximum(0, idx - window)
    return (csum[:, idx] - csum[:, starts]).astype(np.float32)


def build_group_window_arrays(dyn_agg, static_source, sku_list, date_to_idx, n_days, group_col):
    group_map = (
        static_source[["sku_id", group_col]]
        .drop_duplicates("sku_id")
        .copy()
    )
    group_map["sku_id"] = group_map["sku_id"].astype(str)
    group_map[group_col] = group_map[group_col].fillna("Unknown").astype(str)

    merged = dyn_agg[["sku_id", "date", "qty_replenish", "qty_future"]].merge(
        group_map,
        on="sku_id",
        how="left",
    )
    merged[group_col] = merged[group_col].fillna("Unknown").astype(str)
    grouped = (
        merged.groupby([group_col, "date"], as_index=False)
        .agg(
            qty_replenish=("qty_replenish", "sum"),
            qty_future=("qty_future", "sum"),
        )
    )
    group_values = sorted(grouped[group_col].astype(str).unique().tolist())
    group_to_idx = {value: idx for idx, value in enumerate(group_values)}

    repl_matrix = np.zeros((len(group_values), n_days), dtype=np.float32)
    future_matrix = np.zeros((len(group_values), n_days), dtype=np.float32)
    for row in grouped.itertuples(index=False):
        day_idx = date_to_idx.get(row.date)
        if day_idx is None:
            continue
        grp_idx = group_to_idx[str(getattr(row, group_col))]
        repl_matrix[grp_idx, day_idx] = float(row.qty_replenish)
        future_matrix[grp_idx, day_idx] = float(row.qty_future)

    sums = {
        "sum_repl_30": rolling_sum_matrix(repl_matrix, 30),
        "sum_repl_90": rolling_sum_matrix(repl_matrix, 90),
        "sum_future_30": rolling_sum_matrix(future_matrix, 30),
        "sum_future_90": rolling_sum_matrix(future_matrix, 90),
    }

    sku_group_idx = np.zeros(len(sku_list), dtype=np.int32)
    sku_to_group = dict(zip(group_map["sku_id"].astype(str), group_map[group_col].astype(str)))
    default_idx = group_to_idx.get("Unknown", 0)
    for idx, sku in enumerate(sku_list):
        sku_group_idx[idx] = group_to_idx.get(sku_to_group.get(str(sku), "Unknown"), default_idx)

    return sums, sku_group_idx


def qfo_bucket_from_value(value):
    value = float(value)
    if value <= 0:
        return 0.0
    if value <= 1:
        return 1.0
    if value <= 10:
        return 2.0
    if value <= 30:
        return 3.0
    if value <= 100:
        return 4.0
    return 5.0


def build_qfo_arrays(static_source, sku_list):
    source = static_source[["sku_id", "style_id", "category", "qty_first_order"]].drop_duplicates("sku_id").copy()
    source["sku_id"] = source["sku_id"].astype(str)
    source["style_id"] = source["style_id"].fillna("Unknown").astype(str)
    source["category"] = source["category"].fillna("Unknown").astype(str)
    source["qty_first_order"] = source["qty_first_order"].fillna(0.0).astype(float)

    style_stats = source.groupby("style_id")["qty_first_order"].agg(["mean", "std"]).fillna(0.0)
    category_stats = source.groupby("category")["qty_first_order"].agg(["mean", "std"]).fillna(0.0)

    bucket_arr = np.zeros(len(sku_list), dtype=np.float32)
    style_z_arr = np.zeros(len(sku_list), dtype=np.float32)
    category_z_arr = np.zeros(len(sku_list), dtype=np.float32)

    source = source.set_index("sku_id")
    for idx, sku in enumerate(sku_list):
        if str(sku) not in source.index:
            continue
        row = source.loc[str(sku)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        qfo = float(row["qty_first_order"])
        style = str(row["style_id"])
        category = str(row["category"])
        style_mean = float(style_stats.loc[style, "mean"]) if style in style_stats.index else 0.0
        style_std = float(style_stats.loc[style, "std"]) if style in style_stats.index else 0.0
        category_mean = float(category_stats.loc[category, "mean"]) if category in category_stats.index else 0.0
        category_std = float(category_stats.loc[category, "std"]) if category in category_stats.index else 0.0

        bucket_arr[idx] = qfo_bucket_from_value(qfo)
        style_z_arr[idx] = float((qfo - style_mean) / style_std) if style_std > 1e-6 else 0.0
        category_z_arr[idx] = float((qfo - category_mean) / category_std) if category_std > 1e-6 else 0.0

    return {
        "qty_first_order_bucket": bucket_arr,
        "qty_first_order_style_z": style_z_arr,
        "qty_first_order_category_z": category_z_arr,
    }


def load_expected_counts(output_tag):
    candidate_tags = []
    for tag in [output_tag, map_output_tag_to_v5_lite(output_tag)]:
        tag = (tag or "").strip()
        if tag not in candidate_tags:
            candidate_tags.append(tag)

    checked_paths = []
    for tag in candidate_tags:
        suffix = f"_{tag}" if tag else ""
        meta_path = os.path.join(PROJECT_ROOT, "data", f"artifacts_v5_lite{suffix}", "meta_v5_lite.json")
        checked_paths.append(meta_path)
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    raise FileNotFoundError(
        "Missing fair-universe meta. Checked:\n- "
        + "\n- ".join(checked_paths)
    )


def build_tensors_v6_event():
    runtime = get_runtime_dirs(BASE_PROCESSED_DIR, BASE_ARTIFACTS_DIR)
    split_date_str = runtime["split_date"]
    val_mode = runtime["val_mode"]
    calendar_mode = os.environ.get("FEATURE_CALENDAR_MODE", "2025_only").strip().lower()
    processed_dir = runtime["processed_dir"]
    artifacts_dir = runtime["artifacts_dir"]
    output_tag = runtime["output_tag"]

    if calendar_mode not in {"2025_only", "extended"}:
        raise ValueError(
            f"Unsupported FEATURE_CALENDAR_MODE={calendar_mode}. "
            "Expected one of: 2025_only, extended"
        )

    print("=" * 72)
    print("[V6-event] Build tabular event/rolling features for hurdle tree baseline")
    print(
        f"[cfg] split_date={split_date_str} | val_mode={val_mode} | "
        f"calendar_mode={calendar_mode} | processed_dir={processed_dir} | artifacts_dir={artifacts_dir}"
    )
    print("=" * 72)

    keep_skus, keep_source = load_keep_skus_from_v5_lite(output_tag)
    expected_meta = load_expected_counts(output_tag)

    print(f"[{time.strftime('%H:%M:%S')}] load gold/silver on fair-universe")
    gold = load_gold_frame(keep_skus)
    silver = load_silver_frame(keep_skus)

    dyn_agg = (
        gold.groupby(["sku_id", "date"], as_index=False)
        .agg(
            qty_replenish=("qty_replenish", "sum"),
            qty_future=("qty_future", "sum"),
            qty_debt=("qty_debt", "sum"),
            qty_shipped=("qty_shipped", "sum"),
            qty_inbound=("qty_inbound", "sum"),
        )
    )

    static_source = gold[["sku_id"] + STATIC_CAT_COLS[1:] + STATIC_BASE_NUM_COLS].drop_duplicates("sku_id").copy()
    _, static_dict, _, _ = encode_static_table(
        static_source,
        artifacts_dir,
        "label_encoders_v6_event.pkl",
        extra_num_cols=None,
    )

    if calendar_mode == "extended":
        date_min = pd.to_datetime(gold["date"]).min().date()
        date_max = pd.to_datetime(gold["date"]).max().date()
        all_dates = pd.date_range(date_min, date_max, freq="D").date
    else:
        all_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D").date
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_days = len(all_dates)
    split_date = pd.to_datetime(split_date_str).date()

    sku_list = sorted(dyn_agg["sku_id"].astype(str).unique())
    sku_to_idx = {sku: i for i, sku in enumerate(sku_list)}

    print(f"[{time.strftime('%H:%M:%S')}] build dense daily series")
    repl_matrix = np.zeros((len(sku_list), n_days), dtype=np.float32)
    future_matrix = np.zeros((len(sku_list), n_days), dtype=np.float32)
    any_order_matrix = np.zeros((len(sku_list), n_days), dtype=np.float32)
    for row in tqdm(dyn_agg.itertuples(index=False), total=len(dyn_agg), desc="calendar"):
        sku_idx = sku_to_idx.get(str(row.sku_id))
        day_idx = date_to_idx.get(row.date)
        if sku_idx is None or day_idx is None:
            continue
        repl_matrix[sku_idx, day_idx] = float(row.qty_replenish)
        future_matrix[sku_idx, day_idx] = float(row.qty_future)
        any_order_matrix[sku_idx, day_idx] = float(
            (row.qty_replenish > 0) or
            (row.qty_future > 0) or
            (row.qty_debt > 0) or
            (row.qty_shipped > 0) or
            (row.qty_inbound > 0)
        )

    print(f"[{time.strftime('%H:%M:%S')}] build buyer coverage windows")
    buyer_arrays = build_buyer_window_arrays(silver, sku_list, date_to_idx, n_days, window=LOOKBACK)

    print(f"[{time.strftime('%H:%M:%S')}] build phase7 tail/priors caches")
    qfo_arrays = build_qfo_arrays(static_source, sku_list)
    style_group_arrays, style_group_idx = build_group_window_arrays(
        dyn_agg, static_source, sku_list, date_to_idx, n_days, "style_id"
    )
    category_group_arrays, category_group_idx = build_group_window_arrays(
        dyn_agg, static_source, sku_list, date_to_idx, n_days, "category"
    )
    subcat_group_arrays, subcat_group_idx = build_group_window_arrays(
        dyn_agg, static_source, sku_list, date_to_idx, n_days, "sub_category"
    )

    static_cols, core_cols, buyer_cols, activity_cols, qfo_cols, tail_cols, prior_cols = build_feature_columns()
    feature_cols = static_cols + core_cols + buyer_cols + activity_cols + qfo_cols + tail_cols + prior_cols
    feature_groups = {
        "static": static_cols,
        "core": core_cols,
        "buyer": buyer_cols,
        "activity": activity_cols,
        "qfo": qfo_cols,
        "tail": tail_cols,
        "priors": prior_cols,
    }
    feat_dim = len(feature_cols)

    train_cnt = int(expected_meta["train_cnt"])
    val_cnt = int(expected_meta["val_cnt"])
    x_train = open_memmap(os.path.join(processed_dir, "X_train.npy"), mode="w+", dtype=np.float32, shape=(train_cnt, feat_dim))
    y_train_cls = open_memmap(os.path.join(processed_dir, "y_train_cls.npy"), mode="w+", dtype=np.float32, shape=(train_cnt,))
    y_train_reg = open_memmap(os.path.join(processed_dir, "y_train_reg.npy"), mode="w+", dtype=np.float32, shape=(train_cnt,))
    x_val = open_memmap(os.path.join(processed_dir, "X_val.npy"), mode="w+", dtype=np.float32, shape=(val_cnt, feat_dim))
    y_val_cls = open_memmap(os.path.join(processed_dir, "y_val_cls.npy"), mode="w+", dtype=np.float32, shape=(val_cnt,))
    y_val_reg = open_memmap(os.path.join(processed_dir, "y_val_reg.npy"), mode="w+", dtype=np.float32, shape=(val_cnt,))

    train_idx = 0
    val_idx = 0
    val_keys = []
    end_idx = n_days - FORECAST

    print(f"[{time.strftime('%H:%M:%S')}] slide windows")
    for sku_idx, sku in enumerate(tqdm(sku_list, desc="windows")):
        if str(sku) not in static_dict:
            continue

        base_static = static_dict[str(sku)].copy()
        repl = repl_matrix[sku_idx]
        future = future_matrix[sku_idx]
        any_order = any_order_matrix[sku_idx]

        core_cache = {}
        for window in ROLL_WINDOWS:
            core_cache[f"sum_repl_{window}"] = rolling_sum(repl, window)
            core_cache[f"count_repl_{window}"] = rolling_count(repl, window)
            core_cache[f"max_repl_{window}"] = rolling_max(repl, window)
            core_cache[f"sum_future_{window}"] = rolling_sum(future, window)
            core_cache[f"count_future_{window}"] = rolling_count(future, window)
            core_cache[f"max_future_{window}"] = rolling_max(future, window)

        repl_days_90 = core_cache["count_repl_90"]
        future_recency = days_since_last_positive(future)
        repl_recency = days_since_last_positive(repl)
        any_recency = days_since_last_positive(any_order)
        active_any_30 = rolling_count(any_order, 30)
        active_any_90 = rolling_count(any_order, 90)
        activity_bucket = np.array(
            [activity_bucket_from_days(int(day_count)) for day_count in repl_days_90],
            dtype=np.float32,
        )
        tail_cache = {}
        for window in (30, 60, 90):
            tail_cache[f"count_repl_gt10_{window}"] = rolling_sum((repl > 10).astype(np.float32), window)
            tail_cache[f"count_repl_gt25_{window}"] = rolling_sum((repl > 25).astype(np.float32), window)
            tail_cache[f"count_future_gt10_{window}"] = rolling_sum((future > 10).astype(np.float32), window)
            tail_cache[f"count_future_gt25_{window}"] = rolling_sum((future > 25).astype(np.float32), window)

        buyer_repl = buyer_arrays["repl_buyer_count_90"][sku_idx]
        buyer_future = buyer_arrays["future_buyer_count_90"][sku_idx]
        buyer_top1 = buyer_arrays["future_top1_share_90"][sku_idx]
        repl_top1 = buyer_arrays["repl_top1_share_90"][sku_idx]
        repl_top3 = buyer_arrays["repl_top3_share_90"][sku_idx]
        repl_hhi = buyer_arrays["repl_hhi_90"][sku_idx]
        future_top3 = buyer_arrays["future_top3_share_90"][sku_idx]
        future_hhi = buyer_arrays["future_hhi_90"][sku_idx]

        style_idx = style_group_idx[sku_idx]
        category_idx = category_group_idx[sku_idx]
        subcat_idx = subcat_group_idx[sku_idx]
        qfo_bucket_arr = qfo_arrays["qty_first_order_bucket"]
        qfo_style_z_arr = qfo_arrays["qty_first_order_style_z"]
        qfo_category_z_arr = qfo_arrays["qty_first_order_category_z"]

        for i in range(LOOKBACK - 1, end_idx):
            target = float(repl[i + 1 : i + FORECAST + 1].sum())
            anchor_date = all_dates[i]
            is_train, is_val = split_flags(anchor_date, split_date, val_mode)

            if not is_train and not is_val:
                continue

            if target == 0 and is_train and (i % NEG_STEP != 0):
                continue

            static_vec = base_static.copy()
            static_vec[-3] = float(anchor_date.month)
            static_vec[-2] = np.log1p(max(0.0, static_vec[-2]))
            static_vec[-1] = np.log1p(max(0.0, static_vec[-1]))
            qfo_value = float(static_vec[-2])
            repl_flag = float(core_cache["count_repl_90"][i] > 0)
            future_flag = float(core_cache["count_future_90"][i] > 0)

            row_values = list(static_vec)
            for prefix in ("repl", "future"):
                for stat in ("sum", "count", "max"):
                    for window in ROLL_WINDOWS:
                        row_values.append(float(core_cache[f"{stat}_{prefix}_{window}"][i]))
            row_values.extend([
                float(repl_recency[i]),
                float(future_recency[i]),
                float(any_recency[i]),
                float(buyer_repl[i]),
                float(buyer_future[i]),
                float(buyer_top1[i]),
                float(active_any_30[i]),
                float(active_any_90[i]),
                float(activity_bucket[i]),
                float(qfo_bucket_arr[sku_idx]),
                float(qfo_style_z_arr[sku_idx]),
                float(qfo_category_z_arr[sku_idx]),
                qfo_value * float((repl_flag == 0.0) and (future_flag == 0.0)),
                qfo_value * float((repl_flag == 0.0) and (future_flag == 1.0)),
                qfo_value * float((repl_flag == 1.0) and (future_flag == 0.0)),
                qfo_value * float((repl_flag == 1.0) and (future_flag == 1.0)),
                float(tail_cache["count_repl_gt10_30"][i]),
                float(tail_cache["count_repl_gt10_60"][i]),
                float(tail_cache["count_repl_gt10_90"][i]),
                float(tail_cache["count_repl_gt25_30"][i]),
                float(tail_cache["count_repl_gt25_60"][i]),
                float(tail_cache["count_repl_gt25_90"][i]),
                float(tail_cache["count_future_gt10_30"][i]),
                float(tail_cache["count_future_gt10_60"][i]),
                float(tail_cache["count_future_gt10_90"][i]),
                float(tail_cache["count_future_gt25_30"][i]),
                float(tail_cache["count_future_gt25_60"][i]),
                float(tail_cache["count_future_gt25_90"][i]),
                float(style_group_arrays["sum_repl_30"][style_idx, i]),
                float(style_group_arrays["sum_repl_90"][style_idx, i]),
                float(style_group_arrays["sum_future_30"][style_idx, i]),
                float(style_group_arrays["sum_future_90"][style_idx, i]),
                float(category_group_arrays["sum_repl_30"][category_idx, i]),
                float(category_group_arrays["sum_repl_90"][category_idx, i]),
                float(category_group_arrays["sum_future_30"][category_idx, i]),
                float(category_group_arrays["sum_future_90"][category_idx, i]),
                float(subcat_group_arrays["sum_repl_30"][subcat_idx, i]),
                float(subcat_group_arrays["sum_repl_90"][subcat_idx, i]),
                float(subcat_group_arrays["sum_future_30"][subcat_idx, i]),
                float(subcat_group_arrays["sum_future_90"][subcat_idx, i]),
                float(repl_top1[i]),
                float(repl_top3[i]),
                float(repl_hhi[i]),
                float(future_top3[i]),
                float(future_hhi[i]),
            ])

            y_cls = np.float32(1.0 if target > 0 else 0.0)
            y_reg = np.float32(np.log1p(target))

            if is_train:
                x_train[train_idx] = np.asarray(row_values, dtype=np.float32)
                y_train_cls[train_idx] = y_cls
                y_train_reg[train_idx] = y_reg
                train_idx += 1
            elif is_val:
                x_val[val_idx] = np.asarray(row_values, dtype=np.float32)
                y_val_cls[val_idx] = y_cls
                y_val_reg[val_idx] = y_reg
                val_keys.append({"sku_id": sku, "date": anchor_date})
                val_idx += 1

    if train_idx != train_cnt or val_idx != val_cnt:
        raise ValueError(
            f"V6-event sample count mismatch: expected train/val={train_cnt}/{val_cnt}, "
            f"built={train_idx}/{val_idx}"
        )

    x_train.flush()
    x_val.flush()
    y_train_cls.flush()
    y_train_reg.flush()
    y_val_cls.flush()
    y_val_reg.flush()

    pd.DataFrame(val_keys).to_csv(os.path.join(artifacts_dir, "val_keys.csv"), index=False)

    meta = {
        "feature_version": "v6_event",
        "lookback": LOOKBACK,
        "forecast": FORECAST,
        "split_date": split_date_str,
        "val_mode": val_mode,
        "calendar_mode": calendar_mode,
        "calendar_start": str(all_dates[0]) if len(all_dates) else None,
        "calendar_end": str(all_dates[-1]) if len(all_dates) else None,
        "train_cnt": train_cnt,
        "val_cnt": val_cnt,
        "pos_train": int(float(y_train_cls.sum())),
        "pos_val": int(float(y_val_cls.sum())),
        "feature_cols": feature_cols,
        "feature_groups": feature_groups,
        "static_cat_cols": STATIC_CAT_COLS,
        "static_num_cols": STATIC_BASE_NUM_COLS,
        "buyer_num_cols": BUYER_COLS,
        "activity_cols": ACTIVITY_COLS,
        "qfo_cols": QFO_COLS,
        "tail_cols": TAIL_COLS,
        "prior_cols": PRIOR_COLS,
        "keep_source": keep_source,
    }
    with open(os.path.join(artifacts_dir, "meta_v6_event.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print()
    print("=" * 72)
    print("[done] V6-event feature engineering complete")
    print(f"train samples: {train_cnt:,} | positive rate: {meta['pos_train'] / max(train_cnt, 1):.2%}")
    print(f"val samples:   {val_cnt:,} | positive rate: {meta['pos_val'] / max(val_cnt, 1):.2%}")
    print(f"feature dim:   {feat_dim}")
    print(f"processed dir: {processed_dir}")
    print(f"artifacts dir: {artifacts_dir}")
    print("=" * 72)


if __name__ == "__main__":
    build_tensors_v6_event()
