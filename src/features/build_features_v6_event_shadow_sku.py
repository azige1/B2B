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

from src.features.phase53_feature_utils import (
    FORECAST,
    LOOKBACK,
    NEG_STEP,
    load_gold_frame,
    load_keep_skus_from_v5_lite,
    split_flags,
)


EVENT_WINDOWS = (7, 14, 30)
EVENT_COLS = [
    *[f"event_active_buyers_{window}" for window in EVENT_WINDOWS],
    *[f"event_clicks_{window}" for window in EVENT_WINDOWS],
    *[f"event_view_order_{window}" for window in EVENT_WINDOWS],
    *[f"event_cart_adds_{window}" for window in EVENT_WINDOWS],
    *[f"event_order_success_{window}" for window in EVENT_WINDOWS],
    *[f"event_pay_success_{window}" for window in EVENT_WINDOWS],
    *[f"event_order_submit_qty_{window}" for window in EVENT_WINDOWS],
    *[f"event_pay_qty_{window}" for window in EVENT_WINDOWS],
    "event_days_since_last_any",
    "event_days_since_last_strong",
    "event_click_to_cart_rate_30",
    "event_view_to_order_rate_30",
    "event_cart_to_order_rate_30",
    "event_order_to_pay_rate_30",
]


def rolling_sum_matrix(matrix, window):
    if matrix.size == 0:
        return np.zeros_like(matrix, dtype=np.float32)
    padded = np.pad(matrix.astype(np.float64), ((0, 0), (1, 0)), mode="constant")
    csum = np.cumsum(padded, axis=1)
    idx = np.arange(1, matrix.shape[1] + 1)
    starts = np.maximum(0, idx - window)
    return (csum[:, idx] - csum[:, starts]).astype(np.float32)


def days_since_last_positive(arr):
    out = np.full(len(arr), LOOKBACK + FORECAST, dtype=np.float32)
    last_idx = -1
    for i, value in enumerate(arr):
        if value > 0:
            last_idx = i
            out[i] = 0.0
        elif last_idx >= 0:
            out[i] = float(i - last_idx)
    return out


def safe_rate(num, den):
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    out = np.zeros_like(num, dtype=np.float32)
    mask = den > 1e-6
    out[mask] = num[mask] / den[mask]
    return out


def load_base_paths(split_date):
    tag = split_date.replace("-", "")
    base_tag = f"p7b_{tag}_v6_event"
    output_tag = f"p8shadow_{tag}_v6_event"
    base_processed = PROJECT_ROOT / "data" / f"processed_v6_event_{base_tag}"
    base_artifacts = PROJECT_ROOT / "data" / f"artifacts_v6_event_{base_tag}"
    out_processed = PROJECT_ROOT / "data" / f"processed_v6_event_{output_tag}"
    out_artifacts = PROJECT_ROOT / "data" / f"artifacts_v6_event_{output_tag}"
    out_processed.mkdir(parents=True, exist_ok=True)
    out_artifacts.mkdir(parents=True, exist_ok=True)
    return {
        "base_tag": base_tag,
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
        PROJECT_ROOT / "data" / "phase8a_prep" / "event_intent_daily_features.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(missing))


def load_event_style_daily():
    path = PROJECT_ROOT / "data" / "phase8a_prep" / "event_intent_daily_features.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df["date"] = df["date"].dt.date
    df["style_id"] = df["style_id"].astype(str)
    df["buyer_id"] = df["buyer_id"].astype(str)
    raw_cols = [
        "daily_clicks",
        "daily_view_order",
        "daily_cart_adds",
        "daily_order_success",
        "daily_pay_success",
        "daily_order_submit_qty",
        "daily_pay_qty",
    ]
    grouped = (
        df.groupby(["style_id", "date"], as_index=False)
        .agg(
            active_buyer_count=("buyer_id", "nunique"),
            **{col: (col, "sum") for col in raw_cols},
        )
    )
    return grouped


def build_style_event_arrays(style_values, date_to_idx, n_days):
    event_daily = load_event_style_daily()
    style_to_idx = {style: idx for idx, style in enumerate(style_values)}
    metric_names = [
        "active_buyer_count",
        "daily_clicks",
        "daily_view_order",
        "daily_cart_adds",
        "daily_order_success",
        "daily_pay_success",
        "daily_order_submit_qty",
        "daily_pay_qty",
    ]
    matrices = {
        name: np.zeros((len(style_values), n_days), dtype=np.float32)
        for name in metric_names
    }
    for row in event_daily.itertuples(index=False):
        style_idx = style_to_idx.get(str(row.style_id))
        day_idx = date_to_idx.get(row.date)
        if style_idx is None or day_idx is None:
            continue
        for name in metric_names:
            matrices[name][style_idx, day_idx] = float(getattr(row, name))

    rolling = {}
    for base_name in metric_names:
        for window in EVENT_WINDOWS:
            rolling[f"{base_name}_{window}"] = rolling_sum_matrix(matrices[base_name], window)

    any_matrix = (
        matrices["daily_clicks"]
        + matrices["daily_view_order"]
        + matrices["daily_cart_adds"]
        + matrices["daily_order_success"]
        + matrices["daily_pay_success"]
    ) > 0
    strong_matrix = (
        matrices["daily_view_order"]
        + matrices["daily_cart_adds"]
        + matrices["daily_order_success"]
        + matrices["daily_pay_success"]
    ) > 0

    recency_any = np.zeros((len(style_values), n_days), dtype=np.float32)
    recency_strong = np.zeros((len(style_values), n_days), dtype=np.float32)
    for idx in range(len(style_values)):
        recency_any[idx] = days_since_last_positive(any_matrix[idx].astype(np.float32))
        recency_strong[idx] = days_since_last_positive(strong_matrix[idx].astype(np.float32))

    rates = {
        "event_click_to_cart_rate_30": safe_rate(
            rolling["daily_cart_adds_30"], rolling["daily_clicks_30"]
        ),
        "event_view_to_order_rate_30": safe_rate(
            rolling["daily_order_success_30"], rolling["daily_view_order_30"]
        ),
        "event_cart_to_order_rate_30": safe_rate(
            rolling["daily_order_success_30"], rolling["daily_cart_adds_30"]
        ),
        "event_order_to_pay_rate_30": safe_rate(
            rolling["daily_pay_success_30"], rolling["daily_order_success_30"]
        ),
    }

    return {
        "active_buyer_count": rolling,
        "raw": matrices,
        "recency_any": recency_any,
        "recency_strong": recency_strong,
        "rates": rates,
        "style_to_idx": style_to_idx,
    }


def event_row_values(cache, style_idx, day_idx):
    rolling = cache["active_buyer_count"]
    values = [
        float(rolling["active_buyer_count_7"][style_idx, day_idx]),
        float(rolling["active_buyer_count_14"][style_idx, day_idx]),
        float(rolling["active_buyer_count_30"][style_idx, day_idx]),
        float(rolling["daily_clicks_7"][style_idx, day_idx]),
        float(rolling["daily_clicks_14"][style_idx, day_idx]),
        float(rolling["daily_clicks_30"][style_idx, day_idx]),
        float(rolling["daily_view_order_7"][style_idx, day_idx]),
        float(rolling["daily_view_order_14"][style_idx, day_idx]),
        float(rolling["daily_view_order_30"][style_idx, day_idx]),
        float(rolling["daily_cart_adds_7"][style_idx, day_idx]),
        float(rolling["daily_cart_adds_14"][style_idx, day_idx]),
        float(rolling["daily_cart_adds_30"][style_idx, day_idx]),
        float(rolling["daily_order_success_7"][style_idx, day_idx]),
        float(rolling["daily_order_success_14"][style_idx, day_idx]),
        float(rolling["daily_order_success_30"][style_idx, day_idx]),
        float(rolling["daily_pay_success_7"][style_idx, day_idx]),
        float(rolling["daily_pay_success_14"][style_idx, day_idx]),
        float(rolling["daily_pay_success_30"][style_idx, day_idx]),
        float(rolling["daily_order_submit_qty_7"][style_idx, day_idx]),
        float(rolling["daily_order_submit_qty_14"][style_idx, day_idx]),
        float(rolling["daily_order_submit_qty_30"][style_idx, day_idx]),
        float(rolling["daily_pay_qty_7"][style_idx, day_idx]),
        float(rolling["daily_pay_qty_14"][style_idx, day_idx]),
        float(rolling["daily_pay_qty_30"][style_idx, day_idx]),
        float(cache["recency_any"][style_idx, day_idx]),
        float(cache["recency_strong"][style_idx, day_idx]),
        float(cache["rates"]["event_click_to_cart_rate_30"][style_idx, day_idx]),
        float(cache["rates"]["event_view_to_order_rate_30"][style_idx, day_idx]),
        float(cache["rates"]["event_cart_to_order_rate_30"][style_idx, day_idx]),
        float(cache["rates"]["event_order_to_pay_rate_30"][style_idx, day_idx]),
    ]
    return values


def main():
    split_date = os.environ.get("FEATURE_SPLIT_DATE", "").strip()
    if not split_date:
        raise ValueError("FEATURE_SPLIT_DATE is required.")
    val_mode = os.environ.get("FEATURE_VAL_MODE", "single_anchor").strip().lower()
    if val_mode != "single_anchor":
        raise ValueError("Shadow builder only supports FEATURE_VAL_MODE=single_anchor.")

    paths = load_base_paths(split_date)
    preflight(paths)

    print("=" * 72)
    print("[Phase8 shadow] build v6_event + event_intent shadow assets")
    print(f"split_date={split_date} | output_tag={paths['output_tag']}")
    print("=" * 72)

    with open(paths["base_artifacts"] / "meta_v6_event.json", "r", encoding="utf-8") as fh:
        base_meta = json.load(fh)

    base_x_train = np.load(paths["base_processed"] / "X_train.npy", mmap_mode="r")
    base_x_val = np.load(paths["base_processed"] / "X_val.npy", mmap_mode="r")
    train_cnt = int(base_meta["train_cnt"])
    val_cnt = int(base_meta["val_cnt"])
    base_dim = int(base_x_train.shape[1])
    event_dim = len(EVENT_COLS)

    # Reuse the official phase7 anchor universe instead of expecting
    # a separate V5-lite keep-set for the shadow-only output tag.
    keep_skus, _ = load_keep_skus_from_v5_lite(paths["base_tag"])
    gold = load_gold_frame(keep_skus)
    dyn_agg = (
        gold.groupby(["sku_id", "date"], as_index=False)
        .agg(
            qty_replenish=("qty_replenish", "sum"),
            qty_future=("qty_future", "sum"),
        )
    )
    static_source = gold[["sku_id", "style_id"]].drop_duplicates("sku_id").copy()
    static_source["sku_id"] = static_source["sku_id"].astype(str)
    static_source["style_id"] = static_source["style_id"].fillna("Unknown").astype(str)
    style_by_sku = dict(zip(static_source["sku_id"], static_source["style_id"]))

    all_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D").date
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

    style_values = sorted(set(static_source["style_id"].astype(str)))
    style_cache = build_style_event_arrays(style_values, date_to_idx, len(all_dates))
    default_style_idx = style_cache["style_to_idx"].get("Unknown", 0)

    event_train = np.zeros((train_cnt, event_dim), dtype=np.float32)
    event_val = np.zeros((val_cnt, event_dim), dtype=np.float32)
    train_keys = []
    val_keys = []
    train_idx = 0
    val_idx = 0

    print(f"[{time.strftime('%H:%M:%S')}] build event shadow rows")
    for sku in tqdm(sku_list, desc="shadow_windows"):
        sku_idx = sku_to_idx[str(sku)]
        style_id = style_by_sku.get(str(sku), "Unknown")
        style_idx = style_cache["style_to_idx"].get(style_id, default_style_idx)
        repl = repl_matrix[sku_idx]

        for i in range(LOOKBACK - 1, end_idx):
            target = float(repl[i + 1 : i + FORECAST + 1].sum())
            anchor_date = all_dates[i]
            is_train, is_val = split_flags(anchor_date, split_date_obj, val_mode)
            if not is_train and not is_val:
                continue
            if target == 0 and is_train and (i % NEG_STEP != 0):
                continue

            row = np.asarray(event_row_values(style_cache, style_idx, i), dtype=np.float32)
            if is_train:
                event_train[train_idx] = row
                train_keys.append({"sku_id": sku, "date": anchor_date})
                train_idx += 1
            else:
                event_val[val_idx] = row
                val_keys.append({"sku_id": sku, "date": anchor_date})
                val_idx += 1

    if train_idx != train_cnt or val_idx != val_cnt:
        raise ValueError(
            f"Shadow sample count mismatch: expected train/val={train_cnt}/{val_cnt}, built={train_idx}/{val_idx}"
        )

    base_val_keys = pd.read_csv(paths["base_artifacts"] / "val_keys.csv")
    built_val_keys = pd.DataFrame(val_keys)
    built_val_keys["date"] = pd.to_datetime(built_val_keys["date"]).dt.strftime("%Y-%m-%d")
    base_val_norm = base_val_keys.copy()
    base_val_norm["date"] = pd.to_datetime(base_val_norm["date"]).dt.strftime("%Y-%m-%d")
    if not built_val_keys.equals(base_val_norm):
        raise ValueError("Shadow val_keys do not match base val_keys; sample order is not aligned.")

    out_x_train = open_memmap(
        paths["out_processed"] / "X_train.npy",
        mode="w+",
        dtype=np.float32,
        shape=(train_cnt, base_dim + event_dim),
    )
    out_x_val = open_memmap(
        paths["out_processed"] / "X_val.npy",
        mode="w+",
        dtype=np.float32,
        shape=(val_cnt, base_dim + event_dim),
    )
    out_x_train[:, :base_dim] = base_x_train
    out_x_train[:, base_dim:] = event_train
    out_x_val[:, :base_dim] = base_x_val
    out_x_val[:, base_dim:] = event_val
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
    meta["feature_version"] = "v6_event_shadow"
    meta["feature_cols"] = list(base_meta["feature_cols"]) + EVENT_COLS
    feature_groups = dict(base_meta["feature_groups"])
    feature_groups["event"] = EVENT_COLS
    meta["feature_groups"] = feature_groups
    meta["event_cols"] = EVENT_COLS
    meta["shadow_base_tag"] = paths["base_tag"]
    meta["shadow_source"] = "data/phase8a_prep/event_intent_daily_features.csv"
    with open(paths["out_artifacts"] / "meta_v6_event.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    print(f"[OK] shadow processed -> {paths['out_processed']}")
    print(f"[OK] shadow artifacts -> {paths['out_artifacts']}")
    print(f"[OK] feature dim -> {base_dim + event_dim}")


if __name__ == "__main__":
    main()
