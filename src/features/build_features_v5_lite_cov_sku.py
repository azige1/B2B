"""
V5-lite + buyer coverage feature builder.

Purpose:
- Keep the sparse 6-dim dynamic sequence from V5-lite.
- Add buyer coverage features into the per-sample static side.
- Support pooled / attention sequence variants for Phase 5.3.
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.features.phase53_feature_utils import (
    FORECAST,
    LOOKBACK,
    NEG_STEP,
    STATIC_BASE_NUM_COLS,
    STATIC_CAT_COLS,
    build_buyer_window_arrays,
    encode_static_table,
    get_runtime_dirs,
    load_gold_frame,
    load_keep_skus_from_v5_lite,
    load_silver_frame,
    rolling_sum,
    split_flags,
)

warnings.filterwarnings("ignore")

BASE_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed_v5_lite_cov")
BASE_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "data", "artifacts_v5_lite_cov")
DYN_FEAT_DIM = 6
FEATURE_NAMES = [
    "qty_replenish", "roll_repl_7", "roll_repl_30",
    "qty_future", "roll_fut_7", "roll_fut_30",
]
EXTRA_STATIC_NUM_COLS = ["repl_buyer_count_90", "future_buyer_count_90", "future_top1_share_90"]


def build_tensors_v5_lite_cov():
    runtime = get_runtime_dirs(BASE_PROCESSED_DIR, BASE_ARTIFACTS_DIR)
    split_date_str = runtime["split_date"]
    val_mode = runtime["val_mode"]
    processed_dir = runtime["processed_dir"]
    artifacts_dir = runtime["artifacts_dir"]
    output_tag = runtime["output_tag"]

    print("=" * 72)
    print("[V5-lite-cov] Build 6-dim sparse sequence + buyer coverage static features")
    print(
        f"[cfg] split_date={split_date_str} | val_mode={val_mode} | "
        f"processed_dir={processed_dir} | artifacts_dir={artifacts_dir}"
    )
    print("=" * 72)

    keep_skus, keep_source = load_keep_skus_from_v5_lite(output_tag)
    gold = load_gold_frame(keep_skus)
    silver = load_silver_frame(keep_skus)

    dyn_agg = (
        gold.groupby(["sku_id", "date"], as_index=False)
        .agg(
            qty_replenish=("qty_replenish", "sum"),
            qty_future=("qty_future", "sum"),
        )
    )

    static_source = gold[STATIC_CAT_COLS + STATIC_BASE_NUM_COLS].drop_duplicates("sku_id").copy()
    _, static_dict, _, _ = encode_static_table(
        static_source,
        artifacts_dir,
        "label_encoders_v5_lite_cov.pkl",
        extra_num_cols=None,
    )

    all_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D").date
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_days = len(all_dates)
    split_date = pd.to_datetime(split_date_str).date()

    sku_list = sorted(dyn_agg["sku_id"].astype(str).unique())
    sku_to_idx = {sku: i for i, sku in enumerate(sku_list)}

    print(f"[{time.strftime('%H:%M:%S')}] build dense calendar matrix")
    dyn_matrix = np.zeros((len(sku_list), n_days, 2), dtype=np.float32)
    for row in tqdm(dyn_agg.itertuples(index=False), total=len(dyn_agg), desc="calendar"):
        sku_idx = sku_to_idx.get(str(row.sku_id))
        day_idx = date_to_idx.get(row.date)
        if sku_idx is None or day_idx is None:
            continue
        dyn_matrix[sku_idx, day_idx, 0] = float(row.qty_replenish)
        dyn_matrix[sku_idx, day_idx, 1] = float(row.qty_future)

    print(f"[{time.strftime('%H:%M:%S')}] build buyer coverage windows")
    buyer_arrays = build_buyer_window_arrays(silver, sku_list, date_to_idx, n_days, window=LOOKBACK)

    print(f"[{time.strftime('%H:%M:%S')}] compute 6-dim feature pool")
    ts_feat = np.zeros((len(sku_list), n_days, DYN_FEAT_DIM), dtype=np.float32)
    for sku_idx in tqdm(range(len(sku_list)), desc="features"):
        repl = dyn_matrix[sku_idx, :, 0]
        future = dyn_matrix[sku_idx, :, 1]
        roll_repl_7 = rolling_sum(repl, 7)
        roll_repl_30 = rolling_sum(repl, 30)
        roll_future_7 = rolling_sum(future, 7)
        roll_future_30 = rolling_sum(future, 30)

        ts_feat[sku_idx, :, 0] = np.log1p(np.maximum(repl, 0))
        ts_feat[sku_idx, :, 1] = np.log1p(roll_repl_7)
        ts_feat[sku_idx, :, 2] = np.log1p(roll_repl_30)
        ts_feat[sku_idx, :, 3] = np.log1p(np.maximum(future, 0))
        ts_feat[sku_idx, :, 4] = np.log1p(roll_future_7)
        ts_feat[sku_idx, :, 5] = np.log1p(roll_future_30)

    file_in_dir = lambda name: os.path.normpath(os.path.join(processed_dir, name))
    handles = {
        "tr_dyn": open(file_in_dir("X_train_dyn.bin"), "wb"),
        "tr_sta": open(file_in_dir("X_train_static.bin"), "wb"),
        "tr_cls": open(file_in_dir("y_train_cls.bin"), "wb"),
        "tr_reg": open(file_in_dir("y_train_reg.bin"), "wb"),
        "va_dyn": open(file_in_dir("X_val_dyn.bin"), "wb"),
        "va_sta": open(file_in_dir("X_val_static.bin"), "wb"),
        "va_cls": open(file_in_dir("y_val_cls.bin"), "wb"),
        "va_reg": open(file_in_dir("y_val_reg.bin"), "wb"),
    }

    train_cnt = val_cnt = pos_train = pos_val = 0
    val_keys = []
    end_idx = n_days - FORECAST

    print(f"[{time.strftime('%H:%M:%S')}] slide windows")
    for sku_idx, sku in enumerate(tqdm(sku_list, desc="windows")):
        if str(sku) not in static_dict:
            continue

        base_static = static_dict[str(sku)]
        sku_repl = dyn_matrix[sku_idx, :, 0]
        buyer_repl = buyer_arrays["repl_buyer_count_90"][sku_idx]
        buyer_future = buyer_arrays["future_buyer_count_90"][sku_idx]
        buyer_top1 = buyer_arrays["future_top1_share_90"][sku_idx]

        for i in range(LOOKBACK - 1, end_idx):
            target = float(sku_repl[i + 1 : i + FORECAST + 1].sum())
            anchor_date = all_dates[i]
            is_train, is_val = split_flags(anchor_date, split_date, val_mode)

            if not is_train and not is_val:
                continue

            if target == 0 and is_train and (i % NEG_STEP != 0):
                continue

            window = ts_feat[sku_idx, i - LOOKBACK + 1 : i + 1]
            static_vec = base_static.copy()
            static_vec[-3] = float(anchor_date.month)
            static_vec[-2] = np.log1p(max(0.0, static_vec[-2]))
            static_vec[-1] = np.log1p(max(0.0, static_vec[-1]))
            static_vec = np.concatenate(
                [
                    static_vec,
                    np.array(
                        [
                            np.log1p(float(buyer_repl[i])),
                            np.log1p(float(buyer_future[i])),
                            float(buyer_top1[i]),
                        ],
                        dtype=np.float32,
                    ),
                ]
            ).astype(np.float32)

            y_cls = np.array([1.0 if target > 0 else 0.0], dtype=np.float32)
            y_reg = np.array([np.log1p(target)], dtype=np.float32)

            if is_train:
                window.tofile(handles["tr_dyn"])
                static_vec.tofile(handles["tr_sta"])
                y_cls.tofile(handles["tr_cls"])
                y_reg.tofile(handles["tr_reg"])
                train_cnt += 1
                if target > 0:
                    pos_train += 1
            elif is_val:
                window.tofile(handles["va_dyn"])
                static_vec.tofile(handles["va_sta"])
                y_cls.tofile(handles["va_cls"])
                y_reg.tofile(handles["va_reg"])
                val_cnt += 1
                if target > 0:
                    pos_val += 1
                val_keys.append({"sku_id": sku, "date": anchor_date})

    for fh in handles.values():
        fh.close()

    if train_cnt > 0:
        print(f"[{time.strftime('%H:%M:%S')}] normalize dynamic features")
        scaler = MinMaxScaler()
        train_mm = np.memmap(
            file_in_dir("X_train_dyn.bin"),
            dtype=np.float32,
            mode="r+",
            shape=(train_cnt, LOOKBACK, DYN_FEAT_DIM),
        )
        for i in range(0, train_cnt, 20000):
            chunk = train_mm[i : i + 20000].reshape(-1, DYN_FEAT_DIM)
            scaler.partial_fit(chunk)
        for i in range(0, train_cnt, 20000):
            chunk = train_mm[i : i + 20000].reshape(-1, DYN_FEAT_DIM)
            train_mm[i : i + 20000] = scaler.transform(chunk).reshape(train_mm[i : i + 20000].shape)
            train_mm.flush()

        if val_cnt > 0:
            val_mm = np.memmap(
                file_in_dir("X_val_dyn.bin"),
                dtype=np.float32,
                mode="r+",
                shape=(val_cnt, LOOKBACK, DYN_FEAT_DIM),
            )
            for i in range(0, val_cnt, 20000):
                chunk = val_mm[i : i + 20000].reshape(-1, DYN_FEAT_DIM)
                val_mm[i : i + 20000] = scaler.transform(chunk).reshape(val_mm[i : i + 20000].shape)
                val_mm.flush()

        with open(os.path.join(artifacts_dir, "feature_scaler_v5_lite_cov.pkl"), "wb") as fh:
            import pickle
            pickle.dump(scaler, fh)

    meta = {
        "dyn_feat_dim": DYN_FEAT_DIM,
        "static_dim": len(STATIC_CAT_COLS) + 1 + len(STATIC_BASE_NUM_COLS) + len(EXTRA_STATIC_NUM_COLS),
        "static_cat_cols": STATIC_CAT_COLS,
        "static_num_cols": STATIC_BASE_NUM_COLS + EXTRA_STATIC_NUM_COLS,
        "lookback": LOOKBACK,
        "forecast": FORECAST,
        "train_cnt": train_cnt,
        "val_cnt": val_cnt,
        "pos_train": pos_train,
        "pos_val": pos_val,
        "split_date": split_date_str,
        "val_mode": val_mode,
        "feature_names": FEATURE_NAMES,
        "keep_source": keep_source,
        "buyer_feature_names": EXTRA_STATIC_NUM_COLS,
    }
    with open(os.path.join(artifacts_dir, "meta_v5_lite_cov.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    if val_cnt > 0:
        pd.DataFrame(val_keys).to_csv(os.path.join(artifacts_dir, "val_keys.csv"), index=False)

    print()
    print("=" * 72)
    print("[done] V5-lite-cov feature engineering complete")
    print(f"train samples: {train_cnt:,} | positive rate: {pos_train / max(train_cnt, 1):.2%}")
    print(f"val samples:   {val_cnt:,} | positive rate: {pos_val / max(val_cnt, 1):.2%}")
    print(f"processed dir: {processed_dir}")
    print(f"artifacts dir: {artifacts_dir}")
    print("=" * 72)


if __name__ == "__main__":
    build_tensors_v5_lite_cov()
