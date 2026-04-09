"""
V5-lite feature builder for minimal ablation.

Compared with V5:
- Keep only the 3 replenish core features
- Keep only the 3 future lead features
- Drop dense derived features that may distort calibration:
  repl_velocity, fut2repl_ratio, repl_volatility, days_since_last

Output:
- data/processed_v5_lite/
- data/artifacts_v5_lite/
"""
import json
import os
import pickle
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOOKBACK = 90
FORECAST = 30
SPLIT_DATE = "2025-12-01"
NEG_STEP = 1

STATIC_CAT_COLS = [
    "sku_id", "style_id", "product_name", "category",
    "sub_category", "season", "series", "band", "size_id", "color_id"
]
STATIC_NUM_COLS = ["qty_first_order", "price_tag"]
DYN_FEAT_DIM = 6
FEATURE_NAMES = [
    "qty_replenish", "roll_repl_7", "roll_repl_30",
    "qty_future", "roll_fut_7", "roll_fut_30",
]
DROP_ZERO_REPL_SKUS = True

BASE_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed_v5_lite")
BASE_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "data", "artifacts_v5_lite")
GOLD_DIR = os.path.join(PROJECT_ROOT, "data", "gold")


class DummyLE:
    classes_ = np.arange(13)


def get_runtime_config():
    output_tag = os.environ.get("FEATURE_OUTPUT_TAG", "").strip()
    suffix = f"_{output_tag}" if output_tag else ""
    processed_dir = f"{BASE_PROCESSED_DIR}{suffix}"
    artifacts_dir = f"{BASE_ARTIFACTS_DIR}{suffix}"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    calendar_mode = os.environ.get("FEATURE_CALENDAR_MODE", "2025_only").strip().lower()
    if calendar_mode not in {"2025_only", "extended"}:
        raise ValueError(
            f"Unsupported FEATURE_CALENDAR_MODE={calendar_mode}. "
            "Expected one of: 2025_only, extended"
        )
    return {
        "split_date": os.environ.get("FEATURE_SPLIT_DATE", SPLIT_DATE),
        "val_mode": os.environ.get("FEATURE_VAL_MODE", "full_holdout").strip().lower(),
        "calendar_mode": calendar_mode,
        "processed_dir": processed_dir,
        "artifacts_dir": artifacts_dir,
        "output_tag": output_tag,
    }


def rolling_sum(arr, window, n_days):
    cumulative = np.cumsum(np.concatenate([[0.0], arr]))
    result = np.zeros(n_days, dtype=np.float32)
    for j in range(n_days):
        end_idx = j + 1
        start_idx = max(0, end_idx - window)
        result[j] = cumulative[end_idx] - cumulative[start_idx]
    return np.maximum(result, 0)


def build_tensors_v5_lite():
    runtime = get_runtime_config()
    split_date_str = runtime["split_date"]
    val_mode = runtime["val_mode"]
    calendar_mode = runtime["calendar_mode"]
    processed_dir = runtime["processed_dir"]
    artifacts_dir = runtime["artifacts_dir"]

    print("=" * 65)
    print("[V5-lite] Build 6-dim dynamic features: replenish + future only")
    print(
        f"[cfg] split_date={split_date_str} | val_mode={val_mode} | "
        f"calendar_mode={calendar_mode} | processed_dir={processed_dir} | artifacts_dir={artifacts_dir}"
    )
    print("=" * 65)

    if val_mode not in {"full_holdout", "single_anchor"}:
        raise ValueError(
            f"Unsupported FEATURE_VAL_MODE={val_mode}. "
            "Expected one of: full_holdout, single_anchor"
        )

    df_path = os.path.join(GOLD_DIR, "wide_table_sku.csv")
    print(f"[{time.strftime('%H:%M:%S')}] read {df_path}")
    df = pd.read_csv(df_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    if "qty_future" not in df.columns:
        raise ValueError("wide_table_sku.csv missing qty_future")

    print(f"[{time.strftime('%H:%M:%S')}] aggregate national sku-date")
    dyn_agg = (
        df.groupby(["sku_id", "date"])
        .agg(
            qty_replenish=("qty_replenish", "sum"),
            qty_future=("qty_future", "sum"),
        )
        .reset_index()
    )

    sku_activity = (
        dyn_agg.groupby("sku_id", as_index=False)
        .agg(
            repl_days=("qty_replenish", lambda s: int((s > 0).sum())),
            future_days=("qty_future", lambda s: int((s > 0).sum())),
            total_repl=("qty_replenish", "sum"),
            total_future=("qty_future", "sum"),
            last_any_date=("date", "max"),
        )
    )
    sku_activity["future_only"] = (sku_activity["future_days"] > 0) & (sku_activity["repl_days"] == 0)
    sku_activity["kept_for_training"] = True

    if DROP_ZERO_REPL_SKUS:
        keep_mask = sku_activity["repl_days"] > 0
        dropped = int((~keep_mask).sum())
        print(f"[{time.strftime('%H:%M:%S')}] drop zero-replenish SKUs for V5-lite: {dropped:,}")
        sku_activity.loc[~keep_mask, "kept_for_training"] = False
        keep_skus = set(sku_activity.loc[keep_mask, "sku_id"])
        dyn_agg = dyn_agg[dyn_agg["sku_id"].isin(keep_skus)].reset_index(drop=True)
        df = df[df["sku_id"].isin(keep_skus)].reset_index(drop=True)
    else:
        keep_skus = set(sku_activity["sku_id"])

    static_agg = df[STATIC_CAT_COLS + STATIC_NUM_COLS].drop_duplicates("sku_id").copy()
    static_agg["orig_sku"] = static_agg["sku_id"].astype(str)

    encoders = {}
    for col in STATIC_CAT_COLS:
        if col not in static_agg.columns:
            static_agg[col] = "Unknown"
        encoder = LabelEncoder()
        static_agg[col] = encoder.fit_transform(static_agg[col].astype(str))
        encoders[col] = encoder

    encoders["month"] = DummyLE()

    with open(os.path.join(artifacts_dir, "label_encoders_v5_lite.pkl"), "wb") as fh:
        pickle.dump(encoders, fh)
    sku_activity.to_csv(os.path.join(artifacts_dir, "sku_activity_v5_lite.csv"), index=False)

    static_dim = len(STATIC_CAT_COLS) + 1 + len(STATIC_NUM_COLS)
    static_dict = {}
    for _, row in static_agg.iterrows():
        values = [float(row[col]) for col in STATIC_CAT_COLS]
        values.append(0.0)
        values.extend(float(row[col]) if pd.notnull(row[col]) else 0.0 for col in STATIC_NUM_COLS)
        static_dict[str(row["orig_sku"])] = np.array(values, dtype=np.float32)

    if calendar_mode == "extended":
        date_min = pd.to_datetime(df["date"]).min().date()
        date_max = pd.to_datetime(df["date"]).max().date()
        all_dates = pd.date_range(date_min, date_max, freq="D").date
    else:
        all_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D").date
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_days = len(all_dates)
    sku_list = sorted(dyn_agg["sku_id"].unique())
    sku_to_idx = {sku: i for i, sku in enumerate(sku_list)}

    print(f"[{time.strftime('%H:%M:%S')}] build dense calendar matrix")
    dyn_matrix = np.zeros((len(sku_list), n_days, 2), dtype=np.float32)
    for row in tqdm(dyn_agg.itertuples(index=False), total=len(dyn_agg), desc="calendar"):
        sku_idx = sku_to_idx.get(row.sku_id, -1)
        date_idx = date_to_idx.get(row.date, -1)
        if sku_idx >= 0 and date_idx >= 0:
            dyn_matrix[sku_idx, date_idx, 0] = row.qty_replenish
            dyn_matrix[sku_idx, date_idx, 1] = row.qty_future

    print(f"[{time.strftime('%H:%M:%S')}] compute 6-dim feature pool")
    ts_feat = np.zeros((len(sku_list), n_days, DYN_FEAT_DIM), dtype=np.float32)
    val_keys = []

    for sku_idx in tqdm(range(len(sku_list)), desc="features"):
        repl = dyn_matrix[sku_idx, :, 0]
        future = dyn_matrix[sku_idx, :, 1]

        roll_repl_7 = rolling_sum(repl, 7, n_days)
        roll_repl_30 = rolling_sum(repl, 30, n_days)
        roll_future_7 = rolling_sum(future, 7, n_days)
        roll_future_30 = rolling_sum(future, 30, n_days)

        ts_feat[sku_idx, :, 0] = np.log1p(np.maximum(repl, 0))
        ts_feat[sku_idx, :, 1] = np.log1p(roll_repl_7)
        ts_feat[sku_idx, :, 2] = np.log1p(roll_repl_30)
        ts_feat[sku_idx, :, 3] = np.log1p(np.maximum(future, 0))
        ts_feat[sku_idx, :, 4] = np.log1p(roll_future_7)
        ts_feat[sku_idx, :, 5] = np.log1p(roll_future_30)

    split_date = pd.to_datetime(split_date_str).date()
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
    end_idx = n_days - FORECAST

    print(f"[{time.strftime('%H:%M:%S')}] slide windows")
    for sku_idx, sku in enumerate(tqdm(sku_list, desc="windows")):
        if str(sku) not in static_dict:
            continue

        base_static = static_dict[str(sku)]
        sku_repl = dyn_matrix[sku_idx, :, 0]

        for i in range(LOOKBACK - 1, end_idx):
            target = float(sku_repl[i + 1: i + FORECAST + 1].sum())
            anchor_date = all_dates[i]
            is_train = anchor_date < split_date
            is_val = anchor_date == split_date if val_mode == "single_anchor" else anchor_date >= split_date

            if not is_train and not is_val:
                continue

            if target == 0 and is_train and (i % NEG_STEP != 0):
                continue

            window = ts_feat[sku_idx, i - LOOKBACK + 1: i + 1]
            static_vec = base_static.copy()
            static_vec[-3] = float(anchor_date.month)
            static_vec[-2] = np.log1p(max(0.0, static_vec[-2]))
            static_vec[-1] = np.log1p(max(0.0, static_vec[-1]))

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
            chunk = train_mm[i:i + 20000].reshape(-1, DYN_FEAT_DIM)
            scaler.partial_fit(chunk)
        for i in range(0, train_cnt, 20000):
            chunk = train_mm[i:i + 20000].reshape(-1, DYN_FEAT_DIM)
            train_mm[i:i + 20000] = scaler.transform(chunk).reshape(train_mm[i:i + 20000].shape)
            train_mm.flush()

        if val_cnt > 0:
            val_mm = np.memmap(
                file_in_dir("X_val_dyn.bin"),
                dtype=np.float32,
                mode="r+",
                shape=(val_cnt, LOOKBACK, DYN_FEAT_DIM),
            )
            for i in range(0, val_cnt, 20000):
                chunk = val_mm[i:i + 20000].reshape(-1, DYN_FEAT_DIM)
                val_mm[i:i + 20000] = scaler.transform(chunk).reshape(val_mm[i:i + 20000].shape)
                val_mm.flush()

        with open(os.path.join(artifacts_dir, "feature_scaler_v5_lite.pkl"), "wb") as fh:
            pickle.dump(scaler, fh)

    meta = {
        "dyn_feat_dim": DYN_FEAT_DIM,
        "static_dim": static_dim,
        "static_cat_cols": STATIC_CAT_COLS,
        "static_num_cols": STATIC_NUM_COLS,
        "lookback": LOOKBACK,
        "forecast": FORECAST,
        "train_cnt": train_cnt,
        "val_cnt": val_cnt,
        "pos_train": pos_train,
        "pos_val": pos_val,
        "split_date": split_date_str,
        "val_mode": val_mode,
        "calendar_mode": calendar_mode,
        "calendar_start": str(all_dates[0]) if len(all_dates) else None,
        "calendar_end": str(all_dates[-1]) if len(all_dates) else None,
        "feature_names": FEATURE_NAMES,
        "sku_filter": {
            "drop_zero_replenish_skus": DROP_ZERO_REPL_SKUS,
            "kept_sku_count": len(keep_skus),
            "dropped_sku_count": int((~sku_activity["kept_for_training"]).sum()),
        },
    }
    with open(os.path.join(artifacts_dir, "meta_v5_lite.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    if val_cnt > 0:
        pd.DataFrame(val_keys).to_csv(os.path.join(artifacts_dir, "val_keys.csv"), index=False)

    print()
    print("=" * 65)
    print("[done] V5-lite feature engineering complete")
    print(f"train samples: {train_cnt:,} | positive rate: {pos_train / max(train_cnt, 1):.2%}")
    print(f"val samples:   {val_cnt:,} | positive rate: {pos_val / max(val_cnt, 1):.2%}")
    print(f"feature dim:   {DYN_FEAT_DIM}")
    print(f"processed dir: {processed_dir}")
    print(f"artifacts dir: {artifacts_dir}")
    print("=" * 65)


if __name__ == "__main__":
    build_tensors_v5_lite()
