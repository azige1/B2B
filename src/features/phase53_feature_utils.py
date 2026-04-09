import os
import pickle
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOOKBACK = 90
FORECAST = 30
SPLIT_DATE = "2025-12-01"
NEG_STEP = 1
FEATURE_VAL_MODE = "full_holdout"

STATIC_CAT_COLS = [
    "sku_id", "style_id", "product_name", "category",
    "sub_category", "season", "series", "band", "size_id", "color_id",
]
STATIC_BASE_NUM_COLS = ["qty_first_order", "price_tag"]

BASE_GOLD_PATH = os.path.join(PROJECT_ROOT, "data", "gold", "wide_table_sku.csv")
BASE_SILVER_PATH = os.path.join(PROJECT_ROOT, "data", "silver", "clean_orders.csv")


class DummyLE:
    classes_ = np.arange(13)


def get_runtime_dirs(base_processed_dir, base_artifacts_dir):
    output_tag = os.environ.get("FEATURE_OUTPUT_TAG", "").strip()
    suffix = f"_{output_tag}" if output_tag else ""
    processed_dir = f"{base_processed_dir}{suffix}"
    artifacts_dir = f"{base_artifacts_dir}{suffix}"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    return {
        "split_date": os.environ.get("FEATURE_SPLIT_DATE", SPLIT_DATE),
        "val_mode": get_feature_val_mode(),
        "processed_dir": processed_dir,
        "artifacts_dir": artifacts_dir,
        "output_tag": output_tag,
        "suffix": suffix,
    }


def get_feature_val_mode():
    val_mode = os.environ.get("FEATURE_VAL_MODE", FEATURE_VAL_MODE).strip().lower()
    if val_mode not in {"full_holdout", "single_anchor"}:
        raise ValueError(
            f"Unsupported FEATURE_VAL_MODE={val_mode}. "
            "Expected one of: full_holdout, single_anchor"
        )
    return val_mode


def split_flags(anchor_date, split_date, val_mode):
    is_train = anchor_date < split_date
    if val_mode == "single_anchor":
        is_val = anchor_date == split_date
    else:
        is_val = anchor_date >= split_date
    return is_train, is_val


def map_output_tag_to_v5_lite(output_tag=""):
    tag = (output_tag or "").strip()
    if not tag:
        return ""

    version_suffixes = [
        "_v6_event",
        "_v5_lite_cov",
        "_v3_filtered",
        "_v5_lite",
    ]
    for suffix in version_suffixes:
        if tag.endswith(suffix):
            base = tag[: -len(suffix)]
            return f"{base}_v5_lite" if base else "v5_lite"
    return tag


def load_keep_skus_from_v5_lite(output_tag=""):
    candidate_tags = []
    for tag in [output_tag, map_output_tag_to_v5_lite(output_tag)]:
        tag = (tag or "").strip()
        if tag not in candidate_tags:
            candidate_tags.append(tag)

    checked_paths = []
    for tag in candidate_tags:
        suffix = f"_{tag}" if tag else ""
        activity_path = os.path.join(
            PROJECT_ROOT, "data", f"artifacts_v5_lite{suffix}", "sku_activity_v5_lite.csv"
        )
        checked_paths.append(activity_path)
        if not os.path.exists(activity_path):
            continue
        activity = pd.read_csv(activity_path)
        if "kept_for_training" in activity.columns:
            keep = set(activity.loc[activity["kept_for_training"], "sku_id"].astype(str))
        else:
            keep = set(activity["sku_id"].astype(str))
        return keep, activity_path

    raise FileNotFoundError(
        "Missing V5-lite keep-set activity file. Checked:\n- "
        + "\n- ".join(checked_paths)
        + "\nBuild the matching V5-lite features first."
    )


def _normalize_date_col(df, date_col):
    df[date_col] = pd.to_datetime(df[date_col]).dt.date
    return df


def load_gold_frame(keep_skus=None):
    usecols = [
        "date",
        "sku_id",
        "buyer_id",
        "qty_replenish",
        "qty_future",
        "qty_debt",
        "qty_shipped",
        "qty_inbound",
        "style_id",
        "product_name",
        "category",
        "sub_category",
        "season",
        "band",
        "series",
        "price_tag",
        "size_id",
        "color_id",
        "qty_first_order",
    ]
    df = pd.read_csv(BASE_GOLD_PATH, usecols=usecols)
    # Keep the same SKU universe behavior as the original V5-lite builder.
    # The dense calendar is still restricted to 2025 later via date_to_idx.
    df = _normalize_date_col(df, "date")
    df["sku_id"] = df["sku_id"].astype(str)
    if keep_skus is not None:
        df = df[df["sku_id"].isin(keep_skus)].copy()
    return df


def load_silver_frame(keep_skus=None):
    usecols = [
        "buyer_id",
        "sku_id",
        "order_date",
        "qty_replenish",
        "qty_future",
        "qty_debt",
        "qty_shipped",
        "qty_first_order",
        "qty_inbound",
    ]
    df = pd.read_csv(BASE_SILVER_PATH, usecols=usecols)
    # Keep out-of-2025 rows so fair-universe SKU membership matches V5-lite.
    # Non-2025 rows are ignored later when no 2025 date index is found.
    df = _normalize_date_col(df, "order_date")
    df["sku_id"] = df["sku_id"].astype(str)
    df["buyer_id"] = df["buyer_id"].astype(str)
    if keep_skus is not None:
        df = df[df["sku_id"].isin(keep_skus)].copy()
    return df


def encode_static_table(static_df, artifacts_dir, encoder_name, extra_num_cols=None):
    extra_num_cols = extra_num_cols or []
    numeric_cols = STATIC_BASE_NUM_COLS + extra_num_cols
    static_df = static_df[STATIC_CAT_COLS + numeric_cols].drop_duplicates("sku_id").copy()
    static_df["orig_sku"] = static_df["sku_id"].astype(str)

    encoders = {}
    for col in STATIC_CAT_COLS:
        if col not in static_df.columns:
            static_df[col] = "Unknown"
        encoder = LabelEncoder()
        static_df[col] = encoder.fit_transform(static_df[col].astype(str))
        encoders[col] = encoder

    encoders["month"] = DummyLE()
    with open(os.path.join(artifacts_dir, encoder_name), "wb") as fh:
        pickle.dump(encoders, fh)

    static_dict = {}
    for _, row in static_df.iterrows():
        values = [float(row[col]) for col in STATIC_CAT_COLS]
        values.append(0.0)
        values.extend(
            float(row[col]) if pd.notnull(row[col]) else 0.0
            for col in numeric_cols
        )
        static_dict[str(row["orig_sku"])] = np.array(values, dtype=np.float32)

    return static_df, static_dict, encoders, numeric_cols


def rolling_sum(arr, window):
    cumulative = np.cumsum(np.concatenate([[0.0], arr.astype(np.float64)]))
    result = np.zeros(len(arr), dtype=np.float32)
    for j in range(len(arr)):
        end_idx = j + 1
        start_idx = max(0, end_idx - window)
        result[j] = cumulative[end_idx] - cumulative[start_idx]
    return result


def rolling_count(arr, window):
    return rolling_sum((arr > 0).astype(np.float32), window)


def rolling_max(arr, window):
    result = np.zeros(len(arr), dtype=np.float32)
    for j in range(len(arr)):
        start_idx = max(0, j - window + 1)
        result[j] = np.max(arr[start_idx : j + 1]) if j >= start_idx else 0.0
    return result


def days_since_last_positive(arr):
    result = np.full(len(arr), LOOKBACK + FORECAST, dtype=np.float32)
    last_idx = -1
    for i, value in enumerate(arr):
        if value > 0:
            last_idx = i
            result[i] = 0.0
        elif last_idx >= 0:
            result[i] = float(i - last_idx)
    return result


def activity_bucket_from_days(repl_days):
    if repl_days > 30:
        return 3.0
    if repl_days >= 10:
        return 2.0
    if repl_days >= 1:
        return 1.0
    return 0.0


def _sliding_buyer_features(day_dicts, n_days, window):
    active_days = Counter()
    qty_sums = defaultdict(float)
    unique_count = np.zeros(n_days, dtype=np.float32)
    top1_share = np.zeros(n_days, dtype=np.float32)
    top3_share = np.zeros(n_days, dtype=np.float32)
    hhi = np.zeros(n_days, dtype=np.float32)

    for day_idx in range(n_days):
        for buyer_id, qty in day_dicts[day_idx].items():
            active_days[buyer_id] += 1
            qty_sums[buyer_id] += float(qty)

        old_idx = day_idx - window
        if old_idx >= 0:
            for buyer_id, qty in day_dicts[old_idx].items():
                active_days[buyer_id] -= 1
                if active_days[buyer_id] <= 0:
                    del active_days[buyer_id]
                qty_sums[buyer_id] -= float(qty)
                if qty_sums[buyer_id] <= 1e-9:
                    qty_sums.pop(buyer_id, None)

        unique_count[day_idx] = float(len(active_days))
        if qty_sums:
            total_qty = float(sum(qty_sums.values()))
            ordered = sorted(qty_sums.values(), reverse=True)
            top1_qty = float(ordered[0])
            top3_qty = float(sum(ordered[:3]))
            top1_share[day_idx] = top1_qty / total_qty if total_qty > 0 else 0.0
            top3_share[day_idx] = top3_qty / total_qty if total_qty > 0 else 0.0
            hhi[day_idx] = float(sum((qty / total_qty) ** 2 for qty in ordered)) if total_qty > 0 else 0.0

    return unique_count, top1_share, top3_share, hhi


def build_buyer_window_arrays(silver_df, sku_list, date_to_idx, n_days, window=LOOKBACK):
    sku_to_idx = {sku: i for i, sku in enumerate(sku_list)}
    repl_counts = np.zeros((len(sku_list), n_days), dtype=np.float32)
    future_counts = np.zeros((len(sku_list), n_days), dtype=np.float32)
    repl_top1_share = np.zeros((len(sku_list), n_days), dtype=np.float32)
    repl_top3_share = np.zeros((len(sku_list), n_days), dtype=np.float32)
    repl_hhi = np.zeros((len(sku_list), n_days), dtype=np.float32)
    future_top_share = np.zeros((len(sku_list), n_days), dtype=np.float32)
    future_top3_share = np.zeros((len(sku_list), n_days), dtype=np.float32)
    future_hhi = np.zeros((len(sku_list), n_days), dtype=np.float32)

    grouped = (
        silver_df.groupby(["sku_id", "order_date", "buyer_id"], as_index=False)
        .agg(
            qty_replenish=("qty_replenish", "sum"),
            qty_future=("qty_future", "sum"),
        )
        .groupby("sku_id")
    )

    for sku, sku_df in tqdm(grouped, desc="buyer90", total=len(grouped)):
        sku_idx = sku_to_idx.get(str(sku))
        if sku_idx is None:
            continue

        repl_day_maps = [dict() for _ in range(n_days)]
        future_day_maps = [dict() for _ in range(n_days)]

        for row in sku_df.itertuples(index=False):
            day_idx = date_to_idx.get(row.order_date)
            if day_idx is None:
                continue
            if row.qty_replenish > 0:
                repl_day_maps[day_idx][row.buyer_id] = (
                    repl_day_maps[day_idx].get(row.buyer_id, 0.0) + float(row.qty_replenish)
                )
            if row.qty_future > 0:
                future_day_maps[day_idx][row.buyer_id] = (
                    future_day_maps[day_idx].get(row.buyer_id, 0.0) + float(row.qty_future)
                )

        repl_unique, repl_top1, repl_top3, repl_hhi_row = _sliding_buyer_features(repl_day_maps, n_days, window)
        future_unique, future_top, future_top3, future_hhi_row = _sliding_buyer_features(future_day_maps, n_days, window)

        repl_counts[sku_idx] = repl_unique
        future_counts[sku_idx] = future_unique
        repl_top1_share[sku_idx] = repl_top1
        repl_top3_share[sku_idx] = repl_top3
        repl_hhi[sku_idx] = repl_hhi_row
        future_top_share[sku_idx] = future_top
        future_top3_share[sku_idx] = future_top3
        future_hhi[sku_idx] = future_hhi_row

    return {
        "repl_buyer_count_90": repl_counts,
        "future_buyer_count_90": future_counts,
        "repl_top1_share_90": repl_top1_share,
        "repl_top3_share_90": repl_top3_share,
        "repl_hhi_90": repl_hhi,
        "future_top1_share_90": future_top_share,
        "future_top3_share_90": future_top3_share,
        "future_hhi_90": future_hhi,
    }
