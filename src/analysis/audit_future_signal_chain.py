"""
Audit the future-signal data chain from silver -> gold -> feature tensors.

This script answers three questions:
1. Is qty_future present and preserved through ETL?
2. Why does qty_future look sparse after tensorization?
3. Does qty_future have useful lead-signal value for future replenish target?
"""
import json
import os
from datetime import date

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SILVER_PATH = os.path.join(PROJECT_ROOT, "data", "silver", "clean_orders.csv")
GOLD_PATH = os.path.join(PROJECT_ROOT, "data", "gold", "wide_table_sku.csv")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "phase5", "future_signal_audit.txt")
SEED = 42
SAMPLE_N = 50000


def add(lines, text=""):
    print(text)
    lines.append(text)


def section(lines, title):
    add(lines)
    add(lines, "=" * 76)
    add(lines, title)
    add(lines, "=" * 76)


def load_silver():
    usecols = ["buyer_id", "sku_id", "order_date", "qty_replenish", "qty_future"]
    df = pd.read_csv(SILVER_PATH, usecols=usecols, encoding="utf-8-sig")
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def load_gold():
    usecols = ["buyer_id", "sku_id", "date", "qty_replenish", "qty_future"]
    df = pd.read_csv(GOLD_PATH, usecols=usecols, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df


def summarize_silver(lines, silver_df):
    section(lines, "1. Silver Summary")
    total_rows = len(silver_df)
    future_nz = int((silver_df["qty_future"] != 0).sum())
    repl_nz = int((silver_df["qty_replenish"] != 0).sum())
    both_nz = int(((silver_df["qty_replenish"] != 0) & (silver_df["qty_future"] != 0)).sum())
    future_only = int(((silver_df["qty_replenish"] == 0) & (silver_df["qty_future"] != 0)).sum())
    repl_only = int(((silver_df["qty_replenish"] != 0) & (silver_df["qty_future"] == 0)).sum())

    add(lines, f"rows: {total_rows:,}")
    add(lines, f"buyers: {silver_df['buyer_id'].nunique():,}")
    add(lines, f"skus: {silver_df['sku_id'].nunique():,}")
    add(lines, f"dates: {silver_df['order_date'].nunique():,}")
    add(lines, f"qty_replenish != 0 rows: {repl_nz:,}")
    add(lines, f"qty_future != 0 rows: {future_nz:,}")
    add(lines, f"future_only rows: {future_only:,}")
    add(lines, f"repl_only rows: {repl_only:,}")
    add(lines, f"both_nonzero rows: {both_nz:,}")
    add(lines, f"qty_replenish sum: {silver_df['qty_replenish'].sum():,.1f}")
    add(lines, f"qty_future sum: {silver_df['qty_future'].sum():,.1f}")

    sku_date = (
        silver_df.groupby(["sku_id", "order_date"], as_index=False)[["qty_replenish", "qty_future"]]
        .sum()
        .rename(columns={"order_date": "date"})
    )
    future_nz_sku_date = int((sku_date["qty_future"] != 0).sum())
    repl_nz_sku_date = int((sku_date["qty_replenish"] != 0).sum())
    add(lines)
    add(lines, "After aggregating to SKU-date:")
    add(lines, f"sku-date rows: {len(sku_date):,}")
    add(lines, f"sku-date future != 0 rows: {future_nz_sku_date:,}")
    add(lines, f"sku-date replenish != 0 rows: {repl_nz_sku_date:,}")
    add(lines, f"sku-date future_only rows: {int(((sku_date['qty_replenish'] == 0) & (sku_date['qty_future'] != 0)).sum()):,}")
    add(lines, f"sku-date both_nonzero rows: {int(((sku_date['qty_replenish'] != 0) & (sku_date['qty_future'] != 0)).sum()):,}")
    return sku_date


def summarize_gold(lines, silver_df, gold_df):
    section(lines, "2. Silver -> Gold Consistency")
    silver_agg = (
        silver_df.groupby(["buyer_id", "sku_id", "order_date"], as_index=False)[["qty_replenish", "qty_future"]]
        .sum()
        .rename(columns={"order_date": "date"})
        .sort_values(["buyer_id", "sku_id", "date"])
        .reset_index(drop=True)
    )
    gold_cmp = gold_df.sort_values(["buyer_id", "sku_id", "date"]).reset_index(drop=True)

    same_rows = len(silver_agg) == len(gold_cmp)
    same_keys = silver_agg[["buyer_id", "sku_id", "date"]].equals(gold_cmp[["buyer_id", "sku_id", "date"]])
    repl_diff = float((silver_agg["qty_replenish"] - gold_cmp["qty_replenish"]).abs().sum())
    future_diff = float((silver_agg["qty_future"] - gold_cmp["qty_future"]).abs().sum())

    add(lines, f"same row count: {same_rows}")
    add(lines, f"same keys: {same_keys}")
    add(lines, f"abs diff sum qty_replenish: {repl_diff:.1f}")
    add(lines, f"abs diff sum qty_future: {future_diff:.1f}")


def summarize_2025_density(lines, gold_df):
    section(lines, "3. Why qty_future Looks Sparse")
    gold_2025 = gold_df[
        (gold_df["date"] >= pd.Timestamp("2025-01-01")) &
        (gold_df["date"] <= pd.Timestamp("2025-12-31"))
    ].copy()
    sku_date = (
        gold_2025.groupby(["sku_id", "date"], as_index=False)[["qty_replenish", "qty_future"]]
        .sum()
    )
    n_sku = sku_date["sku_id"].nunique()
    total_calendar_cells = n_sku * 365
    future_nz = int((sku_date["qty_future"] != 0).sum())
    repl_nz = int((sku_date["qty_replenish"] != 0).sum())

    add(lines, "Feature builder densifies data into a full SKU x 365-day calendar.")
    add(lines, f"unique SKUs used by feature builder: {n_sku:,}")
    add(lines, f"calendar cells: {total_calendar_cells:,} (= sku_count x 365)")
    add(lines, f"2025 sku-date future != 0 cells: {future_nz:,} ({future_nz / max(total_calendar_cells, 1):.4%})")
    add(lines, f"2025 sku-date replenish != 0 cells: {repl_nz:,} ({repl_nz / max(total_calendar_cells, 1):.4%})")
    add(lines)
    add(lines, "This is why `163,899` future rows can still look sparse in tensors:")
    add(lines, "- `163,899` is at buyer-sku-date grain in silver/gold.")
    add(lines, "- feature engineering first aggregates to SKU-date.")
    add(lines, "- then it expands to a dense 365-day calendar for every SKU.")
    return sku_date


def compute_lead_stats(lines, sku_date):
    section(lines, "4. Lead-Signal Strength at SKU-date Level")
    all_dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    sku_list = sorted(sku_date["sku_id"].unique())
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    sku_to_idx = {s: i for i, s in enumerate(sku_list)}

    mat_r = np.zeros((len(sku_list), len(all_dates)), dtype=np.float32)
    mat_f = np.zeros((len(sku_list), len(all_dates)), dtype=np.float32)

    for row in sku_date.itertuples(index=False):
        si = sku_to_idx[row.sku_id]
        di = date_to_idx.get(row.date.normalize(), -1)
        if di >= 0:
            mat_r[si, di] = row.qty_replenish
            mat_f[si, di] = row.qty_future

    cumsum = np.cumsum(np.pad(mat_r, ((0, 0), (1, 0))), axis=1)
    future30 = np.zeros_like(mat_r)
    for t in range(len(all_dates) - 30):
        future30[:, t] = cumsum[:, t + 31] - cumsum[:, t + 1]

    valid = np.arange(len(all_dates) - 30)
    future_today = mat_f[:, valid].reshape(-1)
    repl_today = mat_r[:, valid].reshape(-1)
    repl_next30 = future30[:, valid].reshape(-1)
    future_flag = future_today > 0

    add(lines, f"samples evaluated: {len(future_today):,}")
    add(lines, f"future>0 rate: {future_flag.mean():.4%}")
    add(lines, f"P(next30_replenish>0 | future>0): {(repl_next30[future_flag] > 0).mean():.4%}")
    add(lines, f"P(next30_replenish>0 | future=0): {(repl_next30[~future_flag] > 0).mean():.4%}")
    add(lines, f"mean(next30_replenish | future>0): {repl_next30[future_flag].mean():.4f}")
    add(lines, f"mean(next30_replenish | future=0): {repl_next30[~future_flag].mean():.4f}")
    add(lines, f"corr(future_t, next30_replenish): {np.corrcoef(future_today, repl_next30)[0, 1]:.4f}")
    add(lines, f"corr(replenish_t, next30_replenish): {np.corrcoef(repl_today, repl_next30)[0, 1]:.4f}")


def compute_buyer_level_hit_rate(lines, silver_df):
    section(lines, "5. Buyer-SKU Event-Level Lead Check")
    agg = (
        silver_df.groupby(["buyer_id", "sku_id", "order_date"], as_index=False)[["qty_replenish", "qty_future"]]
        .sum()
        .sort_values(["buyer_id", "sku_id", "order_date"])
        .reset_index(drop=True)
    )

    future_only_rows = 0
    future_only_hit = 0

    for _, group in agg.groupby(["buyer_id", "sku_id"], sort=False):
        dates = group["order_date"].tolist()
        repl = group["qty_replenish"].to_numpy()
        fut = group["qty_future"].to_numpy()
        n_rows = len(group)

        for i in range(n_rows):
            if fut[i] <= 0 or repl[i] != 0:
                continue
            future_only_rows += 1
            horizon_end = dates[i] + pd.Timedelta(days=30)
            hit = False
            for j in range(i + 1, n_rows):
                if dates[j] > horizon_end:
                    break
                if repl[j] > 0:
                    hit = True
                    break
            future_only_hit += int(hit)

    hit_rate = future_only_hit / future_only_rows if future_only_rows else 0.0
    add(lines, f"future_only buyer-sku-date rows: {future_only_rows:,}")
    add(lines, f"future_only rows followed by replenish within 30 days: {future_only_hit:,}")
    add(lines, f"future_only hit rate within same buyer-sku: {hit_rate:.4%}")


def compute_feature_density(lines):
    section(lines, "6. Feature Tensor Density")
    rng = np.random.default_rng(SEED)
    configs = [
        ("v3", "data/artifacts_v3/meta_v2.json", "data/processed_v3/X_train_dyn.bin",
         ["qty_replenish", "roll_repl_7", "roll_repl_30", "qty_debt", "qty_shipped", "roll_ship_7", "qty_inbound"]),
        ("v5", "data/artifacts_v5/meta_v5.json", "data/processed_v5/X_train_dyn.bin", None),
    ]

    for version, meta_rel, dyn_rel, fallback_names in configs:
        meta_path = os.path.join(PROJECT_ROOT, meta_rel)
        dyn_path = os.path.join(PROJECT_ROOT, dyn_rel)
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        feature_names = meta.get("feature_names", fallback_names)
        train_cnt = meta["train_cnt"]
        lookback = meta.get("lookback", 90)
        dyn_dim = meta.get("dyn_feat_dim", len(feature_names))
        sample_n = min(SAMPLE_N, train_cnt)
        sample_idx = rng.choice(train_cnt, size=sample_n, replace=False)
        arr = np.memmap(dyn_path, dtype=np.float32, mode="r", shape=(train_cnt, lookback, dyn_dim))
        sample = np.asarray(arr[sample_idx])
        nz_rate = (np.abs(sample) > 1e-8).mean(axis=(0, 1))
        mean_abs = np.abs(sample).mean(axis=(0, 1))

        add(lines, f"{version.upper()} train sample size: {sample_n:,}")
        for i, name in enumerate(feature_names):
            add(lines, f"  [{i}] {name:<18} nz_rate={nz_rate[i]:.4%} | mean_abs={mean_abs[i]:.6f}")


def main():
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    lines = []

    silver_df = load_silver()
    gold_df = load_gold()

    sku_date = summarize_silver(lines, silver_df)
    summarize_gold(lines, silver_df, gold_df)
    sku_date_2025 = summarize_2025_density(lines, gold_df)
    compute_lead_stats(lines, sku_date_2025)
    compute_buyer_level_hit_rate(lines, silver_df)
    compute_feature_density(lines)

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print()
    print(f"[saved] {REPORT_PATH}")


if __name__ == "__main__":
    main()
