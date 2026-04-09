"""
B2B 补货系统 V5.0 — SKU 级全国汇总特征工程（10维强化版）
================================================================
核心升级（相对 V2.2）:
  - 废弃 4 个全零死特征: qty_debt, qty_shipped, roll_ship_7, qty_inbound
  - 引入 qty_future (期货/预订数据)，天然的「因果性领先指标」
  - 新特征设计 (3真+4死 → 10真):
      [0]  qty_replenish    当日补货量 (log1p)
      [1]  roll_repl_7      7日滚动补货 (log1p)
      [2]  roll_repl_30     30日滚动补货 (log1p)
      [3]  qty_future       当日期货量 (log1p)   ★ 领先指标
      [4]  roll_fut_7       7日滚动期货 (log1p)  ★ 领先指标
      [5]  roll_fut_30      30日滚动期货 (log1p) ★ 领先指标
      [6]  repl_velocity    补货加速度: log1p域下7日均速-30日均速 (clip)
      [7]  fut2repl_ratio   期货/补货比: 近7日期货 / (近7日补货+ε) (log1p)
      [8]  repl_volatility  7日补货波动率: std / (mean+ε) (log1p)
      [9]  days_since_last  距最近有补货天数 / 30 (归一化到0~1)

输出:
  data/processed_v5/  — 训练/验证张量
  data/artifacts_v5/  — 编码器、meta、val_keys
"""
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import json
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import timedelta

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ======================================================
# 常量
# ======================================================
LOOKBACK        = 90
FORECAST        = 30
SPLIT_DATE      = '2025-12-01'
NEG_STEP        = 1     # 保留全量负样本（与 V2.2 一致）

STATIC_CAT_COLS = [
    'sku_id', 'style_id', 'product_name', 'category',
    'sub_category', 'season', 'series', 'band', 'size_id', 'color_id'
]
STATIC_NUM_COLS = ['qty_first_order', 'price_tag']
DYN_FEAT_DIM    = 10    # ★ 10维（原7维3真4死→10维10真）

# 输出到独立目录，不覆盖 V2/V3 数据
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_v5')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v5')
GOLD_DIR      = os.path.join(PROJECT_ROOT, 'data', 'gold')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


class DummyLE:
    classes_ = np.arange(13)


def rolling_sum(arr, window, n):
    """快速滚动求和（用前缀和实现 O(n)）"""
    cum = np.cumsum(np.concatenate([[0.0], arr]))
    res = np.zeros(n, dtype=np.float32)
    for j in range(n):
        i_end = j + 1
        i_sta = max(0, i_end - window)
        res[j] = cum[i_end] - cum[i_sta]
    return np.maximum(res, 0)


def rolling_log1p(arr, window, n):
    return np.log1p(rolling_sum(arr, window, n))


def build_tensors_v5():
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 65)
    print("🚀 [V5.0 Feature Engineering] 10维强化版 期货信号 + 趋势特征")
    print("=" * 65)

    # ----------------------------------------
    # 1. 读取并聚合数据
    # ----------------------------------------
    df_path = os.path.join(GOLD_DIR, 'wide_table_sku.csv')
    print(f"[{time.strftime('%H:%M:%S')}] 读取 wide_table_sku.csv ...")
    df = pd.read_csv(df_path)
    df['date'] = pd.to_datetime(df['date']).dt.date

    # 确认 qty_future 存在
    if 'qty_future' not in df.columns:
        raise ValueError("❌ wide_table_sku.csv 缺少 qty_future 列！请先重跑 ETL。")

    print(f"[{time.strftime('%H:%M:%S')}] 全国 (sku_id, date) 聚合...")
    # ★ V5: 聚合 qty_replenish + qty_future（去掉全零的 qty_debt/shipped/inbound）
    dyn_agg = df.groupby(['sku_id', 'date']).agg(
        qty_replenish=('qty_replenish', 'sum'),
        qty_future   =('qty_future',    'sum'),
    ).reset_index()

    static_agg = df[STATIC_CAT_COLS + STATIC_NUM_COLS].drop_duplicates('sku_id')
    static_agg['orig_sku'] = static_agg['sku_id'].astype(str)
    unique_skus = dyn_agg['sku_id'].nunique()
    print(f"    唯一 SKU 数: {unique_skus:,}")
    nz_future = (dyn_agg['qty_future'] > 0).sum()
    nz_repl   = (dyn_agg['qty_replenish'] > 0).sum()
    print(f"    qty_replenish 非零行: {nz_repl:,} | qty_future 非零行: {nz_future:,}")

    # ----------------------------------------
    # 2. 编码静态特征（与 V2 完全一致）
    # ----------------------------------------
    encoders = {}
    for col in STATIC_CAT_COLS:
        if col not in static_agg.columns:
            static_agg[col] = 'Unknown'
        le = LabelEncoder()
        static_agg[col] = le.fit_transform(static_agg[col].astype(str))
        encoders[col] = le

    encoders['month'] = DummyLE()

    with open(os.path.join(ARTIFACTS_DIR, 'label_encoders_v5.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    print(f"[{time.strftime('%H:%M:%S')}] LabelEncoder 已保存 → artifacts_v5/label_encoders_v5.pkl")

    STATIC_DIM = len(STATIC_CAT_COLS) + 1 + len(STATIC_NUM_COLS)  # 13
    static_dict = {}
    for _, row in static_agg.iterrows():
        arr = [float(row[c]) for c in STATIC_CAT_COLS]
        arr.append(0.0)   # month 占位（后面逐样本填入）
        for c in STATIC_NUM_COLS:
            arr.append(float(row[c]) if pd.notnull(row[c]) else 0.0)
        static_dict[str(row['orig_sku'])] = np.array(arr, dtype=np.float32)

    # ----------------------------------------
    # 3. 构造致密时间序列 (SKU × 365天 × 2列)
    # ----------------------------------------
    all_dates   = pd.date_range('2025-01-01', '2025-12-31', freq='D').date
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_days      = len(all_dates)       # 365

    print(f"[{time.strftime('%H:%M:%S')}] 填充 365 天致密日历矩阵...")
    sku_list = sorted(dyn_agg['sku_id'].unique())
    sku_to_i = {s: i for i, s in enumerate(sku_list)}

    # dyn_matrix: (n_sku, 365, 2)  — [0]=qty_replenish, [1]=qty_future
    dyn_matrix = np.zeros((len(sku_list), n_days, 2), dtype=np.float32)
    for row in tqdm(dyn_agg.itertuples(index=False), total=len(dyn_agg), desc="填入日历"):
        si = sku_to_i.get(row.sku_id, -1)
        di = date_to_idx.get(row.date, -1)
        if si >= 0 and di >= 0:
            dyn_matrix[si, di, 0] = row.qty_replenish
            dyn_matrix[si, di, 1] = row.qty_future

    # ----------------------------------------
    # 4. 生成 10 维时序特征池
    # ----------------------------------------
    ts_feat = np.zeros((len(sku_list), n_days, DYN_FEAT_DIM), dtype=np.float32)
    val_keys = []

    print(f"[{time.strftime('%H:%M:%S')}] 生成 10 维特征池（含期货信号+趋势）...")
    for si in tqdm(range(len(sku_list)), desc="计算特征"):
        r = dyn_matrix[si, :, 0]   # qty_replenish，shape (365,)
        f = dyn_matrix[si, :, 1]   # qty_future，shape (365,)

        # --- 第一组: 补货核心信号 (3维) ---
        ts_feat[si, :, 0] = np.log1p(np.maximum(r, 0))         # 当日补货
        roll_r7  = rolling_sum(r, 7,  n_days)
        roll_r30 = rolling_sum(r, 30, n_days)
        ts_feat[si, :, 1] = np.log1p(roll_r7)                  # 7日滚动补货
        ts_feat[si, :, 2] = np.log1p(roll_r30)                 # 30日滚动补货

        # --- 第二组: 期货领先信号 (3维) ---
        ts_feat[si, :, 3] = np.log1p(np.maximum(f, 0))         # 当日期货
        roll_f7  = rolling_sum(f, 7,  n_days)
        roll_f30 = rolling_sum(f, 30, n_days)
        ts_feat[si, :, 4] = np.log1p(roll_f7)                  # 7日滚动期货
        ts_feat[si, :, 5] = np.log1p(roll_f30)                 # 30日滚动期货

        # --- 第三组: 趋势与节奏特征 (4维) ---
        # [6] 补货加速度: 7日均速 vs 30日均速（在原始量域比较，再取 log1p 域差异）
        # 用 7/30 天的日均量之差来衡量：正=加速(爆款先兆), 负=减速(换季)
        avg_r7  = roll_r7  / 7.0
        avg_r30 = roll_r30 / 30.0
        velocity = (avg_r7 - avg_r30)
        # clip 防止极端值，scale 到合理数值范围再存储（用 sign×log1p(|x|) 保留方向）
        ts_feat[si, :, 6] = np.sign(velocity) * np.log1p(np.abs(velocity))

        # [7] 期货/补货比: 近7日期货 / (近7日补货 + ε)
        # >1 = 大量预订但尚未补货(即将爆发), <1 = 已在补货(需求已释放)
        fut2repl = roll_f7 / (roll_r7 + 1.0)   # +1 平滑，避免除零
        ts_feat[si, :, 7] = np.log1p(np.clip(fut2repl, 0, 50))  # clip 防极值

        # [8] 7日补货波动率: std(近7天补货) / (mean(近7天) + ε)
        # 需要逐日计算（无法用前缀和，改用滑动窗口近似，用7天的方差近似）
        # 用 E[X^2] - E[X]^2 = Var(X) 的前缀和公式实现 O(n) 计算
        r2 = r ** 2
        cum_r  = np.cumsum(np.concatenate([[0.0], r]))
        cum_r2 = np.cumsum(np.concatenate([[0.0], r2]))
        w = 7
        var7 = np.zeros(n_days, dtype=np.float32)
        for j in range(n_days):
            i_end = j + 1
            i_sta = max(0, i_end - w)
            cnt   = i_end - i_sta
            s1    = cum_r[i_end]  - cum_r[i_sta]
            s2    = cum_r2[i_end] - cum_r2[i_sta]
            mean_ = s1 / cnt
            var7[j] = max(0.0, s2 / cnt - mean_ ** 2)
        std7 = np.sqrt(var7)
        mean7 = roll_r7 / 7.0
        volatility = std7 / (mean7 + 1.0)    # +1 平滑
        ts_feat[si, :, 8] = np.log1p(volatility)

        # [9] 距最近有补货天数（归一化到 [0,1]，1=30天前最后补货）
        last_repl_days = np.zeros(n_days, dtype=np.float32)
        gap = 30  # 初始化为 30（上限）
        for j in range(n_days):
            if r[j] > 0:
                gap = 0
            else:
                gap = min(gap + 1, 30)
            last_repl_days[j] = gap
        ts_feat[si, :, 9] = last_repl_days / 30.0

    # ----------------------------------------
    # 5. 暴力滑窗截取样本（与 V2.2 逻辑完全一致）
    # ----------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] 暴力滑窗扫描样本 (LOOKBACK={LOOKBACK}, FORECAST={FORECAST})...")
    split_date = pd.to_datetime(SPLIT_DATE).date()

    fn = lambda name: os.path.normpath(os.path.join(PROCESSED_DIR, name))
    f_tr_dyn = open(fn('X_train_dyn.bin'), 'wb')
    f_tr_sta = open(fn('X_train_static.bin'), 'wb')
    f_tr_cls = open(fn('y_train_cls.bin'), 'wb')
    f_tr_reg = open(fn('y_train_reg.bin'), 'wb')
    f_va_dyn = open(fn('X_val_dyn.bin'), 'wb')
    f_va_sta = open(fn('X_val_static.bin'), 'wb')
    f_va_cls = open(fn('y_val_cls.bin'), 'wb')
    f_va_reg = open(fn('y_val_reg.bin'), 'wb')

    tr_cnt, va_cnt, pos_tr, pos_va = 0, 0, 0, 0
    end_idx = n_days - FORECAST

    for si, sku in enumerate(tqdm(sku_list, desc="滑窗截取")):
        if str(sku) not in static_dict:
            continue

        base_sf  = static_dict[str(sku)]
        sku_qty  = dyn_matrix[si, :, 0]   # qty_replenish 用于计算 target

        for i in range(LOOKBACK - 1, end_idx):
            target      = float(sku_qty[i + 1: i + FORECAST + 1].sum())
            anchor_date = all_dates[i]
            is_train    = anchor_date < split_date

            # NEG_STEP 降采样（当前=1，即全部保留）
            if target == 0:
                if is_train and (i % NEG_STEP != 0):
                    continue

            # 提取 90×10 动态窗口
            window = ts_feat[si, i - LOOKBACK + 1: i + 1]  # (90, 10)

            # 填入月份 + log1p 数值特征
            sf = base_sf.copy()
            sf[-3] = float(anchor_date.month)   # month（倒数第3列）
            sf[-2] = np.log1p(max(0.0, sf[-2]))  # qty_first_order
            sf[-1] = np.log1p(max(0.0, sf[-1]))  # price_tag

            y_c = np.array([1.0 if target > 0 else 0.0], dtype=np.float32)
            y_r = np.array([np.log1p(target)], dtype=np.float32)

            if is_train:
                window.tofile(f_tr_dyn); sf.tofile(f_tr_sta)
                y_c.tofile(f_tr_cls);   y_r.tofile(f_tr_reg)
                tr_cnt += 1
                if target > 0: pos_tr += 1
            else:
                window.tofile(f_va_dyn); sf.tofile(f_va_sta)
                y_c.tofile(f_va_cls);   y_r.tofile(f_va_reg)
                va_cnt += 1
                if target > 0: pos_va += 1
                val_keys.append({'sku_id': sku_list[si], 'date': anchor_date})

    for fh in [f_tr_dyn, f_tr_sta, f_tr_cls, f_tr_reg,
               f_va_dyn, f_va_sta, f_va_cls, f_va_reg]:
        fh.close()

    # ----------------------------------------
    # 6. 归一化（与 V2.2 一致）
    # ----------------------------------------
    if tr_cnt > 0:
        print(f"[{time.strftime('%H:%M:%S')}] MinMaxScaler 归一化...")
        scaler = MinMaxScaler()
        X_tr = np.memmap(fn('X_train_dyn.bin'), dtype=np.float32, mode='r+',
                         shape=(tr_cnt, LOOKBACK, DYN_FEAT_DIM))
        for i in range(0, tr_cnt, 20000):
            c = np.nan_to_num(X_tr[i:i+20000].reshape(-1, DYN_FEAT_DIM),
                              nan=0, posinf=0, neginf=0)
            scaler.partial_fit(c)
        for i in range(0, tr_cnt, 20000):
            c = np.nan_to_num(X_tr[i:i+20000].reshape(-1, DYN_FEAT_DIM),
                              nan=0, posinf=0, neginf=0)
            X_tr[i:i+20000] = scaler.transform(c).reshape(X_tr[i:i+20000].shape)
            X_tr.flush()

        if va_cnt > 0:
            X_va = np.memmap(fn('X_val_dyn.bin'), dtype=np.float32, mode='r+',
                             shape=(va_cnt, LOOKBACK, DYN_FEAT_DIM))
            for i in range(0, va_cnt, 20000):
                c = np.nan_to_num(X_va[i:i+20000].reshape(-1, DYN_FEAT_DIM),
                                  nan=0, posinf=0, neginf=0)
                X_va[i:i+20000] = scaler.transform(c).reshape(X_va[i:i+20000].shape)
                X_va.flush()

        with open(os.path.join(ARTIFACTS_DIR, 'feature_scaler_v5.pkl'), 'wb') as fh:
            pickle.dump(scaler, fh)

    # ----------------------------------------
    # 7. Meta & Val Keys 写入
    # ----------------------------------------
    meta = {
        'dyn_feat_dim'   : DYN_FEAT_DIM,
        'static_dim'     : STATIC_DIM,
        'static_cat_cols': STATIC_CAT_COLS,
        'static_num_cols': STATIC_NUM_COLS,
        'lookback'       : LOOKBACK,
        'forecast'       : FORECAST,
        'train_cnt'      : tr_cnt,
        'val_cnt'        : va_cnt,
        'pos_train'      : pos_tr,
        'pos_val'        : pos_va,
        'split_date'     : SPLIT_DATE,
        'feature_names'  : [
            'qty_replenish', 'roll_repl_7', 'roll_repl_30',
            'qty_future', 'roll_fut_7', 'roll_fut_30',
            'repl_velocity', 'fut2repl_ratio', 'repl_volatility', 'days_since_last'
        ],
    }
    with open(os.path.join(ARTIFACTS_DIR, 'meta_v5.json'), 'w') as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)

    if va_cnt > 0:
        pd.DataFrame(val_keys).to_csv(
            os.path.join(ARTIFACTS_DIR, 'val_keys.csv'), index=False)
        print(f"  ➤ [Keys] val_keys.csv 已保存 ({len(val_keys)} 行)")

    print("\n" + "=" * 65)
    print("🎉 V5.0 特征工程完毕！")
    print(f"  ➤ [Train] 样本数: {tr_cnt:,} | 正样本率: {pos_tr/max(tr_cnt,1):.2%}")
    print(f"  ➤ [Val]   样本数: {va_cnt:,} | 正样本率: {pos_va/max(va_cnt,1):.2%}")
    print(f"  ➤ [特征]  动态特征维度: {DYN_FEAT_DIM} 维 (含期货信号)")
    print("=" * 65)


if __name__ == "__main__":
    build_tensors_v5()
