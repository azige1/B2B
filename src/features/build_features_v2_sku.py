"""
B2B 补货系统 V2.0 — SKU 级全国汇总特征工程 (V2.2 暴力滑窗版)
=====================================================
完全抛弃寻找"合规锚点"的智能尝试。
核心逻辑：
  1. 为每个 SKU 生成全量 1~365 天的一维补货数组。
  2. 无脑遍历全年的每一个可用天数 (day=30 遍历到 335)。
  3. 如果未来 30 天总和 > 0，作为正样本（100% 录入）。
  4. 如果未来 30 天总和 == 0，作为负样本随机抛弃 90% (即 1:10 采样)。
这就结了。只要这世界上还存在未来 30 天补货总量 >0 的时段，它绝对会生成张量！
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
LOOKBACK        = 90   # [v2.5] 扩展至 90 天，对齐业务 1.7 规范，提供更长的历史记忆
FORECAST        = 30   
SPLIT_DATE      = '2025-12-01'  # ★ 1-11月全部训练，12月作为验证集（锁定）
NEG_STEP        = 1     # [V2.2 终极重构] 关闭截断，100% 保留死库存负样本，全量训练以极限提升 Precision

STATIC_CAT_COLS = [
    'sku_id', 'style_id', 'product_name', 'category',
    'sub_category', 'season', 'series', 'band', 'size_id', 'color_id'
]
STATIC_NUM_COLS = ['qty_first_order', 'price_tag']
DYN_FEAT_DIM    = 7   

PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_v3')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v3')
GOLD_DIR      = os.path.join(PROJECT_ROOT, 'data', 'gold')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


class DummyLE:
    classes_ = np.arange(13)


def build_tensors_v2():
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')
    
    print("=" * 60)
    print("🚀 [V2.2 Feature Engineering] SKU 级全国汇总 (暴力滑窗版)")
    print("=" * 60)

    # ----------------------------------------
    # 1. 读取并聚合数据
    # ----------------------------------------
    df_path = os.path.join(GOLD_DIR, 'wide_table_sku.csv')
    df = pd.read_csv(df_path)
    df['date'] = pd.to_datetime(df['date']).dt.date

    print(f"[{time.strftime('%H:%M:%S')}] 🔄 全国 (sku_id, date) 聚合...")
    dyn_agg = df.groupby(['sku_id', 'date']).agg(
        qty_replenish=('qty_replenish', 'sum'),
        qty_debt     =('qty_debt',      'sum'),
        qty_shipped  =('qty_shipped',   'sum'),
        qty_inbound  =('qty_inbound',   'sum'),
    ).reset_index()

    static_agg = df[STATIC_CAT_COLS + STATIC_NUM_COLS].drop_duplicates('sku_id')
    static_agg['orig_sku'] = static_agg['sku_id'].astype(str)
    unique_skus = dyn_agg['sku_id'].nunique()
    print(f"    唯一 SKU 数: {unique_skus:,}")

    # ----------------------------------------
    # 2. 编码静态特征
    # ----------------------------------------
    encoders = {}
    for col in STATIC_CAT_COLS:
        if col not in static_agg.columns: static_agg[col] = 'Unknown'
        le = LabelEncoder()
        static_agg[col] = le.fit_transform(static_agg[col].astype(str))
        encoders[col] = le

    encoders['month'] = DummyLE()

    with open(os.path.join(ARTIFACTS_DIR, 'label_encoders_v2.pkl'), 'wb') as f:
        pickle.dump(encoders, f)

    STATIC_DIM = len(STATIC_CAT_COLS) + 1 + len(STATIC_NUM_COLS)
    static_dict = {}
    for _, row in static_agg.iterrows():
        arr = [float(row[c]) for c in STATIC_CAT_COLS]
        arr.append(0.0) 
        for c in STATIC_NUM_COLS:
            arr.append(float(row[c]) if pd.notnull(row[c]) else 0.0)
        static_dict[str(row['orig_sku'])] = np.array(arr, dtype=np.float32)

    # ----------------------------------------
    # 3. 构造完美的致密时间序列 (所有 SKU x 365天)
    # ----------------------------------------
    # 2025 全年数据，不超出2025范围
    all_dates = pd.date_range('2025-01-01', '2025-12-31', freq='D').date
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_days = len(all_dates)

    print(f"[{time.strftime('%H:%M:%S')}] 🗓️  填充 365 天致密日历矩阵...")
    sku_list = sorted(dyn_agg['sku_id'].unique())
    sku_to_i = {s: i for i, s in enumerate(sku_list)}

    # dyn_matrix = (10000, 365, 4)
    dyn_matrix = np.zeros((len(sku_list), n_days, 4), dtype=np.float32)
    for row in tqdm(dyn_agg.itertuples(index=False), total=len(dyn_agg), desc="填入"):
        si = sku_to_i.get(row.sku_id, -1)
        di = date_to_idx.get(row.date, -1)
        if si >= 0 and di >= 0:
            dyn_matrix[si, di, 0] = row.qty_replenish
            dyn_matrix[si, di, 1] = row.qty_debt
            dyn_matrix[si, di, 2] = row.qty_shipped
            dyn_matrix[si, di, 3] = row.qty_inbound

    # ----------------------------------------
    # 4. 前向滚动计算所有 7 大特征
    # ----------------------------------------
    def rolling_log1p(arr, window, n):
        cum = np.cumsum(np.concatenate([[0.0], arr]))
        res = np.zeros(n, dtype=np.float32)
        for j in range(n):
            i_end = j + 1
            i_sta = max(0, i_end - window)
            res[j] = cum[i_end] - cum[i_sta]
        return np.log1p(np.maximum(res, 0))

    ts_feat = np.zeros((len(sku_list), n_days, DYN_FEAT_DIM), dtype=np.float32)
    val_keys = []  # 记录验证集的 (sku_id, date) 用于评估对齐
    
    print(f"[{time.strftime('%H:%M:%S')}] ⚙️  生成滑动特征池...")
    for si in range(len(sku_list)):
        r = dyn_matrix[si, :, 0]
        d = dyn_matrix[si, :, 1]
        s = dyn_matrix[si, :, 2]
        n = dyn_matrix[si, :, 3]
        ts_feat[si, :, 0] = np.log1p(np.maximum(r, 0))
        ts_feat[si, :, 1] = rolling_log1p(r, 7,  n_days)
        ts_feat[si, :, 2] = rolling_log1p(r, 30, n_days)
        ts_feat[si, :, 3] = np.log1p(np.maximum(d, 0))
        ts_feat[si, :, 4] = np.log1p(np.maximum(s, 0))
        ts_feat[si, :, 5] = rolling_log1p(s, 7,  n_days)
        ts_feat[si, :, 6] = np.log1p(np.maximum(n, 0))

    # ----------------------------------------
    # 5. 暴力滑窗截取样本
    # ----------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] 🎯 暴力滑窗扫描样本...")
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
    # 只扫描一个全量 SKU 到另一个全量 SKU
    for si, sku in enumerate(tqdm(sku_list, desc="滑窗截取")):
        if str(sku) not in static_dict: continue
        
        base_sf = static_dict[str(sku)]
        sku_qty = dyn_matrix[si, :, 0]   # 每天的补货量本身
        
        # 对于此款 SKU，从第 30 天滑行到倒数第 30 天
        for i in range(LOOKBACK - 1, end_idx):
            target = float(sku_qty[i + 1: i + FORECAST + 1].sum())
            anchor_date = all_dates[i]
            is_train = anchor_date < split_date

            # 如果全是 0，触发降采样
            if target == 0:
                if is_train and (i % NEG_STEP != 0):
                    continue  # 训练集：每 10 个负样本只保留 1 个
                # 验证集 12 月只有 1 天锁点，全部保留，不降采样
            
            # => 提取样本
            window = ts_feat[si, i - LOOKBACK + 1: i + 1]  # (30, 7)
            
            sf = base_sf.copy()
            sf[-3] = float(anchor_date.month)  # V1 的月其实在倒数第3列
            # 连续数值用 log1p
            sf[-2] = np.log1p(max(0.0, sf[-2]))
            sf[-1] = np.log1p(max(0.0, sf[-1]))

            y_c = np.array([1.0 if target > 0 else 0.0], dtype=np.float32)
            y_r = np.array([np.log1p(target)], dtype=np.float32)

            if is_train:
                window.tofile(f_tr_dyn); sf.tofile(f_tr_sta)
                y_c.tofile(f_tr_cls);    y_r.tofile(f_tr_reg)
                tr_cnt += 1
                if target > 0: pos_tr += 1
            else:
                window.tofile(f_va_dyn); sf.tofile(f_va_sta)
                y_c.tofile(f_va_cls);    y_r.tofile(f_va_reg)
                va_cnt += 1
                if target > 0: pos_va += 1
                val_keys.append({'sku_id': sku_list[si], 'date': anchor_date})

    for f in [f_tr_dyn, f_tr_sta, f_tr_cls, f_tr_reg, f_va_dyn, f_va_sta, f_va_cls, f_va_reg]:
        f.close()

    # ----------------------------------------
    # 6. 归一化与收尾
    # ----------------------------------------
    if tr_cnt > 0:
        print(f"[{time.strftime('%H:%M:%S')}] 📏 极速归一化...")
        scaler = MinMaxScaler()
        X_tr = np.memmap(fn('X_train_dyn.bin'), dtype=np.float32, mode='r+', shape=(tr_cnt, LOOKBACK, DYN_FEAT_DIM))
        for i in range(0, tr_cnt, 20000):
            c = np.nan_to_num(X_tr[i:i+20000].reshape(-1, DYN_FEAT_DIM), nan=0, posinf=0, neginf=0)
            scaler.partial_fit(c)
        for i in range(0, tr_cnt, 20000):
            c = np.nan_to_num(X_tr[i:i+20000].reshape(-1, DYN_FEAT_DIM), nan=0, posinf=0, neginf=0)
            X_tr[i:i+20000] = scaler.transform(c).reshape(X_tr[i:i+20000].shape)
            X_tr.flush()
            
        if va_cnt > 0:
            X_va = np.memmap(fn('X_val_dyn.bin'), dtype=np.float32, mode='r+', shape=(va_cnt, LOOKBACK, DYN_FEAT_DIM))
            for i in range(0, va_cnt, 20000):
                c = np.nan_to_num(X_va[i:i+20000].reshape(-1, DYN_FEAT_DIM), nan=0, posinf=0, neginf=0)
                X_va[i:i+20000] = scaler.transform(c).reshape(X_va[i:i+20000].shape)
                X_va.flush()

        with open(os.path.join(ARTIFACTS_DIR, 'feature_scaler_v2.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    meta = {
        'static_dim': STATIC_DIM,
        'static_cat_cols': STATIC_CAT_COLS,
        'static_num_cols': STATIC_NUM_COLS,
        'lookback': LOOKBACK,
        'forecast': FORECAST,
        'train_cnt': tr_cnt, 'val_cnt': va_cnt,
        'pos_train': pos_tr, 'pos_val': pos_va,
        'split_date': SPLIT_DATE,
    }
    with open(os.path.join(ARTIFACTS_DIR, 'meta_v2.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if va_cnt > 0:
        pd.DataFrame(val_keys).to_csv(os.path.join(ARTIFACTS_DIR, 'val_keys.csv'), index=False)
        print(f"  ➤ [Keys]  val_keys.csv 已保存 ({len(val_keys)} 行)")

    print("\n" + "=" * 60)
    print("🎉 V2.2 张量暴力生成完毕！")
    print(f"  ➤ [Train] 样本数: {tr_cnt:,} | 正样本率: {pos_tr/max(tr_cnt,1):.2%}")
    print(f"  ➤ [Val]   样本数: {va_cnt:,} | 正样本率: {pos_va/max(va_cnt,1):.2%}")
    print("=" * 60)

if __name__ == "__main__":
    build_tensors_v2()
