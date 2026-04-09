"""
B2B 补货系统 Phase 4A — 周粒度时间聚合特征工程 (Weekly Aggregation)
=====================================================================
核心改造思路：
  1. 将 (sku_id, date) 的日频数据聚合为 (sku_id, week_key) 的周频数据
  2. 序列长度从 90 天→ 12 周 (约 84 天)，LSTM 步数从 90 压缩到 12
  3. 零值比例从 78% 预期降至 30%~40%（周内聚合后大量 0+0+X→X 变连续）
  4. 显存收益：Batch Size 可从 1536 提升至 8192+，单轮训练预计缩短 70%

隔离原则（铁律）：
  - 读取同一张源表：data/gold/wide_table_sku.csv
  - 输出到独立目录：data/processed_weekly/
  - artifacts 存入：data/artifacts_weekly/
  - 原有 processed_v3/ 和 artifacts_v3/ 完全不碰

控制变量说明 (Phase 4A)：
  - 动态特征维度保持 7 维（对齐现有模型 DYN_FEAT_DIM=7），但改为周频信号
  - 静态特征维度保持不变（复用现有 LabelEncoder）
  - 不加入新特征（std_repl 等留给 Phase 4B 对比）
"""

import argparse
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import json
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ======================================================
# 常量定义 —— 周频版本（部分可通过命令行覆盖）
# ======================================================
FORECAST_DAYS   = 30     # 预测目标：未来 30 天的补货总量（业务定义不变）
SPLIT_DATE      = '2025-12-01'  # 训练/验证切分日期（与日频版本一致）
NEG_STEP        = 1      # 验证集全量保留；训练集负样本全保留（与 V2.2 对齐）
DYN_FEAT_DIM    = 7

STATIC_CAT_COLS = [
    'sku_id', 'style_id', 'product_name', 'category',
    'sub_category', 'season', 'series', 'band', 'size_id', 'color_id'
]
STATIC_NUM_COLS = ['qty_first_order', 'price_tag']
GOLD_DIR        = os.path.join(PROJECT_ROOT, 'data', 'gold')


class DummyLE:
    """月份的哑编码器，保持与日频版本一致。"""
    classes_ = np.arange(13)


def parse_args():
    parser = argparse.ArgumentParser(description='B2B Phase 4 周频特征工程')
    parser.add_argument('--lookback', type=int, default=12,
                        help='回看窗口周数，默认 12 周')
    return parser.parse_args()


def build_week_key(date_series: pd.Series) -> pd.Series:
    """
    将日期转换为滚动周键（整数），锚点为数据集第一天。
    使用滚动窗口而非自然周，避免月末/季末边界效应。
    week_key = 0 表示第 1~7 天，week_key = 1 表示第 8~14 天，依此类推。
    """
    base_date = date_series.min()
    delta_days = (date_series - base_date).dt.days
    return delta_days // 7


def build_tensors_weekly(lookback_weeks: int, processed_dir: str, artifacts_dir: str):
    """主入口：生成周粒度训练/验证张量。"""
    import sys
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

    # 接收参数，覆盖模块级常量
    LOOKBACK_WEEKS = lookback_weeks
    PROCESSED_DIR  = processed_dir
    ARTIFACTS_DIR  = artifacts_dir

    print("=" * 60)
    print("🗓️  [Phase 4A] 周粒度特征工程 (Weekly Aggregation)")
    print(f"    回看窗口: {LOOKBACK_WEEKS} 周 | 预测目标: 未来 {FORECAST_DAYS} 天总量")
    print("=" * 60)

    # --------------------------------------------------
    # 1. 读取日频源数据（复用同一张 wide_table）
    # --------------------------------------------------
    df_path = os.path.join(GOLD_DIR, 'wide_table_sku.csv')
    print(f"[{time.strftime('%H:%M:%S')}] 📂 读取源数据: {df_path}")
    df = pd.read_csv(df_path)
    df['date'] = pd.to_datetime(df['date'])

    # --------------------------------------------------
    # 2. 先做日频 (sku_id, date) 聚合（和日频版本一致，保证数据干净）
    # --------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] 🔄 (sku_id, date) 日频基础聚合...")
    dyn_daily = df.groupby(['sku_id', 'date']).agg(
        qty_replenish=('qty_replenish', 'sum'),
        qty_debt     =('qty_debt',      'sum'),
        qty_shipped  =('qty_shipped',   'sum'),
        qty_inbound  =('qty_inbound',   'sum'),
    ).reset_index()

    # --------------------------------------------------
    # 3. 为每个日期打上"滚动周键"
    # --------------------------------------------------
    dyn_daily['week_key'] = build_week_key(dyn_daily['date'])
    # 每周的锚定日期：取该周内最后一天（用于后续时序切分判断）
    week_anchor = dyn_daily.groupby(['sku_id', 'week_key'])['date'].max().reset_index()
    week_anchor.rename(columns={'date': 'week_end_date'}, inplace=True)

    # --------------------------------------------------
    # 4. (sku_id, week_key) 周频聚合
    # --------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] 📅 (sku_id, week_key) 周频聚合...")
    dyn_weekly = dyn_daily.groupby(['sku_id', 'week_key']).agg(
        qty_replenish_week=('qty_replenish', 'sum'),
        qty_debt_week     =('qty_debt',      'sum'),
        qty_shipped_week  =('qty_shipped',   'sum'),
        qty_inbound_week  =('qty_inbound',   'sum'),
    ).reset_index()
    dyn_weekly = dyn_weekly.merge(week_anchor, on=['sku_id', 'week_key'], how='left')

    unique_skus = dyn_weekly['sku_id'].nunique()
    total_weeks = dyn_weekly['week_key'].nunique()
    print(f"    唯一 SKU: {unique_skus:,} | 总周数跨度: {total_weeks} 周")

    # --------------------------------------------------
    # 5. 编码静态特征（复用日频风格，保存到新 artifacts 目录）
    # --------------------------------------------------
    static_agg = df[STATIC_CAT_COLS + STATIC_NUM_COLS].drop_duplicates('sku_id').copy()
    static_agg['orig_sku'] = static_agg['sku_id'].astype(str)

    encoders = {}
    for col in STATIC_CAT_COLS:
        if col not in static_agg.columns:
            static_agg[col] = 'Unknown'
        le = LabelEncoder()
        static_agg[col] = le.fit_transform(static_agg[col].astype(str))
        encoders[col] = le
    encoders['month'] = DummyLE()

    with open(os.path.join(ARTIFACTS_DIR, 'label_encoders_weekly.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    print(f"[{time.strftime('%H:%M:%S')}] ✅ Label Encoders 已保存至 artifacts_weekly/")

    STATIC_DIM = len(STATIC_CAT_COLS) + 1 + len(STATIC_NUM_COLS)  # +1 = month
    static_dict = {}
    for _, row in static_agg.iterrows():
        arr = [float(row[c]) for c in STATIC_CAT_COLS]
        arr.append(0.0)  # month 占位符，后续在滑窗时按锚点月份填入
        for c in STATIC_NUM_COLS:
            arr.append(np.log1p(max(0.0, float(row[c]))) if pd.notnull(row[c]) else 0.0)
        static_dict[str(row['orig_sku'])] = np.array(arr, dtype=np.float32)

    # --------------------------------------------------
    # 6. 构造周频矩阵 [n_sku, n_weeks, 4]
    # --------------------------------------------------
    sku_list  = sorted(dyn_weekly['sku_id'].unique())
    sku_to_i  = {s: i for i, s in enumerate(sku_list)}
    week_list = sorted(dyn_weekly['week_key'].unique())
    week_to_i = {w: i for i, w in enumerate(week_list)}
    n_weeks   = len(week_list)

    # 记录全局基准日期和最后一天（用于确定锚点边界）
    base_date = dyn_daily['date'].min()   # 数据集第一天
    last_date = dyn_daily['date'].max()   # 数据集最后一天
    print(f"    数据范围: {base_date.date()} → {last_date.date()}")

    print(f"[{time.strftime('%H:%M:%S')}] 🔲 构建周频矩阵 ({len(sku_list):,} SKU × {n_weeks} 周 × 4维)...")
    # dyn_matrix_w: (n_sku, n_weeks, 4) —— raw weekly amounts before log1p
    dyn_matrix_w = np.zeros((len(sku_list), n_weeks, 4), dtype=np.float32)
    # week_anchor_dates: (n_sku, n_weeks) —— week_end_date for each cell
    week_anchor_mat = np.empty((len(sku_list), n_weeks), dtype=object)

    for row in tqdm(dyn_weekly.itertuples(index=False), total=len(dyn_weekly), desc="填入周矩阵"):
        si = sku_to_i.get(row.sku_id, -1)
        wi = week_to_i.get(row.week_key, -1)
        if si >= 0 and wi >= 0:
            dyn_matrix_w[si, wi, 0] = row.qty_replenish_week
            dyn_matrix_w[si, wi, 1] = row.qty_debt_week
            dyn_matrix_w[si, wi, 2] = row.qty_shipped_week
            dyn_matrix_w[si, wi, 3] = row.qty_inbound_week
            week_anchor_mat[si, wi] = row.week_end_date

    # --------------------------------------------------
    # 7. 计算周频的 7 维特征（log1p 变换 + 滚动求和）
    # --------------------------------------------------
    def week_rolling_log1p(arr_1d, window):
        """对一个 1D 周频数组做滑窗求和再 log1p。"""
        n = len(arr_1d)
        cum = np.cumsum(np.concatenate([[0.0], arr_1d]))
        res = np.zeros(n, dtype=np.float32)
        for j in range(n):
            i_end = j + 1
            i_sta = max(0, i_end - window)
            res[j] = cum[i_end] - cum[i_sta]
        return np.log1p(np.maximum(res, 0))

    print(f"[{time.strftime('%H:%M:%S')}] ⚙️  生成周频 7 维特征池 (log1p)...")
    ts_feat_w = np.zeros((len(sku_list), n_weeks, DYN_FEAT_DIM), dtype=np.float32)
    for si in tqdm(range(len(sku_list)), desc="特征池"):
        r       = dyn_matrix_w[si, :, 0]   # qty_replenish_week
        d       = dyn_matrix_w[si, :, 1]   # qty_debt_week
        s       = dyn_matrix_w[si, :, 2]   # qty_shipped_week
        inbound = dyn_matrix_w[si, :, 3]   # qty_inbound_week（避免与 week_rolling_log1p 内的 n 同名）

        ts_feat_w[si, :, 0] = np.log1p(np.maximum(r, 0))               # 本周补货
        ts_feat_w[si, :, 1] = week_rolling_log1p(r, 2)                 # 过去 2 周补货
        ts_feat_w[si, :, 2] = week_rolling_log1p(r, 4)                 # 过去 4 周补货
        ts_feat_w[si, :, 3] = np.log1p(np.maximum(d, 0))               # 本周欠货
        ts_feat_w[si, :, 4] = np.log1p(np.maximum(s, 0))               # 本周发货
        ts_feat_w[si, :, 5] = week_rolling_log1p(s, 2)                 # 过去 2 周发货
        ts_feat_w[si, :, 6] = np.log1p(np.maximum(inbound, 0))         # 本周入库

    # --------------------------------------------------
    # 8. 暴力滑窗截取样本（周版本）
    # --------------------------------------------------
    # 目标: 以锚点周 i 为基础，预测未来 30 天补货总量
    # - 动态特征: ts_feat_w[si, i-LOOKBACK_WEEKS+1 : i+1]  shape=(12, 7)
    # - 目标: sum(qty_replenish[day in week_end+1 .. week_end+30])
    #         使用原始日频 dyn_daily 来精确计算未来 30 天的真实补货量
    print(f"[{time.strftime('%H:%M:%S')}] 🎯 暴力滑窗截取样本（周频, lookback={LOOKBACK_WEEKS}周）...")
    split_date_ts = pd.Timestamp(SPLIT_DATE)
    last_date_ts  = pd.Timestamp(last_date)

    # 预先建立 per-SKU 的日频补货字典，用于精确计算目标变量
    print(f"[{time.strftime('%H:%M:%S')}] 📊 建立 SKU→日频补货字典 (用于目标变量计算)...")
    sku_daily_repl = {}
    for sku_id, grp in tqdm(dyn_daily.groupby('sku_id'), desc="建立日频字典"):
        date_repl = dict(zip(grp['date'], grp['qty_replenish']))
        sku_daily_repl[str(sku_id)] = date_repl

    # ★ 修复核心：提前算出每个 week_key 对应的精确锚定日期（周末）
    # 不依赖 week_anchor_mat，保证即使 SKU 当周无数据也能生成锚点
    # week_key=k 对应从 base_date 起第 7k~7k+6 天，锚定日为最后一天(7k+6)
    week_end_dates = {
        wk: base_date + pd.Timedelta(days=int(wk) * 7 + 6)
        for wk in week_list
    }

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
    val_keys = []

    for si, sku in enumerate(tqdm(sku_list, desc="滑窗截取")):
        sku_str = str(sku)
        if sku_str not in static_dict:
            continue

        base_sf   = static_dict[sku_str]
        # 无数据的 SKU 日频字典为空，目标变量全为 0（负样本），也应生成并保留
        daily_repl = sku_daily_repl.get(sku_str, {})

        # ★ 修复：对每一个周索引都生成锚点，不跳过无数据的周
        for wi in range(LOOKBACK_WEEKS - 1, n_weeks):
            # 直接从 week_key 计算锚定日期，无需依赖 week_anchor_mat
            anchor_date = week_end_dates[week_list[wi]]

            # 边界检查：锚点 + 30 天必须在数据集范围内
            if anchor_date + pd.Timedelta(days=FORECAST_DAYS) > last_date_ts:
                continue

            # 精确计算未来 30 天的真实日频补货总量（目标变量）
            future_total = 0.0
            for delta in range(1, FORECAST_DAYS + 1):
                future_date = anchor_date + pd.Timedelta(days=delta)
                future_total += daily_repl.get(future_date, 0.0)

            is_train = anchor_date < split_date_ts

            # 负样本降采样（仅训练集）
            if future_total == 0.0 and is_train and (wi % NEG_STEP != 0):
                continue

            # 提取周序列窗口: shape = (LOOKBACK_WEEKS, 7)
            window = ts_feat_w[si, wi - LOOKBACK_WEEKS + 1: wi + 1].copy()

            # 填入静态特征（月份用锚点周的结束月份）
            sf = base_sf.copy()
            sf[len(STATIC_CAT_COLS)] = float(anchor_date.month)

            y_c = np.array([1.0 if future_total > 0 else 0.0], dtype=np.float32)
            y_r = np.array([np.log1p(future_total)], dtype=np.float32)

            if is_train:
                window.tofile(f_tr_dyn); sf.tofile(f_tr_sta)
                y_c.tofile(f_tr_cls);    y_r.tofile(f_tr_reg)
                tr_cnt += 1
                if future_total > 0: pos_tr += 1
            else:
                window.tofile(f_va_dyn); sf.tofile(f_va_sta)
                y_c.tofile(f_va_cls);    y_r.tofile(f_va_reg)
                va_cnt += 1
                if future_total > 0: pos_va += 1
                val_keys.append({'sku_id': sku, 'week_end_date': anchor_date.date()})

    for fh in [f_tr_dyn, f_tr_sta, f_tr_cls, f_tr_reg, f_va_dyn, f_va_sta, f_va_cls, f_va_reg]:
        fh.close()

    # --------------------------------------------------
    # 9. 归一化（对齐日频版本的 MinMaxScaler 流程）
    # --------------------------------------------------
    if tr_cnt > 0:
        print(f"[{time.strftime('%H:%M:%S')}] 📏 MinMaxScaler 归一化 (训练集→验证集)...")
        scaler = MinMaxScaler()
        X_tr = np.memmap(fn('X_train_dyn.bin'), dtype=np.float32, mode='r+',
                         shape=(tr_cnt, LOOKBACK_WEEKS, DYN_FEAT_DIM))
        for i in range(0, tr_cnt, 20000):
            c = np.nan_to_num(X_tr[i:i+20000].reshape(-1, DYN_FEAT_DIM), nan=0, posinf=0, neginf=0)
            scaler.partial_fit(c)
        for i in range(0, tr_cnt, 20000):
            c = np.nan_to_num(X_tr[i:i+20000].reshape(-1, DYN_FEAT_DIM), nan=0, posinf=0, neginf=0)
            X_tr[i:i+20000] = scaler.transform(c).reshape(X_tr[i:i+20000].shape)
            X_tr.flush()

        if va_cnt > 0:
            X_va = np.memmap(fn('X_val_dyn.bin'), dtype=np.float32, mode='r+',
                             shape=(va_cnt, LOOKBACK_WEEKS, DYN_FEAT_DIM))
            for i in range(0, va_cnt, 20000):
                c = np.nan_to_num(X_va[i:i+20000].reshape(-1, DYN_FEAT_DIM), nan=0, posinf=0, neginf=0)
                X_va[i:i+20000] = scaler.transform(c).reshape(X_va[i:i+20000].shape)
                X_va.flush()

        with open(os.path.join(ARTIFACTS_DIR, 'feature_scaler_weekly.pkl'), 'wb') as f:
            pickle.dump(scaler, f)

    # --------------------------------------------------
    # 10. 保存元信息（供 run_training_weekly.py 自动读取）
    # --------------------------------------------------
    meta = {
        'static_dim':      STATIC_DIM,
        'static_cat_cols': STATIC_CAT_COLS,
        'static_num_cols': STATIC_NUM_COLS,
        'lookback_weeks':  LOOKBACK_WEEKS,
        'dyn_feat_dim':    DYN_FEAT_DIM,
        'forecast_days':   FORECAST_DAYS,
        'train_cnt':       tr_cnt,
        'val_cnt':         va_cnt,
        'pos_train':       pos_tr,
        'pos_val':         pos_va,
        'split_date':      SPLIT_DATE,
        'processed_dir':   PROCESSED_DIR,
        'artifacts_dir':   ARTIFACTS_DIR,
    }
    meta_path = os.path.join(ARTIFACTS_DIR, 'meta_weekly.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if va_cnt > 0:
        pd.DataFrame(val_keys).to_csv(
            os.path.join(ARTIFACTS_DIR, 'val_keys_weekly.csv'), index=False
        )
        print(f"  ➤ [Keys] val_keys_weekly.csv 已保存 ({len(val_keys)} 行)")

    # --------------------------------------------------
    # 11. 汇报完成
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("🎉 Phase 4A 周频张量生成完毕！")
    print(f"  ➤ [Train] 样本数: {tr_cnt:,} | 正样本率: {pos_tr/max(tr_cnt,1):.2%}")
    print(f"  ➤ [Val]   样本数: {va_cnt:,} | 正样本率: {pos_va/max(va_cnt,1):.2%}")
    print(f"  ➤ [序列]  LOOKBACK={LOOKBACK_WEEKS}周 × DYN_FEAT={DYN_FEAT_DIM}维")
    print(f"  ➤ [输出]  {PROCESSED_DIR}")
    print(f"  ➤ [Meta]  {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    _lookback = args.lookback
    # 不同回看窗口输出到独立目录（12周用默认目录，其余加后缀）
    _suffix = f'_{_lookback}w' if _lookback != 12 else ''
    _proc_dir = os.path.join(PROJECT_ROOT, 'data', f'processed_weekly{_suffix}')
    _art_dir  = os.path.join(PROJECT_ROOT, 'data', f'artifacts_weekly{_suffix}')
    os.makedirs(_proc_dir, exist_ok=True)
    os.makedirs(_art_dir, exist_ok=True)
    build_tensors_weekly(
        lookback_weeks=_lookback,
        processed_dir=_proc_dir,
        artifacts_dir=_art_dir,
    )
