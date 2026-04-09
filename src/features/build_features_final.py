"""
工业级 B2B 时序张量构建脚本 (Event-Driven Sampling)
- 修正了 Off-by-one 数据穿越/缺失 Bug
- 引入 np.cumsum 性能向量化优化
"""
import pandas as pd
import numpy as np
import os
import pickle
import warnings
import time
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import yaml
from datetime import timedelta

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DummyLE:
    classes_ = np.arange(13)

def load_yaml(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_tensors():
    print("=" * 60)
    print("🚀 [Feature Engineering] 开始构建时序张量 (纯订单驱动版)...")
    print("=" * 60)
    
    config_path = os.path.join(PROJECT_ROOT, 'config', 'model_config.yaml')
    try:
        model_config = load_yaml(config_path)
    except FileNotFoundError:
        print("⚠️ 未找到配置文件，采用默认核心配置。")
        model_config = {
            'paths': {'gold_dir': 'data/gold', 'dataset_dir': 'data/processed'},
            'data': {
                'lookback_days': 90, 
                'forecast_days': 30,
                'static_cols': ['category', 'season', 'price_tag']  # 移除原本由外部传入的控制，防止混淆
            }
        }
        
    gold_dir = os.path.join(PROJECT_ROOT, model_config['paths']['gold_dir'])
    processed_dir = os.path.join(PROJECT_ROOT, model_config['paths']['dataset_dir'])
    artifacts_dir = os.path.join(PROJECT_ROOT, 'data', 'artifacts')
    
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    lookback = model_config['data']['lookback_days']
    forecast = model_config['data']['forecast_days']
    
    # --- 静态特征维度定义 ---
    # 彻底补全所有商品画像，并将 price_tag 作为连续数值移至右侧！
    static_cat_cols = ['buyer_id', 'sku_id', 'style_id', 'product_name', 'category', 'sub_category', 'season', 'series', 'band', 'size_id', 'color_id']
    
    # ★ V1.8 战役三核心：全面整合买手空间与商业画像标签
    static_num_cols = [
        'qty_first_order', 
        'price_tag',
        'cooperation_years',
        'monthly_average_replenishment',
        'avg_discount_rate',
        'replenishment_frequency',
        'item_coverage_rate'
    ]
    static_cols_to_extract = static_cat_cols + static_num_cols 
    # 'month' 作为一个额外的动态生成的分类特征处理
    
    # --- 动态特征维度定义 (共7维) ---
    dyn_cols = ['qty_replenish', 'roll_repl_7', 'roll_repl_30', 'qty_debt', 'qty_shipped', 'roll_ship_7', 'qty_inbound']
    
    df_path = os.path.join(gold_dir, "wide_table_sku.csv")
    if not os.path.exists(df_path): return
        
    print(f"[{time.strftime('%H:%M:%S')}] 📂 读取原始稀疏数据...")
    df = pd.read_csv(df_path)
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    if 'qty_replenish' not in df.columns: return
    
    # 要求的特征列
    req_cols = [
        'date', 'buyer_id', 'sku_id', 
        'qty_replenish', 'qty_debt', 'qty_shipped', 'qty_inbound', 
        'style_id', 'product_name', 'category', 'sub_category', 'season', 'series', 'band', 'size_id', 'color_id', 
        'qty_first_order', 'price_tag'
    ]
    
    # 填充 Unknown 预防空值，补充缺失列
    for c in static_cat_cols:
        if c not in df.columns:
            df[c] = 'Unknown'

    print(f"[{time.strftime('%H:%M:%S')}] 🔄 构建惰性查询字典...")
    # 静态特征取唯一，依然以 buyer_id, sku_id 为主键
    static_df = df[static_cols_to_extract].drop_duplicates(subset=['buyer_id', 'sku_id'])
    
    # 🌟 修复 Bug 核心：保存原始的 buyer_id 和 sku_id
    original_keys = list(zip(static_df['buyer_id'], static_df['sku_id']))
    
    # Label Encoder
    encoders = {}
    for col in static_cat_cols + ['month']:  # 给 month 也单独造 encoder
        if col == 'month':
            encoders[col] = DummyLE()
            continue

        le = LabelEncoder()
        static_df[col] = le.fit_transform(static_df[col].astype(str))
        encoders[col] = le
        
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 转换静态特征架构 (Dictionary Cache)...")
    static_dict = {}
    for i, row in enumerate(static_df.itertuples(index=False)):
        orig_buyer, orig_sku = original_keys[i]
        # Array to store static features:
        # [buyer_idx, sku_idx, category_idx, season_idx, month_idx(placeholder), price_tag_idx, qty_first_order_val]
        arr = []
        for col in static_cat_cols:
            arr.append(getattr(row, col))
            
        arr.append(0) # 'month' placeholder
        
        for num_col in static_num_cols:
            if num_col in static_df.columns:
                val = getattr(row, num_col)
                arr.append(val if pd.notnull(val) else 0.0)
            else:
                arr.append(0.0)
                
        # 因为混合了 int 和 float (qty_first_order)，用 float32 存
        static_dict[(orig_buyer, orig_sku)] = np.array(arr, dtype=np.float32)

        
    records = {}
    for row in df.itertuples(index=False):
        # 预抓取四项动态数据组进行缓存: [replenish, debt, shipped, inbound]
        key = (row.buyer_id, row.sku_id)
        if key not in records: records[key] = {}
        
        qty_repl = getattr(row, 'qty_replenish', 0)
        qty_debt = getattr(row, 'qty_debt', 0)
        qty_shi = getattr(row, 'qty_shipped', 0)
        qty_inb = getattr(row, 'qty_inbound', 0)
        
        records[key][row.date] = np.array([float(qty_repl), float(qty_debt), float(qty_shi), float(qty_inb)], dtype=np.float32)

    min_date = pd.to_datetime('2025-01-01').date()
    max_date = pd.to_datetime('2025-12-31').date()  # 数据完整覆盖全年
    full_dates = [min_date + timedelta(days=i) for i in range((max_date - min_date).days + 1)]
    
    # [v1.9.2] 固定切点为 2025-09-01（覆盖 Sep:16K + Oct:8K + Nov:6K + Dec:11K 旺淡混合）
    # 原来 80% ≈ 10月19日，验证集偏淡季；现在覆盖更完整的季节分布
    split_date = pd.to_datetime('2025-09-01').date()
    print(f"[{time.strftime('%H:%M:%S')}] 📅 日期范围: {min_date} → {max_date}，Train/Val 切点: {split_date}")

    print(f"[{time.strftime('%H:%M:%S')}] 🎯 执行事件驱动采样与张量硬盘直写流...")
    
    # 彻底使用内存映射直写硬盘以防止超 18GB Numpy OOM
    fn_tr_dyn = os.path.normpath(os.path.join(processed_dir, 'X_train_dyn.bin'))
    fn_tr_sta = os.path.normpath(os.path.join(processed_dir, 'X_train_static.bin'))
    fn_tr_cls = os.path.normpath(os.path.join(processed_dir, 'y_train_cls.bin'))
    fn_tr_reg = os.path.normpath(os.path.join(processed_dir, 'y_train_reg.bin'))
    
    fn_va_dyn = os.path.normpath(os.path.join(processed_dir, 'X_val_dyn.bin'))
    fn_va_sta = os.path.normpath(os.path.join(processed_dir, 'X_val_static.bin'))
    fn_va_cls = os.path.normpath(os.path.join(processed_dir, 'y_val_cls.bin'))
    fn_va_reg = os.path.normpath(os.path.join(processed_dir, 'y_val_reg.bin'))
    
    f_tr_dyn = open(fn_tr_dyn, 'wb')
    f_tr_sta = open(fn_tr_sta, 'wb')
    f_tr_cls = open(fn_tr_cls, 'wb')
    f_tr_reg = open(fn_tr_reg, 'wb')
    
    f_va_dyn = open(fn_va_dyn, 'wb')
    f_va_sta = open(fn_va_sta, 'wb')
    f_va_cls = open(fn_va_cls, 'wb')
    f_va_reg = open(fn_va_reg, 'wb')

    train_cnt, val_cnt, pos_train, pos_val = 0, 0, 0, 0
    # 黄金折中快攻策略 (大幅剔除训练集死水, 100% 严防验证集保留商业原貌)
    # [v1.8] 因为数据源切换至 V_IRS_ORDER 且无笛卡尔膨胀，绝对稀疏度从 99.6% 降至 70.8%
    # 将负采样从 40 调低为 10，防范正样本失衡导致模型产生大盘暴涨幻觉
    neg_step_train = 20  # [v1.9.2] neg_step=20 → 训练集正样本率约 33%，已足够
    neg_step_val = 1     # 验证集绝不降采样，全量模拟真实线上环境
    
    # 预留 Buffer 计算 Lookback 中的 Lag 特征
    buffer_days = lookback + 30 
    
    for (buyer, sku), daily_sales in tqdm(records.items(), desc="扫描组合", unit="组"):
        if (buyer, sku) not in static_dict:
            continue
        base_static_feat = static_dict[(buyer, sku)]
        
        # [v1.9.1 二级向量化] 预构建 SKU 全量日期矩阵
        sku_full_arr = np.zeros((len(full_dates), 4), dtype=np.float32)
        for d, vals in daily_sales.items():
            d_idx = (d - min_date).days
            if 0 <= d_idx < len(full_dates):
                sku_full_arr[d_idx] = vals
        
        # --- 性能核心：在 SKU 顶层执行全量预计算，锚点循环内仅做切片 ---
        # 基础列
        ts_repl = np.log1p(np.maximum(sku_full_arr[:, 0], 0))
        ts_debt = np.log1p(np.maximum(sku_full_arr[:, 1], 0))
        ts_ship = np.log1p(np.maximum(sku_full_arr[:, 2], 0))
        ts_inbn = np.log1p(np.maximum(sku_full_arr[:, 3], 0))
        
        # 滚动列 (利用全局 Cumsum 快速生成全量序列)
        sku_cum_repl = np.cumsum(np.insert(sku_full_arr[:, 0], 0, 0.0))
        sku_cum_ship = np.cumsum(np.insert(sku_full_arr[:, 2], 0, 0.0))
        
        # 全量生成对应的 365 天滚动 Sum 序列 (长度与 full_dates 一致)
        # 用补齐 0 的方式对齐索引
        def get_rolling_ts(cum_arr, window):
            # j 时刻的回看 window 天，索引在 cum_arr 中是 [j+1-window, j+1]
            res = np.zeros(len(full_dates), dtype=np.float32)
            for j in range(len(full_dates)):
                idx_end = j + 1
                idx_sta = max(0, idx_end - window)
                res[j] = cum_arr[idx_end] - cum_arr[idx_sta]
            return np.log1p(np.maximum(res, 0))

        # 注意：这里内部其实还有一个 loop，但如果是 365 次则无所谓
        # 甚至可以继续向量化：
        def get_rolling_ts_vec(cum_arr, window):
            # 取 cum_arr[window:] - cum_arr[:-window]
            res = np.zeros(len(full_dates), dtype=np.float32)
            # 有效窗口从 index=window-1 开始
            val = cum_arr[window:] - cum_arr[:-window]
            res[window-1:] = val
            # 前 window-1 天需要特殊处理 (只有部分窗口)
            for j in range(window - 1):
                res[j] = cum_arr[j+1]
            return np.log1p(np.maximum(res, 0))

        ts_roll_repl_7  = get_rolling_ts_vec(sku_cum_repl, 7)
        ts_roll_repl_30 = get_rolling_ts_vec(sku_cum_repl, 30)
        ts_roll_ship_7  = get_rolling_ts_vec(sku_cum_ship, 7)

        # 把 7 大维度合并为 (365, 7) 的主特征矩阵
        sku_ts_feat = np.stack([
            ts_repl, ts_roll_repl_7, ts_roll_repl_30,
            ts_debt, ts_ship, ts_roll_ship_7, ts_inbn
        ], axis=1)

        start_idx = buffer_days
        end_idx = len(full_dates) - forecast
        
        for i in range(start_idx, end_idx):
            anchor_date = full_dates[i]
            is_train = anchor_date < split_date
            
            # 1. 计算 Target (未来 30 天切片并求和)
            target_sum = np.sum(sku_full_arr[i+1 : i+forecast+1, 0])
            
            # 2. 负样本降采样
            neg_step = neg_step_train if is_train else neg_step_val
            if target_sum == 0 and i % neg_step != 0: continue 
                
            # 3. 极速切片提取 X (90 天历史直接从预计算矩阵中拉取)
            # 索引 i 对应的 window 结尾是 i，开头是 i-lookback+1
            window_dyn = sku_ts_feat[i - lookback + 1 : i + 1]
            
            # 4. 组装静态 & 写入
            static_feat = base_static_feat.copy()
            static_feat[11] = anchor_date.month
            
            if is_train:
                window_dyn.tofile(f_tr_dyn)
                static_feat.astype(np.float32).tofile(f_tr_sta)
                np.array([1.0 if target_sum > 0 else 0.0], dtype=np.float32).tofile(f_tr_cls)
                np.array([float(np.log1p(target_sum))], dtype=np.float32).tofile(f_tr_reg)
                train_cnt += 1
                if target_sum > 0: pos_train += 1
            else:
                window_dyn.tofile(f_va_dyn)
                static_feat.astype(np.float32).tofile(f_va_sta)
                np.array([1.0 if target_sum > 0 else 0.0], dtype=np.float32).tofile(f_va_cls)
                np.array([float(np.log1p(target_sum))], dtype=np.float32).tofile(f_va_reg)
                val_cnt += 1
                if target_sum > 0: pos_val += 1

    # 关闭所有流直写引擎
    f_tr_dyn.close()
    f_tr_sta.close()
    f_tr_cls.close()
    f_tr_reg.close()
    f_va_dyn.close()
    f_va_sta.close()
    f_va_cls.close()
    f_va_reg.close()

    print(f"[{time.strftime('%H:%M:%S')}] 📏 极速启动 Mmap 分块标度放缩与打包...")
    scaler = MinMaxScaler()
    static_dim = len(static_cols_to_extract) + 1  # 包含 month 占位

    # -- Train 端放缩 --
    if train_cnt > 0:
        X_tr_dyn_mmap = np.memmap(fn_tr_dyn, dtype=np.float32, mode='r+', shape=(train_cnt, lookback, 7))
        X_tr_sta_mmap = np.memmap(fn_tr_sta, dtype=np.float32, mode='r+', shape=(train_cnt, static_dim))
        
        # 数值静态连续列安全 log1p
        num_numeric = len(static_num_cols)
        for i in range(1, num_numeric + 1):
            X_tr_sta_mmap[:, -i] = np.log1p(np.maximum(X_tr_sta_mmap[:, -i], 0))
        X_tr_sta_mmap.flush()

        # 分块 Fit 以处理近亿矩阵，[改小至20000防止400MB以上连续内存分片OOM]
        batch_sz = 20000
        for i in range(0, train_cnt, batch_sz):
            chunk = X_tr_dyn_mmap[i:i+batch_sz].reshape(-1, 7)
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            scaler.partial_fit(chunk)
            
        # 分块 Transform
        for i in range(0, train_cnt, batch_sz):
            c_shape = X_tr_dyn_mmap[i:i+batch_sz].shape
            c_flat = X_tr_dyn_mmap[i:i+batch_sz].reshape(-1, 7)
            c_flat = np.nan_to_num(c_flat, nan=0.0, posinf=0.0, neginf=0.0)
            X_tr_dyn_mmap[i:i+batch_sz] = scaler.transform(c_flat).reshape(c_shape)
            X_tr_dyn_mmap.flush()

    # -- Val 端放缩 --
    if val_cnt > 0:
        X_va_dyn_mmap = np.memmap(fn_va_dyn, dtype=np.float32, mode='r+', shape=(val_cnt, lookback, 7))
        X_va_sta_mmap = np.memmap(fn_va_sta, dtype=np.float32, mode='r+', shape=(val_cnt, static_dim))
        
        for i in range(1, num_numeric + 1):
            X_va_sta_mmap[:, -i] = np.log1p(np.maximum(X_va_sta_mmap[:, -i], 0))
        X_va_sta_mmap.flush()

        # 分块使用统一 scaler 对齐 Valid
        batch_sz = 20000
        for i in range(0, val_cnt, batch_sz):
            c_shape = X_va_dyn_mmap[i:i+batch_sz].shape
            c_flat = X_va_dyn_mmap[i:i+batch_sz].reshape(-1, 7)
            c_flat = np.nan_to_num(c_flat, nan=0.0, posinf=0.0, neginf=0.0)
            X_va_dyn_mmap[i:i+batch_sz] = scaler.transform(c_flat).reshape(c_shape)
            X_va_dyn_mmap.flush()

    print(f"[{time.strftime('%H:%M:%S')}]  [特征生成分析与庞大数据规模报告]")
    neg_train = train_cnt - pos_train
    neg_val = val_cnt - pos_val
    
    if train_cnt > 0:
        print(f"   ➤ [Train / 历史集] 样本总数: {train_cnt}")
        print(f"       - 动态张量 形状 (X_dyn):    {X_tr_dyn_mmap.shape}  -> (Samples, 90天, 7大维度)")
        print(f"       - 正样本(未来30天有补货): {int(pos_train)} 笔 ({pos_train/max(train_cnt,1)*100:.2f}%)")
        print(f"       - 负样本(未来30天无补货): {int(neg_train)} 笔 ({neg_train/max(train_cnt,1)*100:.2f}%)")

    if val_cnt > 0:
        print(f"   ➤ [Val / 测试集] 样本总数: {val_cnt}")
        print(f"       - 动态张量 形状 (X_dyn):    {X_va_dyn_mmap.shape}  -> (Samples, 90天, 7大维度)")
        print(f"       - 正样本(未来30天有补货): {int(pos_val)} 笔 ({pos_val/max(val_cnt,1)*100:.2f}%)")
        print(f"       - 负样本(未来30天无补货): {int(neg_val)} 笔 ({neg_val/max(val_cnt,1)*100:.2f}%)")

    with open(os.path.join(artifacts_dir, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    with open(os.path.join(artifacts_dir, 'feature_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    print("🎉 张量构建大功告成!")

if __name__ == "__main__":
    build_tensors()
