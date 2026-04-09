import pandas as pd
import numpy as np
import os
import glob
import warnings
import re

# 忽略警告
warnings.filterwarnings('ignore')

# ================= 1. 路径配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# 注意：保留您原有的 data_warehouse 路径
RAW_DIR = os.path.join(PROJECT_ROOT, "data_warehouse")
SILVER_DIR = os.path.join(PROJECT_ROOT, "data/silver")

os.makedirs(SILVER_DIR, exist_ok=True)

# ================= 2. 工具函数 =================

def analyze_replenishment(df, name="Orders Data"):
    """
    📊 [新增功能] 数据浓度探针
    统计有多少行是真的在补货 (qty_replenish > 0)
    """
    if 'qty_replenish' not in df.columns:
        return

    total_rows = len(df)
    # 过滤出补货量 > 0 的行
    pos_df = df[df['qty_replenish'] > 0]
    pos_rows = len(pos_df)
    
    ratio = pos_rows / total_rows if total_rows > 0 else 0
    
    print("\n" + "="*50)
    print(f"🧐 {name} 深度透视 (Deep Dive)")
    print("="*50)
    print(f"   -> 📦 总行数 (Total Rows):      {total_rows:,}")
    print(f"   -> 🔥 有补货行数 (Qty > 0):     {pos_rows:,}")
    print(f"   -> 📉 补货稀疏度 (Sparsity):    {ratio:.4%}")
    if pos_rows > 0:
        print(f"   -> 📊 补货量均值 (Mean):        {pos_df['qty_replenish'].mean():.2f}")
        print(f"   -> 🚀 单次最大补货 (Max):       {pos_df['qty_replenish'].max()}")
        # [Added] 分位数分布
        quantiles = pos_df['qty_replenish'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        print(f"   -> 📈 分位数 (P25/P50/P75):     {quantiles[0.25]:.0f} / {quantiles[0.5]:.0f} / {quantiles[0.75]:.0f}")
        print(f"   -> 🚨 长尾分布 (P90/P95/P99):   {quantiles[0.9]:.0f} / {quantiles[0.95]:.0f} / {quantiles[0.99]:.0f}")
    print("="*50 + "\n")
    
    if ratio < 0.01:
        print("⚠️ [严重警告] 正样本极度稀疏 (<1%)！后续训练必须使用动态采样(Step=1)。\n")

def check_continuity(df, date_col='order_date', name="Order Data"):
    """
    📅 [新增功能] 检查时间连续性
    """
    print(f"📅 {name} 时间连续性检查 (Continuity Check)")
    print("-" * 40)
    
    if df.empty:
        print("   ⚠️ Data is empty.")
        return

    dates = pd.to_datetime(df[date_col]).dt.date
    min_date = dates.min()
    max_date = dates.max()
    
    print(f"   -> ⏳ 时间跨度: {min_date} ~ {max_date}")
    
    full_range = pd.date_range(min_date, max_date, freq='D').date
    existing_dates = set(dates.unique())
    
    missing_dates = sorted(list(set(full_range) - existing_dates))
    
    if not missing_dates:
        print("   ✅ 数据连续，无缺失日期！")
    else:
        print(f"   ⚠️ 发现 {len(missing_dates)} 天缺失数据！")
        # 尝试合并连续的缺失日期区间
        from datetime import timedelta
        
        missing_ranges = []
        if missing_dates:
            start = missing_dates[0]
            end = missing_dates[0]
            
            for d in missing_dates[1:]:
                if d == end + timedelta(days=1):
                    end = d
                else:
                    missing_ranges.append((start, end))
                    start = d
                    end = d
            missing_ranges.append((start, end))
            
        print("   -> 🛑 缺失区间 (Top 10):")
        for i, (s, e) in enumerate(missing_ranges[:10]):
            if s == e:
                print(f"      {i+1}. {s}")
            else:
                print(f"      {i+1}. {s} ~ {e} ({(e-s).days + 1} days)")
        
        if len(missing_ranges) > 10:
            print(f"      ... 以及其他 {len(missing_ranges)-10} 个区间")

    # 每日订单量统计
    daily_counts = df.groupby(date_col).size()
    print(f"   -> 📊 日均订单行数: {daily_counts.mean():.1f}")
    print(f"   -> 📉 最少单量日:   {daily_counts.min()} (Date: {daily_counts.idxmin()})")
    print(f"   -> 🚀 最多单量日:   {daily_counts.max()} (Date: {daily_counts.idxmax()})")
    print("-" * 40 + "\n")

def normalize_id(series):
    """
    🧹 ID 标准化清洗
    1. 转大写
    2. 去除首尾空格
    3. 提取前缀 (针对 buyer_id: 'RA0739-xxx' -> 'RA0739', 'RA0363_123' -> 'RA0363')
    """
    # 转换为字符串 -> 大写 -> 去空格
    s = series.astype(str).str.upper().str.strip()
    # 正则提取: 遇到非字母数字字符(-, _, 空格)截断
    # Extract first alphanumeric sequence
    # Pattern: ^([A-Z0-9]+)
    return s.str.extract(r'^([A-Z0-9]+)', expand=False).fillna(s)


def pick_latest_csv(input_folder, pattern="*.csv"):
    all_files = glob.glob(os.path.join(input_folder, pattern))
    if not all_files:
        return None
    return max(all_files, key=os.path.getmtime)

# ================= 3. 清洗逻辑函数 =================

def clean_products():
    """清洗商品表 (dim_product)"""
    print("1️⃣ 正在清洗商品表 (Products)...")
    input_folder = os.path.join(RAW_DIR, "dim_product")
    input_file = pick_latest_csv(input_folder)

    if not input_file:
        print(f"   ⚠️ [跳过] 没找到商品 CSV 文件")
        return

    print(f"   -> 使用最新商品维表: {os.path.basename(input_file)}")
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.lower()

    if 'm_dim10' in df.columns:
        df = df[~df['m_dim10'].astype(str).str.contains('S线', na=False)]

    col_map = {
        'no': 'sku_id',          
        'name': 'style_id',      
        'value': 'product_name', 
        'm_dim5': 'category',    
        'm_dim6': 'sub_category',
        'm_dim3': 'season',      
        'm_dim8': 'band',        
        'm_dim10': 'series',     
        'pricelist': 'price_tag',
        'value1': 'color_id',    
        'value2': 'size_id'      
    }
    
    valid_cols = [c for c in col_map.keys() if c in df.columns]
    df = df[valid_cols].rename(columns=col_map)
    
    # ★ ID 标准化 ★
    if 'style_id' in df.columns:
        df['style_id'] = normalize_id(df['style_id'])

    df.drop_duplicates(subset=['sku_id'], inplace=True)
    df['price_tag'] = pd.to_numeric(df['price_tag'], errors='coerce').fillna(0)
    
    # ==========================================
    # ★ 核心改动：提取历史文件的 qty_first_order
    # ==========================================
    history_file = os.path.join(PROJECT_ROOT, "data_warehouse", "fact_orders", "V_IRS_ORDER_2025.csv")
    if os.path.exists(history_file):
        print("   📦 正在从历史订单抽取 QTYFFO 作为首单数量...")
        # 只读取 NO 和 QTYFFO 两列
        try:
            df_hist = pd.read_csv(history_file, usecols=['NO', 'QTYFFO'], encoding='utf-8')
        except UnicodeDecodeError:
            df_hist = pd.read_csv(history_file, usecols=['NO', 'QTYFFO'], encoding='gbk')
            
        df_hist.columns = ['sku_id', 'qty_first_order']
        # 移除空值并确保为数值
        df_hist['qty_first_order'] = pd.to_numeric(df_hist['qty_first_order'], errors='coerce').fillna(0)
        # 每个 SKU 理论上只有一个首发货量，直接取大值
        sku_ffo = df_hist.groupby('sku_id')['qty_first_order'].max().reset_index()
        
        # 将提取出的首单数量合并入商品表
        df = df.merge(sku_ffo, on='sku_id', how='left')
        df['qty_first_order'] = df['qty_first_order'].fillna(0.0)
        print(f"      - 成功命中历史首发量的 SKU 比例: {(df['qty_first_order'] > 0).mean():.1%}")
    else:
        print("   ⚠️ 找不到 V_IRS_ORDER_2025.csv，默认首单量设为 0")
        df['qty_first_order'] = 0.0

    text_cols = ['category', 'sub_category', 'season', 'band', 'series', 'style_id']
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].fillna('Unknown').astype(str)

    output_path = os.path.join(SILVER_DIR, "clean_products.csv")
    df.to_csv(output_path, index=False)
    print(f"   ✅ 商品表清洗完成 | 行数: {len(df)}")


def clean_stores():
    """清洗门店表 (dim_store)"""
    print("2️⃣ 正在清洗门店表 (Stores)...")
    input_folder = os.path.join(RAW_DIR, "dim_store")
    input_file = pick_latest_csv(input_folder)

    if not input_file:
        return

    print(f"   -> 使用最新门店维表: {os.path.basename(input_file)}")
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.lower()

    # NAME carries the buyer entity (e.g. RA0529-河北唐山孙亚军).
    # STORENAME is a warehouse / store label and should not be used as buyer_id.
    keep_cols = [c for c in ['name', 'storename', 'modifieddate'] if c in df.columns]
    df = df[keep_cols].copy()
    df = df.rename(
        columns={
            'name': 'buyer_name',
            'storename': 'store_label',
            'modifieddate': 'store_modifieddate',
        }
    )

    if 'buyer_name' not in df.columns:
        print("   ⚠️ 门店维表缺少 NAME 字段，无法提取 buyer_id")
        return

    df['buyer_id'] = normalize_id(df['buyer_name'])
    # Keep the production buyer keys, filter out short warehouse/test labels.
    df = df[df['buyer_id'].str.len() >= 6].copy()

    agg_map = {'buyer_name': 'first'}
    if 'store_label' in df.columns:
        agg_map['store_label'] = 'first'
        store_counts = df.groupby('buyer_id')['store_label'].nunique().rename('store_label_count')
    else:
        store_counts = None
    if 'store_modifieddate' in df.columns:
        agg_map['store_modifieddate'] = 'max'

    df = df.groupby('buyer_id', as_index=False).agg(agg_map)
    if store_counts is not None:
        df = df.merge(store_counts.reset_index(), on='buyer_id', how='left')

    # Keep downstream schema stable; distributor_name now carries the buyer display name.
    df['distributor_name'] = df['buyer_name']
    output_cols = ['buyer_id', 'distributor_name']
    for extra_col in ['buyer_name', 'store_label', 'store_label_count', 'store_modifieddate']:
        if extra_col in df.columns:
            output_cols.append(extra_col)
    df = df[output_cols].copy()
    
    output_path = os.path.join(SILVER_DIR, "clean_stores.csv")
    df.to_csv(output_path, index=False)
    print(f"   ✅ 门店表清洗完成 | 行数: {len(df)}")


def clean_orders_new():
    """
    清洗新格式订单表 V_IRS_ORDERFTP.csv
    ★ 新格式说明：
      - NAME     = sku_id
      - STORENAME= buyer_id（含门店全称，需提取前缀）
      - BILLDATE = 日期（YYYYMMDD 整数）
      - QTY      = 数量（现货=qty_replenish, 期货=qty_future）
      - TYPE     = '现货' / '期货'
    ★ 注意：新表无 qty_debt/qty_shipped/qty_first_order/qty_inbound，
           这些字段在 wide_table 中将全部填 0。
    """
    print("3️⃣ 正在清洗新格式订单表 (V_IRS_ORDERFTP.csv)...")
    input_file = os.path.join(RAW_DIR, "fact_orders", "V_IRS_ORDERFTP.csv")

    if not os.path.exists(input_file):
        print(f"   ⚠️ 没找到新格式订单文件: {input_file}，回退到旧格式")
        clean_orders()  # 回退
        return

    try:
        df = pd.read_csv(input_file, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='gbk')

    # ── 字段对齐 ──────────────────────────────────────────
    df = df.rename(columns={
        'NO':        'sku_id',
        'NAME':      'style_id',
        'STORENAME': '_storename_raw',
        'BILLDATE':  'order_date_raw',
        'QTY':       'qty_raw',
        'TYPE':      'type',
    })

    # 提取 buyer_id（取"-"前的前缀，如 RA0136）
    df['buyer_id'] = df['_storename_raw'].astype(str).str.extract(r'^([A-Z0-9]+)-', expand=False)
    df['buyer_id'] = df['buyer_id'].fillna(df['_storename_raw'])
    df = df[df['buyer_id'].str.len() >= 6]

    # 日期解析
    df['order_date'] = pd.to_datetime(
        df['order_date_raw'].astype(str).str[:8], format='%Y%m%d', errors='coerce'
    ).dt.date
    df = df.dropna(subset=['order_date'])

    # 时间范围过滤（保留所有可用数据，不再硬代码截止 2025-12-31）
    start_date = pd.Timestamp('2025-01-01').date()
    df = df[df['order_date'] >= start_date]

    # 数量
    df['qty_raw'] = pd.to_numeric(df['qty_raw'], errors='coerce').fillna(0)
    # 退货/取消为负数，置为 0
    df['qty_raw'] = df['qty_raw'].clip(lower=0)

    # 分拆现货/期货
    df['qty_replenish'] = df['qty_raw'].where(df['type'] == '现货', 0)
    df['qty_future']    = df['qty_raw'].where(df['type'] == '期货',  0)

    # 按 (buyer_id, sku_id, order_date) 聚合
    df_agg = df.groupby(['buyer_id', 'sku_id', 'order_date']).agg(
        qty_replenish=('qty_replenish', 'sum'),
        qty_future=('qty_future', 'sum'),
    ).reset_index()

    # 新表无以下字段，补零
    for col in ['qty_debt', 'qty_shipped', 'qty_first_order', 'qty_inbound']:
        df_agg[col] = 0.0

    # 过滤全零行
    mask = (df_agg['qty_replenish'] > 0) | (df_agg['qty_future'] > 0)
    df_agg = df_agg[mask]

    # 分析
    analyze_replenishment(df_agg, name="新格式订单清洗后")
    check_continuity(df_agg, date_col='order_date')

    output_path = os.path.join(SILVER_DIR, "clean_orders.csv")
    df_agg.to_csv(output_path, index=False)
    print(f"   ✅ 新格式订单表清洗完成 | 行数: {len(df_agg):,} | 日期范围: {df_agg['order_date'].min()} ~ {df_agg['order_date'].max()}")


def clean_orders():
    """
    清洗旧格式订单表 (fact_orders/V_IRS_ORDER_2025.csv)  ← 仅在新格式不存在时调用
    ★ v1.8 升级：锁定 V_IRS_ORDER_2025.csv 唯一底表 + 数据浓度分析 ★
    """
    print("3️⃣ 正在清洗旧格式订单表 (V_IRS_ORDER_2025.csv)...")
    input_file = os.path.join(RAW_DIR, "fact_orders", "V_IRS_ORDER_2025.csv")

    if not os.path.exists(input_file):
        print(f"   ⚠️ 没找到订单文件: {input_file}")
        return

    try:
        df = pd.read_csv(input_file)
        df.columns = df.columns.str.lower()
    except Exception as e:
        print(f"   ⚠️ 读取失败 {input_file}: {e}")
        return

    # 业务过滤
    if 'm_dim10' in df.columns:
        df = df[~df['m_dim10'].astype(str).str.contains('S线', na=False)]

    col_map = {
        'no': 'sku_id',
        'name': 'style_id',
        'storename': 'buyer_id',
        'billdate': 'order_date',
        'qtyso': 'qty_future',
        'qtyspo': 'qty_replenish',
        'qtyrem': 'qty_debt',
        'total_qtyout': 'qty_shipped',
        'qtyffo': 'qty_first_order',
        'qtypur': 'qty_inbound'
    }

    valid_cols = [c for c in col_map.keys() if c in df.columns]
    df = df[valid_cols].rename(columns=col_map)

    if 'buyer_id' in df.columns:
        df['buyer_id'] = normalize_id(df['buyer_id'])
        df = df[df['buyer_id'].str.len() >= 6]

    if 'style_id' in df.columns:
        df['style_id'] = normalize_id(df['style_id'])

    df['order_date'] = pd.to_datetime(df['order_date'].astype(str).str.split('.').str[0], format='%Y%m%d', errors='coerce').dt.date
    df = df.dropna(subset=['order_date'])

    start_date = pd.Timestamp('2025-01-01').date()
    end_date = pd.Timestamp('2025-12-31').date()
    df = df[(df['order_date'] >= start_date) & (df['order_date'] <= end_date)]

    num_cols = ['qty_future', 'qty_replenish', 'qty_debt', 'qty_shipped', 'qty_first_order', 'qty_inbound']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    mask = (df['qty_future'] != 0) | (df['qty_replenish'] != 0) | \
           (df.get('qty_debt', 0) != 0) | (df.get('qty_shipped', 0) != 0)
    df = df[mask]

    if 'qty_replenish' in df.columns:
        df.loc[df['qty_replenish'] < 0, 'qty_replenish'] = 0

    analyze_replenishment(df, name="清洗后的订单数据")
    check_continuity(df, date_col='order_date')

    output_path = os.path.join(SILVER_DIR, "clean_orders.csv")
    df.to_csv(output_path, index=False)
    print(f"   ✅ 订单表清洗完成 | 行数: {len(df)} | 日期范围: {df['order_date'].min()} ~ {df['order_date'].max()}")



def clean_inventory():
    """清洗每日库存快照 (snapshot_inventory)"""
    print("4️⃣ 正在清洗库存快照 (Inventory)...")
    input_folder = os.path.join(RAW_DIR, "snapshot_inventory") 
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not all_files:
        print(f"   ⚠️ [跳过] 库存文件夹为空")
        return

    df_list = []
    processed_count = 0
    
    for f in all_files:
        filename = os.path.basename(f)
        # 简单过滤，防止非storage文件混入
        if "storage" not in filename: continue
        match = re.search(r'(\d{8})', filename)
        if not match: continue
            
        file_date_str = match.group(1)
        try:
            tmp = pd.read_csv(f)
            tmp.columns = tmp.columns.str.lower()
            tmp['inventory_date'] = pd.to_datetime(file_date_str, format='%Y%m%d').date()
            if tmp['inventory_date'].iloc[0] < pd.Timestamp('2025-01-01').date():
                continue
            df_list.append(tmp)
            processed_count += 1
        except Exception as e:
            print(f"   ⚠️ 读取失败 {filename}: {e}")

    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)

    if 'm_dim10' in df.columns:
        df = df[~df['m_dim10'].astype(str).str.contains('S线', na=False)]

    col_map = {
        'no': 'sku_id',          
        'name': 'style_id',      
        'qtycan': 'qty_stock',   
        'inventory_date': 'inventory_date'
    }
    
    valid_cols = [c for c in col_map.keys() if c in df.columns]
    df = df[valid_cols].rename(columns=col_map)
    df['qty_stock'] = pd.to_numeric(df['qty_stock'], errors='coerce').fillna(0)
    
    output_path = os.path.join(SILVER_DIR, "clean_inventory.csv")
    df.to_csv(output_path, index=False)
    print(f"   ✅ 库存表清洗完成 | 行数: {len(df)}")


def clean_events():
    """
    清洗埋点表 (fact_events)
    ★ 功能升级: 聚合每日行为特征 (Clicks, Cart, Orders)
    ★ 修复: 正确处理 SKC ID 映射
    """
    print("5️⃣ 正在清洗埋点表 (Events) -> 聚合每日特征...")
    input_folder = os.path.join(RAW_DIR, "fact_events")
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not all_files: 
        print("   ⚠️ 未找到埋点文件")
        return

    # 1. 读取原始数据
    df_list = []
    use_cols = ['USERNAME', 'PRODUCTNAME', 'CURRENT_STAGE', 'CREATIONDATE', 'ORDER_QTY']
    
    for f in all_files:
        try:
            # 只读取需要的列以节省内存
            tmp = pd.read_csv(f, usecols=lambda c: c.upper() in use_cols)
            tmp.columns = tmp.columns.str.upper() # 统一大写方便处理
            df_list.append(tmp)
        except Exception as e: 
            print(f"   ⚠️ 读取失败 {f}: {e}")

    if not df_list: return
    df = pd.concat(df_list, ignore_index=True)
    
    # 2. 重命名
    df.rename(columns={
        'USERNAME': 'buyer_id',
        'PRODUCTNAME': 'skc_id', # ★★★ 统一重命名为 skc_id ★★★
        'CURRENT_STAGE': 'event_type',
        'CREATIONDATE': 'event_time',
        'ORDER_QTY': 'quantity'
    }, inplace=True)
    
    # 3. 基础清洗
    # ID 标准化
    if 'buyer_id' in df.columns:
        df['buyer_id'] = normalize_id(df['buyer_id'])
        df = df[df['buyer_id'].str.len() >= 6]
        
    if 'skc_id' in df.columns:
        df['skc_id'] = normalize_id(df['skc_id'])

    df.dropna(subset=['buyer_id', 'skc_id', 'event_time'], inplace=True)

    # 日期处理
    df['event_time'] = pd.to_datetime(df['event_time'], errors='coerce')
    df['date'] = df['event_time'].dt.date
    
    # ★ 时间范围过滤 (v1.5 Update: Only 2025) ★
    start_date = pd.Timestamp('2025-01-01').date()
    end_date = pd.Timestamp('2025-12-31').date()
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    print(f"   -> 原始行数 (2024-2025): {len(df)}")
    
    # 4. 聚合特征 (Daily Aggregation)
    # ★★★ 关键修复：Groupby 使用 skc_id ★★★
    
    # Clicks
    clicks = df[df['event_type'] == '商品点击'].groupby(['buyer_id', 'skc_id', 'date']).size().reset_index(name='daily_clicks')
    
    # Cart Adds
    cart_adds = df[df['event_type'] == '加购物车'].groupby(['buyer_id', 'skc_id', 'date']).size().reset_index(name='daily_cart_adds')
    
    # Orders
    orders_df = df[df['event_type'] == '下单成功']
    orders_count = orders_df.groupby(['buyer_id', 'skc_id', 'date']).size().reset_index(name='daily_orders')
    
    # Order Qty (Sum)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
    orders_qty = df[df['event_type'] == '下单成功'].groupby(['buyer_id', 'skc_id', 'date'])['quantity'].sum().reset_index(name='daily_order_qty')
    
    # 5. 合并
    merged = pd.merge(clicks, cart_adds, on=['buyer_id', 'skc_id', 'date'], how='outer')
    merged = pd.merge(merged, orders_count, on=['buyer_id', 'skc_id', 'date'], how='outer')
    merged = pd.merge(merged, orders_qty, on=['buyer_id', 'skc_id', 'date'], how='outer')
    
    # 填补 0
    merged.fillna(0, inplace=True)
    
    # 6. 保存
    # 这里的列名为: buyer_id, skc_id, date, daily_clicks...
    # 注意：build_wide_table.py 读取时会期待 'sku_id' 列来代表 SKC，或者我们这里就输出 'sku_id' 
    # 为了配合之前的 build_wide_table 代码（它把 sku_id 当 SKC 处理），我们这里最好重命名回去，或者告知用户
    # 建议：这里输出就叫 'sku_id'，虽然它存的是 SKC，但这样 build_wide_table 不用改代码就能跑
    merged.rename(columns={'skc_id': 'sku_id'}, inplace=True)
    
    output_path = os.path.join(SILVER_DIR, "clean_events.csv")
    merged.to_csv(output_path, index=False)
    print(f"   ✅ 埋点表清洗聚合完成 | 输出: {output_path} | 行数: {len(merged)}")


def clean_buyer_profile():
    """
    清洗买手多维画像 (Buyer Profile) 
    🌟 [V1.8 战役三核心] 提取门店等级、吞吐量等深层属性
    """
    print("6️⃣ 正在清洗买手画像 (Buyer Profile)...")
    input_folder = os.path.join(RAW_DIR, "snapshot_metrics")
    all_files = glob.glob(os.path.join(input_folder, "*_customer_profile.csv"))
    
    if not all_files:
        print("   ⚠️ 未找到买手画像文件")
        return
        
    df = pd.read_csv(all_files[0])
    
    # 提取 Buyer ID (形如 "RA0884-河南驻马店闺蜜")
    if 'CUSTOMER_NAME' not in df.columns:
        print("   ⚠️ 买手画像文件缺少 CUSTOMER_NAME 字段")
        return
        
    df['buyer_id'] = df['CUSTOMER_NAME'].astype(str).str.split('-').str[0]
    df['buyer_id'] = normalize_id(df['buyer_id'])
    
    # 抽取核心业务数值特征
    numeric_cols = [
        'COOPERATION_YEARS', 
        'MONTHLY_AVERAGE_REPLENISHMENT', 
        'AVG_DISCOUNT_RATE', 
        'REPLENISHMENT_FREQUENCY', 
        'ITEM_COVERAGE_RATE'
    ]
    
    valid_cols = ['buyer_id']
    for c in numeric_cols:
        if c in df.columns:
            # 填缝并转数值
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
            valid_cols.append(c)
            
    df = df[valid_cols]
    
    # 去重：同一个买手可能会因为不同店铺后缀在画像表出现多行，我们取 max() 保留高能上限
    df = df.groupby('buyer_id').max().reset_index()
    
    # 全部转小写列名对接主宽表
    df.columns = df.columns.str.lower()
    
    output_path = os.path.join(SILVER_DIR, "clean_buyer_profile.csv")
    df.to_csv(output_path, index=False)
    print(f"   ✅ 买手画像清洗完成 | 行数: {len(df)}")


def main():
    print("🚀 [Step 2] 开始数据清洗 (新格式 V_IRS_ORDERFTP 优先)...")
    clean_products()
    clean_stores()
    clean_orders_new()   # ★ 优先使用新格式；文件不存在则自动回退到旧格式
    clean_inventory()
    clean_events()
    clean_buyer_profile()
    print("\n🎉 Step 2 完成！")

if __name__ == "__main__":
    main()
