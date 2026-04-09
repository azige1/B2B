import pandas as pd
import os
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')

# ================= 1. 路径配置 =================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
SILVER_DIR = os.path.join(PROJECT_ROOT, "data/silver")
GOLD_DIR = os.path.join(PROJECT_ROOT, "data/gold")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data/processed") 
PHASE8_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "phase8a_prep")

os.makedirs(GOLD_DIR, exist_ok=True)

def print_log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def load_inventory_features():
    """
    优先使用 phase8a 已整理好的 inventory_daily_features。
    该表同时包含 storage + b2b_hq，并且覆盖到 2026-04。
    如果不存在，则回退到 silver/clean_inventory.csv。
    """
    inventory_path = os.path.join(PHASE8_DATA_DIR, "inventory_daily_features.csv")
    if os.path.exists(inventory_path):
        df = pd.read_csv(inventory_path)
        if df.empty:
            return None

        df['date'] = pd.to_datetime(df['date'])
        df['sku_id'] = df['sku_id'].astype(str).str.strip()
        for col in ['qty_storage_stock', 'qty_b2b_hq_stock', 'has_storage_snapshot', 'has_b2b_snapshot']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            else:
                df[col] = 0.0

        # phase8a 的库存表存在 date+sku 重复键；这里聚成唯一键，避免宽表膨胀。
        df = (
            df.groupby(['date', 'sku_id'], as_index=False)
            .agg({
                'qty_storage_stock': 'max',
                'qty_b2b_hq_stock': 'max',
                'has_storage_snapshot': 'max',
                'has_b2b_snapshot': 'max',
            })
        )
        df['qty_stock'] = df['qty_storage_stock'] + df['qty_b2b_hq_stock']
        df['is_real_stock'] = (
            (df['has_storage_snapshot'] > 0) | (df['has_b2b_snapshot'] > 0)
        ).astype(int)
        print_log(
            f"   -> 使用 phase8a 库存特征: {os.path.basename(inventory_path)} | 行数: {len(df)}"
        )
        return df

    fallback_path = os.path.join(SILVER_DIR, "clean_inventory.csv")
    if os.path.exists(fallback_path):
        df = pd.read_csv(fallback_path)
        if df.empty:
            return None
        df['date'] = pd.to_datetime(df['inventory_date'])
        df['sku_id'] = df['sku_id'].astype(str).str.strip()
        df['qty_stock'] = pd.to_numeric(df['qty_stock'], errors='coerce').fillna(0)
        df = (
            df.groupby(['date', 'sku_id'], as_index=False)
            .agg({'qty_stock': 'max'})
        )
        df['qty_storage_stock'] = df['qty_stock']
        df['qty_b2b_hq_stock'] = 0.0
        df['has_storage_snapshot'] = (df['qty_stock'] > 0).astype(int)
        df['has_b2b_snapshot'] = 0
        df['is_real_stock'] = df['has_storage_snapshot']
        print_log(
            f"   -> 回退使用 clean_inventory.csv | 行数: {len(df)}"
        )
        return df

    print_log("   -> 未找到可用库存特征，继续使用占位库存列")
    return None

def load_data():
    """加载 Step 2 清洗好的数据"""
    print_log("1️⃣ 加载银层数据 (Silver Data)...")
    data = {}
    files = {
        'products': 'clean_products.csv',
        'stores': 'clean_stores.csv',
        'orders': 'clean_orders.csv',
        'events': 'clean_events.csv' # 聚合后的日粒度数据 (SKC级)
    }
    
    for key, filename in files.items():
        path = os.path.join(SILVER_DIR, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            # 统一日期格式
            for col in ['order_date', 'event_date', 'date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # 统一 ID 格式 (转字符串)
            for id_col in ['buyer_id', 'style_id', 'sku_id']:
                if id_col in df.columns:
                    df[id_col] = df[id_col].astype(str).str.strip()
            
            data[key] = df
            print(f"   - {key}: {len(df)} 行")
        else:
            print(f"   ⚠️ 缺失: {filename}")
            data[key] = None
    
    return data

def build_wide_table():
    print_log("🚀 [Step 3] 开始合成宽表 (v1.5 Final: Direct Order Backbone)...")
    data = load_data()
    
    if data['orders'] is None:
        print("❌ 错误：缺少订单数据，无法构建骨架")
        return

    # ==========================================
    # 1. 准备骨架 (Order Backbone)
    # ==========================================
    # 我们只关心订单表里出现过的记录 (Buyer, SKU, Date)
    # 根据你的样例，order_date 是日期列
    df_main = data['orders'].rename(columns={'order_date': 'date'})
    
    # ★ 核心：生成关联键 SKC_ID ★
    # 根据你的数据样例: sku_id 'AM3100155336000' -> style_id 'AM31001553'
    # 截取前 10 位作为 skc_id 用来关联埋点
    df_main['skc_id'] = df_main['sku_id'].str[:10]
    
    # 预防性删除订单表自带的 style_id，防止和商品表/埋点表冲突
    # 我们会在最后一步通过商品表 merge 回来最准确的 style_id
    if 'style_id' in df_main.columns:
        df_main = df_main.drop(columns=['style_id'])

    # 聚合：确保骨架唯一 (Date, Buyer, SKU)
    # 保留 skc_id 用于下一步关联
    group_cols = ['date', 'buyer_id', 'sku_id', 'skc_id']
    df_main = df_main.groupby(group_cols).sum(numeric_only=True).reset_index()
    
    print_log(f"   -> 订单骨架行数: {len(df_main)}")

    # ==========================================
    # 2. 关联埋点数据 (Left Join via SKC)
    # ==========================================
    if data['events'] is not None:
        print_log("   -> 关联埋点特征 (Left Join via SKC)...")
        
        # 准备 Events
        # clean_events.csv 里的 'sku_id' 已经被 clean_data.py 统一重命名为 'sku_id' 但存的是 SKC
        # 为了避免混淆，我们读进来后改名为 skc_id
        df_events = data['events'].rename(columns={'sku_id': 'skc_id'})
        
        # 选取需要的列
        evt_cols = ['date', 'buyer_id', 'skc_id', 'daily_clicks', 'daily_cart_adds', 'daily_orders', 'daily_order_qty']
        valid_evt_cols = [c for c in evt_cols if c in df_events.columns]
        df_events = df_events[valid_evt_cols]
        
        # ★ 核心关联 ★
        # 逻辑：订单(SKU) -> 截取前10位(SKC) -> 关联埋点(SKC)
        # Left Join: 严格保证行数不增加
        df_main = pd.merge(
            df_main, 
            df_events, 
            on=['date', 'buyer_id', 'skc_id'], 
            how='left'
        )
        
        # 填充 0 (未关联上的就是没看)
        fill_cols = ['daily_clicks', 'daily_cart_adds', 'daily_orders', 'daily_order_qty']
        for c in fill_cols:
            if c in df_main.columns: df_main[c] = df_main[c].fillna(0)
        
        # ★ 时间闸门 (Time Gating) ★
        # 逻辑：只有 2025-09-18 之后的数据是真实的，之前的强制归零
        EVENT_START_DATE = pd.Timestamp("2025-09-18")
        df_main['is_pv_valid'] = (df_main['date'] >= EVENT_START_DATE).astype(int)
        
        mask_invalid = df_main['is_pv_valid'] == 0
        for c in fill_cols:
                if c in df_main.columns: df_main.loc[mask_invalid, c] = 0
                
    else:
        df_main['is_pv_valid'] = 0
        for c in ['daily_clicks', 'daily_cart_adds', 'daily_orders', 'daily_order_qty']:
            df_main[c] = 0

    # 用完 skc_id 可以删除了，保持宽表干净
    df_main = df_main.drop(columns=['skc_id'])

    # ==========================================
    # 3. 关联商品与门店
    # ==========================================
    print_log("3️⃣ 关联商品与门店...")
    
    # 商品
    if data['products'] is not None:
        prod_cols = ['sku_id', 'style_id', 'product_name', 'category', 'sub_category', 'season', 'band', 'series', 'price_tag', 'size_id', 'color_id', 'qty_first_order']
        valid_cols = [c for c in prod_cols if c in data['products'].columns]
        # 去重，防止 SKU 重复导致膨胀
        df_prod = data['products'][valid_cols].drop_duplicates(subset=['sku_id'])
        product_sku_set = set(df_prod['sku_id'].astype(str))
        
        # 移除由于占位预留的无效属性，准备接收真实属性
        for c in ['style_id', 'qty_first_order']:
            if c in df_main.columns and c in valid_cols:
                df_main = df_main.drop(columns=[c])
                
        df_main = pd.merge(df_main, df_prod, on='sku_id', how='left')
        missing_product_mask = ~df_main['sku_id'].astype(str).isin(product_sku_set)
        missing_product_rows = int(missing_product_mask.sum())
        if missing_product_rows > 0:
            missing_product_skus = int(df_main.loc[missing_product_mask, 'sku_id'].astype(str).nunique())
            print_log(
                f"   -> 过滤缺失商品主数据的 SKU: {missing_product_skus} 个 SKU, {missing_product_rows} 行订单骨架"
            )
            df_main = df_main.loc[~missing_product_mask].copy()
        
    # 门店
    if data['stores'] is not None:
        df_stores = data['stores'].drop_duplicates(subset=['buyer_id'])
        df_main = pd.merge(df_main, df_stores, on='buyer_id', how='left')
        
    # 库存：优先接入 phase8a 的真实 inventory_daily_features
    inventory_df = load_inventory_features()
    if inventory_df is not None:
        inventory_cols = [
            'date', 'sku_id',
            'qty_storage_stock',
            'qty_b2b_hq_stock',
            'has_storage_snapshot',
            'has_b2b_snapshot',
            'qty_stock',
            'is_real_stock',
        ]
        df_main = pd.merge(
            df_main,
            inventory_df[inventory_cols],
            on=['date', 'sku_id'],
            how='left'
        )
        for col in [
            'qty_storage_stock',
            'qty_b2b_hq_stock',
            'has_storage_snapshot',
            'has_b2b_snapshot',
            'qty_stock',
            'is_real_stock',
        ]:
            df_main[col] = pd.to_numeric(df_main[col], errors='coerce').fillna(0)
    else:
        df_main['qty_storage_stock'] = 0
        df_main['qty_b2b_hq_stock'] = 0
        df_main['has_storage_snapshot'] = 0
        df_main['has_b2b_snapshot'] = 0
        df_main['qty_stock'] = 0
        df_main['is_real_stock'] = 0

    # ==========================================
    # 4. 注入高阶特征 (含去重修复)
    # ==========================================
    print_log("4️⃣ 注入高阶特征 (Optional)...")
    
    # --- Style Features ---
    style_feat_path = os.path.join(PROCESSED_DIR, "style_advanced_features.csv")
    if os.path.exists(style_feat_path):
        try:
            df_style = pd.read_csv(style_feat_path)
            if 'style_id' in df_style.columns:
                df_style['style_id'] = df_style['style_id'].astype(str).str.strip()
                # ★ 去重，防止膨胀 ★
                df_style = df_style.drop_duplicates(subset=['style_id'])
                cols = [c for c in df_style.columns if c not in df_main.columns or c == 'style_id']
                df_main = pd.merge(df_main, df_style[cols], on='style_id', how='left')
        except Exception as e: print(f"Warning: {e}")

    # --- Buyer Features ---
    # ★ 已经升级为 SKU 级全国预测，买手画像在后续聚合中无实际意义，故移除。

    # ==========================================
    # 5. 最终清洗与保存
    # ==========================================
    print_log("5️⃣ 最终清洗与保存...")
    
    str_cols = df_main.select_dtypes(include=['object']).columns
    df_main[str_cols] = df_main[str_cols].fillna('Unknown')
    
    num_cols = df_main.select_dtypes(include=[np.number]).columns
    df_main[num_cols] = df_main[num_cols].fillna(0)
    
    # 按时间排序对 LSTM 很重要
    df_main = df_main.sort_values(['buyer_id', 'sku_id', 'date'])
    
    output_path = os.path.join(GOLD_DIR, "wide_table_sku.csv")
    df_main.to_csv(output_path, index=False)
    
    print_log("-" * 30)
    print_log(f"🎉 宽表构建完成: {output_path}")
    print_log(f"📊 最终行数: {len(df_main)} (Strict Order Backbone)")
    print_log(f"🔎 列名: {list(df_main.columns)}")

if __name__ == "__main__":
    build_wide_table()
