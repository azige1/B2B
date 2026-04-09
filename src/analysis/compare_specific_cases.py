import pandas as pd
import numpy as np
import os

def specific_case_audit():
    path_v16 = r'E:\LSTM\B2B\B2B_Replenishment_System\data\gold\wide_table_sku_v1.6.csv'
    path_now = r'E:\LSTM\B2B\B2B_Replenishment_System\data\gold\wide_table_sku.csv'
    
    target_date = '2025-12-15'
    target_buyer = 'RA0001'
    # 从之前的扫描中选取的几个典型 SKU
    target_skus = ['AK1070155736720', 'AK1011076936100', 'AK1060165636613']
    
    print("="*100)
    print(f"🔍 深度个案核查: {target_date} | Buyer: {target_buyer}")
    print("="*100)

    for sku in target_skus:
        print(f"\n📦 SKU: {sku}")
        print("-" * 100)
        
        row_v16 = pd.DataFrame()
        row_now = pd.DataFrame()
        
        # 读取 V1.6
        if os.path.exists(path_v16):
            df_v16 = pd.read_csv(path_v16)
            df_v16['date'] = pd.to_datetime(df_v16['date'])
            match = df_v16[(df_v16['date'] == target_date) & 
                           (df_v16['buyer_id'] == target_buyer) & 
                           (df_v16['sku_id'].astype(str) == sku)]
            if not match.empty:
                row_v16 = match.iloc[0]
        
        # 读取 Current
        if os.path.exists(path_now):
            df_now = pd.read_csv(path_now)
            df_now['date'] = pd.to_datetime(df_now['date'])
            match = df_now[(df_now['date'] == target_date) & 
                           (df_now['buyer_id'] == target_buyer) & 
                           (df_now['sku_id'].astype(str) == sku)]
            if not match.empty:
                row_now = match.iloc[0]

        # 如果两个都没找到，尝试模糊日期
        if row_v16.empty and row_now.empty:
            print("   ⚠️ 两个版本中均未找到该 (Date, Buyer, SKU) 的精确匹配记录。")
            continue

        # 整理对比列
        qty_cols = sorted(list(set([c for c in row_v16.index if c.startswith('qty_')] + 
                                   [c for c in row_now.index if c.startswith('qty_')])))
        
        print(f"{'字段':<25} | {'V1.6':>25} | {'Current':>25} | {'差异':>15}")
        print("-" * 100)
        
        for col in qty_cols:
            val_v16 = row_v16.get(col, "N/A")
            val_now = row_now.get(col, "N/A")
            
            diff = ""
            try:
                if isinstance(val_v16, (int, float)) and isinstance(val_now, (int, float)):
                    diff = f"{val_now - val_v16:+.2f}"
            except: pass
            
            print(f"{col:<25} | {str(val_v16):>25} | {str(val_now):>25} | {diff:>15}")

        # 其他元数据
        meta_cols = ['style_id', 'product_name', 'category', 'sub_category', 'season', 'price_tag']
        print("\n   [ 元数据对比 ]")
        for col in meta_cols:
            v_v16 = row_v16.get(col, "N/A")
            v_now = row_now.get(col, "N/A")
            status = "✅ 一致" if str(v_v16) == str(v_now) else "❌ 冲突"
            print(f"   - {col:<15}: {str(v_v16):<20} vs {str(v_now):<20} ({status})")

if __name__ == "__main__":
    specific_case_audit()
