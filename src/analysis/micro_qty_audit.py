import pandas as pd
import numpy as np
import os

def micro_qty_audit():
    raw_path = r'E:\LSTM\B2B\B2B_Replenishment_System\data_warehouse\fact_orders\V_IRS_ORDER_2025.csv'
    v16_path = r'E:\LSTM\B2B\B2B_Replenishment_System\data/gold/wide_table_sku_v1.6.csv'
    curr_path = r'E:\LSTM\B2B\B2B_Replenishment_System\data/gold/wide_table_sku.csv'
    
    print("="*100)
    print("🔬 原始字段 vs V1.6 vs Current 极细度审计")
    print("="*100)

    # 1. 审计原始文件的字段特征
    if os.path.exists(raw_path):
        print(f"[*] 正在分析原始底表: {os.path.basename(raw_path)}")
        # 读取关键列
        raw_df = pd.read_csv(raw_path, usecols=['BILLDATE', 'MODIFIEDDATE', 'QTYSO', 'QTYSPO', 'TOTAL_QTYOUT', 'STORENAME', 'NO'], low_memory=False)
        raw_df.columns = raw_df.columns.str.upper()
        
        # 统计 QTYSO (qty_future) 的特征
        print("\n--- 原始字段分布 ---")
        for col in ['QTYSO', 'QTYSPO', 'TOTAL_QTYOUT']:
            if col in raw_df.columns:
                s = pd.to_numeric(raw_df[col], errors='coerce').fillna(0)
                print(f"{col:<15}: Sum={s.sum():>12,.0f}, NonZero={len(s[s>0]):>7,}, Mean={s[s>0].mean():>6.2f}")
        
        # 对比 MODIFIEDDATE vs BILLDATE 的去重效应
        print("\n--- 时间维度去重效应 ---")
        raw_df['BILL_D'] = pd.to_datetime(raw_df['BILLDATE'].astype(str), format='%Y%m%d', errors='coerce').dt.date
        raw_df['MOD_D'] = pd.to_datetime(raw_df['MODIFIEDDATE'], errors='ignore').dt.date
        
        pairs_bill = raw_df.groupby(['BILL_D', 'STORENAME', 'NO']).size()
        pairs_mod = raw_df.groupby(['MOD_D', 'STORENAME', 'NO']).size()
        
        print(f"按 BILLDATE 聚合后的总行数: {len(pairs_bill):,}")
        print(f"按 MODIFIEDDATE 聚合后的总行数: {len(pairs_mod):,}")
        print(f"结论: 使用 MODIFIEDDATE 产生的记录数比 BILLDATE 多 {len(pairs_mod) - len(pairs_bill):,} 行")

    # 2. 对比 V1.6 和 Current 在同一批 SKU 上的 QTY_FUTURE 表现
    print("\n" + "="*100)
    print("🎯 同一 (Buyer, SKU) 在不同版本下的字段对齐")
    print("="*100)
    
    if os.path.exists(v16_path) and os.path.exists(curr_path):
        v16 = pd.read_csv(v16_path).head(100000) # 取前10万行做样
        curr = pd.read_csv(curr_path).head(100000)
        
        # 寻找共同的 (buyer, sku)
        merge_test = pd.merge(
            v16[['buyer_id', 'sku_id', 'qty_future', 'qty_replenish', 'qty_shipped']], 
            curr[['buyer_id', 'sku_id', 'qty_future', 'qty_replenish', 'qty_shipped']], 
            on=['buyer_id', 'sku_id'], 
            suffixes=('_v16', '_curr')
        )
        
        if not merge_test.empty:
            print(f"找到 {len(merge_test)} 对跨版本匹配记录")
            # 统计差异
            for col in ['qty_future', 'qty_replenish', 'qty_shipped']:
                diff = merge_test[f'{col}_curr'] - merge_test[f'{col}_v16']
                print(f"\n字段 {col.upper()}:")
                print(f"  - 均值差异 (Curr - V16): {diff.mean():.2f}")
                print(f"  - 两者完全一致比例: {(diff == 0).mean():.2%}")
                
            print("\n[ 差异个案抽样 ]")
            print(merge_test[merge_test['qty_future_v16'] != merge_test['qty_future_curr']].head(10).to_string())
        else:
            print("❌ 两个版本的头 10 万行中没有找到重叠的 (Buyer, SKU) 对，可能是排序或 ID 策略不一致。")

if __name__ == "__main__":
    micro_qty_audit()
