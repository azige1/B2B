import pandas as pd
import os

def analyze_v16_density():
    path_v16 = r'E:\LSTM\B2B\B2B_Replenishment_System\data\gold\wide_table_sku_v1.6.csv'
    
    if not os.path.exists(path_v16):
        print(f"❌ 未找到文件: {path_v16}")
        return

    print(f"[*] 正在读取 {os.path.basename(path_v16)} ...")
    # 只读关键列以节省内存
    df = pd.read_csv(path_v16, usecols=['buyer_id', 'sku_id', 'date'])
    
    # 按照 (buyer_id, sku_id) 分组计数
    counts = df.groupby(['buyer_id', 'sku_id']).size()
    total_pairs = len(counts)
    
    print("\n" + "="*50)
    print("📊 V1.6 宽表 (买手+SKU) 时序深度分布")
    print("="*50)
    
    # 统计分布
    dist = counts.value_counts().sort_index()
    print(f"{'记录行数':<10} | {'这对组合数':<15} | {'占比':<10}")
    print("-" * 45)
    for depth, freq in dist.items():
        print(f"{depth:<10} | {freq:<15,} | {freq/total_pairs:.2%}")
    
    print("-" * 45)
    print(f"总唯一对数 (Unique Pairs): {total_pairs:,}")
    print(f"平均每对行数 (Mean Depth): {counts.mean():.4f}")
    
    # 展示多行记录的例子
    multi_rows = counts[counts > 1]
    if not multi_rows.empty:
        print("\n[ 典型多行案例 (Multi-row Examples) ]")
        # 选一些行数较多的案例
        example_pairs = multi_rows.sort_values(ascending=False).head(5).index
        for b_id, s_id in example_pairs:
            print(f"\n🔹 Buyer: {b_id} | SKU: {s_id} ({counts[(b_id, s_id)]} rows)")
            subset = df[(df['buyer_id'] == b_id) & (df['sku_id'] == s_id)].sort_values('date')
            print(subset['date'].to_string(index=False))

if __name__ == "__main__":
    analyze_v16_density()
