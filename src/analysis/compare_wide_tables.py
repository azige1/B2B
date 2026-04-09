import pandas as pd
import numpy as np
import os
import time

def compare_tables():
    path_v16 = r'E:\LSTM\B2B\B2B_Replenishment_System\data\gold\wide_table_sku_v1.6.csv'
    path_now = r'E:\LSTM\B2B\B2B_Replenishment_System\data\gold\wide_table_sku.csv'
    
    files = {'V1.6': path_v16, 'Current': path_now}
    results = {}

    print("="*80)
    print("🚀 B2B 宽表版本大比武: V1.6 vs Current")
    print("="*80)

    for name, path in files.items():
        if not os.path.exists(path):
            print(f"⚠️ 文件不存在: {path}")
            continue
        
        print(f"[*] 正在分析 {name}...")
        start = time.time()
        
        # 只读取核心维度和指标
        cols = ['date', 'buyer_id', 'sku_id', 'qty_replenish']
        df = pd.read_csv(path, usecols=lambda x: x in cols or x.startswith('qty_'))
        
        df['date'] = pd.to_datetime(df['date'])
        
        # 基础统计
        total_rows = len(df)
        pairs = df.groupby(['buyer_id', 'sku_id']).size()
        unique_pairs = len(pairs)
        avg_depth = pairs.mean()
        max_depth = pairs.max()
        
        # 补货浓度
        q_repl = df['qty_replenish'].fillna(0)
        pos_rows = (q_repl > 0).sum()
        sparsity = pos_rows / total_rows
        
        results[name] = {
            'Rows': total_rows,
            'Unique Pairs': unique_pairs,
            'Avg Depth (Rows/Pair)': avg_depth,
            'Max Depth': max_depth,
            'Pos Rows (Qty>0)': pos_rows,
            'Sparsity (Pos%)': sparsity,
            'Min Date': df['date'].min().date(),
            'Max Date': df['date'].max().date(),
            'Mean Qty (Pos Only)': q_repl[q_repl > 0].mean() if pos_rows > 0 else 0
        }
        print(f"    - 完成，用时: {time.time()-start:.2f}s")

    # 打印对比表格
    print("\n" + "指标项".ljust(25) + " | " + "V1.6".center(20) + " | " + "Current".center(20))
    print("-" * 75)
    
    metrics = ['Rows', 'Unique Pairs', 'Avg Depth (Rows/Pair)', 'Max Depth', 'Pos Rows (Qty>0)', 'Sparsity (Pos%)', 'Min Date', 'Max Date', 'Mean Qty (Pos Only)']
    
    for m in metrics:
        v16_val = results.get('V1.6', {}).get(m, "N/A")
        curr_val = results.get('Current', {}).get(m, "N/A")
        
        # 格式化
        if isinstance(v16_val, float): v16_str = f"{v16_val:.4f}"
        elif isinstance(v16_val, int): v16_str = f"{v16_val:,}"
        else: v16_str = str(v16_val)
        
        if isinstance(curr_val, float): curr_str = f"{curr_val:.4f}"
        elif isinstance(curr_val, int): curr_str = f"{curr_val:,}"
        else: curr_str = str(curr_val)
            
        print(f"{m.ljust(25)} | {v16_str.center(20)} | {curr_str.center(20)}")

    print("\n" + "="*80)
    print("💡 结论透视:")
    if 'V1.6' in results and 'Current' in results:
        depth_diff = results['V1.6']['Avg Depth (Rows/Pair)'] / results['Current']['Avg Depth (Rows/Pair)']
        print(f"1. 时序厚度: V1.6 的平均深度是 Current 的 {depth_diff:.1f} 倍。")
        print(f"2. 样本规模: Current 包含了 {results['Current']['Unique Pairs']:,} 对关系，是对 V1.6 的有力扩充，还是数据漂移？")
        if results['V1.6']['Sparsity (Pos%)'] > results['Current']['Sparsity (Pos%)']:
            print(f"3. 信号浓度: V1.6 补货行占比更高 ({results['V1.6']['Sparsity (Pos%)']:.2%})，模型更容易找到规律。")
    print("="*80)

if __name__ == "__main__":
    compare_tables()
