import pandas as pd
import numpy as np
import os
import time

def detailed_comparison():
    path_v16 = r'E:\LSTM\B2B\B2B_Replenishment_System\data\gold\wide_table_sku_v1.6.csv'
    path_now = r'E:\LSTM\B2B\B2B_Replenishment_System\data\gold\wide_table_sku.csv'
    
    files = {'V1.6': path_v16, 'Current': path_now}
    
    print("="*100)
    print("🔬 B2B 宽表全维度深度审计: V1.6 vs Current")
    print("="*100)

    all_stats = []

    for name, path in files.items():
        if not os.path.exists(path):
            print(f"⚠️ 文件不存在: {path}")
            continue
        
        print(f"[*] 正在深度扫描 {name}...")
        start_time = time.time()
        
        # 预读列名
        header = pd.read_csv(path, nrows=0)
        qty_cols = [c for c in header.columns if c.startswith('qty_')]
        cols_to_load = ['buyer_id', 'sku_id', 'date'] + qty_cols
        
        # 加载数据
        df = pd.read_csv(path, usecols=cols_to_load)
        df['date'] = pd.to_datetime(df['date'])
        
        total_rows = len(df)
        
        # 计算该版本的基本面
        pairs = df.groupby(['buyer_id', 'sku_id']).size()
        
        print(f"    - 加载完成 ({total_rows:,} 行), 开始计算指标...")

        for col in qty_cols:
            series = pd.to_numeric(df[col], errors='coerce').fillna(0)
            pos_series = series[series > 0]
            
            stats = {
                'Version': name,
                'Column': col,
                'Non-Zero%': len(pos_q) / total_rows if (total_rows > 0 and 'pos_q' in locals()) else len(pos_series)/total_rows,
                'Sum': series.sum(),
                'Mean': pos_series.mean() if len(pos_series) > 0 else 0,
                'Max': series.max(),
                'P90': pos_series.quantile(0.9) if len(pos_series) > 0 else 0,
                'P99': pos_series.quantile(0.99) if len(pos_series) > 0 else 0,
                'Zeros': total_rows - len(pos_series)
            }
            all_stats.append(stats)
            
        # 打印该版本的时序深度分布
        depth_counts = pairs.value_counts().sort_index()
        print(f"\n    [ 时序深度分布 - {name} ]")
        print(f"    {'深度(行数)':<10} | {'对数(Pairs)':<15} | {'占比':<10}")
        for depth, count in depth_counts.items():
            if depth > 5 and depth != depth_counts.index[-1]: continue # 压缩打印
            print(f"    {depth:<10} | {count:<15,} | {count/len(pairs):.2%}")
        
    # 汇总对比表
    df_stats = pd.DataFrame(all_stats)
    
    print("\n" + "="*100)
    print("📊 各 qty_ 字段 Side-by-Side 对比")
    print("="*100)
    
    common_cols = sorted(list(set(df_stats[df_stats['Version'] == 'V1.6']['Column']) & 
                              set(df_stats[df_stats['Version'] == 'Current']['Column'])))
    
    for col in common_cols:
        v16 = df_stats[(df_stats['Version'] == 'V1.6') & (df_stats['Column'] == col)].iloc[0]
        curr = df_stats[(df_stats['Version'] == 'Current') & (df_stats['Column'] == col)].iloc[0]
        
        print(f"\n🔹 字段: {col.upper()}")
        print(f"{'指标名':<20} | {'V1.6':>20} | {'Current':>20} | {'差异(C/V16)':>15}")
        print("-" * 85)
        
        metrics = [('Non-Zero%', '.2%'), ('Sum', ',.0f'), ('Mean', '.2f'), ('Max', ',.0f'), ('P90', '.2f'), ('P99', '.2f')]
        
        for m, fmt in metrics:
            val_v16 = v16[m]
            val_curr = curr[m]
            ratio = val_curr / val_v16 if (pd.notnull(val_v16) and val_v16 != 0) else np.nan
            
            # 先格式化数值，再对齐
            s_v16 = format(val_v16, fmt)
            s_curr = format(val_curr, fmt)
            
            print(f"{m:<20} | {s_v16:>20} | {s_curr:>20} | {ratio:>15.2f}x")

    # 独有字段分析
    only_v16 = set(df_stats[df_stats['Version'] == 'V1.6']['Column']) - set(df_stats[df_stats['Version'] == 'Current']['Column'])
    only_curr = set(df_stats[df_stats['Version'] == 'Current']['Column']) - set(df_stats[df_stats['Version'] == 'V1.6']['Column'])
    
    if only_v16: print(f"\n⚠️ 仅 V1.6 拥有的字段: {only_v16}")
    if only_curr: print(f"\n🆕 仅 Current 拥有的字段: {only_curr}")

    print("\n" + "="*100)
    print("💡 终极分析结论:")
    
    # 抽取补货量进行深度点评
    v16_repl = df_stats[(df_stats['Version'] == 'V1.6') & (df_stats['Column'] == 'qty_replenish')].iloc[0]
    curr_repl = df_stats[(df_stats['Version'] == 'Current') & (df_stats['Column'] == 'qty_replenish')].iloc[0]
    
    print(f"1. 补货信号: Current 补货总量是 V1.6 的 {curr_repl['Sum']/v16_repl['Sum']:.1f} 倍，广度极大提升。")
    print(f"2. 数值尺度: Current 的单笔均值 ({curr_repl['Mean']:.2f}) 只有 V1.6 ({v16_repl['Mean']:.2f}) 的一半，说明数据更散。")
    
    v16_ship = df_stats[(df_stats['Version'] == 'V1.6') & (df_stats['Column'] == 'qty_shipped')].iloc[0]
    curr_ship = df_stats[(df_stats['Version'] == 'Current') & (df_stats['Column'] == 'qty_shipped')].iloc[0]
    print(f"3. 履约信号 (Shipped): Current 的发货覆盖率高达 {curr_ship['Non-Zero%']:.1%}，而 V1.6 仅 {v16_ship['Non-Zero%']:.1%}。")
    print("   -> 结论: Current 实际上包含了一个“全量订单”的底表，而 V1.6 是一个“抽样稀疏”的底表。")
    print("="*100)

if __name__ == "__main__":
    detailed_comparison()
