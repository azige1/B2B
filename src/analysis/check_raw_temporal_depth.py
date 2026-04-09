import pandas as pd
import os

def check_raw_depth():
    raw_path = r'E:\LSTM\B2B\B2B_Replenishment_System\data_warehouse\fact_orders\V_IRS_ORDER_2025.csv'
    
    if not os.path.exists(raw_path):
        print(f"❌ 未找到原始文件: {raw_path}")
        return

    print(f"[*] 正在深度审计原始底表: {os.path.basename(raw_path)}")
    
    # 1. 加载关键列 (Store, SKU, Date)
    df = pd.read_csv(raw_path, usecols=['STORENAME', 'NO', 'BILLDATE'], low_memory=False)
    
    # 2. 统计基本面
    total_rows = len(df)
    unique_pairs = df.groupby(['STORENAME', 'NO']).size().shape[0]
    
    # 3. 计算“时序重复度”：同一个对，在不同日期出现的次数
    # 我们看 (STORENAME, NO) 分组后，BILLDATE 的唯一值数量
    date_depth = df.groupby(['STORENAME', 'NO'])['BILLDATE'].nunique()
    
    print("\n" + "="*60)
    print("📊 原始底表 (Store + SKU) 时序深度透视")
    print("="*60)
    print(f"总行数:               {total_rows:,}")
    print(f"唯一 (买手, SKU) 对数:  {unique_pairs:,}")
    print("-" * 60)
    
    # 深度统计
    depth_dist = date_depth.value_counts().sort_index()
    print(f"{'日期深度(天数)':<15} | {'组合数(Pairs)':<15} | {'占比':<10}")
    print("-" * 50)
    for depth, count in depth_dist.items():
        print(f"{depth:<15} | {count:<15,} | {count/unique_pairs:.2%}")
    
    print("-" * 50)
    print(f"结论记录：")
    if depth_dist.get(1, 0) / unique_pairs > 0.99:
        print("💡 铁证: 超过 99% 的 (买手, SKU) 在全年 365 天里【仅在某一天】留下了记录。")
        print("   这意味着对于 AI 来说，它看不到这个 SKU 补货的‘频率’和‘过程’。")
    else:
        print(f"💡 发现有 {unique_pairs - depth_dist.get(1,0):,} 对组合具备多日流水。")

    # 4. 随机抽查一个“有日期但没深度”的例子
    if not date_depth.empty:
        sample_pair = date_depth[date_depth == 1].index[0]
        print(f"\n[ 典型单点核查 ]")
        print(f"组合: {sample_pair}")
        print(df[(df['STORENAME'] == sample_pair[0]) & (df['NO'] == sample_pair[1])])

if __name__ == "__main__":
    check_raw_depth()
