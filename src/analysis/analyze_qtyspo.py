import pandas as pd
import numpy as np
import os
import time

def analyze_qtyspo():
    file_path = r'E:\LSTM\B2B\B2B_Replenishment_System\data_warehouse\fact_orders\V_IRS_ORDER_2025.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    print("="*60)
    print(f"🚀 开始分析 QTYSPO 字段: {os.path.basename(file_path)}")
    print("="*60)

    start_time = time.time()
    
    # 只需要读取部分核心列以节省内存和提高速度
    relevant_cols = ['BILLDATE', 'STORENAME', 'NAME', 'QTYSPO', 'TOTAL_QTYOUT', 'QTYREM']
    
    try:
        # low_memory=False 防止分块读取时的类型推断冲突
        df = pd.read_csv(file_path, usecols=relevant_cols, low_memory=False)
        end_time = time.time()
        print(f"[*] 数据加载完成，用时: {end_time - start_time:.2f}s")
        print(f"[*] 总行数: {len(df):,}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        # 如果列名不对，尝试只读取一两行看列名
        df_head = pd.read_csv(file_path, nrows=0)
        print(f"[*] 现有列名: {list(df_head.columns)}")
        return

    # 数据预处理
    df['BILLDATE'] = pd.to_datetime(df['BILLDATE'].astype(str), format='%Y%m%d', errors='coerce')
    
    # 分析 QTYSPO
    q_all = pd.to_numeric(df['QTYSPO'], errors='coerce').fillna(0)
    pos_q = q_all[q_all > 0]
    
    print("\n📊 [QTYSPO 统计摘要]")
    print(f"   - 非零行数: {len(pos_q):,}")
    print(f"   - 非零比例: {len(pos_q)/len(df):.4%}")
    print(f"   - 数值总量: {q_all.sum():,.0f}")
    if len(pos_q) > 0:
        print("\n   [分布详情]")
        print(pos_q.describe(percentiles=[.25, .5, .75, .9, .95, .99]))

    # 分析 TOTAL_QTYOUT (发货量)
    if 'TOTAL_QTYOUT' in df.columns:
        out_all = pd.to_numeric(df['TOTAL_QTYOUT'], errors='coerce').fillna(0)
        pos_out = out_all[out_all > 0]
        print("\n📊 [TOTAL_QTYOUT 统计摘要]")
        print(f"   - 非零行数: {len(pos_out):,}")
        print(f"   - 数值总量: {out_all.sum():,.0f}")

    # 月度趋势对比
    print("\n📈 [月度趋势对比 (Sum)]")
    df['QTYSPO_num'] = q_all
    df['OUT_num'] = pd.to_numeric(df['TOTAL_QTYOUT'], errors='coerce').fillna(0) if 'TOTAL_QTYOUT' in df.columns else 0
    
    monthly = df.groupby(df['BILLDATE'].dt.to_period('M'))[['QTYSPO_num', 'OUT_num']].sum()
    print(monthly.to_string())

    # 交叉检查: 在同一笔订单中，QTYSPO 和其他字段的关系
    print("\n🔍 [字段关联性抽样 (前 10 行非零记录)]")
    mask = (q_all > 0)
    print(df[mask][['BILLDATE', 'STORENAME', 'NAME', 'QTYSPO', 'TOTAL_QTYOUT', 'QTYREM']].head(10).to_string())

    print("\n" + "="*60)
    print(f"✅ 分析结束，总耗时: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    analyze_qtyspo()
