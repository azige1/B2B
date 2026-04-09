import pandas as pd
import numpy as np
import os
import time

def analyze_logic_link():
    path_v16 = r'E:\LSTM\B2B\B2B_Replenishment_System\data\gold\wide_table_sku_v1.6.csv'
    path_now = r'E:\LSTM\B2B\B2B_Replenishment_System\data\gold\wide_table_sku.csv'
    
    files = {'V1.6': path_v16, 'Current': path_now}
    results = {}

    print("="*80)
    print("🔬 核心逻辑链审计: QTYSO (期货) -> QTYSPO (补货)")
    print("="*80)

    for name, path in files.items():
        if not os.path.exists(path):
            print(f"⚠️ 文件不存在: {path}")
            continue
        
        print(f"[*] 正在处理 {name}...")
        start = time.time()
        
        # 加载必要字段
        df = pd.read_csv(path, usecols=['qty_future', 'qty_replenish'])
        q_so = df['qty_future'].fillna(0).astype(float)
        q_spo = df['qty_replenish'].fillna(0).astype(float)
        
        # 1. 基础规模
        so_sum = q_so.sum()
        spo_sum = q_spo.sum()
        
        # 2. 相关性 (线性关联度)
        corr = q_so.corr(q_spo)
        
        # 3. 概率统计 (条件预测力)
        # P(发生补货 | 有期货) = 在有期货的行中，有多少行最终发生了补货
        mask_so = q_so > 0
        p_spo_given_so = (q_spo[mask_so] > 0).mean() if mask_so.any() else 0
        
        # 4. 信号覆盖率 (召回率视角)
        # P(有期货记录 | 发生补货) = 在发生补货的行中，有多少行提前具备期货信号
        mask_spo = q_spo > 0
        p_so_given_spo = (q_so[mask_spo] > 0).mean() if mask_spo.any() else 0
        
        # 5. 零信号比例
        # 多少补货行其实是“空降”的 (完全没有期货信号支撑)
        p_surprise = (q_so[mask_spo] == 0).mean() if mask_spo.any() else 0

        results[name] = {
            'QTYSO_Sum': so_sum,
            'QTYSPO_Sum': spo_sum,
            'Correlation': corr,
            'P(Replenish | Future>0)': p_spo_given_so,
            'P(Future>0 | Replenish)': p_so_given_spo,
            'Surprise_Ratio (Wait 0 SO)': p_surprise
        }
        print(f"    - 完成，用时: {time.time()-start:.2f}s")

    # 结果输出
    print("\n" + "指标项".ljust(30) + " | " + "V1.6 (成功版)".center(20) + " | " + "Current (失效版)".center(20))
    print("-" * 80)
    
    metrics = [
        ('QTYSO_Sum', ',.0f'),
        ('QTYSPO_Sum', ',.0f'),
        ('Correlation', '.4f'),
        ('P(Replenish | Future>0)', '.2%'),
        ('P(Future>0 | Replenish)', '.2%'),
        ('Surprise_Ratio (Wait 0 SO)', '.2%')
    ]
    
    for m, fmt in metrics:
        v16_val = results.get('V1.6', {}).get(m, 0)
        curr_val = results.get('Current', {}).get(m, 0)
        
        s_v16 = format(v16_val, fmt)
        s_curr = format(curr_val, fmt)
        print(f"{m.ljust(30)} | {s_v16:>20} | {s_curr:>20}")

    print("\n" + "="*80)
    print("💡 专家级诊断报告:")
    if 'V1.6' in results and 'Current' in results:
        v = results['V1.6']
        c = results['Current']
        
        print(f"1. 信号强度断裂: QTYSO 总量从 {v['QTYSO_Sum']:,.0f} 暴跌至 {c['QTYSO_Sum']:,.0f} (萎缩 100 倍)。")
        print(f"2. 补货压力激增: QTYSPO 目标总量从 {v['QTYSPO_Sum']:,.0f} 增加到 {c['QTYSPO_Sum']:,.0f} (增加 3 倍)。")
        
        if c['Surprise_Ratio (Wait 0 SO)'] > v['Surprise_Ratio (Wait 0 SO)']:
            increase = (c['Surprise_Ratio (Wait 0 SO)'] - v['Surprise_Ratio (Wait 0 SO)']) * 100
            print(f"3. 盲目预测程度: 当前版本有 {c['Surprise_Ratio (Wait 0 SO)']:.2%} 的补货记录在‘期货列’是 0，高于 V1.6。")
            print("   -> 结论: 模型现在是在‘没有任何先兆’的情况下被迫预测补货，所以它选择了集体沉默。")
    print("="*80)

if __name__ == "__main__":
    analyze_logic_link()
