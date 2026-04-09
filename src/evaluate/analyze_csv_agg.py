import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(PROJECT_ROOT, "reports", "validation_strategy_details.csv")

def analyze_agg():
    print("=" * 60)
    print("🔍 B2B V1.8 深度分析：[SKU 全国大盘物理件数对齐验证]")
    print(f"数据源: {csv_path}")
    print("=" * 60)

    if not os.path.exists(csv_path):
        print(f"❌ 未找到底表 {csv_path}。请先运行 evaluate_dashboard.py。")
        return

    df = pd.read_csv(csv_path)
    
    # 将全国分散在不同买手的同一件衣服的 真实件数 和 AI预测件数 加起来
    df_agg = df.groupby('sku_name').agg({
        'true_qty': 'sum',
        'pred_qty': 'sum'
    }).reset_index()

    # 计算聚合并合并后的绝对误差和均方根误差
    df_agg['abs_err'] = np.abs(df_agg['true_qty'] - df_agg['pred_qty'])
    df_agg['sq_err'] = df_agg['abs_err'] ** 2
    
    mae = df_agg['abs_err'].mean()
    rmse = np.sqrt(df_agg['sq_err'].mean())
    ratio = df_agg['pred_qty'].sum() / (df_agg['true_qty'].sum() + 1e-5)
    
    print(f"\n[*] 聚合计算完成. 全国覆盖 SKU 总数: {len(df_agg):,}")
    print("-" * 60)
    print(f"➤ 总大盘安全水比 (Ratio): {ratio:.3f}")
    print(f"➤ 全国单一爆款平均预测漂移 (MAE) : {mae:.2f} 件/款")
    print(f"➤ 全国单一爆款极值惩罚误差 (RMSE): {rmse:.2f} 件/款")
    print("-" * 60)
    
    print("\n🔥 【严重失真榜单 Top 5】(断货引起的防守失能)")
    df_bad = df_agg.sort_values('abs_err', ascending=False).head(5)
    for idx, row in df_bad.iterrows():
        print(f"SKU: {row['sku_name']:<18} | 真实爆量: {int(row['true_qty']):<5} | AI保守给库: {int(row['pred_qty']):<5} | 断层差值: {int(row['abs_err'])}")
        
    print("\n🎯 【完美阻击榜单 Top 5】(总仓吞吐极其丝滑)")
    # 找真实销量很大，但误差极小的神预判
    df_good = df_agg[(df_agg['true_qty'] > 1000)].sort_values('abs_err').head(5)
    for idx, row in df_good.iterrows():
        print(f"SKU: {row['sku_name']:<18} | 真实爆量: {int(row['true_qty']):<5} | AI精准给库: {int(row['pred_qty']):<5} | 精准差值: {int(row['abs_err'])}")

    print("\n✅ 第 4 个核心测试引擎 `analyze_csv_agg.py` 诊断完毕。")

if __name__ == "__main__":
    analyze_agg()
