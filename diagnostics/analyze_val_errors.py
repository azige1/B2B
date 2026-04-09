import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from src.train.run_training import DummyLE

report_path = os.path.join(PROJECT_ROOT, "reports", "backtest_20251019.csv")
data_path = os.path.join(PROJECT_ROOT, "data", "gold", "wide_table_sku.csv")

def analyze_old_items_accuracy():
    if not os.path.exists(report_path):
        print(f"❌ 找不到诊断底表: {report_path}")
        return

    print("=" * 60)
    print("🎯 B2B 智能补货系统 - 老款商品专属预测精度分析 (排爆冷启动)")
    print("=" * 60)

    # 1. 加载预测真值表
    df_rep = pd.read_csv(report_path)
    
    # 2. 我们通过一个简易的方法判定“老款”：
    # 如果系统给出了预测值(ai_budget_30d > 0) 或者该组合已经被我们上面验证过有历史
    # 这里我们直接剔除那些 120天0历史的绝对漏报（经过前面的 check_misses_history 验证，纯漏报=ai为0且历史为0）
    # 所以我们定义：只要 AI 有给出预测动作的（即突破了死区和低概率区的），或者本身属于我们核心盘口的，都算。
    # 更严谨的做法：直接关联宽表验证历史。
    
    print("\n[*] 正在关联 120 天历史宽表，剥离纯新款 (0历史)...")
    df_all = pd.read_csv(data_path)
    df_all['date'] = pd.to_datetime(df_all['date'])
    START = pd.Timestamp("2025-06-22")
    ANCHOR = pd.Timestamp("2025-10-19")
    df_hist = df_all[df_all["date"].between(START, ANCHOR)]
    
    hist_keys = set(zip(df_hist["buyer_id"], df_hist["sku_id"]))
    
    # 标记是否为老款
    df_rep["is_old"] = df_rep.apply(lambda row: (row["buyer_id"], row["sku_id"]) in hist_keys, axis=1)
    
    # 过滤出老款大盘
    df_old = df_rep[df_rep["is_old"]].copy()
    
    old_has_real = df_old[df_old["real_qty_30d"] > 0]
    total_real_old = old_has_real["real_qty_30d"].sum()
    total_pred_old = df_old["ai_budget_30d"].sum()
    
    print(f"\n✅ 剥离完成！")
    print(f"  📌 大盘总记录: {len(df_rep)} 组合")
    print(f"  📌 剥离纯新款: {len(df_rep) - len(df_old)} 组合 (不再关注它们)")
    print(f"  📌 聚焦老款组: {len(df_old)} 组合")
    
    print("\n" + "-" * 60)
    print("🔥 老款全局核心表现 🔥")
    print("-" * 60)
    print(f"  总真实需求件数: {total_real_old:.1f} 件")
    print(f"  AI预测的总件数: {total_pred_old:.1f} 件")
    print(f"  老款捕获 Ratio: {total_pred_old / total_real_old:.3f} (完美值为 1.0)")
    
    # 双阳 MAE (预测了也卖了)
    both_mask = (df_old["ai_budget_30d"] > 0) & (df_old["real_qty_30d"] > 0)
    if both_mask.sum() > 0:
        mae = (df_old.loc[both_mask, "ai_budget_30d"] - df_old.loc[both_mask, "real_qty_30d"]).abs().mean()
        print(f"  双阳单店误差 (MAE): {mae:.1f} 件")

    print("\n" + "-" * 60)
    print("📉 老款预测能力阶梯剖析 (按真实件数分层)")
    print("-" * 60)
    
    # 将真实销量分段
    bins = [-1, 0, 10, 50, 200, 9999]
    labels = ["滞销(0件)", "零星(1-10件)", "平销(11-50件)", "畅销(51-200件)", "大爆(>200件)"]
    df_old["real_tier"] = pd.cut(df_old["real_qty_30d"], bins=bins, labels=labels)
    
    tier_stats = df_old.groupby("real_tier").agg(
        组合数=("sku_id", "count"),
        总真实件数=("real_qty_30d", "sum"),
        总预测件数=("ai_budget_30d", "sum")
    ).reset_index()
    
    tier_stats["Ratio"] = (tier_stats["总预测件数"] / tier_stats["总真实件数"].replace(0, np.nan)).fillna(0).round(2)
    
    print(tier_stats.to_string(index=False))
    
    print("\n❓ 诊断结论：")
    print("如果滞销品(0件)预测偏多，说明阈值太低在乱发单（假阳性）；")
    print("如果平销品/畅销品 Ratio 极低，说明 LSTM 对于老款的走势过于保守（被 Loss 函数或者 expm1 给压制了）。")
    print("这决定了我们的下一步模型改进方案。")

if __name__ == "__main__":
    analyze_old_items_accuracy()
