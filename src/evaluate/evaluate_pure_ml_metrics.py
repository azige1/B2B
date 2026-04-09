import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sys.stdout = open("pure_ml_metrics.txt", "w", encoding="utf-8")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv_path = os.path.join(PROJECT_ROOT, "reports", "validation_strategy_details.csv")

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    计算 sMAPE (对称平均绝对百分比误差)
    解决传统 MAPE 遇到真实值为 0 时爆炸的问题，范围 0% ~ 200%
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # 避免除以 0
    diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1.0, denominator)
    # 对于分母本来就是 0 的地方（双 0预测），误差设为 0
    diff[denominator == 0] = 0.0
    return np.mean(diff) * 100

def evaluate_pure_ml_metrics():
    print("=" * 80)
    print("🧪 B2B V1.8 纯机器学习客观度量：【模型张量拟合精度大审判】")
    print("=" * 80)

    if not os.path.exists(csv_path):
        print(f"❌ 未找到底表 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # 纯数据科学视角：只考察那些有波动的样本（抛弃大量静态的 0-0 无效对局）
    df_eval = df[(df['true_qty'] > 0) | (df['ai_pred_qty'] > 0)].copy()

    y_true = df_eval['true_qty'].values
    y_pred = df_eval['ai_pred_qty'].values

    # ---------------------------------------------------------
    # 核心统计学三大件 (The Big Three)
    # ---------------------------------------------------------
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred) # 衡量模型对数据的方差解释力
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)

    print("\n[PART 1: 标准回归度量仪 (Standard Regression Metrics)]")
    print(f"  ➤ 参评有效活跃矩阵数 (Active Tensors): {len(df_eval):,}")
    print(f"  ➤ 均方根误差 (RMSE): {rmse:.4f} (对大偏差极度敏感，越小越贴合曲线)")
    print(f"  ➤ 平均绝对误差 (MAE) : {mae:.4f} (模型预测点偏离真实点的平均绝对物理距离)")
    print(f"  ➤ 对称百分比误差 (sMAPE): {smape:.2f}% (模型在数量级上的缩放稳定性，0%最完美)")
    print(f"  ➤ 拟合优度 (R-squared): {r2:.4f} (范围通常为0~1，负数说明拟合烂穿)")

    # ---------------------------------------------------------
    # 分箱探测：在不同的真实分布曲面上的性能 (Decile Analysis)
    # ---------------------------------------------------------
    print("\n[PART 2: 分层探针测试 (Stratified Error Analysis)]")
    print("  观察模型对于不同量级的信号，其解析力和失真度是如何变化的：")
    print("-" * 80)
    print(f"{'真实需求量级区间 (True Target)':<28} | {'样本量':<8} | {'模型平均推断值 (Mean Pred)':<22} | {'分箱 MAE':<10}")
    print("-" * 80)
    
    bins = [
        ("0 件 (静默幽灵，理应休眠)", df_eval['true_qty'] == 0),
        ("1 ~ 5 件 (底层低频噪声)", (df_eval['true_qty'] >= 1) & (df_eval['true_qty'] <= 5)),
        ("6 ~ 20 件 (腰部常委，时序主力)", (df_eval['true_qty'] > 5) & (df_eval['true_qty'] <= 20)),
        ("21 ~ 100 件 (中上层震荡波)", (df_eval['true_qty'] > 20) & (df_eval['true_qty'] <= 100)),
        ("> 100 件 (极值爆点，离群值)", df_eval['true_qty'] > 100)
    ]

    for label, condition in bins:
        subset = df_eval[condition]
        if len(subset) == 0:
            continue
            
        sub_true = subset['true_qty']
        sub_pred = subset['ai_pred_qty']
        
        sub_mae = mean_absolute_error(sub_true, sub_pred)
        sub_mean_pred = sub_pred.mean()
        
        print(f"{label:<28} | {len(subset):<8} | {sub_mean_pred:<22.2f} | {sub_mae:<10.2f}")

    # ---------------------------------------------------------
    # 散点收敛度分析 (Under-forecast vs Over-forecast)
    # ---------------------------------------------------------
    print("\n[PART 3: 向量阻尼偏离度 (Bias Direction Analysis)]")
    # 预测比真实低了 (低估，偏截断属性)
    under_mask = y_pred < y_true
    # 预测比真实高了 (高估，偏噪点激化)
    over_mask = y_pred > y_true

    under_mae = y_true[under_mask] - y_pred[under_mask]
    over_mae = y_pred[over_mask] - y_true[over_mask]

    print(f"  ➤ 模型陷入【迟钝/极值截断】的样本占比: {(under_mask.sum()/len(df_eval))*100:.1f}% (平均偏小: {under_mae.mean() if len(under_mae)>0 else 0:.2f}件)")
    print(f"  ➤ 模型陷入【过敏/噪点放散】的样本占比: {(over_mask.sum()/len(df_eval))*100:.1f}% (平均偏大: {over_mae.mean() if len(over_mae)>0 else 0:.2f}件)")

    print("\n" + "=" * 80)
    print("科学结论 (单纯基于数据拟合表现)：")
    
    if r2 < 0:
        print("判定: [异常破窗] R² 击穿负数，模型在全局层面完全无法解释当前离散极值的方差。")
    elif smape > 150:
         print("判定: [量级撕裂] sMAPE 过高，模型经常性在 0 和 大数 之间产生对立级预测崩塌。")
    print(f"核心弱点: {under_mae.mean() if len(under_mae)>0 else 0:.1f} 件的高昂“低估惩罚均值”说明，模型受限于 Clamp 和 零膨胀屏蔽，被砍断了捕获长尾大值（>100）的触角。")
    print("=" * 80)

if __name__ == "__main__":
    evaluate_pure_ml_metrics()
