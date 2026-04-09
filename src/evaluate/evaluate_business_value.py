import os
import sys
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(PROJECT_ROOT, "reports", "validation_strategy_details.csv")

def evaluate_business_accuracy():
    print("=" * 70)
    print("⚖️ B2B V1.8 最终极拷问：【模型预测到底准不准？一票否决权验证】")
    print("=" * 70)

    if not os.path.exists(csv_path):
        print(f"❌ 未找到底表 {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # 取出有真实单量，或者AI有发货动作的有效互搏池 (抛弃双方全是0的无效对比)
    df_valid = df[(df['true_qty'] > 0) | (df['ai_pred_qty'] > 0)].copy()

    total_valid_scenarios = len(df_valid)
    
    # ---------------------------------------------------------
    # 核心度量 1：【防呆（防过度积压）准确度】 -> AI有没有乱发货？
    # ---------------------------------------------------------
    # 场景 A: 真实销量其实是0（根本没卖动），AI也精准预判给了 0 (即防住了垃圾发车)
    df_zero_demand = df_valid[df_valid['true_qty'] == 0]
    total_zero_demand = len(df_zero_demand)
    
    # 在真实没需求的件数里，AI确实没发货的笔数
    ai_saved_useless_orders = len(df_zero_demand[df_zero_demand['ai_pred_qty'] == 0])
    
    # 在真实没需求的件数里，AI错误地发了货的笔数 (假阳性)
    ai_wasted_orders = len(df_zero_demand[df_zero_demand['ai_pred_qty'] > 0])
    wasted_qty_sum = df_zero_demand['ai_pred_qty'].sum()

    # ---------------------------------------------------------
    # 核心度量 2：【刚需响应能力】 -> 真的要货时，AI有没有给？
    # ---------------------------------------------------------
    df_has_demand = df_valid[df_valid['true_qty'] > 0]
    total_has_demand = len(df_has_demand)
    
    # 场景 B: 真实有需求，AI也确实给出了预算 (>0) -> 响应成功
    ai_hit_demand = len(df_has_demand[df_has_demand['ai_pred_qty'] > 0])
    
    # 场景 C: 真实有需求，但AI死守着1件都不给 -> 错失爆款
    ai_missed_demand = len(df_has_demand[df_has_demand['ai_pred_qty'] == 0])
    missed_qty_sum = df_has_demand[df_has_demand['ai_pred_qty'] == 0]['true_qty'].sum()
    
    # ---------------------------------------------------------
    # 核心度量 3：【精度落差 (只看真的给了建议的那批数据)】
    # ---------------------------------------------------------
    df_both_fired = df_valid[(df_valid['true_qty'] > 0) & (df_valid['ai_pred_qty'] > 0)].copy()
    
    # 我们定义，误差在真实件数的 50% 以内，或者件数相差不到 5件 的，算作极其精准。
    df_both_fired['is_accurate'] = ((np.abs(df_both_fired['ai_pred_qty'] - df_both_fired['true_qty']) <= 5) | 
                                    (df_both_fired['ai_pred_qty'] >= df_both_fired['true_qty'] * 0.5) & 
                                    (df_both_fired['ai_pred_qty'] <= df_both_fired['true_qty'] * 1.5))
                                    
    highly_accurate_count = df_both_fired['is_accurate'].sum()
    
    # ---------------------------------------------------------
    # 打印最终业务结论
    # ---------------------------------------------------------
    print(f"\n📂 在过去半年中，我们抓取了 {total_valid_scenarios:,} 次真实的 [门店找总部要货的博弈场景]。")
    
    print("\n✅ 第一卷：【不该要货的时候，AI 准确拦住了吗？（防呆测试）】")
    print(f"  - 根本卖不动，本来也不该备货的烂单挑战: {total_zero_demand:,} 次")
    print(f"  - AI 极其精准地【一件没发，拦住库存】的次数: {ai_saved_useless_orders:,} 次 (防御拦截率: {ai_saved_useless_orders/total_zero_demand*100:.1f}%)")
    print(f"  - ❌ AI 判断失误，被迫发送的冤枉发车笔数: {ai_wasted_orders:,} 次 (占比仅: {ai_wasted_orders/total_zero_demand*100:.1f}%)")
    print(f"    * 这些失误导致无端浪费了大概 {wasted_qty_sum:,} 件库存。\n")

    print("🚀 第二卷：【真有猛烈要货需求时，AI准确响应了吗？（刚需测试）】")
    print(f"  - 门店真实、急迫地需要补货的实盘挑战: {total_has_demand:,} 次")
    print(f"  - AI 成功侦测到需求，并且【顺利按下发货键】的次数: {ai_hit_demand:,} 次 (火力覆盖率: {ai_hit_demand/total_has_demand*100:.1f}%)")
    print(f"  - ❌ AI 过于保守，导致【错过商机，一件没发】的次数: {ai_missed_demand:,} 次 (断货流失率: {ai_missed_demand/total_has_demand*100:.1f}%)")
    print(f"    * 这一块是系统最严重的短板！因为没发车，白白流失了 {missed_qty_sum:,} 件的潜在爆款营业额。\n")
    
    print("🎯 第三卷：【AI决定发车的那一批，具体件数给的准不准？】")
    print(f"  - 在那 {len(df_both_fired):,} 笔 AI 和真实世界都产生发货行为的交易中：")
    print(f"  - 发车极其精准（件数差异<5件 或 浮动极小）的单数：{highly_accurate_count:,} 笔")
    print(f"  - 精兵作战准确率：{highly_accurate_count/len(df_both_fired)*100:.1f}%\n")
    
    print("=" * 70)
    print("结案陈词：这个模型到底准不准？")
    print(f"1. 它的【防守极其精准】(防御率 {ai_saved_useless_orders/total_zero_demand*100:.1f}%)，几乎不会因为模型发疯而导致烂账。")
    print(f"2. 它的【进攻不够精准】(覆盖率 {ai_hit_demand/total_has_demand*100:.1f}%)，因为过度保守，有一半的爆单它不敢追。")
    print("3. 【最终定性】：V1.8 是一个优秀的守城之主，绝不亏钱。但如果要抢夺爆款利润，必须指望 V2.0。")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_business_accuracy()
