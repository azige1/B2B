import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
report_path = os.path.join(PROJECT_ROOT, "reports", "backtest_20251019.csv")
data_path = os.path.join(PROJECT_ROOT, "data", "gold", "wide_table_sku.csv")

if not os.path.exists(report_path):
    print("找不到测试报告，验证中止。")
    exit(1)

print("[*] 加载合并预测真值表...")
df_rep = pd.read_csv(report_path)

# 抓出 515 个纯漏报（AI=0，但真实现实补货的）
misses = df_rep[(df_rep["ai_budget_30d"] == 0) & (df_rep["real_qty_30d"] > 0)]
print(f"[*] 提取出漏报模型组合: {len(misses)} 个")

if len(misses) == 0:
    print("没有漏报数据，中止。")
    exit(0)

print("\n[*] 加载系统底盘事实大表...")
df_all = pd.read_csv(data_path)
df_all['date'] = pd.to_datetime(df_all['date'])

# 模拟那个预测切点：2025-06-22 ~ 2025-10-19 
START = pd.Timestamp("2025-06-22")
ANCHOR = pd.Timestamp("2025-10-19")
df_hist = df_all[df_all["date"].between(START, ANCHOR)]

print(f"[*] 切出 120 天全量历史骨架: {len(df_hist)} 条日志")

# 统计这些买手-sku在纯历史期的销售/补货动作是否为 0
miss_keys = set(zip(misses["buyer_id"], misses["sku_id"]))

# 现在看看历史大表里，包含这些 key 的总活跃度
df_hist_misses = df_hist[df_hist.set_index(['buyer_id', 'sku_id']).index.isin(miss_keys)]

if len(df_hist_misses) == 0:
    print("\n✅ 铁证如山：在大表的过去 120 天内，这 515 个漏报组合的出现次数为 0！它们属于纯粹的冷启动脱档。")
else:
    print(f"\n❌ 等等！发现不妥！这 515 个漏报组合里，有 {len(df_hist_misses)} 条历史日志！代码某处把有历史的也当 0 了！")
    agg_hist = df_hist_misses.groupby(['buyer_id', 'sku_id'])['qty_replenish'].sum().reset_index()
    print("这是明明有宽表日志却漏测的组合：")
    print(agg_hist.head(10))
