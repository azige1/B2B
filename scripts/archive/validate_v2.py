import pandas as pd
import numpy as np

# Load V2.3 Inference
inf_df = pd.read_csv('reports/daily_sku_orders_v2_2025-12-31.csv')

print('=== V2.3 MVP Inference Validation (2025-12-31) ===')
print(f'Total Valid Predictions (Qty > 0): {len(inf_df)}')
print(f'Total Pieces Recommended: {inf_df["ai_budget_qty_30d"].sum():.0f}')

print('\nDistribution of Predicted Quantities:')
print(inf_df['ai_budget_qty_30d'].describe())

# Check against historical performance
hist_df = pd.read_csv('data/gold/wide_table_sku.csv')
hist_agg = hist_df.groupby('sku_id')['qty_replenish'].sum().reset_index()

merged = pd.merge(inf_df, hist_agg, on='sku_id', how='left')
merged['hist_to_pred_ratio'] = merged['ai_budget_qty_30d'] / (merged['qty_replenish'] + 1)

print('\nTop 10 Predictions vs Historical Total Replenishment:')
top10 = merged.sort_values(by='ai_budget_qty_30d', ascending=False).head(10)[['sku_id', 'ai_budget_qty_30d', 'qty_replenish']]
print(top10.to_string(index=False))
