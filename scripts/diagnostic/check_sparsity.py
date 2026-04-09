import pandas as pd
import numpy as np

data_path = r'E:\LSTM\B2B\B2B_Replenishment_System\data\gold\wide_table_sku.csv'

print("Loading data...")
df = pd.read_csv(data_path, usecols=['sku_id', 'date', 'qty_replenish'])

print(f"Total rows before aggregation (Micro Level): {len(df)}")
pos_before = (df['qty_replenish'] > 0).sum()
density_before = pos_before / len(df) * 100
print(f"Replenishment > 0 before aggregation: {pos_before} ({density_before:.4f}%)")
print(f"Sparsity before aggregation: {100 - density_before:.4f}%\n")

print("Aggregating by sku_id and date (Macro Level)...")
agg_df = df.groupby(['sku_id', 'date'])['qty_replenish'].sum().reset_index()

print(f"Total rows after aggregation: {len(agg_df)}")
pos_after = (agg_df['qty_replenish'] > 0).sum()
density_after = pos_after / len(agg_df) * 100
print(f"Replenishment > 0 after aggregation: {pos_after} ({density_after:.4f}%)")
print(f"Sparsity after aggregation: {100 - density_after:.4f}%")
