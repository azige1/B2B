import pandas as pd
import numpy as np

data_path = r'E:\LSTM\B2B\B2B_Replenishment_System\data\gold\wide_table_sku.csv'

with open('calc_result.txt', 'w', encoding='utf-8') as f:
    f.write("Loading data...\n")
    df = pd.read_csv(data_path, usecols=['sku_id', 'date', 'qty_replenish'])

    f.write(f"Total rows before aggregation (Micro Level): {len(df)}\n")
    pos_before = (df['qty_replenish'] > 0).sum()
    density_before = pos_before / len(df) * 100
    f.write(f"Replenishment > 0 before aggregation: {pos_before} ({density_before:.4f}%)\n")
    f.write(f"Sparsity before aggregation: {100 - density_before:.4f}%\n\n")

    f.write("Aggregating by sku_id and date (Macro Level)...\n")
    agg_df = df.groupby(['sku_id', 'date'])['qty_replenish'].sum().reset_index()

    f.write(f"Total rows after aggregation: {len(agg_df)}\n")
    pos_after = (agg_df['qty_replenish'] > 0).sum()
    density_after = pos_after / len(agg_df) * 100
    f.write(f"Replenishment > 0 after aggregation: {pos_after} ({density_after:.4f}%)\n")
    f.write(f"Sparsity after aggregation: {100 - density_after:.4f}%\n")
