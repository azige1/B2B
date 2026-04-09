import pandas as pd

data_path = r'E:\LSTM\B2B\B2B_Replenishment_System\data\gold\wide_table_sku.csv'

with open('calc_result_2025.txt', 'w', encoding='utf-8') as f:
    f.write("Loading data...\n")
    df = pd.read_csv(data_path, usecols=['sku_id', 'date', 'qty_replenish'])

    # Filter for 2025 only
    df['date'] = pd.to_datetime(df['date'])
    df_2025 = df[df['date'].dt.year == 2025].copy()

    f.write(f"Total rows before aggregation in 2025 (Micro Level): {len(df_2025)}\n")
    if len(df_2025) == 0:
        f.write("No data found for 2025!\n")
    else:
        pos_before = (df_2025['qty_replenish'] > 0).sum()
        density_before = pos_before / len(df_2025) * 100
        f.write(f"Replenishment > 0 before aggregation: {pos_before} ({density_before:.4f}%)\n")
        f.write(f"Sparsity before aggregation: {100 - density_before:.4f}%\n\n")

        f.write("Aggregating by sku_id and date (Macro Level)...\n")
        agg_df = df_2025.groupby(['sku_id', 'date'])['qty_replenish'].sum().reset_index()

        f.write(f"Total rows after aggregation: {len(agg_df)}\n")
        pos_after = (agg_df['qty_replenish'] > 0).sum()
        density_after = pos_after / len(agg_df) * 100
        f.write(f"Replenishment > 0 after aggregation: {pos_after} ({density_after:.4f}%)\n")
        f.write(f"Sparsity after aggregation: {100 - density_after:.4f}%\n")
