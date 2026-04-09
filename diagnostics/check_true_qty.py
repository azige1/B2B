import pandas as pd
import numpy as np

file_path = "e:/LSTM/B2B/B2B_Replenishment_System/reports/v1.8_demo/detail_buyer_sku.csv"
print(f"Reading file: {file_path}")
try:
    df = pd.read_csv(file_path)
    total_len = len(df)
    print(f"Total rows: {total_len:,}")
    
    gt_zero = df[df['true_qty'] > 0]
    gt_zero_len = len(gt_zero)
    print(f"Rows where true_qty > 0: {gt_zero_len:,}")
    
    if total_len > 0:
        sparsity = (total_len - gt_zero_len) / total_len * 100
        print(f"Sparsity (Zero percentage): {sparsity:.4f}%")
        
    print(f"Max true_qty: {df['true_qty'].max()}")
    print(f"Sum true_qty: {df['true_qty'].sum()}")
    
    if gt_zero_len > 0:
        print("\nTop 5 rows with highest true_qty:")
        print(gt_zero.sort_values(by='true_qty', ascending=False).head(5))
except Exception as e:
    print(f"Error reading file: {e}")
