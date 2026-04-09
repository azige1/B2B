import pandas as pd
import numpy as np

def check():
    df = pd.read_csv('data/gold/wide_table_sku.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    unique_pairs = df[['buyer_id', 'sku_id']].drop_duplicates()
    num_pairs = len(unique_pairs)
    
    full_dates = sorted(list(df['date'].unique()))
    if len(full_dates) > 0:
        date_range = (full_dates[-1] - full_dates[0]).days + 1
    else:
        date_range = 0
        
    num_events = len(df)
    num_repl = len(df[df['qty_replenish'] > 0])
    
    # 模拟 sliding window
    # 假设每个 pair 在 (date_range - 120 - 30) 天内滑动
    slide_days = max(1, date_range - 120 - 30 + 1)
    total_windows = num_pairs * slide_days
    
    # 正样本大致数量 = 每次补货会落在它前 30 天的任意一个 anchor 的三十天窗口中
    # 所以每个补货事件，如果它位于边界内部，最多可以造就 30 个正样本窗口
    max_pos_windows = num_repl * 30
    
    print(f"唯一 (买手, SKU) 组合数: {num_pairs}")
    print(f"数据总跨度 (天): {date_range}")
    print(f"每个组合滑动次数: {slide_days}")
    print(f"总计大约生成的滑动窗口数 (无降采样): {total_windows}")
    print(f"最大可能的正样本窗口数 (qty>0): {max_pos_windows}")
    print(f"如果没有降采样，天然的正样本比例: 至少 {max_pos_windows/total_windows*100:.2f}% (因为多次补货可能重叠在一个窗口，实际稍低)")

if __name__ == "__main__":
    check()
