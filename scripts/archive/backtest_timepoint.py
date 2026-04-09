"""
B2B 补货系统 V2.3 -- 时间点回测脚本 (backtest_timepoint.py)
==============================================================
核心思路: 站在历史某天 T (如 2025-10-31), 用 [T-59, T] 共 60 天的全量历史
         模拟"今天就是那一天"，执行推断，然后下拉 [T+1, T+30] 真实数据进行
         逐 SKU 精度对比——这是最直观、最有业务说服力的验证方式。

使用方法:
  python backtest_timepoint.py --date 2025-10-31
  python backtest_timepoint.py --date 2025-11-30
  python backtest_timepoint.py  # 默认测试两个日期并对比

输出:
  - 控制台打印 SKU 级对比 + 大盘指标
  - reports/backtest_{DATE}.csv  详细逐 SKU 比对文件
"""
import sys
import os
import json
import pickle
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
warnings.filterwarnings('ignore')

from src.models.enhanced_model import EnhancedTwoTowerLSTM

# 强制 UTF-8 输出 (兼容 Windows GBK 终端)
sys.stdout.reconfigure(encoding='utf-8')

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v3')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_v3')
GOLD_DIR      = os.path.join(PROJECT_ROOT, 'data', 'gold')
REPORTS_DIR   = os.path.join(PROJECT_ROOT, 'reports')
LOOKBACK      = 90
FORECAST      = 30

class DummyLE:
    classes_ = np.arange(13)

def sep(title='', char='=', width=66):
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{char*side} {title} {char*side}\n")
    else:
        print(char * width)

def load_model(device, cat_cols, num_cols, encoders):
    """初始化并加载 V2 模型权重"""
    static_vocab_sizes = {c: len(encoders[c].classes_) + 5 for c in cat_cols if c in encoders}
    static_vocab_sizes['month'] = 18
    model = EnhancedTwoTowerLSTM(
        dyn_feat_dim=7,
        static_vocab_sizes=static_vocab_sizes,
        num_numeric_feats=len(num_cols),
        lstm_hidden=256
    ).to(device)
    st = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
    model.load_state_dict(st)
    model.eval()
    return model

def rolling_log1p(arr, window, n):
    cum = np.cumsum(np.concatenate([[0.0], arr]))
    res = np.zeros(n, dtype=np.float32)
    for j in range(n):
        i_end = j + 1
        i_sta = max(0, i_end - window)
        res[j] = cum[i_end] - cum[i_sta]
    return np.log1p(np.maximum(res, 0))

def run_backtest_for_date(anchor_date_str, df_full, model, device, meta, encoders, scaler):
    """
    站在 anchor_date 做推断，对比 [anchor+1, anchor+30] 的真实补货
    Returns:
        result_df: 含 sku_id / pred_qty / actual_qty 的 DataFrame
    """
    anchor = pd.to_datetime(anchor_date_str).date()
    cat_cols = meta['static_cat_cols']
    num_cols = meta['static_num_cols']
    static_keys = cat_cols + ['month'] + num_cols

    sep(f"回测时间点: T = {anchor} (站在此日做预测)")

    # 1. 切出 [T-59, T] 输入窗口
    train_start = pd.to_datetime(anchor - pd.Timedelta(days=LOOKBACK-1)).date()
    df_hist = df_full[(df_full['date'] >= train_start) & (df_full['date'] <= anchor)].copy()
    print(f"  输入特征窗口: [{train_start}] ~ [{anchor}]  ({len(df_hist)} 行)")

    # 2. 切出 [T+1, T+30] 真实标签
    label_start = pd.to_datetime(anchor + pd.Timedelta(days=1)).date()
    label_end   = pd.to_datetime(anchor + pd.Timedelta(days=FORECAST)).date()
    df_label = df_full[(df_full['date'] >= label_start) & (df_full['date'] <= label_end)].copy()
    actual_agg = df_label.groupby('sku_id')['qty_replenish'].sum().reset_index()
    actual_agg.columns = ['sku_id', 'actual_qty_30d']
    print(f"  真实标签窗口: [{label_start}] ~ [{label_end}]  ({len(df_label)} 行)")

    # 3. 构建动态特征矩阵
    dyn_agg = df_hist.groupby(['sku_id', 'date']).agg({
        'qty_replenish': 'sum',
        'qty_debt':      'sum',
        'qty_shipped':   'sum',
        'qty_inbound':   'sum'
    }).reset_index()

    static_sub = df_hist[cat_cols + num_cols].drop_duplicates('sku_id').copy()
    static_sub['orig_sku'] = static_sub['sku_id'].astype(str)

    all_dates = pd.date_range(end=anchor, periods=LOOKBACK, freq='D').date
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    sku_list = sorted(dyn_agg['sku_id'].unique())
    sku_to_i = {s: i for i, s in enumerate(sku_list)}

    dyn_matrix = np.zeros((len(sku_list), LOOKBACK, 4), dtype=np.float32)
    for row in dyn_agg.itertuples(index=False):
        si = sku_to_i.get(row.sku_id, -1)
        di = date_to_idx.get(row.date, -1)
        if si >= 0 and di >= 0:
            dyn_matrix[si, di, 0] = row.qty_replenish
            dyn_matrix[si, di, 1] = row.qty_debt
            dyn_matrix[si, di, 2] = row.qty_shipped
            dyn_matrix[si, di, 3] = row.qty_inbound

    ts_feat = np.zeros((len(sku_list), LOOKBACK, 7), dtype=np.float32)
    for si in range(len(sku_list)):
        r = dyn_matrix[si, :, 0]
        d = dyn_matrix[si, :, 1]
        s = dyn_matrix[si, :, 2]
        n = dyn_matrix[si, :, 3]
        ts_feat[si, :, 0] = np.log1p(np.maximum(r, 0))
        ts_feat[si, :, 1] = rolling_log1p(r, 7,  LOOKBACK)
        ts_feat[si, :, 2] = rolling_log1p(r, 30, LOOKBACK)
        ts_feat[si, :, 3] = np.log1p(np.maximum(d, 0))
        ts_feat[si, :, 4] = np.log1p(np.maximum(s, 0))
        ts_feat[si, :, 5] = rolling_log1p(s, 7,  LOOKBACK)
        ts_feat[si, :, 6] = np.log1p(np.maximum(n, 0))

    # 4. 构建静态特征字典
    anchor_month = anchor.month
    static_dict = {}
    for _, row in static_sub.iterrows():
        arr = []
        for c in cat_cols:
            val = str(row[c]) if pd.notnull(row[c]) else 'Unknown'
            enc = encoders.get(c)
            if enc and val in enc.classes_:
                arr.append(float(enc.transform([val])[0]))
            else:
                arr.append(0.0)
        arr.append(float(anchor_month))
        for c in num_cols:
            val = row[c]
            arr.append(np.log1p(max(0.0, float(val))) if pd.notnull(val) else 0.0)
        static_dict[str(row['orig_sku'])] = np.array(arr, dtype=np.float32)

    # 5. 批量推断
    print(f"  正在对 {len(sku_list)} 款 SKU 进行推断...")
    valid_skus = [s for s in sku_list if str(s) in static_dict]
    pred_rows = []
    BZ = 2048

    for i in range(0, len(valid_skus), BZ):
        batch_skus = valid_skus[i:i+BZ]
        X_dyn_list, X_sta_list = [], []
        for sku in batch_skus:
            si = sku_to_i[sku]
            X_dyn_list.append(ts_feat[si])
            X_sta_list.append(static_dict[str(sku)])

        X_dyn_np = np.array(X_dyn_list)
        sh = X_dyn_np.shape
        c = np.nan_to_num(X_dyn_np.reshape(-1, 7), nan=0, posinf=0, neginf=0)
        X_dyn_np = scaler.transform(c).reshape(sh)

        X_dyn_t = torch.tensor(X_dyn_np, dtype=torch.float32).to(device)
        X_sta_t = torch.tensor(np.array(X_sta_list), dtype=torch.float32).to(device)

        with torch.no_grad(), torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            logits, preds = model(X_dyn_t, X_sta_t, static_keys)
            prob = torch.sigmoid(logits)

        dead_mask = (X_dyn_t.abs().sum(dim=(1, 2)) < 1e-4)
        prob[dead_mask] = 0.0
        preds = torch.clamp(preds, max=8.5)
        pred_qty = (torch.expm1(preds) * (prob > 0.10).float() * 0.8).cpu().numpy().flatten()
        pred_qty = np.nan_to_num(pred_qty, nan=0.0, posinf=0.0, neginf=0.0)

        for sku, q in zip(batch_skus, pred_qty):
            pred_rows.append({'sku_id': str(sku), 'pred_qty_30d': max(0.0, round(float(q), 1))})

    # 6. 合并预测与真实标签
    pred_df  = pd.DataFrame(pred_rows)
    actual_agg['sku_id'] = actual_agg['sku_id'].astype(str)
    result = pd.merge(pred_df, actual_agg, on='sku_id', how='outer').fillna(0)
    result['anchor_date']  = anchor_date_str
    result['abs_err']      = (result['pred_qty_30d'] - result['actual_qty_30d']).abs()
    result['has_pred']     = (result['pred_qty_30d'] > 0)
    result['has_actual']   = (result['actual_qty_30d'] > 0)

    # 7. 输出报告
    sep("主要指标 (Key Metrics)")
    total_pred   = result['pred_qty_30d'].sum()
    total_actual = result['actual_qty_30d'].sum()
    ratio        = total_pred / (total_actual + 1e-5)
    mae          = result['abs_err'].mean()
    # 正样本 MAE (真实有补货的 SKU)
    pos_mask = result['actual_qty_30d'] > 0
    mae_pos  = result.loc[pos_mask, 'abs_err'].mean() if pos_mask.sum() > 0 else 0

    print(f"  AI 预测有效 SKU 数:    {(result['pred_qty_30d'] > 0).sum():>6}")
    print(f"  实际发生补货 SKU 数:   {pos_mask.sum():>6}")
    print(f"  AI 预测总量 (30d):     {total_pred:>10,.0f} 件")
    print(f"  实际真实补货 (30d):    {total_actual:>10,.0f} 件")
    print(f"  大盘充盈率 (Ratio):   {ratio:>10.3f}  (1.0~1.3 最佳, <1 偏保守, >1.5 偏激进)")
    print(f"  全量 MAE:             {mae:>10.2f}  件/SKU")
    print(f"  正样本 MAE:           {mae_pos:>10.2f}  件/有补货SKU")

    sep("Top 15: AI 预测量最大 vs 真实对比")
    top_preds = result.nlargest(15, 'pred_qty_30d')
    print(f"  {'SKU_ID':<20} {'AI预测':>10} {'实际补货':>10} {'差值':>8} {'准确度':>8}")
    print(f"  {'-'*60}")
    for _, r in top_preds.iterrows():
        diff   = r['pred_qty_30d'] - r['actual_qty_30d']
        accuracy = max(0, 1 - abs(diff) / (r['actual_qty_30d'] + 1e-2)) * 100
        diff_s = f"+{diff:.0f}" if diff >= 0 else f"{diff:.0f}"
        print(f"  {r['sku_id']:<20} {r['pred_qty_30d']:>10.0f} {r['actual_qty_30d']:>10.0f} {diff_s:>8} {accuracy:>7.1f}%")

    sep("漏网之鱼: 实际爆款 AI 完全漏报 Top 10")
    missed = result[result['pred_qty_30d'] == 0].nlargest(10, 'actual_qty_30d')
    if len(missed) > 0:
        for _, r in missed.iterrows():
            print(f"  SKU: {r['sku_id']:<22} 真实补货 {r['actual_qty_30d']:.0f} 件  -- AI 预测为 0 (完全漏报)")
    else:
        print("  无漏报爆款，模型覆盖良好！")

    sep("虚报: AI 报了量但实际无补货 Top 5 (精准率损耗)")
    overshot = result[(result['pred_qty_30d'] > 10) & (result['actual_qty_30d'] == 0)].nlargest(5, 'pred_qty_30d')
    for _, r in overshot.iterrows():
        print(f"  SKU: {r['sku_id']:<22} AI 预测 {r['pred_qty_30d']:.0f} 件  -- 实际真实补货 0 件 (误报)")

    return result

def main():
    parser = argparse.ArgumentParser(description='B2B V2 时间点回测')
    parser.add_argument('--date', type=str, default=None, help='回测日期 YYYY-MM-DD, 不填则测试 10-31 和 11-30 双日期')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    # 加载元数据
    with open(os.path.join(ARTIFACTS_DIR, 'meta_v2.json'), 'r') as f:
        meta = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'label_encoders_v2.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'feature_scaler_v2.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    # 加载全量宽表
    sep("读取历史宽表")
    df_full = pd.read_csv(os.path.join(GOLD_DIR, 'wide_table_sku.csv'))
    df_full['date'] = pd.to_datetime(df_full['date']).dt.date
    print(f"  总记录数: {len(df_full):,}  日期范围: [{df_full['date'].min()}] ~ [{df_full['date'].max()}]")

    # 加载模型
    cat_cols = meta['static_cat_cols']
    num_cols = meta['static_num_cols']
    model = load_model(device, cat_cols, num_cols, encoders)
    print(f"  模型已加载: {MODEL_PATH}")

    # 确定回测日期列表
    if args.date:
        dates = [args.date]
    else:
        dates = ['2025-10-31', '2025-11-30']

    all_results = []
    for d in dates:
        res = run_backtest_for_date(d, df_full, model, device, meta, encoders, scaler)
        all_results.append(res)

        out_path = os.path.join(REPORTS_DIR, f"backtest_{d.replace('-','')}.csv")
        res.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n  详细报表已保存: {out_path}")

    # 双日期对比摘要
    if len(all_results) == 2:
        sep("双日期回测摘要对比")
        for d, res in zip(dates, all_results):
            t = res['pred_qty_30d'].sum()
            a = res['actual_qty_30d'].sum()
            r = t / (a + 1e-5)
            mae = res['abs_err'].mean()
            print(f"  T={d}:  预测总量={t:,.0f}件  实际={a:,.0f}件  Ratio={r:.3f}  MAE={mae:.2f}")
        sep()

if __name__ == '__main__':
    main()
