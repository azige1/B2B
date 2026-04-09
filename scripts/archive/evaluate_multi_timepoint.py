"""
评估套件 - 模块3: 连续多时间点稳定性回测
evaluate_multi_timepoint.py
============================================================
核心问题: 模型在全年不同的月份回测点是否稳定？
         哪个季节的 Ratio 会崩塌？错误是规律性的还是随机的？

自动遍历多个时间点（每月最后一天），统计每个月的:
  - Ratio (充盈率)
  - MAE  (绝对误差)
  - TP/FP/FN 数量
  - 热力图分析：哪个月份推断最准

使用: python evaluate_multi_timepoint.py
"""
import sys, os, json, pickle, time
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.stdout.reconfigure(encoding='utf-8')

from src.models.enhanced_model import EnhancedTwoTowerLSTM

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v3')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_v3')
GOLD_DIR      = os.path.join(PROJECT_ROOT, 'data', 'gold')
REPORTS_DIR   = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

LOOKBACK = 60
FORECAST = 30

class DummyLE:
    classes_ = np.arange(13)

def sep(title='', width=68, char='='):
    side = max(3, (width - len(title) - 2) // 2) if title else 0
    print(f"\n{char*side} {title} {char*side}\n" if title else char*width)

def rolling_log1p(arr, window, n):
    cum = np.cumsum(np.concatenate([[0.0], arr]))
    res = np.zeros(n, dtype=np.float32)
    for j in range(n):
        i_end = j + 1
        i_sta = max(0, i_end - window)
        res[j] = cum[i_end] - cum[i_sta]
    return np.log1p(np.maximum(res, 0))

def get_month_end_dates(df, skip_last=1):
    """从宽表里自动发现每月最后一天，并跳过最近 skip_last 个月（无法拿到 30 天真实标签）"""
    df['yearmonth'] = df['date'].apply(lambda d: (d.year, d.month))
    pivots = []
    for ym, grp in df.groupby('yearmonth'):
        anchor = grp['date'].max()
        label_end = pd.to_datetime(anchor + pd.Timedelta(days=FORECAST)).date()
        if df['date'].max() >= label_end:
            pivots.append(anchor)
    return sorted(pivots)[:-skip_last] if skip_last > 0 else sorted(pivots)

def fast_inference(anchor, df_full, model, device, meta, encoders, scaler):
    """对单个时间点执行推断并返回 pred/actual 汇总"""
    cat_cols = meta['static_cat_cols']
    num_cols = meta['static_num_cols']
    static_keys = cat_cols + ['month'] + num_cols

    train_start = (pd.to_datetime(anchor) - pd.Timedelta(days=LOOKBACK-1)).date()
    df_hist = df_full[(df_full['date'] >= train_start) & (df_full['date'] <= anchor)]
    label_start = (pd.to_datetime(anchor) + pd.Timedelta(days=1)).date()
    label_end   = (pd.to_datetime(anchor) + pd.Timedelta(days=FORECAST)).date()
    df_label = df_full[(df_full['date'] >= label_start) & (df_full['date'] <= label_end)]

    if len(df_hist) < 10:
        return None

    actual_agg = df_label.groupby('sku_id')['qty_replenish'].sum().reset_index()
    actual_agg.columns = ['sku_id', 'actual_qty']

    # 动态矩阵
    dyn_agg = df_hist.groupby(['sku_id', 'date']).agg({
        'qty_replenish': 'sum', 'qty_debt': 'sum',
        'qty_shipped': 'sum', 'qty_inbound': 'sum'
    }).reset_index()
    static_sub = df_hist[cat_cols + num_cols].drop_duplicates('sku_id').copy()

    all_dates = pd.date_range(end=anchor, periods=LOOKBACK, freq='D').date
    d2i = {d: i for i, d in enumerate(all_dates)}
    sku_list = sorted(dyn_agg['sku_id'].unique())
    s2i = {s: i for i, s in enumerate(sku_list)}

    dyn_mat = np.zeros((len(sku_list), LOOKBACK, 4), dtype=np.float32)
    for row in dyn_agg.itertuples(index=False):
        si = s2i.get(row.sku_id, -1)
        di = d2i.get(row.date, -1)
        if si >= 0 and di >= 0:
            dyn_mat[si, di] = [row.qty_replenish, row.qty_debt, row.qty_shipped, row.qty_inbound]

    ts = np.zeros((len(sku_list), LOOKBACK, 7), dtype=np.float32)
    for si in range(len(sku_list)):
        r, d, s, n = dyn_mat[si, :, 0], dyn_mat[si, :, 1], dyn_mat[si, :, 2], dyn_mat[si, :, 3]
        ts[si, :, 0] = np.log1p(np.maximum(r, 0))
        ts[si, :, 1] = rolling_log1p(r, 7, LOOKBACK)
        ts[si, :, 2] = rolling_log1p(r, 30, LOOKBACK)
        ts[si, :, 3] = np.log1p(np.maximum(d, 0))
        ts[si, :, 4] = np.log1p(np.maximum(s, 0))
        ts[si, :, 5] = rolling_log1p(s, 7, LOOKBACK)
        ts[si, :, 6] = np.log1p(np.maximum(n, 0))

    static_dict = {}
    for _, row in static_sub.iterrows():
        arr = []
        for c in cat_cols:
            val = str(row[c]) if pd.notnull(row[c]) else 'Unknown'
            enc = encoders.get(c)
            arr.append(float(enc.transform([val])[0]) if enc and val in enc.classes_ else 0.0)
        arr.append(float(anchor.month))
        for c in num_cols:
            arr.append(np.log1p(max(0.0, float(row[c]))) if pd.notnull(row[c]) else 0.0)
        static_dict[str(row['sku_id'])] = np.array(arr, dtype=np.float32)

    valid_skus = [s for s in sku_list if str(s) in static_dict]
    pred_rows = []
    BZ = 2048
    for i in range(0, len(valid_skus), BZ):
        batch = valid_skus[i:i+BZ]
        Xd = np.array([ts[s2i[s]] for s in batch])
        Xs = np.array([static_dict[str(s)] for s in batch])
        sh = Xd.shape
        Xd = scaler.transform(np.nan_to_num(Xd.reshape(-1, 7))).reshape(sh)
        Xd_t = torch.tensor(Xd, dtype=torch.float32).to(device)
        Xs_t = torch.tensor(Xs, dtype=torch.float32).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
            logits, preds = model(Xd_t, Xs_t, static_keys)
            prob = torch.sigmoid(logits)
        dead = (Xd_t.abs().sum(dim=(1, 2)) < 1e-4)
        prob[dead] = 0.0
        preds = torch.clamp(preds, max=8.5)
        q = np.nan_to_num((torch.expm1(preds) * (prob > 0.20).float() * 0.8).cpu().numpy().flatten())
        for sku, qq in zip(batch, q):
            pred_rows.append({'sku_id': str(sku), 'pred_qty': max(0.0, round(float(qq), 1))})

    pred_df  = pd.DataFrame(pred_rows)
    actual_agg['sku_id'] = actual_agg['sku_id'].astype(str)
    res = pd.merge(pred_df, actual_agg, on='sku_id', how='outer').fillna(0)

    total_pred   = res['pred_qty'].sum()
    total_actual = res['actual_qty'].sum()
    ratio  = total_pred / (total_actual + 1e-5)
    mae    = (res['pred_qty'] - res['actual_qty']).abs().mean()
    hit    = ((res['actual_qty'] > 0) & (res['pred_qty'] > 0)).sum()
    miss   = ((res['actual_qty'] > 0) & (res['pred_qty'] == 0)).sum()
    fp     = ((res['actual_qty'] == 0) & (res['pred_qty'] > 0)).sum()
    recall = hit / ((hit + miss) + 1e-5) * 100

    return {
        'anchor_date': str(anchor),
        'month'      : f"{anchor.year}-{anchor.month:02d}",
        'sku_count'  : len(valid_skus),
        'pred_total' : round(total_pred, 0),
        'actual_total': round(total_actual, 0),
        'ratio'      : round(ratio, 3),
        'mae'        : round(mae, 2),
        'TP'         : int(hit),
        'FN'         : int(miss),
        'FP'         : int(fp),
        'recall_pct' : round(recall, 1),
    }

def main():
    sep("连续多时间点稳定性回测")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")

    with open(os.path.join(ARTIFACTS_DIR, 'meta_v2.json')) as f:
        meta = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'label_encoders_v2.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'feature_scaler_v2.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    cat_cols = meta['static_cat_cols']
    num_cols = meta['static_num_cols']
    sv = {c: len(encoders[c].classes_) + 5 for c in cat_cols if c in encoders}
    sv['month'] = 18
    model = EnhancedTwoTowerLSTM(dyn_feat_dim=7, static_vocab_sizes=sv,
                                  num_numeric_feats=len(num_cols),
        lstm_hidden=256
).to(device)
    st = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in st.items()})
    model.eval()

    df_full = pd.read_csv(os.path.join(GOLD_DIR, 'wide_table_sku.csv'))
    df_full['date'] = pd.to_datetime(df_full['date']).dt.date
    pivots = get_month_end_dates(df_full, skip_last=1)
    print(f"  发现可回测月末节点: {[str(p) for p in pivots]}\n")

    results = []
    for anchor in pivots:
        print(f"  > 正在回测 T={anchor} ...", end='', flush=True)
        r = fast_inference(anchor, df_full, model, device, meta, encoders, scaler)
        if r:
            results.append(r)
            print(f" Ratio={r['ratio']:.3f}  MAE={r['mae']:.2f}  Recall={r['recall_pct']:.1f}%")
        else:
            print(" (数据不足，跳过)")

    if not results:
        print("没有有效的回测月份！")
        return

    rdf = pd.DataFrame(results)
    sep("逐月稳定性汇总表")
    print(rdf[['month','sku_count','pred_total','actual_total','ratio','mae','TP','FN','FP','recall_pct']].to_string(index=False))

    sep("稳定性分析结论")
    best  = rdf.loc[rdf['ratio'].sub(1.1).abs().idxmin(), 'month']
    worst = rdf.loc[rdf['ratio'].sub(1.1).abs().idxmax(), 'month']
    avg_ratio = rdf['ratio'].mean()
    std_ratio = rdf['ratio'].std()
    print(f"  平均充盈率:    {avg_ratio:.3f}  (理想为 1.0~1.3)")
    print(f"  充盈率标准差:  {std_ratio:.3f}  (<0.2 为稳定, >0.5 为不稳定)")
    print(f"  最佳回测月份:  {best}  (充盈率最接近 1.1)")
    print(f"  最差回测月份:  {worst}  (充盈率偏差最大)")
    print()
    good_months = rdf[rdf['ratio'].between(0.7, 1.5)]['month'].tolist()
    bad_months  = rdf[~rdf['ratio'].between(0.7, 1.5)]['month'].tolist()
    print(f"  健康月份 (0.7~1.5): {good_months}")
    print(f"  异常月份 (<0.7或>1.5): {bad_months}")

    # 简易热力图：ASCII 条形图
    sep("月度 Ratio 可视化 (ASCII 热力图)")
    for _, row in rdf.iterrows():
        r = row['ratio']
        bar_len = int(r * 20)
        bar = '#' * bar_len
        flag = '<<' if r > 1.5 else ('>>' if r < 0.7 else '  ')
        print(f"  {row['month']}  [{bar:<40}] {r:.3f} {flag}")

    out = os.path.join(REPORTS_DIR, f'multi_timepoint_backtest_{time.strftime("%Y%m%d")}.csv')
    rdf.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n  详细结果已保存: {out}")

if __name__ == '__main__':
    main()
