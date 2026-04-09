"""
评估套件 - 模块4: SKU 生命周期阶段评估
evaluate_sku_lifecycle.py
============================================================
核心问题:
  - 新款 SKU (历史记录少): AI 有多盲目？
  - 爆款 SKU (高频补货): AI 有多低估？
  - 衰退款 SKU (趋势下行): AI 有多高估？

使用: python evaluate_sku_lifecycle.py
"""
import sys, os, json, pickle, time
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src', 'train'))
sys.stdout.reconfigure(encoding='utf-8')

from src.models.enhanced_model import EnhancedTwoTowerLSTM
from dataset import create_lazy_dataloaders

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v3')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_v3')
GOLD_DIR      = os.path.join(PROJECT_ROOT, 'data', 'gold')
MODEL_PATH    = os.path.join(PROJECT_ROOT, 'models_v2', 'best_enhanced_model.pth')
REPORTS_DIR   = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

SPLIT_DATE = '2025-11-01'

class DummyLE:
    classes_ = np.arange(13)

def sep(title='', width=68, char='='):
    s = max(3, (width - len(title) - 2) // 2) if title else 0
    print(f"\n{char*s} {title} {char*s}\n" if title else char * width)

def classify_sku_lifecycle(df_train, split_date):
    split = pd.to_datetime(split_date).date()
    recent_cut = (pd.to_datetime(split) - pd.Timedelta(days=30)).date()
    old_cut    = (pd.to_datetime(split) - pd.Timedelta(days=90)).date()
    stats = df_train.groupby('sku_id').agg(
        total_records=('qty_replenish', 'count'),
        total_qty=('qty_replenish', 'sum'),
        last_date=('date', 'max'),
    ).reset_index()
    def classify(row):
        if row['total_records'] < 5:
            return '新款(<5条)'
        elif row['last_date'] < old_cut:
            return '衰退款(>90天无活动)'
        elif row['total_records'] > 30 and row['last_date'] >= recent_cut:
            return '爆款(高频且近期活跃)'
        else:
            return '活跃款(5-30条)'
    stats['lifecycle'] = stats.apply(classify, axis=1)
    return stats[['sku_id', 'lifecycle', 'total_records', 'total_qty']]

def collect_val_preds():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(ARTIFACTS_DIR, 'meta_v2.json')) as f:
        meta = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, 'label_encoders_v2.pkl'), 'rb') as f:
        encoders = pickle.load(f)
    cat_cols = meta['static_cat_cols']
    num_cols = meta['static_num_cols']
    sv = {c: len(encoders[c].classes_) + 5 for c in cat_cols if c in encoders}
    sv['month'] = 18
    model = EnhancedTwoTowerLSTM(dyn_feat_dim=7, static_vocab_sizes=sv,
                                  num_numeric_feats=len(num_cols), lstm_hidden=256).to(device)
    st = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in st.items()})
    model.eval()
    _, val_loader = create_lazy_dataloaders(PROCESSED_DIR, batch_size=2048, num_workers=0)
    static_keys = cat_cols + ['month'] + num_cols

    all_p, all_a, all_r, all_s = [], [], [], []
    with torch.no_grad():
        for xd, xs, yc, yr in val_loader:
            oc, or_ = model(xd.to(device), xs.to(device), static_keys)
            all_p.append(torch.sigmoid(oc).cpu().numpy())
            all_a.append(yr.numpy())
            all_r.append(or_.cpu().numpy())
            all_s.append(xs.numpy())

    probs   = np.concatenate(all_p).flatten()
    actuals = np.expm1(np.concatenate(all_a).flatten())
    regs    = np.clip(np.concatenate(all_r).flatten(), None, 8.5)
    preds   = np.nan_to_num(np.expm1(regs) * (probs > 0.20) * 0.8)
    static  = np.concatenate(all_s, axis=0)

    enc_sku = encoders.get('sku_id') if 'sku_id' in cat_cols else None
    if enc_sku:
        idx = static[:, cat_cols.index('sku_id')].astype(int).clip(0, len(enc_sku.classes_)-1)
        sku_ids = enc_sku.inverse_transform(idx)
    else:
        sku_ids = np.arange(len(preds)).astype(str)

    return pd.DataFrame({'sku_id': sku_ids, 'actual_qty': actuals,
                         'pred_qty': preds, 'prob': probs})

def main():
    sep("SKU 生命周期阶段评估")
    print("  加载模型推断验证集 ...")
    val_df = collect_val_preds()

    df_full = pd.read_csv(os.path.join(GOLD_DIR, 'wide_table_sku.csv'))
    df_full['date'] = pd.to_datetime(df_full['date']).dt.date
    split = pd.to_datetime(SPLIT_DATE).date()
    df_train = df_full[df_full['date'] < split].copy()

    lc_map = classify_sku_lifecycle(df_train, SPLIT_DATE)
    print(f"  训练集 SKU 生命周期分布:\n{lc_map['lifecycle'].value_counts().to_string()}\n")

    val_df['sku_id'] = val_df['sku_id'].astype(str)
    lc_map['sku_id'] = lc_map['sku_id'].astype(str)
    merged = pd.merge(val_df, lc_map, on='sku_id', how='left')
    merged['lifecycle'] = merged['lifecycle'].fillna('未匹配款')

    # --- 各阶段汇总指标 ---
    sep("各生命周期阶段: 预测精度汇总")
    header = f"  {'生命周期阶段':<26} {'样本':>6} {'真实补货':>9} {'AI预测':>9} {'充盈率':>8} {'MAE':>7} {'召回%':>8}"
    print(header)
    print(f"  {'-'*75}")
    summary_rows = []
    for lc, grp in merged.groupby('lifecycle'):
        n      = len(grp)
        actual = grp['actual_qty'].sum()
        pred   = grp['pred_qty'].sum()
        ratio  = pred / (actual + 1e-5)
        mae    = (grp['pred_qty'] - grp['actual_qty']).abs().mean()
        tp     = ((grp['actual_qty'] > 0) & (grp['pred_qty'] > 0)).sum()
        fn_    = ((grp['actual_qty'] > 0) & (grp['pred_qty'] == 0)).sum()
        recall = tp / (tp + fn_ + 1e-5) * 100
        flag   = ''
        if ratio < 0.5:   flag = ' << 严重低估'
        elif ratio > 1.8: flag = ' >> 严重高估'
        elif ratio < 0.8: flag = ' < 轻微低估'
        print(f"  {lc:<26} {n:>6,} {actual:>9,.0f} {pred:>9,.0f} {ratio:>8.3f} {mae:>7.2f} {recall:>7.1f}%{flag}")
        summary_rows.append({'lifecycle': lc, 'samples': n, 'actual': round(actual,0),
                              'pred': round(pred,0), 'ratio': round(ratio,3),
                              'mae': round(mae,2), 'recall_pct': round(recall,1)})

    # --- 爆款低估放大镜 ---
    sep("爆款低估诊断 - Top 15 最严重低估 SKU")
    high_demand = merged[(merged['actual_qty'] >= 8)]
    underest = high_demand[high_demand['pred_qty'] < high_demand['actual_qty'] * 0.6]
    top_under = underest.nlargest(15, 'actual_qty')[
        ['sku_id', 'lifecycle', 'total_records', 'actual_qty', 'pred_qty', 'prob']
    ].copy()
    if not top_under.empty:
        top_under['低估率%'] = ((1 - top_under['pred_qty'] / top_under['actual_qty']) * 100).round(0)
        print(top_under.to_string(index=False))
    else:
        print("  无严重低估爆款，模型表现良好！")

    # --- 衰退款高估放大镜 ---
    sep("衰退款高估诊断 - Top 10 AI 误报虚单")
    overest = merged[(merged['pred_qty'] > 10) & (merged['actual_qty'] == 0)]
    top_over = overest.nlargest(10, 'pred_qty')[['sku_id', 'lifecycle', 'total_records', 'pred_qty', 'prob']]
    if not top_over.empty:
        print(top_over.to_string(index=False))
    else:
        print("  无严重虚报，精准率良好！")

    # --- 新款专项分析 ---
    sep("新款冷启动分析 (历史 < 5 条记录)")
    new_df = merged[merged['lifecycle'] == '新款(<5条)']
    if len(new_df) > 0:
        pos_new = new_df[new_df['actual_qty'] > 0]
        print(f"  新款样本总数: {len(new_df):,}  其中真实有补货: {len(pos_new):,} ({len(pos_new)/len(new_df):.1%})")
        hit = ((pos_new['pred_qty'] > 0)).sum()
        print(f"  AI 正确命中新款补货: {hit} / {len(pos_new)} ({hit/max(len(pos_new),1):.1%})")
        fp_new = ((new_df['actual_qty'] == 0) & (new_df['pred_qty'] > 0)).sum()
        print(f"  AI 对零补货新款的误报: {fp_new} 条 ({fp_new/max(len(new_df),1):.1%})")
        print()
        print("  [结论] 新款冷启动是当前版本的主要弱点，建议:")
        print("    1. 期货铺货时同步将首单量作为强特征注入")
        print("    2. 用同款型历史均值做先验初始化")
    else:
        print("  验证集中无新款数据。")

    sep("综合建议")
    sdf = pd.DataFrame(summary_rows)
    worst_lc = sdf.loc[sdf['ratio'].sub(1.0).abs().idxmax(), 'lifecycle']
    best_lc  = sdf.loc[sdf['ratio'].sub(1.0).abs().idxmin(), 'lifecycle']
    print(f"  模型预测最准: [{best_lc}]  (充盈率最接近 1.0)")
    print(f"  最需优化阶段: [{worst_lc}]  (充盈率偏差最大)")

    out = os.path.join(REPORTS_DIR, f'lifecycle_eval_{time.strftime("%Y%m%d")}.csv')
    sdf.to_csv(out, index=False, encoding='utf-8-sig')
    print(f"\n  汇总报表已保存: {out}")

if __name__ == '__main__':
    main()
