"""
评估套件 - 模块2: 季节与品类分层精度分析
evaluate_by_segment.py
============================================================
核心问题: AI 对哪些季节/品类预测最准？哪些是重灾区？
维度:
  - 按 season 分层（春/夏/秋/冬/四季款）
  - 按 category 分层（T恤/裤类/裙子/外套/毛衣...）
  - 按价格带分层（<999 / 1000-1999 / 2000+）
  - 各维度的命中率/漏报率/误报率

使用: python evaluate_by_segment.py
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

class DummyLE:
    classes_ = np.arange(13)

def sep(title='', width=68, char='='):
    side = max(3, (width - len(title) - 2) // 2) if title else 0
    print(f"\n{char*side} {title} {char*side}\n" if title else char*width)

def load_model_and_predict():
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
                                  num_numeric_feats=len(num_cols),
        lstm_hidden=256
).to(device)
    st = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    st = {k.replace('_orig_mod.', ''): v for k, v in st.items()}
    model.load_state_dict(st)
    model.eval()

    _, val_loader = create_lazy_dataloaders(PROCESSED_DIR, batch_size=2048, num_workers=0)
    static_keys = cat_cols + ['month'] + num_cols

    all_actual, all_prob, all_pred_raw, all_static = [], [], [], []
    with torch.no_grad():
        for x_dyn, x_static, y_cls, y_reg in val_loader:
            x_dyn    = x_dyn.to(device)
            x_static_gpu = x_static.to(device)
            o_cls, o_reg = model(x_dyn, x_static_gpu, static_keys)
            all_actual.append(y_reg.cpu().numpy())
            all_prob.append(torch.sigmoid(o_cls).cpu().numpy())
            all_pred_raw.append(o_reg.cpu().numpy())
            all_static.append(x_static.numpy())

    actual_qty  = np.expm1(np.concatenate(all_actual).flatten())
    pred_prob   = np.concatenate(all_prob).flatten()
    pred_raw    = np.clip(np.concatenate(all_pred_raw).flatten(), None, 8.5)
    pred_qty    = np.nan_to_num(np.expm1(pred_raw) * (pred_prob > 0.20) * 0.8)
    static_mat  = np.concatenate(all_static, axis=0)

    # 把静态矩阵里的分类列索引解码回标签
    decoded = {}
    cat_label_map = {}
    for i, col in enumerate(cat_cols):
        if col in encoders:
            enc = encoders[col]
            idx = static_mat[:, i].astype(int).clip(0, len(enc.classes_)-1)
            decoded[col] = enc.inverse_transform(idx)
        else:
            decoded[col] = static_mat[:, i].astype(str)

    df = pd.DataFrame({
        'actual_qty'  : actual_qty,
        'pred_qty'    : pred_qty,
        'pred_prob'   : pred_prob,
        **decoded,
    })

    # 拼接价格（静态数值列里 price_tag 是第 1 个，经 log1p，需 expm1 还原）
    num_start = len(cat_cols) + 1      # +1 for month
    if 'price_tag' in num_cols:
        pt_idx = num_cols.index('price_tag') + num_start
        df['price_tag'] = np.expm1(static_mat[:, pt_idx].clip(0, 20))

    return df

def report_segment(df, group_col, title, top_n=99):
    sep(title)
    records = []
    for gp, grp in df.groupby(group_col, dropna=False):
        n          = len(grp)
        actual_pos = (grp['actual_qty'] > 0).sum()
        pred_pos   = (grp['pred_qty']   > 0).sum()
        hit        = ((grp['actual_qty'] > 0) & (grp['pred_qty'] > 0)).sum()
        miss       = ((grp['actual_qty'] > 0) & (grp['pred_qty'] == 0)).sum()
        fp         = ((grp['actual_qty'] == 0) & (grp['pred_qty'] > 0)).sum()
        recall     = hit / (actual_pos + 1e-5) * 100
        precision  = hit / (pred_pos + 1e-5) * 100
        mae        = (grp['pred_qty'] - grp['actual_qty']).abs().mean()
        ratio      = grp['pred_qty'].sum() / (grp['actual_qty'].sum() + 1e-5)
        records.append({
            '分类': str(gp), '样本数': n,
            '真实补货样本': int(actual_pos), 'AI发出补货': int(pred_pos),
            '命中(TP)': int(hit), '漏报(FN)': int(miss), '误报(FP)': int(fp),
            '召回率%': round(recall, 1), '精确率%': round(precision, 1),
            'MAE': round(mae, 2), '充盈率': round(ratio, 3)
        })
    rdf = pd.DataFrame(records).sort_values('真实补货样本', ascending=False).head(top_n)
    print(rdf.to_string(index=False))
    return rdf

def main():
    sep("模型分层精度分析 -- 季节 / 品类 / 价格带")
    print("  正在加载模型并批量推断验证集...")
    df = load_model_and_predict()
    print(f"  完成. 验证集总样本: {len(df):,}")

    all_tables = {}

    # -- 1. 按季节 --
    r1 = report_segment(df, 'season', "维度A: 按季节分层 (Season Breakdown)")
    all_tables['season'] = r1

    # -- 2. 按大品类 --
    r2 = report_segment(df, 'category', "维度B: 按大品类分层 (Category Breakdown)")
    all_tables['category'] = r2

    # -- 3. 按子品类 --
    r3 = report_segment(df, 'sub_category', "维度C: 按子品类分层 (Sub-Category Breakdown)", top_n=20)
    all_tables['sub_category'] = r3

    # -- 4. 按价格带 --
    if 'price_tag' in df.columns:
        df['price_band'] = pd.cut(
            df['price_tag'],
            bins=[0, 999, 1999, 2999, 99999],
            labels=['经济款(<999)', '中档款(1000-1999)', '高档款(2000-2999)', '奢品款(>3000)']
        )
        r4 = report_segment(df, 'price_band', "维度D: 按价格带分层 (Price Band Breakdown)")
        all_tables['price_band'] = r4

    # -- 5. 按波段(band) --
    r5 = report_segment(df, 'band', "维度E: 按波段分层 (Band Breakdown)")
    all_tables['band'] = r5

    # 综合说明
    sep("综合解读指南")
    print("  召回率 (Recall)  : 真实补货中被 AI 正确捕捉到的比例。")
    print("                     目标 > 60%, 低于 40% 说明该细分市场数据稀疏，AI 基本看不懂。")
    print("  精确率 (Precision): AI 发出的补货指令中命中真实需求的比例。")
    print("                     目标 > 50%, 低于 30% 说明 AI 虚报严重，需提高阈值。")
    print("  充盈率 (Ratio)   : <1.0 保守, 1.0-1.3 黄金区间, >1.5 激进高估。")
    print()

    # 保存
    out_path = os.path.join(REPORTS_DIR, f'segment_analysis_{time.strftime("%Y%m%d")}.xlsx')
    try:
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            for sheet, tbl in all_tables.items():
                tbl.to_excel(writer, sheet_name=sheet[:31], index=False)
        print(f"  分层分析报表已保存: {out_path}")
    except Exception as e:
        print(f"  Excel 保存失败 ({e})，改为 CSV：")
        for sheet, tbl in all_tables.items():
            p = os.path.join(REPORTS_DIR, f'segment_{sheet}_{time.strftime("%Y%m%d")}.csv')
            tbl.to_csv(p, index=False, encoding='utf-8-sig')
            print(f"    {p}")

if __name__ == '__main__':
    main()
