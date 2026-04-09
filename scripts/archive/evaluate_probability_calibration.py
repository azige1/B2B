"""
评估套件 - 模块4: 概率校准与阈值敏感性分析
evaluate_probability_calibration.py
============================================================
核心问题:
  1. AI 说"补货概率 80%"时，真实补货发生率是多少？(校准曲线)
  2. 不同概率阈值（0.10 ~ 0.80）下的精确率/召回率变化？
  3. 最优阈值应该设在哪里？（F1最大点 / Precision-Recall权衡点）
  4. 当前 0.20 阈值是否是最佳选择？

使用: python evaluate_probability_calibration.py
"""
import sys, os, json, pickle
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
MODEL_PATH    = os.path.join(PROJECT_ROOT, 'models_v2', 'best_enhanced_model.pth')
REPORTS_DIR   = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

class DummyLE:
    classes_ = np.arange(13)

def sep(title='', width=68, char='='):
    s = max(3,(width-len(title)-2)//2) if title else 0
    print(f"\n{char*s} {title} {char*s}\n" if title else char*width)

def collect_probs():
    """获取验证集所有样本的预测概率和真实标签"""
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
    all_prob, all_actual_reg, all_pred_reg = [], [], []
    with torch.no_grad():
        for x_dyn, x_static, y_cls, y_reg in val_loader:
            x_dyn = x_dyn.to(device)
            x_static = x_static.to(device)
            o_cls, o_reg = model(x_dyn, x_static, static_keys)
            all_prob.append(torch.sigmoid(o_cls).cpu().numpy())
            all_actual_reg.append(y_reg.numpy())
            all_pred_reg.append(o_reg.cpu().numpy())
    probs   = np.concatenate(all_prob).flatten()
    actuals = np.expm1(np.concatenate(all_actual_reg).flatten())
    regs    = np.clip(np.concatenate(all_pred_reg).flatten(), None, 8.5)
    y_true  = (actuals > 0).astype(float)
    return probs, y_true, actuals, regs

def main():
    sep("概率校准 & 阈值敏感性分析")
    print("  加载模型并推断验证集...")
    probs, y_true, actual_qty, pred_regs = collect_probs()

    # ====== 1. 概率校准曲线 ======
    sep("维度A: 概率校准曲线 (Probability Calibration)")
    print("  说明: AI 预测概率 X% 的样本中，真实发生补货的实际比例是多少？")
    print("        理想校准: 80%置信度 -> 80%真实补货率 (完美对角线)\n")
    bins = np.arange(0, 1.1, 0.1)
    bin_labels = [f"{int(a*100)}-{int(b*100)}%" for a, b in zip(bins[:-1], bins[1:])]
    df_cal = pd.DataFrame({'prob': probs, 'true': y_true, 'actual_qty': actual_qty})
    df_cal['bin'] = pd.cut(df_cal['prob'], bins=bins, labels=bin_labels, include_lowest=True)
    cal_table = df_cal.groupby('bin', observed=True).agg(
        样本数=('true', 'count'),
        平均预测概率=('prob', 'mean'),
        真实补货率=('true', 'mean'),
        平均真实补货件数=('actual_qty', 'mean'),
    ).round(3).reset_index()
    cal_table['校准偏差'] = (cal_table['平均预测概率'] - cal_table['真实补货率']).round(3)
    cal_table['状态'] = cal_table['校准偏差'].apply(
        lambda x: '过度自信(高估)' if x > 0.1 else ('过于保守(低估)' if x < -0.1 else '校准良好')
    )
    print(cal_table.to_string(index=False))

    # ASCII 校准图
    print("\n  [校准直方图]: 每行 = 一个概率区间，#数量 = 真实补货率")
    for _, row in cal_table.iterrows():
        rate = row['真实补货率']
        if pd.isna(rate): continue
        ideal = (cal_table['bin'] == row['bin']).idxmax()
        bar = '#' * int(rate * 40)
        print(f"  {row['bin']:>8}: [{bar:<40}] 真实率={rate:.1%}")

    # ====== 2. 阈值扫描 ======
    sep("维度B: 阈值敏感性扫描 (Threshold Sweep)")
    print("  说明: 改变补货判断阈值 (0.05 ~ 0.70)，观察 Precision/Recall/F1 变化\n")
    thresholds = np.arange(0.05, 0.75, 0.05)
    th_rows = []
    best_f1, best_th = 0, 0
    for th in thresholds:
        pred_pos = (probs >= th)
        tp = (y_true == 1) & pred_pos
        fp = (y_true == 0) & pred_pos
        fn = (y_true == 1) & ~pred_pos
        recall    = tp.sum() / (tp.sum() + fn.sum() + 1e-5)
        precision = tp.sum() / (tp.sum() + fp.sum() + 1e-5)
        f1        = 2 * precision * recall / (precision + recall + 1e-5)
        # 同时计算这个阈值下的业务充盈率（Ratio）
        pred_qty = np.clip(pred_regs, None, 8.5)
        pred_qty = np.nan_to_num(np.expm1(pred_qty) * pred_pos.astype(float) * 0.8)
        ratio = pred_qty.sum() / (actual_qty.sum() + 1e-5)
        if f1 > best_f1:
            best_f1, best_th = f1, th
        th_rows.append({
            '阈值': round(th, 2), 'TP': int(tp.sum()),
            'FP': int(fp.sum()), 'FN': int(fn.sum()),
            '召回率': round(recall, 3), '精确率': round(precision, 3),
            'F1分数': round(f1, 3), '充盈率': round(ratio, 3),
        })
    th_df = pd.DataFrame(th_rows)
    print(th_df.to_string(index=False))

    print(f"\n  ★ 最优 F1 阈值: {best_th:.2f} (F1 = {best_f1:.3f})")
    print(f"  ★ 当前使用阈值: 0.20")
    print()
    cur = th_df[th_df['阈值'] == 0.20]
    opt = th_df[th_df['阈值'] == round(best_th, 2)].head(1)
    if not cur.empty and not opt.empty:
        print(f"  当前 0.20 阈值: 召回={cur.iloc[0]['召回率']:.3f}  精确={cur.iloc[0]['精确率']:.3f}  F1={cur.iloc[0]['F1分数']:.3f}  Ratio={cur.iloc[0]['充盈率']:.3f}")
        print(f"  最优 {best_th:.2f} 阈值: 召回={opt.iloc[0]['召回率']:.3f}  精确={opt.iloc[0]['精确率']:.3f}  F1={opt.iloc[0]['F1分数']:.3f}  Ratio={opt.iloc[0]['充盈率']:.3f}")

    # ====== 3. 实战建议 ======
    sep("维度C: 业务阈值选择建议")
    # 找 Recall >= 0.55 且 Ratio 最接近 1.1 的阈值
    sub = th_df[th_df['召回率'] >= 0.50]
    if not sub.empty:
        biz_th = sub.loc[sub['充盈率'].sub(1.1).abs().idxmin()]
        print(f"  [建议阈值]: {biz_th['阈值']}  (在召回≥50%的前提下，充盈率最接近 1.1)")
        print(f"    召回率={biz_th['召回率']:.3f}  精确率={biz_th['精确率']:.3f}  F1={biz_th['F1分数']:.3f}  充盈率={biz_th['充盈率']:.3f}")
    print()
    print("  [策略说明]")
    print("  * 如果业务目标是'不漏爆款' -> 调低阈值(0.10~0.15), 召回率上升")
    print("  * 如果业务目标是'控制库存' -> 调高阈值(0.40~0.50), 精确率上升")
    print("  * 综合 F1 最大点 通常是'默认最优'")

    out = os.path.join(REPORTS_DIR, 'probability_calibration.csv')
    th_df.to_csv(out, index=False, encoding='utf-8-sig')
    cal_table.to_csv(out.replace('.csv', '_calibration.csv'), index=False, encoding='utf-8-sig')
    print(f"\n  结果已保存: {out}")

if __name__ == '__main__':
    main()
