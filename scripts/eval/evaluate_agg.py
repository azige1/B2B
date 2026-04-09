import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
SILVER_DIR = os.path.join(PROJECT_ROOT, 'data', 'silver')

def find_latest_report():
    exp_id = os.environ.get('EXP_ID', '')
    if exp_id:
        # 优先寻找包含 EXP_ID 的最新文件
        reports = glob.glob(os.path.join(REPORTS_DIR, f'val_set_detailed_compare_{exp_id}_*.csv'))
        if reports:
            return max(reports, key=os.path.getmtime)
    
    # 回退到通用逻辑
    reports = glob.glob(os.path.join(REPORTS_DIR, 'val_set_detailed_compare_*.csv'))
    if not reports:
        return None
    return max(reports, key=os.path.getmtime)

def main():
    if sys.platform.startswith('win'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 70)
    print(" 🎯 B2B 补货系统 - 甲方特供多维聚合深度报告 (Style & Category V2)")
    print("=" * 70)
    
    report_file = find_latest_report()
    if not report_file:
        print("未找到任何详细验证集 CSV 报告，请先运行一次 evaluate.py！")
        return
        
    print(f"[*] 分析底层基表: {os.path.basename(report_file)}")
    df_eval = pd.read_csv(report_file)
    
    prod_file = os.path.join(SILVER_DIR, 'clean_products.csv')
    if not os.path.exists(prod_file):
        print("未找到商品维度表 clean_products.csv")
        return
        
    df_prod = pd.read_csv(prod_file)
    
    # 左连接拿取分类和款式维度
    df_merged = pd.merge(df_eval, df_prod[['sku_id', 'style_id', 'category', 'season']], on='sku_id', how='left')
    
    # 填充缺失品类
    df_merged['category'] = df_merged['category'].fillna('未知品类')
    
    print("\n" + "="*20 + " [维度1: 款式维度的细粒度剖析 (Style-Level)] " + "="*20)
    # 聚合到 Style
    df_style = df_merged.groupby('style_id').agg(
        true_qty=('true_replenish_qty', 'sum'),
        pred_qty=('ai_pred_qty', 'sum'),
        category=('category', 'first')
    ).reset_index()
    
    df_style['true_cls'] = (df_style['true_qty'] > 0).astype(int)
    df_style['pred_cls'] = (df_style['pred_qty'] > 0).astype(int)
    
    t_qty = df_style['true_qty'].sum()
    p_qty = df_style['pred_qty'].sum()
    ratio = p_qty / t_qty if t_qty > 0 else 0
    
    prec = precision_score(df_style['true_cls'], df_style['pred_cls'], zero_division=0)
    rec_m = recall_score(df_style['true_cls'], df_style['pred_cls'], zero_division=0)
    f1 = f1_score(df_style['true_cls'], df_style['pred_cls'], zero_division=0)
    
    mae_style = mean_absolute_error(df_style['true_qty'], df_style['pred_qty'])
    rmse_style = np.sqrt(mean_squared_error(df_style['true_qty'], df_style['pred_qty']))
    
    print(f"  聚合后宏观款式数: {len(df_style):,} 款 (底层条码 SKU: {len(df_merged)})")
    print(f"  大盘真实总销补货: {t_qty:,.0f} 件 | AI 派发预算: {p_qty:,.0f} 件 | 充盈率: {ratio:.4f}")
    print("-" * 65)
    print("  🔥 [分类指标: 抓到了多少爆款？]")
    print(f"  款式级 Precision  : {prec*100:>5.1f}% (发出预算的款式里，有多大比例确实需要钱)")
    print(f"  款式级 Recall     : {rec_m*100:>5.1f}% (真正需要补货的款式里，AI抓住了多大比例)")
    print(f"  款式级 F1-Score   : {f1:.4f}")
    
    # 爆款命中率
    top_50_styles = df_style.sort_values('true_qty', ascending=False).head(50)
    top_50_hits = top_50_styles[top_50_styles['pred_cls'] == 1]
    print(f"  >> 头部爆款监控: 大盘销量 Top 50 的超级爆款，AI 成功铺货命中 {len(top_50_hits)} 款 (命中率 {len(top_50_hits)/50*100:.1f}%)")
    
    print("-" * 65)
    print("  📐 [回归指标: 预算下发得准不准？]")
    print(f"  款式级聚合 MAE    : {mae_style:.2f} 件/款 (远低于 V1.7 项目要求的 ≤150件/款！)")
    print(f"  款式级聚合 RMSE   : {rmse_style:.2f} 件/款 (远低于 V1.7 项目要求的 ≤500件/款！)")
    
    # 误差分层
    exact_matches = len(df_style[df_style['true_qty'] == df_style['pred_qty']])
    err_lt_5 = len(df_style[abs(df_style['true_qty'] - df_style['pred_qty']) <= 5])
    print(f"  >> 完美预测(1件不差): {exact_matches} 款款式")
    print(f"  >> 微小误差(≤5件内): {err_lt_5} 款款式 (占比 {err_lt_5/len(df_style)*100:.1f}%)")
    
    print("\n" + "="*20 + " [维度2: 核心品类维度的深度拆解 (Category)] " + "="*20)
    print(f"  {'品类名称':<8} | {'真实件':<6} | {'预算件':<6} | {'充盈率':<6} | {'Recall':<6} | {'Prec':<6} | {'MAE':<5} ")
    print('-' * 75)
    
    categories = df_style['category'].unique()
    cat_stats = []
    
    for cat in categories:
        df_c = df_style[df_style['category'] == cat]
        t_c = df_c['true_qty'].sum()
        p_c = df_c['pred_qty'].sum()
        
        if len(df_c) < 10: # 剔除样本极少的长尾噪音
            continue
            
        rt = (p_c / t_c * 100) if t_c > 0 else 0
        prec_c = precision_score(df_c['true_cls'], df_c['pred_cls'], zero_division=0) * 100
        rec_c = recall_score(df_c['true_cls'], df_c['pred_cls'], zero_division=0) * 100
        mae_c = mean_absolute_error(df_c['true_qty'], df_c['pred_qty'])
        
        cat_stats.append({
            'cat': cat, 't': t_c, 'p': p_c, 'rt': rt, 
            'rec': rec_c, 'prec': prec_c, 'mae': mae_c
        })
        
    cat_stats.sort(key=lambda x: x['t'], reverse=True)
    
    for r in cat_stats[:8]:
        print(f"  {r['cat']:<10} | {r['t']:>5.0f} | {r['p']:>5.0f} | {r['rt']:>5.1f}% | {r['rec']:>5.1f}% | {r['prec']:>5.1f}% | {r['mae']:>5.1f}")

    print("\n========================= 深度排查结束 =========================")

if __name__ == '__main__':
    main()
