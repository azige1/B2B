"""
evaluate_agg.py — Phase 5 SKU 级 Ratio 聚合分析（方案A）
==========================================================
用途:
  对 evaluate.py 输出的 val_set_detailed_compare_{EXP_ID}.csv
  按 SKU 聚合（方案A: 先聚合再算Ratio），输出 SKU 级中位 Ratio 分布、
  品类横截面 Ratio、严重偏差 SKU 名单。

使用方法:
  python evaluate_agg.py                     # 自动找最新的 CSV
  python evaluate_agg.py --exp e54_bilstm_l3_v5  # 指定实验ID
  python evaluate_agg.py --csv reports/val_set_detailed_compare_e54_bilstm_l3_v5_20250331.csv
"""
import os
import sys
import glob
import argparse
import json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR  = os.environ.get('EXP_REPORT_DIR', os.path.join(PROJECT_ROOT, 'reports'))
LOOKBACK_DAYS = 90
_GOLD_CACHE = None

# =========================================================
# 辅助函数
# =========================================================
def sep(title='', width=65, char='─'):
    if title:
        side = max(2, (width - len(title) - 2) // 2)
        print(f"\n{char*side} {title} {char*side}\n")
    else:
        print(char * width)

def find_latest_csv(exp_id=None):
    """自动找最新的 val_set_detailed_compare_*.csv"""
    pattern = os.path.join(REPORTS_DIR, 'val_set_detailed_compare_*.csv')
    if exp_id:
        pattern = os.path.join(REPORTS_DIR, f'val_set_detailed_compare_{exp_id}_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    # 按修改时间取最新
    return max(files, key=os.path.getmtime)


def resolve_phase_report_dir(exp_id):
    """Prefer the same phase subdir that already contains this experiment's text report."""
    candidate_dirs = [
        REPORTS_DIR,
        os.path.join(REPORTS_DIR, 'phase5_3'),
        os.path.join(REPORTS_DIR, 'phase5_2'),
        os.path.join(REPORTS_DIR, 'phase5_1'),
        os.path.join(REPORTS_DIR, 'phase5'),
    ]
    for base_dir in candidate_dirs:
        if not os.path.isdir(base_dir):
            continue
        if (
            os.path.exists(os.path.join(base_dir, f'eval_{exp_id}.txt')) or
            os.path.exists(os.path.join(base_dir, f'agg_{exp_id}.txt'))
        ):
            return base_dir
    return os.path.join(REPORTS_DIR, 'phase5')


def load_gold_context():
    global _GOLD_CACHE
    if _GOLD_CACHE is not None:
        return _GOLD_CACHE

    gold_path = os.path.join(PROJECT_ROOT, 'data', 'gold', 'wide_table_sku.csv')
    if not os.path.exists(gold_path):
        _GOLD_CACHE = (None, None)
        return _GOLD_CACHE

    usecols = [
        'sku_id', 'date', 'qty_replenish', 'qty_future',
        'category', 'style_id', 'season', 'series', 'band'
    ]
    gold = pd.read_csv(gold_path, usecols=usecols)
    gold['date'] = pd.to_datetime(gold['date'])

    daily = (
        gold.groupby(['sku_id', 'date'], as_index=False)
        .agg(
            qty_replenish=('qty_replenish', 'sum'),
            qty_future=('qty_future', 'sum'),
        )
    )
    static_cols = ['sku_id', 'category', 'style_id', 'season', 'series', 'band']
    static_info = gold[static_cols].drop_duplicates('sku_id')
    _GOLD_CACHE = (daily, static_info)
    return _GOLD_CACHE


def attach_eval_context(df):
    """Enrich detailed eval rows with SKU metadata and 90d lookback context."""
    enriched = df.copy()

    if 'anchor_date' in enriched.columns:
        enriched['anchor_date'] = pd.to_datetime(enriched['anchor_date'])

    daily, static_info = load_gold_context()
    if static_info is not None:
        merge_cols = [c for c in ['category', 'style_id', 'season', 'series', 'band'] if c not in enriched.columns]
        if merge_cols:
            enriched = enriched.merge(static_info[['sku_id'] + merge_cols], on='sku_id', how='left')

    needed_cols = {
        'lookback_repl_days_90',
        'lookback_future_days_90',
        'lookback_repl_sum_90',
        'lookback_future_sum_90',
    }
    if daily is not None and 'anchor_date' in enriched.columns and not needed_cols.issubset(set(enriched.columns)):
        context_frames = []
        for anchor_date in enriched['anchor_date'].dropna().drop_duplicates():
            start = anchor_date - pd.Timedelta(days=LOOKBACK_DAYS)
            hist = daily[(daily['date'] >= start) & (daily['date'] < anchor_date)]
            context = (
                hist.groupby('sku_id', as_index=False)
                .agg(
                    lookback_repl_days_90=('qty_replenish', lambda s: int((s > 0).sum())),
                    lookback_future_days_90=('qty_future', lambda s: int((s > 0).sum())),
                    lookback_repl_sum_90=('qty_replenish', 'sum'),
                    lookback_future_sum_90=('qty_future', 'sum'),
                )
            )
            context['anchor_date'] = anchor_date
            context_frames.append(context)

        if context_frames:
            context_all = pd.concat(context_frames, ignore_index=True)
            enriched = enriched.merge(context_all, on=['sku_id', 'anchor_date'], how='left')

    for col in ['lookback_repl_days_90', 'lookback_future_days_90', 'lookback_repl_sum_90', 'lookback_future_sum_90']:
        if col in enriched.columns:
            enriched[col] = enriched[col].fillna(0)

    if 'lookback_repl_days_90' in enriched.columns:
        def _activity_bucket(days):
            if days > 30:
                return 'hot'
            if days >= 10:
                return 'warm'
            if days >= 1:
                return 'cold'
            return 'ice'
        enriched['activity_bucket'] = enriched['lookback_repl_days_90'].astype(int).map(_activity_bucket)

    if 'lookback_repl_days_90' in enriched.columns and 'lookback_future_days_90' in enriched.columns:
        enriched['signal_quadrant'] = 'repl0_fut0'
        enriched.loc[
            (enriched['lookback_repl_days_90'] > 0) & (enriched['lookback_future_days_90'] == 0),
            'signal_quadrant'
        ] = 'repl1_fut0'
        enriched.loc[
            (enriched['lookback_repl_days_90'] == 0) & (enriched['lookback_future_days_90'] > 0),
            'signal_quadrant'
        ] = 'repl0_fut1'
        enriched.loc[
            (enriched['lookback_repl_days_90'] > 0) & (enriched['lookback_future_days_90'] > 0),
            'signal_quadrant'
        ] = 'repl1_fut1'

    return enriched

# =========================================================
# Part 1: 样本统计（大盘全局）
# =========================================================
def part1_global(df):
    sep("Part 1: 全局样本统计")
    total = len(df)
    pos_true = (df['true_replenish_qty'] > 0).sum()
    pos_pred = (df['ai_pred_qty'] > 0).sum()
    both_pos = ((df['true_replenish_qty'] > 0) & (df['ai_pred_qty'] > 0)).sum()

    sum_true = df['true_replenish_qty'].sum()
    sum_pred = df['ai_pred_qty'].sum()
    global_ratio = sum_pred / (sum_true + 1e-9)

    mae_all = np.mean(np.abs(df['ai_pred_qty'] - df['true_replenish_qty']))
    wmape = np.sum(np.abs(df['ai_pred_qty'] - df['true_replenish_qty'])) / (sum_true + 1e-9)

    print(f"  样本总数:         {total:>12,}")
    print(f"  真实有补货样本:   {int(pos_true):>12,}  ({pos_true/total*100:.1f}%)")
    print(f"  AI预测有补货:     {int(pos_pred):>12,}  ({pos_pred/total*100:.1f}%)")
    print(f"  双方均有补货:     {int(both_pos):>12,}")
    print()
    print(f"  真实总补货量:     {sum_true:>12,.0f} 件")
    print(f"  AI总预测量:       {sum_pred:>12,.0f} 件")
    print(f"  大盘 Ratio:       {global_ratio:>12.4f}  ({'⚠ 高估' if global_ratio>1.5 else ('⚠ 低估' if global_ratio<0.8 else '✔ 健康')})")
    print()
    print(f"  全样本 MAE:       {mae_all:>12.4f} 件/样本")
    print(f"  WMAPE:            {wmape:>12.4f}  (加权绝对百分比误差)")

    return global_ratio, wmape

# =========================================================
# Part 2: SKU 级 Ratio 分布（方案A: 先聚合再算Ratio）
# =========================================================
def part2_sku_ratio(df):
    sep("Part 2: SKU 级 Ratio 分布（方案A: 聚合后计算）")
    print("  说明: 先将同一SKU的所有样本预测值求和/真实值求和，得到SKU级别Ratio")
    print("       再对所有SKU的Ratio取中位数/分位数分布\n")

    # 按 sku_id 聚合
    df_sku = df.groupby('sku_id').agg(
        total_true=('true_replenish_qty', 'sum'),
        total_pred=('ai_pred_qty', 'sum'),
        n_samples =('true_replenish_qty', 'count'),
    ).reset_index()

    # 只对"真实有补货"的 SKU 计算 Ratio（方案A核心）
    valid = df_sku[df_sku['total_true'] > 0].copy()
    valid['sku_ratio'] = valid['total_pred'] / valid['total_true']

    if len(valid) == 0:
        print("  ⚠ 无有效正样本SKU，跳过")
        return {}

    ratios = valid['sku_ratio'].values
    pcts   = np.percentile(ratios, [10, 25, 50, 75, 90])

    print(f"  有效SKU数（真实有补货）: {len(valid):,} / {len(df_sku):,} 总SKU")
    print()
    print(f"  {'指标':<20} {'值':>10}")
    print(f"  {'─'*32}")
    print(f"  {'P10 (低估区边界)':<20} {pcts[0]:>10.4f}")
    print(f"  {'P25':<20} {pcts[1]:>10.4f}")
    print(f"  {'中位数 P50 ⭐':<20} {pcts[2]:>10.4f}  ← 核心指标")
    print(f"  {'P75':<20} {pcts[3]:>10.4f}")
    print(f"  {'P90 (高估区边界)':<20} {pcts[4]:>10.4f}")
    print(f"  {'均值':<20} {ratios.mean():>10.4f}")
    print()
    print(f"  健康区间标准: P50 介于 0.8 ~ 1.3 为理想状态")

    # SKU Ratio 分布区间统计
    bins = [
        ("严重低估 (ratio < 0.3)",  ratios < 0.3),
        ("低估     (0.3~0.8)",       (ratios >= 0.3) & (ratios < 0.8)),
        ("健康     (0.8~1.3)",       (ratios >= 0.8) & (ratios < 1.3)),
        ("高估     (1.3~2.0)",       (ratios >= 1.3) & (ratios < 2.0)),
        ("严重高估 (ratio > 2.0)",   ratios >= 2.0),
    ]
    print()
    print(f"  {'区间':<28} {'SKU数':>8} {'占比':>8}")
    print(f"  {'─'*48}")
    for label, mask in bins:
        cnt = mask.sum()
        pct = cnt / len(ratios) * 100
        bar = '█' * int(pct / 5)
        print(f"  {label:<28} {cnt:>8,} {pct:>7.1f}%  {bar}")

    return {
        'sku_count'    : int(len(valid)),
        'median_ratio' : float(pcts[2]),
        'p25_ratio'    : float(pcts[1]),
        'p75_ratio'    : float(pcts[3]),
        'sku_ratio_df' : valid,
    }

# =========================================================
# Part 3: 品类横截面 Ratio
# =========================================================
def part3_category(df):
    if 'category' not in df.columns or 'sku_id' not in df.columns:
        print("  ⚠ 缺少 category 列，跳过品类分析")
        return
    sep("Part 3: 品类横截面 Ratio")

    # 先按品类+SKU级聚合，再计算Ratio（保持方案A一致性）
    df_cat = df.groupby('category').agg(
        total_true=('true_replenish_qty', 'sum'),
        total_pred=('ai_pred_qty', 'sum'),
        sku_count =('sku_id', 'nunique'),
    ).reset_index()
    df_cat['ratio'] = df_cat['total_pred'] / (df_cat['total_true'] + 1e-9)
    df_cat = df_cat.sort_values('total_true', ascending=False)

    print(f"  {'品类':<16} {'实际件数':>10} {'预测件数':>10} {'Ratio':>8} {'SKU数':>7} 状态")
    print(f"  {'─'*65}")
    for _, row in df_cat.iterrows():
        r = row['ratio']
        if r > 2.0:
            status = '⚠ 严重高估'
        elif r > 1.3:
            status = '↑ 轻度高估'
        elif r < 0.3:
            status = '⚠ 严重低估'
        elif r < 0.8:
            status = '↓ 轻度低估'
        else:
            status = '✔ 健康'
        cat = str(row['category'])[:14]
        print(f"  {cat:<16} {int(row['total_true']):>10,} {int(row['total_pred']):>10,}"
              f" {r:>8.3f} {int(row['sku_count']):>7} {status}")

# =========================================================
# Part 4: 严重偏差 SKU 名单
# =========================================================
def part4_outliers(sku_ratio_df):
    if sku_ratio_df is None or len(sku_ratio_df) == 0:
        return
    sep("Part 4: 严重偏差 SKU 名单")

    under = sku_ratio_df[sku_ratio_df['sku_ratio'] < 0.3].sort_values('total_true', ascending=False)
    over  = sku_ratio_df[sku_ratio_df['sku_ratio'] > 2.0].sort_values('total_true', ascending=False)

    print(f"  严重低估 SKU（Ratio < 0.3）: {len(under)} 个  ← 这些SKU可能严重断货")
    if len(under) > 0:
        print(f"  {'SKU':<25} {'实际':>8} {'预测':>8} {'Ratio':>7}")
        print(f"  {'─'*55}")
        for _, row in under.head(10).iterrows():
            print(f"  {str(row['sku_id']):<25} {int(row['total_true']):>8,}"
                  f" {int(row['total_pred']):>8,} {row['sku_ratio']:>7.3f}")

    print()
    print(f"  严重高估 SKU（Ratio > 2.0）: {len(over)} 个  ← 这些SKU可能积压库存")
    if len(over) > 0:
        print(f"  {'SKU':<25} {'实际':>8} {'预测':>8} {'Ratio':>7}")
        print(f"  {'─'*55}")
        for _, row in over.head(10).iterrows():
            print(f"  {str(row['sku_id']):<25} {int(row['total_true']):>8,}"
                  f" {int(row['total_pred']):>8,} {row['sku_ratio']:>7.3f}")

# =========================================================
# Part 5: 爆款专项（真实补货 > 25 件的 SKU）
# =========================================================
def part5_big_sku(sku_ratio_df):
    if sku_ratio_df is None or len(sku_ratio_df) == 0:
        return
    sep("Part 5: 爆款专项（真实总补货 > 25 件的 SKU）")

    big = sku_ratio_df[sku_ratio_df['total_true'] > 25].sort_values('total_true', ascending=False)
    if len(big) == 0:
        print("  无满足条件的爆款 SKU")
        return

    big_ratios = big['sku_ratio'].values
    big_mae    = np.abs(big['total_pred'] - big['total_true']).mean()
    print(f"  爆款 SKU 总数: {len(big):,}  (真实补货>25件)")
    print(f"  爆款中位 Ratio: {np.median(big_ratios):.4f}")
    print(f"  爆款平均 MAE:   {big_mae:.2f} 件/款")
    print()
    print(f"  Top 10 真实爆款:")
    print(f"  {'SKU':<25} {'实际':>8} {'预测':>8} {'Ratio':>7} {'绝对误差':>8}")
    print(f"  {'─'*65}")
    for _, row in big.head(10).iterrows():
        err = abs(row['total_pred'] - row['total_true'])
        print(f"  {str(row['sku_id']):<25} {int(row['total_true']):>8,}"
              f" {int(row['total_pred']):>8,} {row['sku_ratio']:>7.3f} {err:>8.0f}")

# =========================================================
# Part 6: 冷启动诊断（Cold Start Diagnosis）
# =========================================================
def part6_signal_quadrants(df):
    if 'signal_quadrant' not in df.columns:
        print("  ⚠ 缺少 signal_quadrant，上游上下文未准备好，跳过信号象限分析")
        return

    sep("Part 6: 信号象限诊断")
    print("  说明: 结合过去90天补货/期货历史，将样本拆成四类，定位期货信息真正起作用的区域\n")
    print(f"  {'象限':<14} {'样本数':>8} {'正样本率':>9} {'真实总量':>10} {'预测总量':>10} {'Ratio':>8} {'MAE':>8}")
    print(f"  {'─'*76}")

    order = ['repl0_fut0', 'repl0_fut1', 'repl1_fut0', 'repl1_fut1']
    for name in order:
        grp = df[df['signal_quadrant'] == name]
        if len(grp) == 0:
            continue
        true_sum = grp['true_replenish_qty'].sum()
        pred_sum = grp['ai_pred_qty'].sum()
        ratio = pred_sum / (true_sum + 1e-9)
        pos_rate = (grp['true_replenish_qty'] > 0).mean()
        mae = np.abs(grp['ai_pred_qty'] - grp['true_replenish_qty']).mean()
        print(f"  {name:<14} {len(grp):>8,} {pos_rate:>8.1%} {true_sum:>10,.0f} {pred_sum:>10,.0f} {ratio:>8.3f} {mae:>8.3f}")


def part7_decision_consistency(df):
    required = {'cls_pred_best_f1', 'ai_pred_positive_qty'}
    if not required.issubset(set(df.columns)):
        print("  ⚠ 缺少分类/数量口径列，跳过决策一致性分析")
        return

    sep("Part 7: 分类口径 vs 数量口径")
    cls_positive = df['cls_pred_best_f1'].astype(bool)
    qty_positive = df['ai_pred_positive_qty'].astype(bool)
    cls_only = cls_positive & (~qty_positive)
    qty_only = qty_positive & (~cls_positive)
    both_positive = cls_positive & qty_positive

    print(f"  分类判正样本:   {int(cls_positive.sum()):>8,}")
    print(f"  数量>0样本:     {int(qty_positive.sum()):>8,}")
    print(f"  二者同时为正:   {int(both_positive.sum()):>8,}")
    print(f"  仅分类为正:     {int(cls_only.sum()):>8,}")
    print(f"  仅数量为正:     {int(qty_only.sum()):>8,}")
    if 'dead_blocked' in df.columns:
        print(f"  死库存硬拦截:   {int(df['dead_blocked'].sum()):>8,}")


def part6_cold_start(df, sku_ratio_df):
    """
    冷启动评估:
    统计每个 SKU 在验证集 lookback 窗口内的历史补货活跃天数，
    按活跃度分组，对比各组的预测准确率。
    
    活跃度分组:
    - Hot (>30天活跃): 成熟SKU，充足历史信号
    - Warm (10-30天): 中等历史，有一定参考
    - Cold (1-9天): 稀疏历史，冷启动风险区
    - Ice (0天): 零历史，纯冷启动（仅靠静态特征预测）
    """
    sep("Part 8: 冷启动诊断（Cold Start Diagnosis）")
    print("  说明: 评估「历史补货活跃度」对预测准确率的影响")
    print("        为后续是否需要专门解决冷启动问题提供数据依据\n")

    if sku_ratio_df is None or len(sku_ratio_df) == 0:
        print("  ⚠ 无 SKU Ratio 数据，跳过")
        return

    # 优先复用逐样本附加的 90 天上下文，避免再走固定日期切窗的近似逻辑。
    if 'lookback_repl_days_90' in df.columns:
        sku_context = (
            df.groupby('sku_id', as_index=False)
            .agg(active_days=('lookback_repl_days_90', 'max'))
        )
        merged = sku_ratio_df.merge(sku_context, on='sku_id', how='left')
    else:
        gold_path = os.path.join(PROJECT_ROOT, 'data', 'gold', 'wide_table_sku.csv')
        if not os.path.exists(gold_path):
            print(f"  ⚠ 找不到 {gold_path}，跳过冷启动分析")
            return

        try:
            wt = pd.read_csv(gold_path, usecols=['sku_id', 'date', 'qty_replenish'])
            wt['date'] = pd.to_datetime(wt['date']).dt.date
        except Exception as e:
            print(f"  ⚠ 读取 wide_table 失败: {e}")
            return

        from datetime import date, timedelta
        split_date = date(2025, 12, 1)
        lookback_start = split_date - timedelta(days=90)
        wt_window = wt[(wt['date'] >= lookback_start) & (wt['date'] < split_date)]
        active_days = wt_window[wt_window['qty_replenish'] > 0].groupby('sku_id')['date'].nunique()
        active_days.name = 'active_days'
        merged = sku_ratio_df.merge(active_days.reset_index(), on='sku_id', how='left')

    merged['active_days'] = merged['active_days'].fillna(0).astype(int)

    # 3. 分组
    def classify(days):
        if days > 30:  return 'Hot (>30天)'
        if days >= 10: return 'Warm (10-30天)'
        if days >= 1:  return 'Cold (1-9天)'
        return 'Ice (0天)'

    merged['warmth'] = merged['active_days'].apply(classify)

    # 4. 按分组输出
    groups = ['Hot (>30天)', 'Warm (10-30天)', 'Cold (1-9天)', 'Ice (0天)']
    print(f"  {'温度组':<18} {'SKU数':>7} {'占比':>7} {'中位Ratio':>10} {'P25':>7} {'P75':>7} {'判断':>8}")
    print(f"  {'─'*72}")

    cold_total = 0
    cold_over  = 0
    for g in groups:
        subset = merged[merged['warmth'] == g]
        cnt = len(subset)
        pct = cnt / len(merged) * 100 if len(merged) > 0 else 0

        if cnt > 0 and cnt > 0:
            valid = subset[subset['total_true'] > 0]
            if len(valid) > 0:
                ratios = valid['sku_ratio'].values
                med = np.median(ratios)
                p25 = np.percentile(ratios, 25)
                p75 = np.percentile(ratios, 75)
            else:
                med = p25 = p75 = float('nan')
        else:
            med = p25 = p75 = float('nan')

        # 判断
        if np.isnan(med):
            verdict = '—'
        elif 0.7 <= med <= 1.4:
            verdict = '✔ 正常'
        elif med < 0.7:
            verdict = '⚠ 低估'
        else:
            verdict = '⚠ 高估'

        if g in ['Cold (1-9天)', 'Ice (0天)']:
            cold_total += cnt
            cold_over += len(subset[(subset['total_true'] > 0) &
                                    ((subset['sku_ratio'] > 2.0) | (subset['sku_ratio'] < 0.3))]) if cnt > 0 else 0

        print(f"  {g:<18} {cnt:>7,} {pct:>6.1f}%  {med:>10.4f} {p25:>7.4f} {p75:>7.4f} {verdict:>8}")

    # 5. 冷启动影响总结
    print()
    cold_pct = cold_total / len(merged) * 100 if len(merged) > 0 else 0
    print(f"  冷启动区域 (Cold+Ice): {cold_total:,} / {len(merged):,} SKU ({cold_pct:.1f}%)")
    if cold_total > 0:
        cold_err_pct = cold_over / cold_total * 100
        print(f"  冷启动严重偏差率 (ratio<0.3 或 >2.0): {cold_over}/{cold_total} = {cold_err_pct:.1f}%")
    print()

    # 活跃天数分布
    all_active = merged['active_days'].values
    print(f"  活跃天数分布:")
    print(f"    min={all_active.min()}, P25={np.percentile(all_active,25):.0f}, "
          f"P50={np.percentile(all_active,50):.0f}, P75={np.percentile(all_active,75):.0f}, "
          f"max={all_active.max()}")

    # 6. 诊断结论
    print()
    if cold_pct > 40:
        print(f"  🔴 冷启动占比 {cold_pct:.0f}% (>40%) — 冷启动是严重问题，需优先解决")
        print(f"     建议: 引入 SKU 相似度迁移学习 / 品类级先验 / Meta-Learning")
    elif cold_pct > 20:
        print(f"  🟡 冷启动占比 {cold_pct:.0f}% (20-40%) — 冷启动有一定影响")
        print(f"     建议: 考虑品类级别的默认预测策略作为 fallback")
    else:
        print(f"  🟢 冷启动占比 {cold_pct:.0f}% (<20%) — 冷启动影响较小")
        print(f"     建议: 当前暂不需要专门处理，后续版本迭代再考虑")


# =========================================================
# 主函数
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="SKU级Ratio聚合分析（方案A）")
    parser.add_argument('--exp', type=str, default=None, help="实验ID（如 e54_bilstm_l3_v5）")
    parser.add_argument('--csv', type=str, default=None, help="直接指定CSV文件路径")
    args = parser.parse_args()

    # 找到目标 CSV
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = find_latest_csv(args.exp)

    if not csv_path or not os.path.exists(csv_path):
        print(f"\n❌ 找不到评估CSV！请先运行 evaluate.py 生成 val_set_detailed_compare_*.csv")
        print(f"   或使用 --csv 参数指定路径")
        sys.exit(1)

    exp_id = os.path.basename(csv_path).replace('val_set_detailed_compare_', '').rsplit('_', 2)[0]
    print(f"\n{'='*65}")
    print(f"  📊 B2B Phase 5 SKU 级 Ratio 聚合分析")
    print(f"  实验ID: {exp_id}")
    print(f"  数据源: {os.path.relpath(csv_path)}")
    print(f"{'='*65}")

    # 读取
    df = pd.read_csv(csv_path)
    print(f"\n  已读取 {len(df):,} 行数据，列: {list(df.columns)}")

    # 字段标准化（兼容不同版本的列名）
    col_map = {
        'true_qty'       : 'true_replenish_qty',
        'pred_qty'       : 'ai_pred_qty',
        'ai_budget_qty_30d': 'ai_pred_qty',
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    report_out_dir = resolve_phase_report_dir(exp_id)
    df = attach_eval_context(df)
    context_out_dir = report_out_dir
    os.makedirs(context_out_dir, exist_ok=True)
    context_out_path = os.path.join(context_out_dir, f'eval_context_{exp_id}.csv')
    df.to_csv(context_out_path, index=False, encoding='utf-8-sig')
    print(f"  [OK] enriched eval context -> {os.path.relpath(context_out_path, PROJECT_ROOT)}")

    # 执行各维度分析
    global_ratio, wmape = part1_global(df)
    sku_info  = part2_sku_ratio(df)
    part3_category(df)
    sku_ratio_df = sku_info.get('sku_ratio_df') if sku_info else None
    part4_outliers(sku_ratio_df)
    part5_big_sku(sku_ratio_df)
    part6_signal_quadrants(df)
    part7_decision_consistency(df)
    part6_cold_start(df, sku_ratio_df)  # ★ 冷启动诊断

    # 保存 SKU 级 Ratio CSV 供进一步分析
    if sku_ratio_df is not None:
        out_dir  = report_out_dir
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'sku_ratio_{exp_id}.csv')
        sku_ratio_df.to_csv(out_path, index=False, encoding='utf-8-sig')
        print(f"\n  ✅ SKU级Ratio明细已保存: {os.path.relpath(out_path)}")

    # 汇总
    sep("汇总")
    median_r = sku_info.get('median_ratio', 0) if sku_info else 0
    print(f"  大盘 Ratio:        {global_ratio:.4f}")
    print(f"  SKU 中位 Ratio:    {median_r:.4f}  ← Phase 5 核心指标")
    print(f"  WMAPE:             {wmape:.4f}")
    print(f"  有效评估 SKU 数:   {sku_info.get('sku_count', 0):,}")
    print(f"\n  目标: SKU中位Ratio 在 0.80 ~ 1.30 为健康区间")
    print()


if __name__ == '__main__':
    main()
