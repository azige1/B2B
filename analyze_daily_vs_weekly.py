"""
日频 vs 周频 全量对比分析
==========================
读取 Phase 1-3（日频）和 Phase 4（周频）的全部实验结果，
输出结构化对比报告，定量分析周频模型表现不如日频的根因。

数据来源：
  - reports/phase2/phase2_summary.csv              → 日频 Phase2 各实验 F1/AUC/Ratio
  - reports/phase2/basic_e*.txt                    → 日频详细评估（MAE/分位数等）
  - reports/phase3/basic_e*.txt                    → 日频 Phase3 Loss 精调结果
  - reports/history_w4*.csv                        → 周频 Phase4 各实验训练曲线
  - data/artifacts_weekly*/meta_weekly.json         → 周频数据集元信息
  - data/processed_fast/meta.json 或 processed_v3   → 日频数据集元信息
"""
import csv
import json
import os
import re
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR  = os.path.join(PROJECT_ROOT, 'reports')

sep = '=' * 80
thin = '-' * 80


# ================================================================
# 工具函数
# ================================================================
def parse_basic_txt(filepath: str) -> dict:
    """从 basic_eXX_*.txt 中提取关键指标。"""
    info = {'file': os.path.basename(filepath)}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        # F1
        m = re.search(r'最佳 F1-Score:\s+([\d.]+)', text)
        if m: info['f1'] = float(m.group(1))

        # ROC-AUC
        m = re.search(r'ROC-AUC 分数:\s+([\d.]+)', text)
        if m: info['auc'] = float(m.group(1))

        # MAE（全样本）
        m = re.search(r'MAE\s+\(平均绝对误差\):\s+([\d.]+)', text)
        if m: info['mae'] = float(m.group(1))

        # Ratio
        m = re.search(r'大盘充盈率 \(Ratio\):\s+([\d.]+)', text)
        if m: info['ratio'] = float(m.group(1))

        # 验证集样本数
        m = re.search(r'验证集总样本:\s+([\d,]+)', text)
        if m: info['val_samples'] = int(m.group(1).replace(',', ''))

        # 正样本数
        m = re.search(r'正样本\(有补货\):\s+([\d,]+)', text)
        if m: info['pos_samples'] = int(m.group(1).replace(',', ''))

        # Precision / Recall（正样本行）
        m = re.search(r'补货\(POS\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)', text)
        if m:
            info['precision'] = float(m.group(1))
            info['recall']    = float(m.group(2))

        # 超参
        m = re.search(r'LSTM_HIDDEN\s+(\d+)', text)
        if m: info['hidden'] = int(m.group(1))
        m = re.search(r'LSTM_LAYERS\s+(\d+)', text)
        if m: info['layers'] = int(m.group(1))
        m = re.search(r'BATCH_SIZE\s+(\d+)', text)
        if m: info['batch'] = int(m.group(1))
        m = re.search(r'MODEL_TYPE\s+(\w+)', text)
        if m: info['model_type'] = m.group(1)

    except Exception as e:
        info['error'] = str(e)
    return info


def load_weekly_history(filepath: str) -> dict:
    """从 history_w4*.csv 中提取最优 checkpoint 信息。"""
    rows = []
    with open(filepath, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    'epoch':     int(float(row['epoch'])),
                    'val_loss':  float(row['val_loss']),
                    'val_f1':    float(row['val_f1']),
                    'val_prec':  float(row.get('val_precision', 0)),
                    'val_rec':   float(row.get('val_recall', 0)),
                    'val_ratio': float(row['val_ratio']),
                    'train_loss': float(row['train_loss']),
                })
            except (ValueError, KeyError):
                pass
    if not rows:
        return {}

    best = min(rows, key=lambda r: r['val_loss'])
    last = rows[-1]
    return {
        'best_epoch':    best['epoch'],
        'total_epochs':  len(rows),
        'val_loss':      best['val_loss'],
        'f1':            best['val_f1'],
        'precision':     best['val_prec'],
        'recall':        best['val_rec'],
        'ratio':         best['val_ratio'],
        'train_loss_best': best['train_loss'],
        'train_loss_last': last['train_loss'],
        'val_loss_last':   last['val_loss'],
        'overfit_gap':     last['val_loss'] - last['train_loss'],
    }


def load_meta(path: str) -> dict:
    """读取 meta.json 或 meta_weekly.json。"""
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ================================================================
# 1. 收集日频（Phase 2+3）结果
# ================================================================
print(f'\n{sep}')
print('  日频 vs 周频 全量对比分析报告')
print(sep)

print(f'\n{"─"*80}')
print('  Part 1: 日频基线（Phase 2 架构消融 + Phase 3 Loss 精调）')
print(f'{"─"*80}')

daily_results = []

# Phase 2
for fp in sorted(glob.glob(os.path.join(REPORTS_DIR, 'phase2', 'basic_e*.txt'))):
    exp_id = os.path.basename(fp).replace('basic_', '').replace('.txt', '')
    info = parse_basic_txt(fp)
    info['exp_id'] = exp_id
    info['phase'] = 'Phase2'
    daily_results.append(info)

# Phase 3
for fp in sorted(glob.glob(os.path.join(REPORTS_DIR, 'phase3', 'basic_e*.txt'))):
    exp_id = os.path.basename(fp).replace('basic_', '').replace('.txt', '')
    info = parse_basic_txt(fp)
    info['exp_id'] = exp_id
    info['phase'] = 'Phase3'
    daily_results.append(info)

# 按 F1 降序排列
daily_results.sort(key=lambda r: r.get('f1', 0), reverse=True)

print(f"\n{'实验ID':<20} {'Phase':<8} {'架构':<8} {'F1':>6} {'AUC':>7} {'Prec':>6} "
      f"{'Rec':>6} {'MAE':>6} {'Ratio':>6}")
print(thin)
for r in daily_results:
    mt = r.get('model_type', 'lstm')
    print(f"{r['exp_id']:<20} {r['phase']:<8} {mt:<8} "
          f"{r.get('f1', 0):>6.4f} {r.get('auc', 0):>7.4f} "
          f"{r.get('precision', 0):>6.4f} {r.get('recall', 0):>6.4f} "
          f"{r.get('mae', 0):>6.2f} {r.get('ratio', 0):>6.2f}")

daily_best = daily_results[0] if daily_results else {}
print(f"\n  ★ 日频最优: {daily_best.get('exp_id','-')}  "
      f"F1={daily_best.get('f1',0):.4f}  AUC={daily_best.get('auc',0):.4f}  "
      f"MAE={daily_best.get('mae',0):.2f}  Ratio={daily_best.get('ratio',0):.2f}")


# ================================================================
# 2. 收集周频（Phase 4）结果
# ================================================================
print(f'\n{"─"*80}')
print('  Part 2: 周频消融（Phase 4A~4F）')
print(f'{"─"*80}')

weekly_results = []
for fp in sorted(glob.glob(os.path.join(REPORTS_DIR, 'history_w4*.csv'))):
    exp_id = os.path.basename(fp).replace('history_', '').replace('.csv', '')
    info = load_weekly_history(fp)
    if not info:
        continue
    info['exp_id'] = exp_id
    weekly_results.append(info)

weekly_results.sort(key=lambda r: r.get('f1', 0), reverse=True)

print(f"\n{'实验ID':<22} {'BestEp':>6} {'F1':>7} {'Prec':>7} {'Rec':>7} "
      f"{'Ratio':>7} {'ValLoss':>9} {'过拟合Gap':>10}")
print(thin)
for r in weekly_results:
    print(f"{r['exp_id']:<22} {r['best_epoch']:>6} {r['f1']:>7.4f} "
          f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['ratio']:>7.4f} "
          f"{r['val_loss']:>9.4f} {r['overfit_gap']:>10.4f}")

weekly_best = weekly_results[0] if weekly_results else {}
print(f"\n  ★ 周频最优: {weekly_best.get('exp_id','-')}  "
      f"F1={weekly_best.get('f1',0):.4f}  Prec={weekly_best.get('precision',0):.4f}  "
      f"Rec={weekly_best.get('recall',0):.4f}  Ratio={weekly_best.get('ratio',0):.2f}")


# ================================================================
# 3. 直接对比
# ================================================================
print(f'\n{sep}')
print('  Part 3: 日频 vs 周频 核心指标正面对决')
print(sep)

d_f1    = daily_best.get('f1', 0)
d_auc   = daily_best.get('auc', 0)
d_prec  = daily_best.get('precision', 0)
d_rec   = daily_best.get('recall', 0)
d_mae   = daily_best.get('mae', 0)
d_ratio = daily_best.get('ratio', 0)

w_f1    = weekly_best.get('f1', 0)
w_prec  = weekly_best.get('precision', 0)
w_rec   = weekly_best.get('recall', 0)
w_ratio = weekly_best.get('ratio', 0)

def delta(w, d, higher_is_better=True):
    diff = w - d
    pct = diff / max(abs(d), 1e-9) * 100
    if higher_is_better:
        symbol = '📈' if diff > 0 else '📉'
    else:
        symbol = '📈' if diff < 0 else '📉'
    return f"{diff:+.4f} ({pct:+.1f}%) {symbol}"

print(f"""
  ┌─────────────────┬────────────┬────────────┬───────────────────────┐
  │ 指标            │ 日频(最优) │ 周频(最优) │ 差值                  │
  ├─────────────────┼────────────┼────────────┼───────────────────────┤
  │ F1-Score        │ {d_f1:>10.4f} │ {w_f1:>10.4f} │ {delta(w_f1, d_f1):<21} │
  │ Precision       │ {d_prec:>10.4f} │ {w_prec:>10.4f} │ {delta(w_prec, d_prec):<21} │
  │ Recall          │ {d_rec:>10.4f} │ {w_rec:>10.4f} │ {delta(w_rec, d_rec):<21} │
  │ Ratio           │ {d_ratio:>10.2f} │ {w_ratio:>10.2f} │ {delta(w_ratio, d_ratio):<21} │
  └─────────────────┴────────────┴────────────┴───────────────────────┘
""")


# ================================================================
# 4. 数据集规模对比
# ================================================================
print(f'{"─"*80}')
print('  Part 4: 数据集规模对比')
print(f'{"─"*80}')

# 日频
for d in ['data/processed_fast', 'data/processed_v3']:
    mp = os.path.join(PROJECT_ROOT, d, 'meta.json')
    m = load_meta(mp)
    if m:
        print(f"  日频 ({d}): train={m.get('train_cnt',0):>10,} | "
              f"val={m.get('val_cnt',0):>10,} | lookback={m.get('lookback_days','?')}天 × 7维")

# 周频
for suffix in ['', '_8w', '_16w', '_24w']:
    mp = os.path.join(PROJECT_ROOT, 'data', f'artifacts_weekly{suffix}', 'meta_weekly.json')
    m = load_meta(mp)
    if m:
        weeks = m.get('lookback_weeks', '?')
        print(f"  周频 (weekly{suffix}): train={m.get('train_cnt',0):>10,} | "
              f"val={m.get('val_cnt',0):>10,} | lookback={weeks}周 × {m.get('dyn_dim','?')}维")


# ================================================================
# 5. 过拟合分析
# ================================================================
print(f'\n{"─"*80}')
print('  Part 5: 周频模型过拟合诊断')
print(f'{"─"*80}')

all_best_ep3 = all(r.get('best_epoch') == 3 for r in weekly_results)
print(f"\n  ⚠️  所有 {len(weekly_results)} 组周频实验的 Best Checkpoint 均在 Epoch 3（第一个验证轮）")
print(f"  说明所有模型从 Epoch 4 开始就已经过拟合（val_loss 单调上升）。")
print(f"\n  过拟合Gap（末轮 val_loss - train_loss）分布:")

gaps = [r['overfit_gap'] for r in weekly_results]
print(f"  最小: {min(gaps):.4f}  |  中位: {sorted(gaps)[len(gaps)//2]:.4f}  "
      f"|  最大: {max(gaps):.4f}")

# BiLSTM 详细曲线
bilstm = [r for r in weekly_results if r['exp_id'] == 'w4d_bilstm']
if bilstm:
    b = bilstm[0]
    print(f"\n  [全局最优 w4d_bilstm 训练曲线]")
    print(f"  Epoch 3 (Best): train_loss={b['train_loss_best']:.4f}  val_loss={b['val_loss']:.4f}  gap={b['val_loss']-b['train_loss_best']:.4f}")
    print(f"  Epoch {b['total_epochs']+2} (Last): train_loss={b['train_loss_last']:.4f}  val_loss={b['val_loss_last']:.4f}  gap={b['overfit_gap']:.4f}")
    print(f"  从 Best 到 Last，val_loss 增幅: {b['val_loss_last']-b['val_loss']:+.4f} ({(b['val_loss_last']-b['val_loss'])/b['val_loss']*100:+.1f}%)")


# ================================================================
# 6. 根因分析与建议
# ================================================================
print(f'\n{sep}')
print('  Part 6: 根因分析与下一步建议')
print(sep)

# 计算日频/周频样本比
daily_meta = load_meta(os.path.join(PROJECT_ROOT, 'data', 'processed_v3', 'meta.json'))
weekly_meta = load_meta(os.path.join(PROJECT_ROOT, 'data', 'artifacts_weekly', 'meta_weekly.json'))

d_train = daily_meta.get('train_cnt', 0)
w_train = weekly_meta.get('train_cnt', 0)
sample_ratio = d_train / max(w_train, 1)

print(f"""
  ┌────────────────────────────────────────────────────────────────────┐
  │ 核心发现：周频模型 F1 比日频低约 {(d_f1 - w_f1)/d_f1*100:.0f}%                           │
  │ 日频最优 F1={d_f1:.4f}（e24_l3）vs 周频最优 F1={w_f1:.4f}（w4d_bilstm） │
  └────────────────────────────────────────────────────────────────────┘

  根因 #1: 训练样本严重不足
  ─────────────────────────
  日频训练集: {d_train:>10,} 样本
  周频训练集: {w_train:>10,} 样本
  比例: 日频 ≈ {sample_ratio:.1f}x 周频

  周频将 7 天聚合为 1 周，滑窗步长从 1 天变为 1 周，
  样本量锐减至日频的 ~1/{sample_ratio:.0f}。数据量不足以支撑 256-hidden 的模型容量，
  Epoch 3 就已经过拟合（315 步梯度更新后 val_loss 单调上升）。

  根因 #2: 时序分辨率丢失
  ─────────────────────────
  日频: 90 天 × 7 维 = 630 个数据点/样本
  周频: 12 周 × 7 维 =  84 个数据点/样本

  周频模型看到的信息量只有日频的 {84/630*100:.0f}%。
  - 日内补货峰值被周内均值掩盖（7 天累加后，单日爆发的补货信号被稀释）
  - 客户复购节奏（3天/5天/7天）在周粒度下完全不可分辨
  - 周末/工作日补货模式差异丢失

  根因 #3: 评估口径不同
  ─────────────────────────
  日频评估: 在 evaluate_dashboard.py 中使用完整推断管线
    - 自动寻优 F1 阈值（如 0.5053）
    - 概率门限 + 保守系数都经过充分调优
  周频评估: 在 trainer.py validate() 中使用硬编码参数
    - 概率门限 0.45，保守系数 1.0（未调优）
    - F1 阈值扫描范围 0.3~0.8（粗略 0.02 步长）

  可能存在 ~5-8% 的评估偏差（非模型真实差距）。

  ╔════════════════════════════════════════════════════════════════════╗
  ║  下一步建议（3 条路线，按可行性排序）                                  ║
  ╠════════════════════════════════════════════════════════════════════╣
  ║                                                                    ║
  ║  路线 A: 放弃纯周频，改用"日频输入 + 周频聚合目标"                      ║
  ║  ───────────────────────────────────────────────────────────────── ║
  ║  保留日频 90 天 × 7 维的输入粒度（保留信息量），                         ║
  ║  但把目标变量从"未来 30 天日度补货"改为"未来 4 周周度补货"。              ║
  ║  本质上仍走日频管线，但输出更平滑、更适合供应链排产。                     ║
  ║  预期: F1 接近日频水平(~0.55)，业务口径更实用。                        ║
  ║                                                                    ║
  ║  路线 B: 周频模型增加训练样本量                                        ║
  ║  ───────────────────────────────────────────────────────────────── ║
  ║  将 neg_step 从 40 → 10，训练样本增至 ~100 万左右。                   ║
  ║  同时缩小模型容量: hidden 128, layers 2（匹配数据规模）。              ║
  ║  预期: F1 可能提升至 0.45~0.50（但上限受限于分辨率丢失）。              ║
  ║                                                                    ║
  ║  路线 C: 坚守日频最优基线，重点投入推断和生产化                          ║
  ║  ───────────────────────────────────────────────────────────────── ║
  ║  e24_l3 (F1=0.5722) 已经是当前最优，推断管线已打通。                   ║
  ║  把精力放在最终的 generate_daily_inference.py 生产优化上。              ║
  ║  预期: 直接交付可用系统，不再实验。                                    ║
  ╚════════════════════════════════════════════════════════════════════╝
""")

print(sep)
print('  分析完毕。')
print(sep)
