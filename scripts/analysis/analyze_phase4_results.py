"""
Phase 4 实验结果全量分析
=========================
读取 reports/history_w4*.csv，按最优 F1 排名，
分组打印各阶段消融结论，并给出最优配置推荐。
"""
import csv
import os
import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR  = os.path.join(PROJECT_ROOT, 'reports')


# ==============================================================
# 1. 读取所有实验历史
# ==============================================================
def load_history(fp: str) -> list[dict]:
    rows = []
    with open(fp, encoding='utf-8') as f:
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
                })
            except (ValueError, KeyError):
                pass
    return rows


results = []
for fp in sorted(glob.glob(os.path.join(REPORTS_DIR, 'history_w4*.csv'))):
    exp_id = os.path.basename(fp).replace('history_', '').replace('.csv', '')
    rows = load_history(fp)
    if not rows:
        continue

    # 以 val_loss 最低的轮次作为"最优 checkpoint"
    best = min(rows, key=lambda r: r['val_loss'])
    # 最后一轮（反映收敛终态）
    last = rows[-1]

    results.append({
        'exp_id':       exp_id,
        'best_epoch':   best['epoch'],
        'total_epochs': len(rows),
        'best_val_loss': best['val_loss'],
        'best_f1':      best['val_f1'],
        'best_prec':    best['val_prec'],
        'best_rec':     best['val_rec'],
        'best_ratio':   best['val_ratio'],
        'last_f1':      last['val_f1'],
        'last_ratio':   last['val_ratio'],
    })


# ==============================================================
# 2. 全量排名（按 Best F1 降序）
# ==============================================================
results.sort(key=lambda r: r['best_f1'], reverse=True)

sep = '=' * 78
print(f'\n{sep}')
print('  Phase 4 实验全量排名（按 Best F1 降序，Checkpoint=val_loss 最低轮）')
print(sep)
print(f"{'实验 ID':<22}  {'最优Epoch':>8}  {'Val Loss':>9}  "
      f"{'Best F1':>8}  {'Precision':>9}  {'Recall':>8}  {'Ratio':>7}  {'总轮数':>5}")
print('-' * 78)
for r in results:
    print(f"{r['exp_id']:<22}  {r['best_epoch']:>8}  {r['best_val_loss']:>9.6f}  "
          f"{r['best_f1']:>8.4f}  {r['best_prec']:>9.4f}  {r['best_rec']:>8.4f}  "
          f"{r['best_ratio']:>7.4f}  {r['total_epochs']:>5}")

# 方便引用
def get(eid):
    for r in results:
        if r['exp_id'] == eid:
            return r
    return {}


# ==============================================================
# 3. 分组消融对比
# ==============================================================
def group_summary(title, exp_ids):
    print(f'\n{"─"*78}')
    print(f'  {title}')
    print(f'{"─"*78}')
    group = [r for r in results if r['exp_id'] in exp_ids]
    group.sort(key=lambda r: r['best_f1'], reverse=True)
    for r in group:
        marker = '  ★ 最优' if r == group[0] else ''
        print(f"  {r['exp_id']:<22}  F1={r['best_f1']:.4f}  "
              f"Prec={r['best_prec']:.4f}  Rec={r['best_rec']:.4f}  "
              f"Ratio={r['best_ratio']:.4f}  Epoch={r['best_epoch']}{marker}")

group_summary(
    'Phase 4A — 回看窗口消融（固定 LSTM L3H256）',
    ['w4a_look_8', 'w4a_look_12', 'w4a_look_16', 'w4a_look_24']
)
group_summary(
    'Phase 4B — 架构深度/宽度消融（固定 12w，LSTM）',
    ['w4b_arch_l2', 'w4a_look_12', 'w4b_arch_l4', 'w4b_arch_h384']
)
group_summary(
    'Phase 4C — Loss 权重精调（固定 LSTM L3H256, 12w）',
    ['w4c_rw_low', 'w4a_look_12', 'w4c_rw_high', 'w4c_sf_high']
)
group_summary(
    'Phase 4D — RNN 类型消融（固定 L3H256, 12w）',
    ['w4a_look_12', 'w4d_gru', 'w4d_bilstm', 'w4d_attn']
)
group_summary(
    'Phase 4E — 学习率消融（固定 LSTM L3H256, 12w）',
    ['w4e_lr_low', 'w4a_look_12', 'w4e_lr_high']
)
group_summary(
    'Phase 4F — Dropout 消融（固定 LSTM L3H256, 12w）',
    ['w4f_drop_low', 'w4a_look_12', 'w4f_drop_high']
)

# ==============================================================
# 4. 最优配置推荐
# ==============================================================
print(f'\n{sep}')
print('  综合最优配置推荐')
print(sep)

top3 = results[:3]
print('\n  Top-3 实验（全局 Best F1）:')
for i, r in enumerate(top3, 1):
    print(f'  {i}. {r["exp_id"]}  F1={r["best_f1"]:.4f}  '
          f'Prec={r["best_prec"]:.4f}  Rec={r["best_rec"]:.4f}  '
          f'Ratio={r["best_ratio"]:.4f}')

# 4A 最优
look_group = sorted(
    [r for r in results if r['exp_id'].startswith('w4a_look')],
    key=lambda r: r['best_f1'], reverse=True
)
# 4D 最优
rnn_group = sorted(
    [r for r in results if r['exp_id'] in ['w4a_look_12', 'w4d_gru', 'w4d_bilstm', 'w4d_attn']],
    key=lambda r: r['best_f1'], reverse=True
)

best_look = look_group[0]['exp_id'] if look_group else 'N/A'
best_rnn  = rnn_group[0]['exp_id']  if rnn_group  else 'N/A'

print(f'\n  ▶ 最优回看窗口 : {best_look}')
print(f'  ▶ 最优 RNN 类型: {best_rnn}')
print(f'  ▶ 全局最优     : {results[0]["exp_id"]}  (F1={results[0]["best_f1"]:.4f})')
print(f'\n  建议: 在全局最优配置上，进行下一阶段超参精调或验证集回测。')
print(f'{sep}\n')
