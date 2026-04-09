"""analyze_phase2_timing.py — 从历史日志分析Phase2单epoch耗时（供Phase5实验估时）"""
import os, glob, csv

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("="*65)
print("  Phase 2/4 历史实验耗时分析")
print("="*65)

# 1. 列出所有 history_*.csv
for pattern in ['reports/history_e2*.csv', 'reports/history_e4*.csv']:
    files = sorted(glob.glob(os.path.join(PROJECT_ROOT, pattern)))
    for f in files:
        size = os.path.getsize(f)
        basename = os.path.basename(f)
        # 读取行数和最后一行
        with open(f, 'r', encoding='utf-8') as fh:
            lines = fh.readlines()
        n_epochs = len(lines) - 1  # 减掉 header
        if n_epochs > 0:
            header = lines[0].strip()
            last = lines[-1].strip()
            # 解析 duration_min
            parts = last.split(',')
            dur_min = parts[-1] if parts else '?'
            print(f"\n  [{basename}]  {n_epochs} epochs, 末行 duration_min={dur_min}")
            # 解析所有epoch的duration
            durations = []
            for line in lines[1:]:
                p = line.strip().split(',')
                try:
                    durations.append(float(p[-1]))
                except:
                    pass
            if durations:
                avg_min = sum(durations) / len(durations)
                total_min = sum(durations)
                print(f"    均耗时: {avg_min:.1f} min/epoch | 总耗时: {total_min:.0f} min ({total_min/60:.1f}h)")
                print(f"    F1(最后): {p[4] if len(p)>4 else '?'} | Ratio(最后): {p[7] if len(p)>7 else '?'}")

# 2. 估算 Phase 5 时间
print("\n" + "="*65)
print("  Phase 5 耗时估算")
print("="*65)
# 假设 LSTM L3 batch=1024 大约 25 min/epoch
# BiLSTM L3 大约 35 min/epoch (2x 参数)
# V5 10维增加约 15% 开销
configs = [
    ("e51 LSTM L3 V3",        25, 12),   # 25 min/ep, 预计12 epochs
    ("e52 BiLSTM L3 V3",      35, 12),
    ("e53 LSTM L3 V5",        29, 15),   # +15% for 10dim
    ("e54 BiLSTM L3 V5",      40, 15),
    ("e55 BiLSTM L3 V5 lr",   40, 15),
    ("e56 BiLSTM L3 V5 lr",   40, 15),
    ("e57 LSTM L3 V5 lr",     29, 15),
    ("e58 BiLSTM L3 V5 +",    40, 15),
    ("e59 BiLSTM L3 V5 +",    40, 15),
    ("e60 BiLSTM L3 V5 +",    40, 15),
]
total_h = 0
for name, min_ep, est_ep in configs:
    est_h = min_ep * est_ep / 60
    total_h += est_h
    print(f"  {name:<28} ~{min_ep} min/ep × {est_ep} ep = {est_h:.1f}h")
print(f"\n  总预估: {total_h:.1f}h ({total_h/12:.1f} 个12小时晚上)")
