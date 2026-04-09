"""
B2B 补货系统 Phase 4 — 周粒度挂机实验总脚本
============================================================
设计目标：10+ 小时无人值守，自动完成三轮消融：

Phase 4A：回看窗口消融（4 组）
  w8 / w12 / w16 / w24 — 探明最优历史记忆长度

Phase 4B：架构消融（在最优回看上进行，3 组）
  L2H256 / L3H256(基准) / L4H256 — 探明深度上限

Phase 4C：Loss 精调（在当前最优配置上，3 组）
  reg_weight 0.3 / 0.5(基准) / 0.8 — 对齐之前日频的结论

运行方式：
  python run_weekly_experiments.py

隔离原则（铁律）：
  - 每组实验模型保存到 models_weekly/<exp_id>/
  - 特征数据保存到 data/processed_weekly_Xw/（不覆盖 12w 基准）
  - 所有日志记录到 reports/phase4_timing_log.csv
"""

import os
import sys
import subprocess
import time
import csv
from datetime import datetime

# Windows UTF-8
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR  = os.path.join(PROJECT_ROOT, 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

TIMING_LOG = os.path.join(REPORTS_DIR, 'phase4_timing_log.csv')


def log_timing(exp_id, lookback, epochs, start_t, end_t, status, note=''):
    elapsed = (end_t - start_t) / 60.0
    with open(TIMING_LOG, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            exp_id, lookback, epochs,
            datetime.fromtimestamp(start_t).strftime('%H:%M:%S'),
            datetime.fromtimestamp(end_t).strftime('%H:%M:%S'),
            f'{elapsed:.2f}', status, note
        ])


def init_log():
    """初始化时间日志（追加模式，不会清空已有内容）。"""
    if not os.path.exists(TIMING_LOG):
        with open(TIMING_LOG, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['exp_id', 'lookback_weeks', 'total_epochs',
                             'start_time', 'end_time', 'elapsed_min', 'status', 'note'])


def build_features(lookback_weeks: int) -> bool:
    """
    为指定的回看窗口生成周频特征张量。
    如果对应目录已存在且**完整**，则跳过（避免重复消耗时间）。
    完整性判断：8 个 bin 文件全部存在 + meta_weekly.json 中 train_cnt > 0。
    """
    suffix = f'_{lookback_weeks}w' if lookback_weeks != 12 else ''
    proc_dir = os.path.join(PROJECT_ROOT, 'data', f'processed_weekly{suffix}')
    art_dir  = os.path.join(PROJECT_ROOT, 'data', f'artifacts_weekly{suffix}')

    # 1. 检查 8 个 bin 文件均存在且大小合理（> 1MB 表明不是空壳）
    required_bins = [
        'X_train_dyn.bin', 'X_train_static.bin', 'y_train_cls.bin', 'y_train_reg.bin',
        'X_val_dyn.bin',   'X_val_static.bin',   'y_val_cls.bin',   'y_val_reg.bin',
    ]
    bins_ok = all(
        os.path.exists(os.path.join(proc_dir, f)) and
        os.path.getsize(os.path.join(proc_dir, f)) > 1024 * 1024  # > 1MB
        for f in required_bins
    )

    # 2. 检查 meta_weekly.json 中 train_cnt > 0（排除空跑产物）
    meta_path = os.path.join(art_dir, 'meta_weekly.json')
    meta_ok = False
    if os.path.exists(meta_path):
        try:
            import json
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            meta_ok = meta.get('train_cnt', 0) > 0
        except Exception:
            meta_ok = False

    if bins_ok and meta_ok:
        with open(meta_path, 'r', encoding='utf-8') as f:
            import json; m = json.load(f)
        print(f"  ✅ [跳过特征工程] processed_weekly{suffix}/ 完整（train={m['train_cnt']:,}，"
              f"val={m['val_cnt']:,}），直接使用。")
        return True

    # 3. 如有残缺文件，先清理避免 memmap 读取错误
    if os.path.exists(proc_dir):
        import shutil
        print(f"  ⚠️  [清理残缺] processed_weekly{suffix}/ 不完整，重新生成...")
        shutil.rmtree(proc_dir)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(art_dir, exist_ok=True)

    print(f"\n  🔨 [特征工程] 生成 {lookback_weeks} 周回看窗口数据...")
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, 'src', 'features', 'build_features_weekly_sku.py'),
        '--lookback', str(lookback_weeks)
    ]
    ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if ret.returncode != 0:
        print(f"  ❌ 特征工程失败！returncode={ret.returncode}")
        return False
    print(f"  ✅ 特征工程完成: lookback={lookback_weeks}周")
    return True



def run_training(exp_id: str, lookback_weeks: int,
                 hidden: int, layers: int, dropout: float,
                 batch: int, lr: float, epochs: int, patience: int,
                 model_type: str = 'lstm',
                 reg_weight: float = 0.5, soft_f1: float = 0.5, huber: float = 1.5) -> int:
    """
    调用 run_training_weekly.py 训练一组实验，自动指向对应的周频数据目录。
    返回进程退出码。
    """
    suffix = f'_{lookback_weeks}w' if lookback_weeks != 12 else ''
    proc_dir   = os.path.join(PROJECT_ROOT, 'data', f'processed_weekly{suffix}')
    art_dir    = os.path.join(PROJECT_ROOT, 'data', f'artifacts_weekly{suffix}')
    save_dir   = os.path.join(PROJECT_ROOT, 'models_weekly', exp_id)
    os.makedirs(save_dir, exist_ok=True)

    env = os.environ.copy()
    env['EXP_ID']         = exp_id
    env['EXP_MODEL_TYPE'] = model_type          # ★ 展比参数化，不再硬编码’lstm’
    env['EXP_HIDDEN']     = str(hidden)
    env['EXP_LAYERS']     = str(layers)
    env['EXP_DROPOUT']    = str(dropout)
    env['EXP_BATCH']      = str(batch)
    env['EXP_LR']         = str(lr)
    env['EXP_EPOCHS']     = str(epochs)
    env['EXP_PATIENCE']   = str(patience)
    # Loss 热注入（★ 全部显式设置，防止父进程残留影响对照实验）
    env['EXP_REG_WEIGHT'] = str(reg_weight)
    env['EXP_SOFT_F1']    = str(soft_f1)
    env['EXP_HUBER']      = str(huber)
    env['EXP_POS_WEIGHT'] = '5.85'     # ★ 显式设置，对照实验组固定此值
    env['EXP_FP_PENALTY'] = '0.15'     # ★ 显式设置，对照实验组固定此值
    # 数据目录覆盖（让 run_training_weekly.py 读对应的目录）
    env['WEEKLY_PROC_DIR'] = proc_dir
    env['WEEKLY_ART_DIR']  = art_dir
    env['WEEKLY_SAVE_DIR'] = save_dir

    cmd = [sys.executable, os.path.join(PROJECT_ROOT, 'run_training_weekly.py')]
    ret = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env)
    return ret.returncode


def run_experiment(exp_id: str, desc: str, lookback_weeks: int,
                   hidden: int, layers: int, dropout: float,
                   batch: int, lr: float, epochs: int, patience: int,
                   model_type: str = 'lstm',
                   reg_weight: float = 0.5, soft_f1: float = 0.5, huber: float = 1.5):
    """完整的一组实验流程：特征工程 + 训练 + 记录日志。"""
    bar = "=" * 65
    print(f"\n{bar}")
    print(f"  🚀 {exp_id}")
    print(f"  📖 {desc}")
    print(f"  🔧 lookback={lookback_weeks}w | {model_type.upper()} L{layers}H{hidden} | batch={batch} | lr={lr}")
    print(f"     Loss: reg_w={reg_weight} | soft_f1={soft_f1} | huber={huber}")
    print(bar)

    # Step 1: 特征工程（如已有则跳过）
    feat_ok = build_features(lookback_weeks)
    if not feat_ok:
        log_timing(exp_id, lookback_weeks, 0, time.time(), time.time(), 'feat_error')
        return

    # Step 2: 训练
    t0 = time.time()
    rc = run_training(
        exp_id=exp_id, lookback_weeks=lookback_weeks,
        hidden=hidden, layers=layers, dropout=dropout,
        batch=batch, lr=lr, epochs=epochs, patience=patience,
        model_type=model_type,
        reg_weight=reg_weight, soft_f1=soft_f1, huber=huber
    )
    t1 = time.time()

    status = 'success' if rc == 0 else f'error(rc={rc})'
    log_timing(exp_id, lookback_weeks, epochs, t0, t1, status)
    print(f"\n  ⏱️  耗时 {(t1-t0)/60:.1f} 分钟 | 状态: {status}")


# ==============================================================
# 实验定义
# ==============================================================

# 共用超参（控制变量时固定，变量实验中只改一个维度）
# ★ 关键修正：周频每轮只有 105 个 batch（日频约 1894），
#   需要提高 LR 和 Epochs 才能达到相近的梯度步数：
#   50 epochs × 105 batch = 5,250 步  vs  日频 15~20 epoch × 1894 = ~28,000 步
#   虽然仍有差距，但配合更高 LR 可以在更少步数内完成收敛
BASE_HIDDEN   = 256
BASE_LAYERS   = 3
BASE_DROPOUT  = 0.3
BASE_BATCH    = 4096
BASE_LR       = 2e-4    # ★ 修正: 从 5e-5 → 2e-4，与日频 e24_l3 基准一致
BASE_EPOCHS   = 50      # ★ 修正: 从 30 → 50，补偿每轮 batch 数少的问题
BASE_PATIENCE = 8       # ★ 修正: 从 5 → 8，给 ReduceLROnPlateau 足够时间恢复


EXPERIMENTS = [

    # ----------------------------------------------------------
    # Phase 4A：回看窗口消融（固定 L3H256，只改 lookback）
    # 目标：找出最优历史记忆长度
    # ----------------------------------------------------------
    dict(
        exp_id='w4a_look_8',
        desc='回看窗口消融：8周（约2个月短记忆）',
        lookback_weeks=8,
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),
    # w12 基准在外部已经运行，这里仍然补跑一次做记录对齐
    dict(
        exp_id='w4a_look_12',
        desc='回看窗口消融：12周（约3个月，当前基准）',
        lookback_weeks=12,
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),
    dict(
        exp_id='w4a_look_16',
        desc='回看窗口消融：16周（约4个月，跨换季）',
        lookback_weeks=16,
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),
    dict(
        exp_id='w4a_look_24',
        desc='回看窗口消融：24周（约6个月，完整半年周期）',
        lookback_weeks=24,
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),

    # ----------------------------------------------------------
    # Phase 4B：架构消融（固定 lookback=12，只改层数）
    # 目标：周频数据下，深度是否仍然是关键（对标日频 Phase 2）
    # ----------------------------------------------------------
    dict(
        exp_id='w4b_arch_l2',
        desc='架构消融：2层LSTM（对比日频 Phase2 基准 L2H256）',
        lookback_weeks=12,
        hidden=256, layers=2,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),
    dict(
        exp_id='w4b_arch_l4',
        desc='架构消融：4层LSTM（深度上探，只有周频显存够用）',
        lookback_weeks=12,
        hidden=256, layers=4,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),
    dict(
        exp_id='w4b_arch_h384',
        desc='架构消融：H384（宽度扩展，对标日频 e22_h384）',
        lookback_weeks=12,
        hidden=384, layers=3,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),

    # ----------------------------------------------------------
    # Phase 4C：Loss 精调（固定最优架构，只改损失函数权重）
    # 目标：验证日频得到的 Loss 配置是否在周频下同样成立
    # ----------------------------------------------------------
    dict(
        exp_id='w4c_rw_low',
        desc='Loss精调：reg_weight=0.3（同日频 e31，分类优先）',
        lookback_weeks=12,
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
        reg_weight=0.3, soft_f1=0.5, huber=1.5,
    ),
    dict(
        exp_id='w4c_rw_high',
        desc='Loss精调：reg_weight=0.8（同日频 e32，回归增强）',
        lookback_weeks=12,
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
        reg_weight=0.8, soft_f1=0.5, huber=1.5,
    ),
    dict(
        exp_id='w4c_sf_high',
        desc='Loss精调：soft_f1=1.0（高 Soft-F1 驱动精准率）',
        lookback_weeks=12,
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
        reg_weight=0.5, soft_f1=1.0, huber=1.5,
    ),
]

# ==============================================================
# Phase 4D: RNN 架构类型消融（固定 L3H256 + 12w）
# 目标：GRU / BiLSTM / 注意力 对比，找最适合周频稀疏时序的 RNN 类型
# ==============================================================
EXPERIMENTS += [
    dict(
        exp_id='w4d_gru',
        desc='RNN类型消融：GRU（参数少25%，稀疏时序泛化性可能更好）',
        lookback_weeks=12, model_type='gru',
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),
    dict(
        exp_id='w4d_bilstm',
        desc='RNN类型消融：BiLSTM（双向读取12周历史，尾期爆款信号对早期可见）',
        lookback_weeks=12, model_type='bilstm',
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),
    dict(
        exp_id='w4d_attn',
        desc='RNN类型消融：LSTM+Attn（对12周加权聚合，学习哪几周订单波峰最关键）',
        lookback_weeks=12, model_type='attn',
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),
]

# ==============================================================
# Phase 4E: 学习率消融（固定 LSTM L3H256 + 12w）
# 目标：确认周频模型的最优 LR，补充 2e-4 基准两侧的数据点
# ==============================================================
EXPERIMENTS += [
    dict(
        exp_id='w4e_lr_low',
        desc='LR消融：LR=1e-4（保守收敛，梯度更小心）',
        lookback_weeks=12,
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=1e-4, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),
    dict(
        exp_id='w4e_lr_high',
        desc='LR消融：LR=3e-4（进攻收敛，依赖 ReduceLROnPlateau 自动救回）',
        lookback_weeks=12,
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=BASE_DROPOUT, batch=BASE_BATCH,
        lr=3e-4, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),
]

# ==============================================================
# Phase 4F: Dropout 消融（固定 LSTM L3H256 + 12w）
# 目标：周频数据零占比从78%降至30%+，正则化需求可能与日频不同
# ==============================================================
EXPERIMENTS += [
    dict(
        exp_id='w4f_drop_low',
        desc='Dropout消融：dropout=0.1（弱正则化，周频数据较密时可能更优）',
        lookback_weeks=12,
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=0.1, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),
    dict(
        exp_id='w4f_drop_high',
        desc='Dropout消融：dropout=0.5（强正则化，防止低稀疏度数据过拟合）',
        lookback_weeks=12,
        hidden=BASE_HIDDEN, layers=BASE_LAYERS,
        dropout=0.5, batch=BASE_BATCH,
        lr=BASE_LR, epochs=BASE_EPOCHS, patience=BASE_PATIENCE,
    ),
]


def main():
    init_log()

    total = len(EXPERIMENTS)
    print("=" * 65)
    print("  🌙 B2B Phase 4 挂机实验 — 周粒度全量消融")
    print(f"  📋 共 {total} 组实验 | 预计耗时 8~12 小时")
    print(f"  ⛰  4A回看窗口 | 4B LSTM深度/宽度 | 4C Loss精调")
    print(f"      4D RNN类型 | 4E 学习率 | 4F Dropout")
    print(f"  ⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"\n[{i}/{total}] ▶ 即将启动: {exp['exp_id']}")
        run_experiment(**exp)
        print(f"[{i}/{total}] ✅ 完成: {exp['exp_id']}")

    print("\n" + "=" * 65)
    print("  🎉 全部实验完成！")
    print(f"  ⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  📊 时间日志: {TIMING_LOG}")
    print("=" * 65)


if __name__ == '__main__':
    main()
