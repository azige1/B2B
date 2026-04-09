"""
B2B Phase 5 挂机实验流水线 (run_phase5_experiments.py)
=======================================================
12 组实验，分 3 个 Group，严格控制变量。

Group A (2×2 阶乘基线): 分离「架构效果」和「特征效果」
  e51: LSTM L3 + V3[7维]     — 绝对基线
  e52: BiLSTM L3 + V3[7维]   — 纯架构效果
  e53: LSTM L3 + V5[10维]    — 纯特征效果
  e54: BiLSTM L3 + V5[10维]  — 架构×特征交叉

Group B (学习率搜索): 10维新特征下的最优LR
  e55: BiLSTM L3 V5 lr=1e-5  — 极保守学习率
  e56: BiLSTM L3 V5 lr=1e-4  — 2×基线
  e57: BiLSTM L3 V5 lr=2e-4  — 4×基线(周频甜点)
  e58: LSTM L3 V5 lr=1e-4    — 对冲(万一LSTM更优)

Group C (正则化压榨): 基于最优arch+feat，压榨最后性能
  e59: BiLSTM L3 V5 + d=0.4        — 更强Dropout
  e60: BiLSTM L3 V5 + LS=0.08      — Label Smoothing
  e61: BiLSTM L3 V5 + WD=1e-4      — Weight Decay
  e62: BiLSTM L3 V5 + d=0.4+LS=0.08— 组合拳

统一训练设置: epochs=30, patience=7 (相对Phase2的20/5)
"""
import os
import sys
import subprocess
import time
import shutil
import csv
import json
from datetime import datetime, timedelta

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ======================================================================
# 统一训练超参（Phase 5 全局覆盖）
# ======================================================================
GLOBAL_EPOCHS   = os.environ.get("PHASE5_EPOCHS", "30")   # Phase 5 keeps longer training budget
GLOBAL_PATIENCE = os.environ.get("PHASE5_PATIENCE", "7")  # Align with the Phase 5 runbook
GLOBAL_WORKERS  = os.environ.get("PHASE5_WORKERS", "2")

# 默认 Loss 超参（Phase 2 最优配置，全组统一）
DEFAULT_LOSS = {
    "pos_w"  : "5.85",
    "fp"     : "0.15",
    "gamma"  : "1.5",
    "reg_w"  : "0.5",
    "sf1"    : "0.5",
    "hub"    : "1.5",
}

# ======================================================================
# 实验定义
# ======================================================================
EXPERIMENTS = [

    # =================================================================
    # Group A: 2×2 阶乘基线
    # 控制变量: lr=5e-5, batch=1024, d=0.3, epochs=30, patience=7
    # 分析方法:
    #   特征主效应 = (e53+e54)/2 - (e51+e52)/2
    #   架构主效应 = (e52+e54)/2 - (e51+e53)/2
    #   交互效应   = e54 - e53 - e52 + e51
    # =================================================================
    {
        "id"         : "e51_lstm_l3_v3",
        "version"    : "v3",
        "model_type" : "lstm",
        "hidden"     : "256",
        "layers"     : "3",
        "dropout"    : "0.3",
        "batch"      : "1024",
        "lr"         : "0.00005",
        "desc"       : "GroupA 绝对基线: LSTM L3 + V3(7维) + 新指标体系。Phase5的参照原点",
    },
    {
        "id"         : "e52_bilstm_l3_v3",
        "version"    : "v3",
        "model_type" : "bilstm",
        "hidden"     : "256",
        "layers"     : "3",
        "dropout"    : "0.3",
        "batch"      : "1024",
        "lr"         : "0.00005",
        "desc"       : "GroupA 架构效果: BiLSTM L3 + V3(7维)。日频首测BiLSTM L3，隔离架构增益",
    },
    {
        "id"         : "e53_lstm_l3_v5",
        "version"    : "v5",
        "model_type" : "lstm",
        "hidden"     : "256",
        "layers"     : "3",
        "dropout"    : "0.3",
        "batch"      : "1024",
        "lr"         : "0.00005",
        "desc"       : "GroupA 特征效果: LSTM L3 + V5(10维 期货信号)。隔离特征增益",
    },
    {
        "id"         : "e54_bilstm_l3_v5",
        "version"    : "v5",
        "model_type" : "bilstm",
        "hidden"     : "256",
        "layers"     : "3",
        "dropout"    : "0.3",
        "batch"      : "1024",
        "lr"         : "0.00005",
        "desc"       : "GroupA 交叉: BiLSTM L3 + V5(10维)。架构×特征协同效应",
    },

    # =================================================================
    # Group B: 学习率搜索
    # 固定: V5(10维), d=0.3, batch=1024
    # e54 已覆盖 lr=5e-5，故此处搜 {1e-5, 1e-4, 2e-4} 三个新点
    # =================================================================
    {
        "id"         : "e55_bilstm_l3_v5_lr1e5",
        "version"    : "v5",
        "model_type" : "bilstm",
        "hidden"     : "256",
        "layers"     : "3",
        "dropout"    : "0.3",
        "batch"      : "1024",
        "lr"         : "0.00001",
        "desc"       : "GroupB 极慢LR: BiLSTM L3 V5 lr=1e-5。测试保守学习率是否更稳",
    },
    {
        "id"         : "e56_bilstm_l3_v5_lr1e4",
        "version"    : "v5",
        "model_type" : "bilstm",
        "hidden"     : "256",
        "layers"     : "3",
        "dropout"    : "0.3",
        "batch"      : "1024",
        "lr"         : "0.0001",
        "desc"       : "GroupB 中速LR: BiLSTM L3 V5 lr=1e-4。Phase2的e23用过此LR",
    },
    {
        "id"         : "e57_bilstm_l3_v5_lr2e4",
        "version"    : "v5",
        "model_type" : "bilstm",
        "hidden"     : "256",
        "layers"     : "3",
        "dropout"    : "0.3",
        "batch"      : "1024",
        "lr"         : "0.0002",
        "desc"       : "GroupB 快速LR: BiLSTM L3 V5 lr=2e-4。config默认值/周频甜点",
    },
    {
        "id"         : "e58_lstm_l3_v5_lr1e4",
        "version"    : "v5",
        "model_type" : "lstm",
        "hidden"     : "256",
        "layers"     : "3",
        "dropout"    : "0.3",
        "batch"      : "1024",
        "lr"         : "0.0001",
        "desc"       : "GroupB 对冲: LSTM L3 V5 lr=1e-4。万一LSTM更优时的LR验证",
    },

    # =================================================================
    # Group C: 正则化精调
    # 固定: BiLSTM L3 V5 lr=5e-5 (Group A 的安全基线)
    # =================================================================
    {
        "id"         : "e59_bilstm_l3_v5_d04",
        "version"    : "v5",
        "model_type" : "bilstm",
        "hidden"     : "256",
        "layers"     : "3",
        "dropout"    : "0.4",
        "batch"      : "1024",
        "lr"         : "0.00005",
        "desc"       : "GroupC Dropout: BiLSTM L3 V5 + dropout=0.4。10维特征可能过拟合",
    },
    {
        "id"         : "e60_bilstm_l3_v5_ls",
        "version"    : "v5",
        "model_type" : "bilstm",
        "hidden"     : "256",
        "layers"     : "3",
        "dropout"    : "0.3",
        "batch"      : "1024",
        "lr"         : "0.00005",
        "label_smooth": "0.08",
        "desc"       : "GroupC LabelSmooth: BiLSTM L3 V5 + LS=0.08。改善概率校准/Precision",
    },
    {
        "id"         : "e61_bilstm_l3_v5_wd",
        "version"    : "v5",
        "model_type" : "bilstm",
        "hidden"     : "256",
        "layers"     : "3",
        "dropout"    : "0.3",
        "batch"      : "1024",
        "lr"         : "0.00005",
        "weight_decay": "0.0001",
        "desc"       : "GroupC WeightDecay: BiLSTM L3 V5 + WD=1e-4 (AdamW)。L2正则",
    },
    {
        "id"         : "e62_bilstm_l3_v5_d04_ls",
        "version"    : "v5",
        "model_type" : "bilstm",
        "hidden"     : "256",
        "layers"     : "3",
        "dropout"    : "0.4",
        "batch"      : "1024",
        "lr"         : "0.00005",
        "label_smooth": "0.08",
        "desc"       : "GroupC 组合: BiLSTM L3 V5 + d=0.4+LS=0.08。如59/60各有效则尝试叠加",
    },
]

# ======================================================================
# 路径
# ======================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TIMING_LOG   = os.path.join(PROJECT_ROOT, "reports", "phase5_timing_log.csv")


def get_group_name(exp_id):
    if exp_id.startswith(("e51", "e52", "e53", "e54")):
        return "A"
    if exp_id.startswith(("e55", "e56", "e57", "e58")):
        return "B"
    return "C"


def get_model_dir(version):
    if version == "v5":
        return os.path.join(PROJECT_ROOT, "models_v5")
    if version == "v3":
        return os.path.join(PROJECT_ROOT, "models_v2")
    return os.path.join(PROJECT_ROOT, f"models_{version}")


def ensure_phase5_dirs():
    os.makedirs(get_model_dir("v3"), exist_ok=True)
    os.makedirs(get_model_dir("v5"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "reports", "phase5"), exist_ok=True)


def run_preflight():
    print("\n[Preflight] 检查 Phase 5 运行前置条件...")
    ensure_phase5_dirs()

    required_files = [
        ("Runner", os.path.join(PROJECT_ROOT, "run_phase5_experiments.py")),
        ("训练入口", os.path.join(PROJECT_ROOT, "src", "train", "run_training_v2.py")),
        ("评估入口", os.path.join(PROJECT_ROOT, "evaluate.py")),
        ("聚合评估", os.path.join(PROJECT_ROOT, "evaluate_agg.py")),
        ("自检脚本", os.path.join(PROJECT_ROOT, "check_phase5_code.py")),
        ("V3 train dyn", os.path.join(PROJECT_ROOT, "data", "processed_v3", "X_train_dyn.bin")),
        ("V3 val dyn", os.path.join(PROJECT_ROOT, "data", "processed_v3", "X_val_dyn.bin")),
        ("V3 meta", os.path.join(PROJECT_ROOT, "data", "artifacts_v3", "meta_v2.json")),
        ("V5 train dyn", os.path.join(PROJECT_ROOT, "data", "processed_v5", "X_train_dyn.bin")),
        ("V5 val dyn", os.path.join(PROJECT_ROOT, "data", "processed_v5", "X_val_dyn.bin")),
        ("V5 meta", os.path.join(PROJECT_ROOT, "data", "artifacts_v5", "meta_v5.json")),
    ]

    missing = []
    for label, path in required_files:
        if os.path.exists(path):
            print(f"  [OK] {label}: {os.path.relpath(path, PROJECT_ROOT)}")
        else:
            print(f"  [MISSING] {label}: {os.path.relpath(path, PROJECT_ROOT)}")
            missing.append(path)

    for label, meta_path in (
        ("V3", os.path.join(PROJECT_ROOT, "data", "artifacts_v3", "meta_v2.json")),
        ("V5", os.path.join(PROJECT_ROOT, "data", "artifacts_v5", "meta_v5.json")),
    ):
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(
                f"  [META] {label}: lookback={meta.get('lookback', '?')} | "
                f"dyn_feat_dim={meta.get('dyn_feat_dim', 7)} | "
                f"train_cnt={meta.get('train_cnt', '?')} | val_cnt={meta.get('val_cnt', '?')}"
            )
        except Exception as e:
            print(f"  [BROKEN] {label} meta 读取失败: {e}")
            missing.append(meta_path)

    if missing:
        print("\n[Preflight] 阻塞：缺少 Phase 5 必需文件，停止运行。")
        return False

    print("[Preflight] 通过。")
    return True


def init_timing_log():
    os.makedirs(os.path.dirname(TIMING_LOG), exist_ok=True)
    if not os.path.exists(TIMING_LOG):
        with open(TIMING_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "exp_id", "group", "version", "model_type", "hidden", "layers",
                "lr", "dropout", "label_smooth", "weight_decay",
                "start_time", "end_time", "elapsed_min",
                "actual_epochs", "min_per_epoch", "status", "note"
            ])
    print(f"[时间日志] 记录至: {os.path.relpath(TIMING_LOG)}")


def log_timing(exp, t_start, t_end, actual_epochs, status, note=""):
    elapsed_min   = round((t_end - t_start) / 60, 2)
    min_per_epoch = round(elapsed_min / max(actual_epochs, 1), 2)
    eid = exp["id"]
    group = get_group_name(eid)
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            eid, group, exp.get("version", "v3"), exp["model_type"],
            exp["hidden"], exp["layers"], exp["lr"], exp["dropout"],
            exp.get("label_smooth", "0"), exp.get("weight_decay", "0"),
            datetime.fromtimestamp(t_start).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(t_end).strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_min, actual_epochs, min_per_epoch, status, note
        ])
    return elapsed_min, min_per_epoch


def get_actual_epochs(exp_id):
    history_file = os.path.join(PROJECT_ROOT, "reports", f"history_{exp_id}.csv")
    if not os.path.exists(history_file):
        return 0
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            return max(0, len(f.readlines()) - 1)
    except Exception:
        return 0


def print_eta(completed_mins, remaining, total):
    if not completed_mins:
        return
    avg   = sum(completed_mins) / len(completed_mins)
    eta_m = avg * remaining
    eta_t = datetime.now() + timedelta(minutes=eta_m)
    done  = total - remaining
    print(f"\n  [进度] 已完成 {done}/{total} 组 | 均耗时 {avg:.1f} min/组")
    print(f"  [ETA]  剩余 {remaining} 组 | 预计还需 {eta_m:.0f} min (~{eta_m/60:.1f}h)")
    print(f"  [ETA]  预计完成: {eta_t.strftime('%Y-%m-%d %H:%M')}\n")


def check_v5_features():
    v5_train = os.path.join(PROJECT_ROOT, "data", "processed_v5", "X_train_dyn.bin")
    v5_meta  = os.path.join(PROJECT_ROOT, "data", "artifacts_v5", "meta_v5.json")
    return os.path.exists(v5_train) and os.path.exists(v5_meta)


def is_already_done(exp):
    """检查实验是否已完成（断点续跑支持）"""
    exp_id = exp["id"]
    version = exp.get("version", "v3")
    models_dir = get_model_dir(version)
    backup = os.path.join(models_dir, f"best_{exp_id}.pth")
    return os.path.exists(backup)


def run_experiment(exp, env):
    train_cmd = [sys.executable,
                 os.path.join("src", "train", "run_training_v2.py")]
    proc = subprocess.Popen(train_cmd, env=env, cwd=PROJECT_ROOT)
    proc.wait()
    return proc.returncode, get_actual_epochs(exp["id"])


def backup_model(exp):
    version    = exp.get("version", "v3")
    models_dir = get_model_dir(version)
    model_src  = os.path.join(models_dir, "best_enhanced_model.pth")
    if not os.path.exists(model_src):
        print(f"  [备份失败] 未找到模型文件: {os.path.relpath(model_src, PROJECT_ROOT)}")
        return False
    backup = os.path.join(models_dir, f"best_{exp['id']}.pth")
    try:
        shutil.copy(model_src, backup)
    except Exception as e:
        print(f"  [备份失败] {e}")
        return False
    size_mb = os.path.getsize(backup) / 1024**2
    print(f"  [备份] {os.path.relpath(backup)}  ({size_mb:.1f} MB)")
    return True


def run_evaluation(exp, env):
    reports_dir = os.path.join(PROJECT_ROOT, "reports", "phase5")
    os.makedirs(reports_dir, exist_ok=True)
    out_path = os.path.join(reports_dir, f"eval_{exp['id']}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        result = subprocess.run(
            [sys.executable, "evaluate.py"],
            stdout=f, stderr=subprocess.STDOUT, env=env, cwd=PROJECT_ROOT
        )
    eval_ok = result.returncode == 0
    status = "OK" if eval_ok else f"ERR({result.returncode})"
    print(f"  [评估] {os.path.relpath(out_path)}  [{status}]")

    if not eval_ok:
        return False, False

    # 同步跑 SKU 聚合分析
    agg_path = os.path.join(reports_dir, f"agg_{exp['id']}.txt")
    with open(agg_path, "w", encoding="utf-8") as f:
        agg_result = subprocess.run(
            [sys.executable, "evaluate_agg.py", "--exp", exp["id"]],
            stdout=f, stderr=subprocess.STDOUT, env=env, cwd=PROJECT_ROOT
        )
    agg_ok = agg_result.returncode == 0
    agg_status = "OK" if agg_ok else f"ERR({agg_result.returncode})"
    print(f"  [聚合] {os.path.relpath(agg_path)}  [{agg_status}]")
    return eval_ok, agg_ok


def main():
    print("=" * 70)
    print("  B2B Phase 5 挂机实验流水线（12组 × 3 Group）")
    print(f"  统一设置: epochs={GLOBAL_EPOCHS}, patience={GLOBAL_PATIENCE}, workers={GLOBAL_WORKERS}")
    print("=" * 70)

    if not run_preflight():
        sys.exit(1)

    # 检查 V5 特征数据
    need_v5 = any(e.get("version") == "v5" for e in EXPERIMENTS)
    if need_v5 and not check_v5_features():
        print("\n[WARN] Phase 5B/5C 实验需要 V5 特征数据，但尚未生成！")
        print("   请先运行: python -m src.features.build_features_v5_sku")
        ans = input("   是否现在自动运行特征工程？(y/n): ").strip().lower()
        if ans == "y":
            t0 = time.time()
            result = subprocess.run(
                [sys.executable, "-m", "src.features.build_features_v5_sku"],
                cwd=PROJECT_ROOT
            )
            if result.returncode != 0:
                print("[FAIL] 特征工程失败，退出。")
                sys.exit(1)
            print(f"[OK] 特征工程完成 ({(time.time()-t0)/60:.1f} min)")
        else:
            print("跳过。V5 实验将失败。")

    init_timing_log()
    pipeline_start  = time.time()
    completed_times = []
    total           = len(EXPERIMENTS)
    current_group   = ""
    pipeline_failed = False
    stop_reason     = ""
    stop_exp_id     = ""

    print(f"\n共 {total} 组实验 | 启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Group A (e51~e54): 2×2 阶乘基线")
    print(f"   Group B (e55~e58): 学习率搜索")
    print(f"   Group C (e59~e62): 正则化精调\n")

    for idx, exp in enumerate(EXPERIMENTS):
        exp_id    = exp["id"]
        version   = exp.get("version", "v3")
        remaining = total - idx - 1

        # Group 分隔标题
        eid_num = exp_id[1:3]
        if eid_num in ("51",) and current_group != "A":
            current_group = "A"
            print(f"\n{'─'*70}")
            print(f"  ┌─ Group A: 2×2 阶乘基线 ──────────────────────────────────────┐")
        elif eid_num in ("55",) and current_group != "B":
            current_group = "B"
            print(f"\n{'─'*70}")
            print(f"  ┌─ Group B: 学习率搜索 ────────────────────────────────────────┐")
        elif eid_num in ("59",) and current_group != "C":
            current_group = "C"
            print(f"\n{'─'*70}")
            print(f"  ┌─ Group C: 正则化精调 ────────────────────────────────────────┐")

        # 断点续跑检查
        if is_already_done(exp):
            print(f"\n  [SKIP] [{idx+1}/{total}] {exp_id} 已完成，跳过")
            continue

        print(f"\n{'='*70}")
        print(f"  [RUN] [{idx+1}/{total}] {exp_id}")
        print(f"     {exp['model_type'].upper()} L{exp['layers']} H{exp['hidden']} D{exp['dropout']}"
              f" | {version.upper()} | lr={exp['lr']}"
              f"{' | LS='+exp.get('label_smooth','') if exp.get('label_smooth') else ''}"
              f"{' | WD='+exp.get('weight_decay','') if exp.get('weight_decay') else ''}")
        print(f"     {exp['desc']}")
        print(f"     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 构造环境变量
        env = os.environ.copy()
        env["PYTHONIOENCODING"]  = "utf-8"
        env["PYTHONUNBUFFERED"]  = "1"
        env["EXP_ID"]            = exp_id
        env["EXP_VERSION"]       = version
        env["EXP_MODEL_TYPE"]    = exp["model_type"]
        env["EXP_HIDDEN"]        = exp["hidden"]
        env["EXP_LAYERS"]        = exp["layers"]
        env["EXP_DROPOUT"]       = exp["dropout"]
        env["EXP_BATCH"]         = exp["batch"]
        env["EXP_LR"]            = exp["lr"]
        env["EXP_EPOCHS"]        = GLOBAL_EPOCHS
        env["EXP_PATIENCE"]      = GLOBAL_PATIENCE
        env["EXP_WORKERS"]       = GLOBAL_WORKERS
        env["EXP_EVAL_WORKERS"]  = GLOBAL_WORKERS
        # Loss 超参
        env["EXP_POS_WEIGHT"]    = DEFAULT_LOSS["pos_w"]
        env["EXP_FP_PENALTY"]    = DEFAULT_LOSS["fp"]
        env["EXP_GAMMA"]         = DEFAULT_LOSS["gamma"]
        env["EXP_REG_WEIGHT"]    = DEFAULT_LOSS["reg_w"]
        env["EXP_SOFT_F1"]       = DEFAULT_LOSS["sf1"]
        env["EXP_HUBER"]         = DEFAULT_LOSS["hub"]
        # Phase 5 特有
        env["EXP_LABEL_SMOOTH"]  = exp.get("label_smooth", "0.0")
        env["EXP_WEIGHT_DECAY"]  = exp.get("weight_decay", "0.0")

        t_start = time.time()
        returncode, actual_epochs = run_experiment(exp, env)
        train_end = time.time()

        if returncode != 0:
            elapsed, _ = log_timing(exp, t_start, train_end, actual_epochs, "train_error")
            print(f"  [FAIL] 异常终止 | {elapsed:.1f} min | {actual_epochs} epochs")
            pipeline_failed = True
            stop_reason = "训练失败"
            stop_exp_id = exp_id
            break

        train_elapsed = round((train_end - t_start) / 60, 2)
        train_mpe = round(train_elapsed / max(actual_epochs, 1), 2)
        print(f"  [OK] 训练完成 | {train_elapsed:.1f} min | {actual_epochs} ep | {train_mpe:.1f} min/ep")
        
        if not backup_model(exp):
            log_timing(exp, t_start, time.time(), actual_epochs, "backup_error")
            pipeline_failed = True
            stop_reason = "备份失败"
            stop_exp_id = exp_id
            break

        print(f"  [评估] 启动 evaluate.py + evaluate_agg.py ...")
        eval_ok, agg_ok = run_evaluation(exp, env)
        if not eval_ok:
            log_timing(exp, t_start, time.time(), actual_epochs, "eval_error")
            pipeline_failed = True
            stop_reason = "评估失败"
            stop_exp_id = exp_id
            break
        if not agg_ok:
            log_timing(exp, t_start, time.time(), actual_epochs, "agg_error")
            pipeline_failed = True
            stop_reason = "聚合评估失败"
            stop_exp_id = exp_id
            break

        exp_end = time.time()
        elapsed, mpe = log_timing(exp, t_start, exp_end, actual_epochs, "success")
        print(f"  [OK] 实验收口完成 | {elapsed:.1f} min | {actual_epochs} ep | {mpe:.1f} min/ep")

        completed_times.append(elapsed)
        print_eta(completed_times, remaining, total)

    # 汇总
    total_elapsed = (time.time() - pipeline_start) / 60
    print("\n" + "=" * 70)
    if pipeline_failed:
        print(f"  [STOP] Phase 5 在 {stop_exp_id} 终止：{stop_reason}")
        print(f"     已停止，不会进入后续 Group。")
        print(f"     当前组别: Group {get_group_name(stop_exp_id)}")
        print(f"     时间日志: {os.path.relpath(TIMING_LOG)}")
        print(f"     报告目录: reports/phase5/")
        print("=" * 70)
        sys.exit(1)

    print(f"  Phase 5 全部 {total} 组实验完成！")
    print(f"     总耗时: {total_elapsed:.0f} min ({total_elapsed/60:.1f}h)")
    print(f"     时间日志: {os.path.relpath(TIMING_LOG)}")
    print(f"     报告目录: reports/phase5/")
    print(f"\n  下一步: python evaluate_agg.py 查看 SKU 级 Ratio 分析")
    print("=" * 70)


if __name__ == "__main__":
    main()

