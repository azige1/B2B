import os
import sys
import subprocess
import time
import shutil
import csv
from datetime import datetime, timedelta

print("=" * 75)
print(" 🚀 B2B 补货系统 - 第三期 Loss 因子精调挂机跑批 (Phase 3: Loss Tuning)")
print(" 🏗️  基础架构锁定: LSTM | Layers=3 | Hidden=256 | Batch=1024")
print("=" * 75)

# ============================================================
# 第三期：锁定第二期冠军架构 (e24_l3: 3层LSTM)，精调 Loss 矩阵
# ============================================================
EXPERIMENTS = [
    {
        "id":         "e31_rw_low",
        "reg_w":      "0.3",
        "sf1":        "0.5",
        "hub":        "1.5",
        "desc": "分类优先：降低回归权重至 0.3，强迫模型优先判准“要不要补”。"
    },
    {
        "id":         "e32_rw_high",
        "reg_w":      "0.8",
        "sf1":        "0.5",
        "hub":        "1.5",
        "desc": "回归优先：提高回归权重至 0.8，加强对补货件数预测的物理约束。"
    },
    {
        "id":         "e33_sf_low",
        "reg_w":      "0.5",
        "sf1":        "0.25",
        "hub":        "1.5",
        "desc": "稳健 F1：降低 Soft-F1 权重，缓解 Recall 虚高导致的虚假订单。"
    },
    {
        "id":         "e34_sf_high",
        "reg_w":      "0.5",
        "sf1":        "1.0",
        "hub":        "1.5",
        "desc": "极限 F1：全速 F1 驱动，挑战业务指标 F1-Score 的最高天花板。"
    },
    {
        "id":         "e35_hub_low",
        "reg_w":      "0.5",
        "sf1":        "0.5",
        "hub":        "1.0",
        "desc": "精准打击：收紧 Huber Delta 至 1.0，让预测对“件数偏差”更敏感。"
    },
    {
        "id":         "e36_optimal",
        "reg_w":      "0.6",
        "sf1":        "0.4",
        "hub":        "1.2",
        "desc": "综合收敛：基于 e24 数据直觉的“黄金配比”尝试，追求平衡发展。"
    },
]

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TIMING_LOG   = os.path.join(PROJECT_ROOT, "reports", "phase3_timing_log.csv")

def init_timing_log():
    if not os.path.exists(TIMING_LOG):
        with open(TIMING_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "exp_id", "reg_w", "sf1", "hub",
                "start_time", "end_time", "elapsed_min",
                "actual_epochs", "min_per_epoch", "status", "note"
            ])
    print(f"[⏱️  时间日志] 将记录至: {os.path.relpath(TIMING_LOG)}")

def run_with_epoch_tracking(cmd, env, exp_id):
    proc = subprocess.Popen(cmd, env=env)
    proc.wait()
    
    max_epoch = 0
    history_file = os.path.join(PROJECT_ROOT, 'reports', f'history_{exp_id}.csv')
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    max_epoch = len(lines) - 1
        except:
            pass
    return proc.returncode, max_epoch

def log_timing(exp_id, reg_w, sf1, hub, t_start, t_end, actual_epochs, status, note=""):
    elapsed_sec = t_end - t_start
    elapsed_min = round(elapsed_sec / 60, 2)
    min_per_epoch = round(elapsed_min / max(actual_epochs, 1), 2)
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            exp_id, reg_w, sf1, hub,
            datetime.fromtimestamp(t_start).strftime("%H:%M:%S"),
            datetime.fromtimestamp(t_end).strftime("%H:%M:%S"),
            elapsed_min, actual_epochs, min_per_epoch, status, note
        ])
    return elapsed_min, min_per_epoch

def print_eta(completed_times_min, remaining_experiments):
    if not completed_times_min:
        return
    avg_min = sum(completed_times_min) / len(completed_times_min)
    eta_min  = avg_min * remaining_experiments
    eta_dt   = datetime.now() + timedelta(minutes=eta_min)
    print(f"\n  📊 [时间预测] 已完成 {len(completed_times_min)} 组 | 平均每组 {avg_min:.1f} min")
    print(f"  📊 [时间预测] 预期全部完成时间: {eta_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    init_timing_log()
    
    pipeline_start  = time.time()
    completed_times = []
    total           = len(EXPERIMENTS)
    
    print(f"\n📋 本次共 {total} 组精调实验 | patience=5 | 基础: 3层LSTM")
    print(f"⏰ 流水线启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for idx, exp in enumerate(EXPERIMENTS):
        exp_id     = exp['id']
        remaining  = total - idx - 1
        
        print(f"\n🚀 =========== [{idx+1}/{total}] 启动实验: [{exp_id}] ===========")
        print(f"📖 核心目的: {exp['desc']}")
        print(f"🔧 参数注入: Reg_W={exp['reg_w']} | Soft_F1={exp['sf1']} | Huber={exp['hub']}")
        
        env = os.environ.copy()
        # 锁定冠军架构参数
        env['EXP_ID']         = exp_id
        env['EXP_MODEL_TYPE'] = 'lstm'
        env['EXP_HIDDEN']     = '256'
        env['EXP_LAYERS']     = '3'
        env['EXP_DROPOUT']    = '0.3'
        env['EXP_BATCH']      = '1536'
        env['EXP_LR']         = '0.00005'
        env['EXP_POS_WEIGHT'] = '5.85'
        env['EXP_FP_PENALTY'] = '0.15'
        env['EXP_GAMMA']      = '1.5'
        
        # 注入 Loss 变变量
        env['EXP_REG_WEIGHT'] = exp['reg_w']
        env['EXP_SOFT_F1']    = exp['sf1']
        env['EXP_HUBER']      = exp['hub']

        t_start = time.time()
        train_cmd  = [sys.executable, os.path.join("src", "train", "run_training_v2.py")]
        returncode, actual_epochs = run_with_epoch_tracking(train_cmd, env, exp_id)
        t_end      = time.time()
        
        status = "success" if returncode == 0 else "error"
        elapsed, mpe = log_timing(exp_id, exp['reg_w'], exp['sf1'], exp['hub'], t_start, t_end, actual_epochs, status)
        
        if returncode == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 完成: 耗时 {elapsed:.1f} min | Epochs: {actual_epochs}")
            
            # 备份模型
            model_src   = os.path.join(PROJECT_ROOT, "models_v2", "best_enhanced_model.pth")
            backup_path = os.path.join(PROJECT_ROOT, "models_v2", f"best_{exp_id}.pth")
            if os.path.exists(model_src):
                shutil.copy(model_src, backup_path)
            
            # 自动化评估 (使用新版 evaluate.py)
            reports_dir = os.path.join(PROJECT_ROOT, "reports")
            print(f"📊 正在生成评估报告 {exp_id}...")
            with open(os.path.join(reports_dir, f"basic_{exp_id}.txt"), "w", encoding="utf-8") as f:
                subprocess.run([sys.executable, "evaluate.py"], stdout=f, env=env)
            with open(os.path.join(reports_dir, f"agg_{exp_id}.txt"), "w", encoding="utf-8") as f:
                subprocess.run([sys.executable, "evaluate_agg.py"], stdout=f, env=env)
        else:
            print(f"❌ 实验 {exp_id} 异常 (Exit Code: {returncode})")

        completed_times.append(elapsed)
        print_eta(completed_times, remaining)

    total_elapsed = (time.time() - pipeline_start) / 60
    print("\n" + "="*60)
    print(f"🌅 第三期精调全部完成！总耗时: {total_elapsed:.1f} min")
    print("="*60)

if __name__ == "__main__":
    main()
