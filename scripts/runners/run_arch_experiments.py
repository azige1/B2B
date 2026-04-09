import os
import sys
import subprocess
import time
import shutil
import csv
import re
from datetime import datetime, timedelta

print("=" * 75)
print(" 🌙 B2B 补货系统 - 第二期架构对比挂机跑批 (Phase 2: Architecture Search)")
print("=" * 75)

# ============================================================
# 第二期：锁定第一期最优优化超参（e10_Slow_Magic），只变模型架构
# ============================================================
EXPERIMENTS = [
    {
        "id":         "e11_GRU",
        "model_type": "gru",
        "hidden":     "256",
        "layers":     "2",
        "dropout":    "0.3",
        "batch":      "2048",
        "pos_w":      "5.85",
        "fp":         "0.15",
        "gamma":      "1.5",
        "lr":         "0.00005",
        "desc": "GRU 替换 LSTM：参数少25%，验证零膨胀稀疏数据上的泛化表现"
    },
    {
        "id":         "e12_BiLSTM",
        "model_type": "bilstm",
        "hidden":     "256",
        "layers":     "2",
        "dropout":    "0.3",
        "batch":      "1024",  # [防 OOM 降配] 计算量翻倍，缩小 Batch
        "pos_w":      "5.85",
        "fp":         "0.15",
        "gamma":      "1.5",
        "lr":         "0.00005",
        "desc": "双向 LSTM：同时感知历史趋势与逆向衰减，增强季节性识别"
    },
    {
        "id":         "e13_Attn",
        "model_type": "attn",
        "hidden":     "256",
        "layers":     "2",
        "dropout":    "0.3",
        "batch":      "2048",
        "pos_w":      "5.85",
        "fp":         "0.15",
        "gamma":      "1.5",
        "lr":         "0.00005",
        "desc": "LSTM + 时序注意力：自主学习关键补货峰值天，直击零稀释痛点"
    },
    # ============================================================
    # 额外补充的第三期：结构超参消融（基于目前最强基线 LSTM）
    # ============================================================
    {
        "id":         "e21_h128",
        "model_type": "lstm",
        "hidden":     "128",
        "layers":     "2",
        "dropout":    "0.3",
        "batch":      "2048",
        "pos_w":      "5.85",
        "fp":         "0.15",
        "gamma":      "1.5",
        "lr":         "0.00005",
        "desc": "结构消融(宽)：缩小hidden到128，测试能否通过限制容量来压低假阳性"
    },
    {
        "id":         "e22_h384",
        "model_type": "lstm",
        "hidden":     "384",
        "layers":     "2",
        "dropout":    "0.3",
        "batch":      "1024",  # [防 OOM 降配]
        "pos_w":      "5.85",
        "fp":         "0.15",
        "gamma":      "1.5",
        "lr":         "0.00005",
        "desc": "结构消融(宽)：扩大hidden到384，测试能否记住更多复杂长尾模式"
    },
    {
        "id":         "e23_l1",
        "model_type": "lstm",
        "hidden":     "256",
        "layers":     "1",
        "dropout":    "0.3",
        "batch":      "2048",
        "pos_w":      "5.85",
        "fp":         "0.15",
        "gamma":      "1.5",
        "lr":         "0.00005",
        "desc": "结构消融(深)：减少到1层LSTM，测试是否能防止深层过拟合"
    },
    {
        "id":         "e24_l3",
        "model_type": "lstm",
        "hidden":     "256",
        "layers":     "3",
        "dropout":    "0.3",
        "batch":      "1024",  # [防 OOM 降配]
        "pos_w":      "5.85",
        "fp":         "0.15",
        "gamma":      "1.5",
        "lr":         "0.00005",
        "desc": "结构消融(深)：增加到3层LSTM，测试是否能抽象更长周期的季节性"
    },
    {
        "id":         "e25_d05",
        "model_type": "lstm",
        "hidden":     "256",
        "layers":     "2",
        "dropout":    "0.5",
        "batch":      "2048",
        "pos_w":      "5.85",
        "fp":         "0.15",
        "gamma":      "1.5",
        "lr":         "0.00005",
        "desc": "结构消融(正则)：提升 Dropout 至 0.5，强迫模型提取泛化规则，观察能否提升 Precision"
    },
    {
        "id":         "e26_d07",
        "model_type": "lstm",
        "hidden":     "256",
        "layers":     "2",
        "dropout":    "0.7",
        "batch":      "2048",
        "pos_w":      "5.85",
        "fp":         "0.15",
        "gamma":      "1.5",
        "lr":         "0.00005",
        "desc": "结构消融(正则)：极限 Dropout 0.7，测试在极度稀疏遮挡下的性能边界"
    },
    {
        "id":         "e27_b1024",
        "model_type": "lstm",
        "hidden":     "256",
        "layers":     "2",
        "dropout":    "0.3",
        "batch":      "1024",
        "pos_w":      "5.85",
        "fp":         "0.15",
        "gamma":      "1.5",
        "lr":         "0.00005",
        "desc": "结构消融(批次)：在同等模型下，对半劈开 Batch = 1024，测试梯度噪音能否帮助跳出局部极小值"
    },
    # ============================================================
    # 额外补充的第四期：Loss 组件因子精调
    # ============================================================
    {
        "id":         "e31_rw03",   "model_type": "lstm", "hidden": "256", "layers": "2", "dropout": "0.3", "batch": "2048",
        "pos_w": "5.85", "fp": "0.15", "gamma": "1.5", "lr": "0.00005", "reg_w": "0.3", "sf1": "0.5", "hub": "1.5",
        "desc": "Loss消融：降低回归惩罚权重至0.3，观察是否能提高分类准度（Precision优先）"
    },
    {
        "id":         "e32_rw07",   "model_type": "lstm", "hidden": "256", "layers": "2", "dropout": "0.3", "batch": "2048",
        "pos_w": "5.85", "fp": "0.15", "gamma": "1.5", "lr": "0.00005", "reg_w": "0.7", "sf1": "0.5", "hub": "1.5",
        "desc": "Loss消融：提高回归惩罚权重至0.7，强迫模型更精确摸底件数曲线"
    },
    {
        "id":         "e33_sf025",  "model_type": "lstm", "hidden": "256", "layers": "2", "dropout": "0.3", "batch": "2048",
        "pos_w": "5.85", "fp": "0.15", "gamma": "1.5", "lr": "0.00005", "reg_w": "0.5", "sf1": "0.25", "hub": "1.5",
        "desc": "Loss消融：Soft-F1抑制到0.25，缓解因为强行关注召回导致的梯度激惹"
    },
    {
        "id":         "e34_sf10",   "model_type": "lstm", "hidden": "256", "layers": "2", "dropout": "0.3", "batch": "2048",
        "pos_w": "5.85", "fp": "0.15", "gamma": "1.5", "lr": "0.00005", "reg_w": "0.5", "sf1": "1.0", "hub": "1.5",
        "desc": "Loss消融：Soft-F1火力全开(1.0)，在当前稳定架构下测试能否直击F1业务核心点"
    },
    {
        "id":         "e35_hub10",  "model_type": "lstm", "hidden": "256", "layers": "2", "dropout": "0.3", "batch": "2048",
        "pos_w": "5.85", "fp": "0.15", "gamma": "1.5", "lr": "0.00005", "reg_w": "0.5", "sf1": "0.5", "hub": "1.0",
        "desc": "Loss消融：缩紧 Huber Delta 至 1.0 (原1.5)，看看对极端大单的宽容度是否有必要"
    },
]

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
TIMING_LOG   = os.path.join(PROJECT_ROOT, "reports", "phase2_timing_log.csv")

# ============================================================
# 时间日志模块：记录每个实验的实际耗时，供后续实验时间预测使用
# ============================================================
def init_timing_log():
    """初始化时间记录 CSV（如果已存在则追加）"""
    if not os.path.exists(TIMING_LOG):
        with open(TIMING_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "exp_id", "model_type", "hidden", "layers", "batch",
                "start_time", "end_time",
                "elapsed_min",          # 实际训练分钟数
                "actual_epochs",        # 实际完成的 epoch 数
                "min_per_epoch",        # 平均每 epoch 跑多长时间
                "status",               # success / oom / error
                "note"
            ])
    print(f"[⏱️  时间日志] 将记录至: {os.path.relpath(TIMING_LOG)}")

def run_with_epoch_tracking(cmd, env, exp_id):
    """
    运行训练脚本，完全释放原生终端 IO 控制权，确保最高训练速度。
    跑完后通过解析 CSV 历史记录获取最终执行了多少 epoch。
    """
    max_epoch = 0

    # 彻底告别管道拦截（stdout=None），释放 GPU / 子进程 IO
    proc = subprocess.Popen(
        cmd, env=env
    )
    
    proc.wait()
    
    # 训练结束后，聪明的做法是从刚生成的 history_csv 里拿跑了多少轮
    history_file = os.path.join(PROJECT_ROOT, 'reports', f'history_{exp_id}.csv')
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    # 减去表头，粗略等于实际跑的 Epoch 数
                    max_epoch = len(lines) - 1
        except:
            pass
            
    return proc.returncode, max_epoch

def log_timing(exp_id, model_type, hidden, layers, batch, t_start, t_end, actual_epochs, status, note=""):
    """追加一行时间记录（含真实 epoch 数和每 epoch 耗时）"""
    elapsed_sec   = t_end - t_start
    elapsed_min   = round(elapsed_sec / 60, 2)
    min_per_epoch = round(elapsed_min / max(actual_epochs, 1), 2)
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            exp_id, model_type, hidden, layers, batch,
            datetime.fromtimestamp(t_start).strftime("%H:%M:%S"),
            datetime.fromtimestamp(t_end).strftime("%H:%M:%S"),
            elapsed_min, actual_epochs, min_per_epoch, status, note
        ])
    return elapsed_min, min_per_epoch

def print_eta(completed_times_min, remaining_experiments):
    """根据已完成实验的平均用时预测剩余时间"""
    if not completed_times_min:
        return
    avg_min = sum(completed_times_min) / len(completed_times_min)
    eta_min  = avg_min * remaining_experiments
    eta_dt   = datetime.now() + timedelta(minutes=eta_min)
    print(f"\n  📊 [时间预测] 已完成 {len(completed_times_min)} 组 | 平均每组 {avg_min:.1f} min")
    print(f"  📊 [时间预测] 剩余 {remaining_experiments} 组，预计还需 {eta_min:.0f} min (~{eta_min/60:.1f}h)")
    print(f"  📊 [时间预测] 预计全部完成时间: {eta_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")

def main():
    init_timing_log()
    
    pipeline_start  = time.time()
    completed_times = []
    total           = len(EXPERIMENTS)
    
    print(f"\n📋 本次共 {total} 组实验，patience=10，目前无先验时间基准")
    print(f"⏰ 流水线启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    for idx, exp in enumerate(EXPERIMENTS):
        exp_id     = exp['id']
        remaining  = total - idx - 1
        
        print(f"\n\n🚀 =========== [{idx+1}/{total}] 启动实验: [{exp_id}] ===========")
        print(f"📖 实验目的: {exp['desc']}")
        print(f"🔧 架构: {exp['model_type'].upper()} | hidden={exp['hidden']} | layers={exp['layers']} | dropout={exp['dropout']}")
        print(f"🔧 优化超参: LR={exp['lr']} | pos_weight={exp['pos_w']} | fp={exp['fp']}")
        
        env = os.environ.copy()
        env['EXP_ID']         = exp_id
        env['EXP_MODEL_TYPE'] = exp['model_type']
        env['EXP_HIDDEN']     = exp['hidden']
        env['EXP_LAYERS']     = exp['layers']
        env['EXP_DROPOUT']    = exp['dropout']
        env['EXP_BATCH']      = "1024" # 强制砍半：实测 2048 会占 3.8GB 显存，由于 Windows 桌面本身要占几百兆，超过 4GB 后会被发配到极慢的共享系统内存（掉速 3 倍以上）
        env['EXP_POS_WEIGHT'] = exp['pos_w']
        env['EXP_FP_PENALTY'] = exp['fp']
        env['EXP_GAMMA']      = exp['gamma']
        env['EXP_LR']         = exp['lr']
        # 第四期新增 Loss 超参（如果有的话，否则给默认值）
        env['EXP_REG_WEIGHT'] = exp.get('reg_w', '0.5')
        env['EXP_SOFT_F1']    = exp.get('sf1', '0.5')
        env['EXP_HUBER']      = exp.get('hub', '1.5')

        t_start = time.time()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ▶️  训练开始（原生速度运行）...")
        
        train_cmd  = [sys.executable, os.path.join("src", "train", "run_training_v2.py")]
        returncode, actual_epochs = run_with_epoch_tracking(train_cmd, env, exp['id'])
        t_end      = time.time()
        
        if returncode != 0:
            elapsed, mpe = log_timing(
                exp_id, exp['model_type'], exp['hidden'], exp['layers'], exp['batch'],
                t_start, t_end, actual_epochs, "error"
            )
            print(f"❌ [{elapsed:.1f} min / {actual_epochs} epochs] 实验 {exp_id} 异常终止（可能 OOM），跳至下一组...")
            completed_times.append(elapsed)
            print_eta(completed_times, remaining)
            continue

        elapsed, mpe = log_timing(
            exp_id, exp['model_type'], exp['hidden'], exp['layers'], exp['batch'],
            t_start, t_end, actual_epochs, "success"
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 训练完成")
        print(f"   ⏱️  总耗时: {elapsed:.1f} min  |  实际 Epoch 数: {actual_epochs}  |  平均每 Epoch: {mpe:.2f} min")

        # 备份最优模型
        model_src   = os.path.join(PROJECT_ROOT, "models_v2", "best_enhanced_model.pth")
        backup_path = os.path.join(PROJECT_ROOT, "models_v2", f"best_{exp_id}.pth")
        if os.path.exists(model_src):
            shutil.copy(model_src, backup_path)
            print(f"💾 模型备份: models_v2/best_{exp_id}.pth")

        # 自动评估
        reports_dir = os.path.join(PROJECT_ROOT, "reports")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 📊 正在生成评估报告...")
        with open(os.path.join(reports_dir, f"basic_{exp_id}.txt"), "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "evaluate.py"], stdout=f, env=env)
        with open(os.path.join(reports_dir, f"agg_{exp_id}.txt"), "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "evaluate_agg.py"], stdout=f, env=env)

        completed_times.append(elapsed)
        print(f"🎉 实验 {exp_id} 完成！")
        
        # 每完成一组就打印 ETA
        print_eta(completed_times, remaining)

    # 总结
    total_elapsed = (time.time() - pipeline_start) / 60
    print("\n" + "="*60)
    print(f"🌅 第二期全部 {total} 组实验完成！")
    print(f"⏱️  总计用时: {total_elapsed:.1f} min ({total_elapsed/60:.2f}h)")
    print(f"📄 时间记录已保存至: {os.path.relpath(TIMING_LOG)}")
    print(f"   → 可用于精准预测第三期（结构消融）和第四期（Loss精调）的时间预算！")
    print("="*60)

if __name__ == "__main__":
    main()
