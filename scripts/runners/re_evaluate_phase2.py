import os
import sys
import subprocess
import shutil
import re
import csv
from datetime import datetime

print("=" * 70)
print(" 🩺 B2B 补货系统 - 架构消融实验 一键评估与指标汇总表生成")
print("=" * 70)

EXPERIMENTS = [
    {"id": "e11_GRU",    "model_type": "gru",    "hidden": "256", "layers": "2", "dropout": "0.3", "batch": "2048"},
    {"id": "e12_BiLSTM", "model_type": "bilstm", "hidden": "256", "layers": "2", "dropout": "0.3", "batch": "1024"},
    {"id": "e13_Attn",   "model_type": "attn",   "hidden": "256", "layers": "2", "dropout": "0.3", "batch": "2048"},
    {"id": "e21_h128",   "model_type": "lstm",   "hidden": "128", "layers": "2", "dropout": "0.3", "batch": "2048"},
    {"id": "e22_h384",   "model_type": "lstm",   "hidden": "384", "layers": "2", "dropout": "0.3", "batch": "1024"},
    {"id": "e23_l1",     "model_type": "lstm",   "hidden": "256", "layers": "1", "dropout": "0.3", "batch": "2048"},
    {"id": "e24_l3",     "model_type": "lstm",   "hidden": "256", "layers": "3", "dropout": "0.3", "batch": "1024"},
    {"id": "e25_d05",    "model_type": "lstm",   "hidden": "256", "layers": "2", "dropout": "0.5", "batch": "2048"},
    {"id": "e26_d07",    "model_type": "lstm",   "hidden": "256", "layers": "2", "dropout": "0.7", "batch": "2048"},
    {"id": "e27_b1024",  "model_type": "lstm",   "hidden": "256", "layers": "2", "dropout": "0.3", "batch": "1024"}
]

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models_v2")
REPORTS_DIR  = os.path.join(PROJECT_ROOT, "reports")
TARGET_MODEL = os.path.join(MODELS_DIR, "best_enhanced_model.pth")
SUMMARY_CSV  = os.path.join(REPORTS_DIR, "phase2_summary.csv")

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    success_count = 0
    
    # 准备写入汇总表
    with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(["实验ID", "架构", "参数", "最佳F1", "ROC-AUC", "全量MAE", "大盘Ratio"])
        
        for exp in EXPERIMENTS:
            exp_id = exp['id']
            backup_model_path = os.path.join(MODELS_DIR, f"best_{exp_id}.pth")

            if not os.path.exists(backup_model_path):
                print(f"\n⏭️ 找不到权重文件 {backup_model_path}，跳过 {exp_id}...")
                continue

            print(f"\n" + "-"*60)
            print(f"🚀 正在重新评估并提取指标: [{exp_id}]")

            shutil.copy(backup_model_path, TARGET_MODEL)

            env = os.environ.copy()
            env['EXP_ID']         = exp_id
            env['EXP_MODEL_TYPE'] = exp['model_type']
            env['EXP_HIDDEN']     = exp['hidden']
            env['EXP_LAYERS']     = exp['layers']
            env['EXP_DROPOUT']    = exp['dropout']
            env['EXP_BATCH']      = exp['batch']

            # 运行 evaluate.py 并捕获输出文本
            try:
                result = subprocess.run(
                    [sys.executable, "evaluate.py"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    env=env, 
                    text=True,
                    encoding='utf-8'
                )
                output = result.stdout
                
                # 保存完整的评估日志备查
                with open(os.path.join(REPORTS_DIR, f"basic_{exp_id}.txt"), "w", encoding="utf-8") as text_f:
                    text_f.write(output)
                
                # ==== 使用正则表达式自动提取关键指标 ====
                # 提取最佳 F1
                f1_match = re.search(r'最佳 F1-Score:\s*([0-9.]+)', output)
                best_f1 = f1_match.group(1) if f1_match else "N/A"
                
                # 提取 ROC-AUC
                auc_match = re.search(r'ROC-AUC:\s*([0-9.]+)', output)
                if not auc_match: # 备用匹配正则
                    auc_match = re.search(r'ROC-AUC=([0-9.]+)', output)
                auc = auc_match.group(1) if auc_match else "N/A"
                
                # 提取 MAE
                mae_match = re.search(r'全量 MAE:\s*([0-9.]+)', output)
                mae = mae_match.group(1) if mae_match else "N/A"
                
                # 提取 Ratio
                ratio_match = re.search(r'Ratio:\s*([0-9.]+)', output)
                ratio = ratio_match.group(1) if ratio_match else "N/A"
                
                # 写入汇总 CSV
                writer.writerow([
                    exp_id, 
                    exp['model_type'].upper(),
                    f"h{exp['hidden']}_l{exp['layers']}_d{exp['dropout']}",
                    best_f1, 
                    auc, 
                    mae, 
                    ratio
                ])
                
                print(f"✅ 提取成功 -> F1: {best_f1} | AUC: {auc} | MAE: {mae} | Ratio: {ratio}")
                success_count += 1
                
            except Exception as e:
                print(f"❌ {exp_id} 评估出错: {e}")

    print("\n" + "="*70)
    print(f"🎉 汇总完成！共成功处理 {success_count} 个模型。")
    print(f"📂 终极架构对比表已生成: {os.path.relpath(SUMMARY_CSV)}")
    print("="*70)

if __name__ == "__main__":
    main()