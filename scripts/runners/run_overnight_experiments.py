import os
import sys
import subprocess
import time
import shutil

print("=" * 75)
print(" 🌙 B2B 补货系统 - 工业级彻夜网格搜索全矩阵跑批 (Nightly Auto-ML)")
print("=" * 75)

# 我们精心挑选 10 个具备明确控制变量意图的高潜力组合，预估熬夜 10 小时：
EXPERIMENTS = [
    # --- 组1：尊重物理分布的真理法则 (最正统的路线) ---
    {"id": "e01_True_Loose", "pos_w": "1.0", "fp": "0.05", "gamma": "2.0", "lr": "0.0002", "desc": "正统派：放任回归爆仓，主攻底层隐层特征发散"},
    {"id": "e02_True_Norm",  "pos_w": "1.0", "fp": "0.15", "gamma": "2.0", "lr": "0.0002", "desc": "正统派：极度平衡的中庸工业级基线"},
    {"id": "e03_True_Strict","pos_w": "1.0", "fp": "0.35", "gamma": "2.0", "lr": "0.0002", "desc": "正统派：严刑峻法压制积压风险"},
    {"id": "e04_True_HardF", "pos_w": "1.0", "fp": "0.15", "gamma": "3.5", "lr": "0.0002", "desc": "正统派：调高 Focal 防线，疯狂惩罚易区分样本，死磕长尾段"},

    # --- 组2：数学贝叶斯逆向补偿 (严谨的理论偏置) ---
    {"id": "e05_Bayes_Loose","pos_w": "0.104", "fp": "0.08", "gamma": "2.0", "lr": "0.0002", "desc": "贝叶斯补偿：逆向10%配资"},
    {"id": "e06_Bayes_Norm", "pos_w": "0.104", "fp": "0.20", "gamma": "2.0", "lr": "0.0002", "desc": "贝叶斯补偿：收紧水龙头"},

    # --- 组3：经验主义的暴力美学 (对抗式纯化特征) ---
    {"id": "e07_Magic_Loose","pos_w": "5.85", "fp": "0.08", "gamma": "1.5", "lr": "0.0002", "desc": "对抗魔法：利用 BCE 挤压 Soft-F1，重现此前的高亮峰值"},
    {"id": "e08_Magic_Norm", "pos_w": "5.85", "fp": "0.15", "gamma": "1.5", "lr": "0.0002", "desc": "对抗魔法：魔数权重带枷锁"},

    # --- 组4：深坑细粒度探索 (使用更小的步长在鞍点滑行) ---
    {"id": "e09_Slow_True",  "pos_w": "1.0", "fp": "0.15", "gamma": "2.0", "lr": "0.00005", "desc": "微步长深勘：在物理真理下用极小学习率探寻极限极值点"},
    {"id": "e10_Slow_Magic", "pos_w": "5.85", "fp": "0.15", "gamma": "1.5", "lr": "0.00005", "desc": "微步长深勘：在对抗区间微搜"}
]

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

def main():
    for exp in EXPERIMENTS:
        exp_id = exp['id']
        print(f"\n\n🚀 =========== 正在启动实验: [{exp_id}] ===========")
        print(f"📖 实验目的: {exp['desc']}")
        print(f"🔧 注入参数: LR={exp['lr']}, POS_W={exp['pos_w']}, GAMMA={exp['gamma']}, FP_PENALTY={exp['fp']}")
        
        # 挂接环境外挂
        env = os.environ.copy()
        env['EXP_POS_WEIGHT'] = exp['pos_w']
        env['EXP_FP_PENALTY'] = exp['fp']
        env['EXP_GAMMA']      = exp['gamma']
        env['EXP_LR']         = exp['lr']
        
        print(f"[{time.strftime('%H:%M:%S')}] 🏃 开始百转千回的深层拉练 (Early_Stopping=40 可保无虞)...")
        train_cmd = [sys.executable, os.path.join("src", "train", "run_training_v2.py")]
        train_proc = subprocess.run(train_cmd, env=env)
        
        if train_proc.returncode != 0:
            print(f"❌ 实验 {exp_id} 崩溃或遭中断，开始回收现场并跳至下一组...")
            continue
            
        print(f"[{time.strftime('%H:%M:%S')}] ✅ 模型收敛完成。")
        
        # 保存节点战利品
        model_src = os.path.join(PROJECT_ROOT, "models_v2", "best_enhanced_model.pth")
        backup_path = os.path.join(PROJECT_ROOT, "models_v2", f"best_{exp_id}.pth")
        if os.path.exists(model_src):
            shutil.copy(model_src, backup_path)
            
        print(f"[{time.strftime('%H:%M:%S')}] 📊 执行自动化全集性能快照...")
        # 标准全量评估
        with open(os.path.join(PROJECT_ROOT, "reports", f"basic_{exp_id}.txt"), "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "evaluate.py"], stdout=f, env=env)
        
        # 款式聚合下钻评估
        with open(os.path.join(PROJECT_ROOT, "reports", f"agg_{exp_id}.txt"), "w", encoding="utf-8") as f:
            subprocess.run([sys.executable, "evaluate_agg.py"], stdout=f, env=env)
            
        print(f"🎉 实验 {exp_id} 圆满杀青！指标密档已落锁归库。\n")

    print("\n" + "="*50)
    print("🌅 全套 10 组重型矩阵大巡礼已完毕！您的赛博矿工正式收工。")
    print("==================================================")

if __name__ == "__main__":
    main()
