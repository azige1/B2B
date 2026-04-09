"""
B2B 时序补货双塔模型 - 全域可视化业务评测台 (v1.5 专版)
功能: 
1. 全局 MAE 与 Ratio 验证
2. 多级逆向聚合: (买手, SKU) -> SKU -> 大类(Category)
3. 动态时间轴 120 天心电图抽样可视化
"""
import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from tqdm import tqdm

# 强制中文输出与字体显示
sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei'] # Windows 字体
plt.rcParams['axes.unicode_minus'] = False 

# 将根目录导入 (在 src/evaluate/ 下需后退两层)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.train.dataset import create_lazy_dataloaders
from src.models.enhanced_model import EnhancedTwoTowerLSTM
from src.train.run_training import DummyLE # 兼容反序列化

def load_label_encoders():
    encoder_path = os.path.join(PROJECT_ROOT, 'data', 'artifacts', 'label_encoders.pkl')
    try:
        with open(encoder_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[!] LabelEncoder 加载失败: {e}")
        return {}

def inverse_transform_le(le_dict, col_name, encoded_vals):
    if col_name in le_dict and hasattr(le_dict[col_name], 'inverse_transform'):
        try:
            return le_dict[col_name].inverse_transform(encoded_vals.astype(int))
        except:
            return encoded_vals
    return encoded_vals

def run_evaluation():
    print("="*60)
    print("📊 B2B 模型全维度业务评审启动 (Global -> SKU -> Category)")
    print("="*60)
    
    # 1. 初始化环境与读取
    config_path = os.path.join(PROJECT_ROOT, 'config', 'model_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    static_order_keys = [
        'buyer_id', 'sku_id', 'style_id', 'product_name', 'category', 
        'sub_category', 'season', 'series', 'band', 'size_id', 'color_id', 
        'month', 
        'qty_first_order', 'price_tag',
        'cooperation_years', 'monthly_average_replenishment', 
        'avg_discount_rate', 'replenishment_frequency', 'item_coverage_rate'
    ]
    sku_idx = static_order_keys.index('sku_id')
    cat_idx = static_order_keys.index('category')
    
    # 2. 拉取验证集
    processed_dir = os.path.join(PROJECT_ROOT, config['paths']['dataset_dir'])
    _, val_loader = create_lazy_dataloaders(processed_dir, batch_size=1024, num_workers=0)
    
    # 3. 拉取模型架构与最佳权重
    le_dict = load_label_encoders()
    dynamic_vocab_sizes = {
        'buyer_id': 1000, 'sku_id': 15000, 'style_id': 3000, 'product_name': 2000, 
        'category': 50, 'sub_category': 100, 'season': 10, 'series': 50,
        'band': 50, 'size_id': 50, 'color_id': 100, 'month': 13
    }
    for col, le in le_dict.items():
        if col in dynamic_vocab_sizes: dynamic_vocab_sizes[col] = len(le.classes_) + 5
            
    num_numeric_feats = len(static_order_keys) - len(dynamic_vocab_sizes)
    model = EnhancedTwoTowerLSTM(
        dyn_feat_dim=7, lstm_hidden=config['model']['hidden_size'],
        lstm_layers=config['model']['num_layers'], static_vocab_sizes=dynamic_vocab_sizes,
        static_emb_dim=16, num_numeric_feats=num_numeric_feats, dropout=0.0
    ).to(device)
    
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_enhanced_model.pth')
    if not os.path.exists(model_path):
        print(f"❌ 找不到权重文件: {model_path}，请先运行完成 run_training.py！")
        return
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    # 取消 torch.compile 在保存时带来的 '_orig_mod.' 前缀，防止 key 找不到
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k.replace('_orig_mod.', '')] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    print("[*] 模型权重加载完成，开始执行全验证集推理遍历...")
    
    # [BUG-D 修复] 加载 MinMaxScaler 用于还原历史序列的真实件数
    import pickle as _pkl
    scaler_path = os.path.join(PROJECT_ROOT, 'data', 'artifacts', 'feature_scaler.pkl')
    _feature_scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as _f:
            _feature_scaler = _pkl.load(_f)
        print("[*] feature_scaler.pkl 加载成功，将还原历史序列为真实件数")
    else:
        print("[!] 未找到 feature_scaler.pkl，心电图将展示归一化值")

    # 结果容器
    results = []
    viz_samples = []

    with torch.no_grad():
        for x_dyn, x_static, y_cls, y_reg in tqdm(val_loader, desc="推理中"):
            x_dyn = x_dyn.to(device)
            x_static = x_static.to(device)
            logits, preds = model(x_dyn, x_static, static_order_keys)
            
            prob = torch.sigmoid(logits)
            
            # --- 同步 trainer.py 及 generate_strategy 里的强控逻辑 ---
            dyn_sum = x_dyn.abs().sum(dim=(1,2))
            dead_mask = (dyn_sum < 1e-4)
            prob[dead_mask] = 0.0
            
            # [v1.9] prob>0.35 校准，scale=1.0
            preds = torch.clamp(preds, max=8.5)
            pred_qty = torch.expm1(preds) * (prob > 0.35).float() * 1.0
            true_qty = torch.expm1(y_reg.to(device))
            
            # 转储到 CPU Numpy，并过滤 inf/nan 保证安全
            pred_np = np.nan_to_num(pred_qty.cpu().numpy().flatten(), nan=0.0, posinf=0.0, neginf=0.0)
            true_np = np.nan_to_num(true_qty.cpu().numpy().flatten(), nan=0.0, posinf=0.0, neginf=0.0)
            sku_ids = x_static[:, sku_idx].cpu().numpy().flatten()
            cat_ids = x_static[:, cat_idx].cpu().numpy().flatten()
            
            # 记录历史 90 天单子，用于心电图
            hist_repl = x_dyn[:, :, 0].cpu().numpy() # 特征0是补货
            
            for i in range(len(pred_np)):
                results.append({
                    'sku_encoded': sku_ids[i],
                    'cat_encoded': cat_ids[i],
                    'true_qty': true_np[i],
                    'pred_qty': pred_np[i]
                })
                # 随机抓取几个两边都有销量的真正的爆单正样本，防止抽到被阈值拦截成 0 的部分
                if true_np[i] > 10 and pred_np[i] > 10 and len(viz_samples) < 5 and np.random.rand() < 0.1:
                    viz_samples.append({
                        'hist_90d': hist_repl[i],
                        'true_qty': true_np[i],
                        'pred_qty': pred_np[i],
                        'sku': sku_ids[i]
                    })
                    
    df_res = pd.DataFrame(results)
    
    # 翻译 ID 到文本名
    df_res['sku_name'] = inverse_transform_le(le_dict, 'sku_id', df_res['sku_encoded'].values)
    df_res['category_name'] = inverse_transform_le(le_dict, 'category', df_res['cat_encoded'].values)
    
    # ==========================================
    # 评审 1: 全局北极星指标
    # ==========================================
    print("\n" + "="*40)
    print("🌟 评审纬度 1：[全局商业大盘兜底评估]")
    print("="*40)
    global_mae = np.mean(np.abs(df_res['pred_qty'] - df_res['true_qty']))
    sum_true = df_res['true_qty'].sum()
    sum_pred = df_res['pred_qty'].sum()
    global_ratio = sum_pred / sum_true if sum_true > 0 else 0
    ratio_mark = "🟢(完美安全区)" if 1.0 <= global_ratio <= 1.50 else ("🔴(断货警报)" if global_ratio < 1.0 else "🟡(轻度积压)")
    print(f"➤ 总预测发件误差 (MAE): {global_mae:.2f} 件/单")
    print(f"➤ 总大盘安全水位 (Ratio): {global_ratio:.2f} {ratio_mark}")
    
    # ==========================================
    # 评审 2: 品类聚合 (Category)
    # ==========================================
    print("\n" + "="*40)
    print("📦 评审纬度 2：[大品类 (Category) 横切表现]")
    print("="*40)
    df_cat = df_res.groupby('category_name').agg(
        total_true=('true_qty', 'sum'),
        total_pred=('pred_qty', 'sum'),
        count=('true_qty', 'count')
    ).reset_index()
    df_cat['ratio'] = df_cat['total_pred'] / (df_cat['total_true'] + 1e-5)
    df_cat = df_cat.sort_values(by='total_true', ascending=False).head(10)
    for _, row in df_cat.iterrows():
        r = row['ratio']
        mk = "✔️" if 1.0 <= r <= 1.50 else "⚠️"
        print(f"品类: {row['category_name'][:12]:<12} | 真实销量: {int(row['total_true']):<6} | 模型预算: {int(row['total_pred']):<6} | 备货水比: {r:.2f} {mk}")

    # ==========================================
    # 评审 3: SKU 微观爆款检测 
    # ==========================================
    print("\n" + "="*40)
    print("🔥 评审纬度 3：[头部 SKU (单款单色) 爆单追踪能力]")
    print("="*40)
    df_sku = df_res.groupby('sku_name').agg(
        total_true=('true_qty', 'sum'),
        total_pred=('pred_qty', 'sum')
    ).reset_index()
    df_sku['mae'] = np.abs(df_sku['total_pred'] - df_sku['total_true'])
    df_sku['ratio'] = df_sku['total_pred'] / (df_sku['total_true'] + 1e-5)
    df_sku_top = df_sku.sort_values(by='total_true', ascending=False).head(5)
    for _, row in df_sku_top.iterrows():
        print(f"SKU: {row['sku_name']} | 真实总火爆度: {int(row['total_true'])}件 | 模型给库: {int(row['total_pred'])}件 | 误差差值: {row['mae']:.1f}件")
        
    # [新增维] 评审 3.5: SKU 全国全买手聚合误差平均值
    print("\n" + "="*40)
    print("🎯 评审纬度 3.5：[单款 SKU 全国总单量对齐能力 (中央仓视角)]")
    print("="*40)
    # df_sku 已经是按 sku_name group_by 求和过了的
    sku_global_mae = df_sku['mae'].mean()
    sku_global_rmse = np.sqrt((df_sku['mae']**2).mean())
    print(f"全量参与的 SKU 总数: {len(df_sku):,}")
    print(f"说明: 这是把全国所有买手对同一个 SKU 的 [真实需求] 加在一起，然后把 AI 对所有买手的 [零碎预测] 也加在一起进行对比。")
    print(f"➤ SKU 全国总盘平均绝对偏差 (MAE): {sku_global_mae:.2f} 件/单款")
    print(f"➤ SKU 全国总盘均方根误差 (RMSE): {sku_global_rmse:.2f} 件/单款")

    # ==========================================
    # ★ 自动存档：结果保存到 reports/v1.8_demo/
    # ==========================================
    demo_dir = os.path.join(PROJECT_ROOT, 'reports', 'v1.8_demo')
    os.makedirs(demo_dir, exist_ok=True)
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. 全局大盘指标 JSON
    global_metrics = {
        "run_time": now_str,
        "model_version": "V1.8",
        "prob_threshold": 0.45,
        "scale_factor": 0.50,
        "global_mae": round(float(global_mae), 4),
        "global_ratio": round(float(global_ratio), 4),
        "sum_true_qty": int(sum_true),
        "sum_pred_qty": int(sum_pred),
        "sku_count": int(len(df_sku)),
        "sku_global_mae": round(float(sku_global_mae), 2),
        "sku_global_rmse": round(float(sku_global_rmse), 2),
    }
    metrics_path = os.path.join(demo_dir, 'global_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(global_metrics, f, ensure_ascii=False, indent=2)
    print(f"\n✅ [存档] 全局指标 → {metrics_path}")

    # 2. 品类明细 CSV
    df_cat_save = df_cat.copy()
    df_cat_save.columns = ['category', 'true_qty', 'pred_qty', 'count', 'ratio']
    cat_path = os.path.join(demo_dir, 'category_breakdown.csv')
    df_cat_save.to_csv(cat_path, index=False, encoding='utf-8-sig')
    print(f"✅ [存档] 品类明细 → {cat_path}")

    # 3. 头部 SKU 追踪 CSV（全量，按真实销量降序）
    df_sku_save = df_sku.sort_values(by='total_true', ascending=False).copy()
    df_sku_save.columns = ['sku_id', 'true_qty', 'pred_qty', 'abs_error', 'ratio']
    sku_path = os.path.join(demo_dir, 'top_sku_chase.csv')
    df_sku_save.to_csv(sku_path, index=False, encoding='utf-8-sig')
    print(f"✅ [存档] SKU 追踪表 → {sku_path}")

    # 4. 全明细结果（买手×SKU 维度）CSV
    detail_path = os.path.join(demo_dir, 'detail_buyer_sku.csv')
    df_res[['sku_name', 'category_name', 'true_qty', 'pred_qty']].to_csv(detail_path, index=False, encoding='utf-8-sig')
    print(f"✅ [存档] 买手SKU明细 → {detail_path}")
        
    # ==========================================
    # 评审 4: 120天时序绘制
    # ==========================================
    print("\n" + "="*40)
    print("📈 评审纬度 4：[绘制 120 天时序心电图]")
    print("="*40)
    if viz_samples:
        fig, axes = plt.subplots(len(viz_samples), 1, figsize=(12, 3 * len(viz_samples)))
        fig.suptitle('B2B 双塔模型: 120 天买手级补货轨迹探测照妖镜', fontsize=16)
        
        if len(viz_samples) == 1: axes = [axes]
        
        for ax, s in zip(axes, viz_samples):
            sku_name = inverse_transform_le(le_dict, 'sku_id', np.array([s['sku']]))[0]
            timeline_history = np.arange(1, 91)
            history_data = s['hist_90d']  # shape (90,), 当前是 log1p + MinMaxScaler 归一化值
            
            # [BUG-D 修复] 若 scaler 可用，还原为真实件数
            if _feature_scaler is not None:
                try:
                    dummy_shape = (90, _feature_scaler.scale_.shape[0])
                    dummy = np.zeros(dummy_shape, dtype=np.float32)
                    dummy[:, 0] = history_data
                    restored = _feature_scaler.inverse_transform(dummy)
                    history_data = np.expm1(np.maximum(restored[:, 0], 0))  # 逆 log1p
                    y_label = "每日补货量（件）"
                except Exception as e:
                    print(f"[!] Scaler 解析异常: {e}，将采用默认形式作图")
                    y_label = "归一化补货量"
            else:
                y_label = "归一化补货量"
            
            # 画左边历史 90 天
            ax.bar(timeline_history, history_data, color='gray', alpha=0.6, label='过往 90 天真实散单')
            
            # 画右边未来 30 天预测对决
            ax.plot([90, 120], [s['pred_qty'], s['pred_qty']], 'r--', lw=3, label=f"🧠 预测总量: {s['pred_qty']:.1f}件")
            ax.plot([90, 120], [s['true_qty'], s['true_qty']], 'g-', lw=2, label=f"🎯 真实总量: {s['true_qty']:.1f}件")
            
            # 装饰
            ax.axvline(x=90, color='blue', linestyle='-.', alpha=0.5)
            ax.set_title(f"Target SKU: {sku_name}", fontsize=10)
            ax.set_ylabel(y_label, fontsize=8)
            ax.set_xlim(0, 125)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # 同时保存至 reports/ 根目录（兼容旧引用）及 v1.8_demo/
        img_path = os.path.join(PROJECT_ROOT, 'reports', 'evaluation_120days.png')
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        plt.savefig(img_path, dpi=150)
        # ★ Demo 版本存档（带时间戳）
        demo_img_path = os.path.join(PROJECT_ROOT, 'reports', 'v1.8_demo', 'evaluation_120days.png')
        plt.savefig(demo_img_path, dpi=150)
        print(f"✅ 心电图已保存至: {img_path}")
        print(f"✅ [存档] 心电图 Demo 版 → {demo_img_path}")
    else:
        print("未发现适合绘制的长序列正样本。")

if __name__ == "__main__":
    run_evaluation()

