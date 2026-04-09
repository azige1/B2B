import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# 添加项目根目录到 sys.path (在 src/evaluate/ 下需后退两层)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.models.enhanced_model import EnhancedTwoTowerLSTM
from src.train.dataset import create_lazy_dataloaders
from utils.common import load_yaml
from src.train.run_training import DummyLE # 兼容反序列化

def generate_validation_report():
    print("=" * 60)
    print("🚀 B2B 智能补货系统 - 验证期明细导出流水线")
    print("=" * 60)

    # 1. 加载配置
    config_path = os.path.join(PROJECT_ROOT, 'config', 'model_config.yaml')
    config = load_yaml(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用设备: {device}")

    # 2. 准备数据加载器
    processed_dir = os.path.join(PROJECT_ROOT, config['paths']['dataset_dir'])
    # 我们只需要 val_loader
    print(f"[*] 正在挂载验证集张量 (mmap_mode)...")
    _, val_loader = create_lazy_dataloaders(
        processed_dir, 
        batch_size=2048, 
        num_workers=0
    )

    # 3. 加载标签解码器
    encoder_path = os.path.join(PROJECT_ROOT, 'data', 'artifacts', 'label_encoders.pkl')
    with open(encoder_path, 'rb') as f:
        encoders = pickle.load(f)
    print(f"[*] 解码器加载完成。")

    # 4. 初始化模型
    static_order_keys = [
        'buyer_id', 'sku_id', 'style_id', 'product_name', 'category', 
        'sub_category', 'season', 'series', 'band', 'size_id', 'color_id', 
        'month', 
        'qty_first_order', 'price_tag',
        'cooperation_years', 'monthly_average_replenishment', 
        'avg_discount_rate', 'replenishment_frequency', 'item_coverage_rate'
    ]
    
    dynamic_vocab_sizes = {
        'buyer_id': 1000, 'sku_id': 15000, 'style_id': 3000, 'product_name': 2000, 
        'category': 50, 'sub_category': 100, 'season': 10, 'series': 50,
        'band': 50, 'size_id': 50, 'color_id': 100, 'month': 13
    }
    for col, le in encoders.items():
        if col in dynamic_vocab_sizes: dynamic_vocab_sizes[col] = len(le.classes_) + 5
            
    num_numeric_feats = len(static_order_keys) - len(dynamic_vocab_sizes)
    model = EnhancedTwoTowerLSTM(
        dyn_feat_dim=7, lstm_hidden=config['model']['hidden_size'],
        lstm_layers=config['model']['num_layers'], static_vocab_sizes=dynamic_vocab_sizes,
        static_emb_dim=16, num_numeric_feats=num_numeric_feats, dropout=0.0
    ).to(device)

    # 加载最佳权重
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_enhanced_model.pth')
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型权重文件: {model_path}")
        return
        
    print(f"[*] 正在加载最佳模型权重: {model_path}")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            new_state_dict[k.replace('_orig_mod.', '')] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # 5. 执行推理
    all_buyers = []
    all_skus = []
    all_months = []
    all_probs = []
    all_preds_qty = []
    all_true_qty = []

    print("[*] 开始执行模型前向推理，抓取验证期明细...")
    with torch.no_grad():
        for x_dyn, x_static, y_cls, y_reg in tqdm(val_loader, desc="Inference"):
            x_dyn = x_dyn.to(device)
            x_static = x_static.to(device)

            logits, preds = model(x_dyn, x_static, static_order_keys)
            
            # --- 与 trainer.py 保持完全一致的强控策略 ---
            prob = torch.sigmoid(logits)
            
            # 动态死区过滤
            dyn_sum = x_dyn.abs().sum(dim=(1,2))
            dead_mask = (dyn_sum < 1e-4)
            prob[dead_mask] = 0.0
            
            # [v1.9] prob>0.35, scale=1.0
            preds = torch.clamp(preds, max=8.5)
            pred_qty = torch.expm1(preds) * (prob > 0.35).float() * 1.0
            true_qty = torch.expm1(y_reg)
            
            # --- 抽取静态特征 (0:buyer, 1:sku, 11:month) ---
            buyers = x_static[:, 0].cpu().numpy().astype(int)
            skus = x_static[:, 1].cpu().numpy().astype(int)
            months = x_static[:, 11].cpu().numpy().astype(int)
            
            # 存入列表
            all_buyers.extend(buyers)
            all_skus.extend(skus)
            all_months.extend(months)
            all_probs.extend(prob.cpu().numpy().flatten())
            # inf/nan 安全过滤
            pq = pred_qty.cpu().numpy().flatten()
            tq = true_qty.cpu().numpy().flatten()
            pq = np.nan_to_num(pq, nan=0.0, posinf=0.0, neginf=0.0)
            tq = np.nan_to_num(tq, nan=0.0, posinf=0.0, neginf=0.0)
            all_preds_qty.extend(pq)
            all_true_qty.extend(tq)

    # 6. 数据解码与成表
    print("[*] 推理完成，正在构建 Pandas 数据框并解码...")
    df_report = pd.DataFrame({
        'buyer_idx': all_buyers,
        'sku_idx': all_skus,
        'month': all_months,
        'ai_prob': all_probs,
        'ai_pred_qty': all_preds_qty,
        'true_qty': all_true_qty
    })

    # 逆向解码 Buyer 和 SKU
    try:
        df_report['buyer_id'] = encoders['buyer_id'].inverse_transform(df_report['buyer_idx'])
        df_report['sku_id'] = encoders['sku_id'].inverse_transform(df_report['sku_idx'])
    except Exception as e:
        print(f"⚠️ 解码警告: {e} (可能包含不可见类别)")
        
    df_report.drop(columns=['buyer_idx', 'sku_idx'], inplace=True)
    
    # 将列重新排序便于查看
    cols = ['month', 'buyer_id', 'sku_id', 'ai_prob', 'ai_pred_qty', 'true_qty']
    df_report = df_report[cols]

    # 为了方便业务查看，我们只保留有“真实发生销量”或由“AI提出补货预警”的重点行
    # 这将极大减少冗余的“真阴性（全0）”数据行
    print("[*] 正在精简冗余数据...")
    df_report['ai_pred_qty'] = df_report['ai_pred_qty'].round(1)
    df_report['true_qty'] = df_report['true_qty'].round(1)
    
    # 保留 AI预测 > 0 或 真实 > 0 的记录
    mask_keep = (df_report['ai_pred_qty'] > 0) | (df_report['true_qty'] > 0)
    df_focused = df_report[mask_keep].copy()

    # 输出数值分布摘要，帮助业务判断是否合理
    # 输出数值分布摘要
    pos_preds = df_focused[df_focused['ai_pred_qty'] > 0]['ai_pred_qty']
    if len(pos_preds) > 0:
        print(f"[*] 📊 AI建议件数分布: 中位数={pos_preds.median():.1f}件, 90%={pos_preds.quantile(0.9):.1f}件, 最大={pos_preds.max():.1f}件")

    sum_pred = df_focused['ai_pred_qty'].sum()
    sum_true = df_focused['true_qty'].sum()
    global_ratio = sum_pred / sum_true if sum_true > 0 else 0

    both_mask = (df_focused['ai_pred_qty'] > 0) & (df_focused['true_qty'] > 0)
    buyer_mae = (df_focused.loc[both_mask, 'ai_pred_qty'] - df_focused.loc[both_mask, 'true_qty']).abs().mean()

    sku_agg = df_focused.groupby('sku_id')[['ai_pred_qty', 'true_qty']].sum()
    sku_both = sku_agg[(sku_agg['ai_pred_qty'] > 0) & (sku_agg['true_qty'] > 0)]
    sku_mae = (sku_both['ai_pred_qty'] - sku_both['true_qty']).abs().mean() if len(sku_both) > 0 else 0

    # 7. 导出至 Excel / CSV
    reports_dir = os.path.join(PROJECT_ROOT, 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    output_csv = os.path.join(reports_dir, 'validation_strategy_details.csv')
    df_focused.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print("\n" + "=" * 60)
    print(f"✅ 验证期模型测试大盘指标生成完毕！")
    print("-" * 60)
    print(f"  📌 关注样本池: {len(df_focused):,} 条记录")
    print(f"  📌 模型预测比 (Ratio): {global_ratio:.3f}")
    if len(df_focused[both_mask]) > 0:
        print(f"  📌 (门店视角) 单店单款平均误差: {buyer_mae:.1f} 件/单")
    if len(sku_both) > 0:
        print(f"  📌 (甲方视角) SKU全国总量平均误差: {sku_mae:.1f} 件/款")
    print("-" * 60)
    print(f"📁 保存路径: {output_csv}")
    print("=" * 60)

if __name__ == "__main__":
    generate_validation_report()
