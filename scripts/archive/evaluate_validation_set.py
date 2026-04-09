import sys
import os
import yaml
import torch
import numpy as np
import pandas as pd

# Add source dirs to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src', 'train'))

from run_training_v2 import EnhancedTwoTowerLSTM
from dataset import create_lazy_dataloaders

class DummyLE:
    classes_ = np.arange(13)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Config
    with open(os.path.join(PROJECT_ROOT, 'config', 'model_config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    import json
    import pickle

    with open(os.path.join(PROJECT_ROOT, 'data', 'artifacts_v2', 'meta_v2.json'), 'r') as f:
        meta = json.load(f)
    with open(os.path.join(PROJECT_ROOT, 'data', 'artifacts_v2', 'label_encoders_v2.pkl'), 'rb') as f:
        encoders = pickle.load(f)

    cat_cols = meta['static_cat_cols']
    num_cols = meta['static_num_cols']

    static_vocab_sizes = {c: len(encoders[c].classes_) + 5 for c in cat_cols if c in encoders}
    static_vocab_sizes['month'] = 18
    num_numeric_feats = len(num_cols)

    # Initialize Model
    model = EnhancedTwoTowerLSTM(
        dyn_feat_dim=7,
        static_vocab_sizes=static_vocab_sizes,
        num_numeric_feats=num_numeric_feats,
        lstm_hidden=128
    ).to(device)
    model_path = os.path.join(PROJECT_ROOT, 'models_v2', 'best_enhanced_model.pth')
    state_dict = torch.load(model_path, map_location=device, weights_only=False)

    # Strip '_orig_mod.' prefix if compiled
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('_orig_mod.', '') if k.startswith('_orig_mod.') else k
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    # Load Validation Set
    _, val_loader = create_lazy_dataloaders(
        processed_dir=os.path.join(PROJECT_ROOT, 'data', 'processed_v2'),
        batch_size=1024,
        num_workers=0
    )

    print("\n=== 开始在验证集 (Validation Set) 上进行批量推断 (Inference)... ===")
    all_actual_reg = []
    all_pred_cls = []
    all_pred_reg = []

    with torch.no_grad():
        for i, (x_dyn, x_static, y_cls, y_reg) in enumerate(val_loader):
            x_dyn = x_dyn.to(device)
            x_static = x_static.to(device)
            static_keys = cat_cols + ['month'] + num_cols
            o_cls, o_reg = model(x_dyn, x_static, static_keys)
            
            all_actual_reg.append(y_reg.cpu().numpy())
            all_pred_cls.append(torch.sigmoid(o_cls).cpu().numpy())
            all_pred_reg.append(o_reg.cpu().numpy())
            if i % 10 == 0:
                print(f"   已处理 Bath {i}...")

    # Flatten
    y_true_reg = np.concatenate(all_actual_reg).flatten()
    y_pred_reg = np.concatenate(all_pred_reg).flatten()
    y_pred_prob = np.concatenate(all_pred_cls).flatten()

    # Reverse Log1p
    actual_qty = np.expm1(y_true_reg)
    
    # Apply Clamping and Decoding
    preds_clamped = np.clip(y_pred_reg, a_min=None, a_max=8.5)
    pred_qty_raw = np.expm1(preds_clamped) * (y_pred_prob > 0.20) * 0.8
    pred_qty = np.nan_to_num(pred_qty_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Build DataFrame
    res_df = pd.DataFrame({
        'Actual_Qty': actual_qty,
        'Predicted_Qty': pred_qty,
        'Prob_Score': y_pred_prob
    })

    print('\n======================================================')
    print('=== [V2.3 验证集评测报告] 预测件数 vs 真实标签对比展示 ===')
    print('======================================================')

    # Positive samples (where actual > 5)
    pos_mask = res_df['Actual_Qty'] >= 5
    if pos_mask.sum() > 0:
        pos_samples = res_df[pos_mask].sample(n=min(10, pos_mask.sum()), random_state=42)
        print("\n=== [场景 1: 模型对‘历史真实爆单款’的预测表现] ===")
        print("   (筛选实际补货量 >= 5件的样本，打乱随机抽出10个)")
        for _, r in pos_samples.iterrows():
            diff = r['Predicted_Qty'] - r['Actual_Qty']
            diff_str = f"+{diff:4.0f}" if diff >= 0 else f"{diff:5.0f}"
            print(f"-> 实际真实补货: {r['Actual_Qty']:6.0f} 件 | AI 预算排产: {r['Predicted_Qty']:6.0f} 件 (差值 {diff_str}) | AI判断补货概率: {r['Prob_Score']*100:4.1f}%")

    # Negative samples (actual == 0)
    neg_mask = res_df['Actual_Qty'] == 0
    if neg_mask.sum() > 0:    
        neg_samples = res_df[neg_mask].sample(n=5, random_state=42)
        print("\n=== [场景 2: 模型对‘历史冷板凳款’的克制表现] ===")
        print("   (筛选实际补货量为 0 件的样本，打乱随机抽出5个)")
        for _, r in neg_samples.iterrows():
            print(f"-> 实际真实补货: {r['Actual_Qty']:6.0f} 件 | AI 预算排产: {r['Predicted_Qty']:6.0f} 件 | AI判断补货概率: {r['Prob_Score']*100:4.1f}%")

    # Large predictions 
    large_preds = res_df.sort_values('Predicted_Qty', ascending=False).head(10)
    print("\n=== [场景 3: AI 预测中最敢‘放量’的 Top 10 个款] ===")
    print("   (验证这些款在未来 30 天是否真实发生了大批量补货)")
    for _, r in large_preds.iterrows():
        diff = r['Predicted_Qty'] - r['Actual_Qty']
        diff_str = f"+{diff:4.0f}" if diff >= 0 else f"{diff:5.0f}"
        print(f"-> 实际真实补货: {r['Actual_Qty']:6.0f} 件 | AI 预算排产: {r['Predicted_Qty']:6.0f} 件 (差值 {diff_str}) | AI判断补货概率: {r['Prob_Score']*100:4.1f}%")

    print("\n======================================================")
    print("=== 大盘终局指标 (Macro Metrics) ===")
    abs_err = np.abs(res_df['Predicted_Qty'] - res_df['Actual_Qty'])
    print(f"* 日均/样本 MAE 误差:  {abs_err.mean():.2f} 件/样本")
    print(f"* 验证集全盘实际总补货: {res_df['Actual_Qty'].sum():.0f} 件")
    print(f"* 验证集全盘 AI 总预算: {res_df['Predicted_Qty'].sum():.0f} 件")
    ratio = res_df['Predicted_Qty'].sum() / (res_df['Actual_Qty'].sum() + 1e-5)
    print(f"* 发货充盈率 (Ratio):   {ratio:.2f} (该值在 1.0~1.5 之间为最佳)")
    print("======================================================\n")

if __name__ == '__main__':
    main()
