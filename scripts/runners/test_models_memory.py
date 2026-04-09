import os
import torch
import torch.nn as nn

from src.models.enhanced_model import (
    EnhancedTwoTowerLSTM, TwoTowerGRU, TwoTowerBiLSTM, TwoTowerLSTMWithAttn
)

# 映射与挂机脚本保持一致
MODEL_REGISTRY = {
    'lstm':    EnhancedTwoTowerLSTM,
    'gru':     TwoTowerGRU,
    'bilstm':  TwoTowerBiLSTM,
    'attn':    TwoTowerLSTMWithAttn,
}

EXPERIMENTS = [
    {"id": "e11_GRU",    "type": "gru",    "h": 256, "l": 2, "d": 0.3, "b": 2048},
    {"id": "e12_BiLSTM", "type": "bilstm", "h": 256, "l": 2, "d": 0.3, "b": 1024},
    {"id": "e13_Attn",   "type": "attn",   "h": 256, "l": 2, "d": 0.3, "b": 2048},
    {"id": "e21_h128",   "type": "lstm",   "h": 128, "l": 2, "d": 0.3, "b": 2048},
    {"id": "e22_h384",   "type": "lstm",   "h": 384, "l": 2, "d": 0.3, "b": 1024},
    {"id": "e23_l1",     "type": "lstm",   "h": 256, "l": 1, "d": 0.3, "b": 2048},
    {"id": "e24_l3",     "type": "lstm",   "h": 256, "l": 3, "d": 0.3, "b": 1024},
    {"id": "e25_d05",    "type": "lstm",   "h": 256, "l": 2, "d": 0.5, "b": 2048},
    {"id": "e26_d07",    "type": "lstm",   "h": 256, "l": 2, "d": 0.7, "b": 2048},
    {"id": "e27_b1024",  "type": "lstm",   "h": 256, "l": 2, "d": 0.3, "b": 1024},
    {"id": "e31_rw03",   "type": "lstm",   "h": 256, "l": 2, "d": 0.3, "b": 2048},
    {"id": "e32_rw07",   "type": "lstm",   "h": 256, "l": 2, "d": 0.3, "b": 2048},
    {"id": "e33_sf025",  "type": "lstm",   "h": 256, "l": 2, "d": 0.3, "b": 2048},
    {"id": "e34_sf10",   "type": "lstm",   "h": 256, "l": 2, "d": 0.3, "b": 2048},
    {"id": "e35_hub10",  "type": "lstm",   "h": 256, "l": 2, "d": 0.3, "b": 2048},
]

# 模拟 13 个静态特征维度顺序
STATIC_KEYS = [
    'sku_id', 'style_id', 'product_name', 'category', 'sub_category', 
    'season', 'series', 'band', 'size_id', 'color_id', 'month', 
    'qty_first_order', 'price_tag'
]

# 模拟实际字典
VOCAB_SIZES = {
    'sku_id': 15000, 'style_id': 5000, 'product_name': 3000,
    'category': 50, 'sub_category': 150, 'season': 10,
    'series': 100, 'band': 50, 'size_id': 50, 'color_id': 200, 'month': 13
}

def test_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"============================================================")
    print(f" 🧪 B2B 模型体检极速探针 (Device: {device})")
    print(f"============================================================")
    
    for exp in EXPERIMENTS:
        model_cls = MODEL_REGISTRY[exp['type']]
        batch_size = exp['b']
        
        # 构造对应 batch 的假数据
        x_dyn = torch.randn(batch_size, 90, 7).to(device)
        x_sta_cat = torch.zeros((batch_size, 11)).float()
        x_sta_num = torch.randn(batch_size, 2).float()
        x_static = torch.cat([x_sta_cat, x_sta_num], dim=1).to(device)
        y_dummy = torch.randn(batch_size, 1).to(device)
        
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(0)
            
            model = model_cls(
                dyn_feat_dim=7,
                lstm_hidden=exp['h'],
                lstm_layers=exp['l'],
                static_vocab_sizes=VOCAB_SIZES,
                static_emb_dim=16,
                num_numeric_feats=2,
                dropout=exp['d']
            ).to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            # --- 测试前向传播 ---
            logits, preds = model(x_dyn, x_static, STATIC_KEYS)
            
            # --- 测试反向传播 ---
            loss = nn.BCEWithLogitsLoss()(logits, y_dummy) + nn.HuberLoss()(preds, y_dummy)
            loss.backward()
            optimizer.step()
            
            peak_mb = torch.cuda.max_memory_allocated(0) / 1024 / 1024
            
            print(f"✅ [PASS] {exp['id']:<10} | {model_cls.__name__:<20} | 峰值显存: {peak_mb:4.0f} MB")
            
            # 收工清场防 OOM
            del model, logits, preds, loss, optimizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ [FAIL] {exp['id']} 崩溃！原因: {str(e)}")

    print(f"============================================================")
    print("探雷结束！所有标记为 PASS 的模型今夜均可安全挂机。")

if __name__ == "__main__":
    test_models()
