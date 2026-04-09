import os
import yaml
import torch
import numpy as np
import pickle
import sys
        
# 从 src/train 返回两层拿到项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.train.dataset import create_lazy_dataloaders
from src.models.enhanced_model import EnhancedTwoTowerLSTM
from src.train.trainer import fit_model

class DummyLE:
    classes_ = np.arange(13)

def main():
    import sys, time
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    print("="*60)
    print("🚀 B2B Replenishment System v1.5 - Training Pipeline")
    print("="*60)
    
    # 1. 读取配置文件
    config_path = os.path.join(PROJECT_ROOT, 'config', 'model_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Base device detected: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("[*] cuDNN benchmark mode: ON")
    
    # 2. 静态特征顺序（前12分类Embedding，后7数值Linear）
    static_order_keys = [
        'buyer_id', 'sku_id', 'style_id', 'product_name', 'category', 
        'sub_category', 'season', 'series', 'band', 'size_id', 'color_id', 
        'month', 
        'qty_first_order', 'price_tag',
        'cooperation_years', 'monthly_average_replenishment', 
        'avg_discount_rate', 'replenishment_frequency', 'item_coverage_rate'
    ]
    
    # 3. 构建 DataLoaders
    processed_dir = os.path.join(PROJECT_ROOT, config['paths']['dataset_dir'])
    print(f"[{time.strftime('%H:%M:%S')}] 构建 DataLoader...")
    train_loader, val_loader = create_lazy_dataloaders(
        processed_dir, 
        batch_size=config['train']['batch_size'],
        num_workers=0  # Windows 下 0 最安全
    )
    
    # [方案1] 创建固定种子的 20% 验证子集，训练中快速监控用（节省 ~80% 验证时间）
    val_subset_ratio = config['train'].get('val_subset_ratio', 1.0)
    if val_subset_ratio < 1.0:
        import random
        seed = config.get('seed', 42)
        random.seed(seed)
        n_val = len(val_loader.dataset)
        n_subset = max(1, int(n_val * val_subset_ratio))
        subset_idx = sorted(random.sample(range(n_val), n_subset))
        from torch.utils.data import Subset
        val_subset_ds = Subset(val_loader.dataset, subset_idx)
        val_monitor_loader = torch.utils.data.DataLoader(
            val_subset_ds, batch_size=config['train']['batch_size'],
            shuffle=False, num_workers=0, pin_memory=True
        )
        print(f"[*] 验证子集已创建: {n_subset:,}/{n_val:,} 样本 (ratio={val_subset_ratio}, seed={seed})")
    else:
        val_monitor_loader = val_loader  # 全量验证
    
    # 4. 组装双塔模型
    print("[*] Assembling Enhanced Two-Tower Neural Network...")
    artifacts_dir = os.path.join(PROJECT_ROOT, 'data', 'artifacts')
    encoder_path = os.path.join(artifacts_dir, 'label_encoders.pkl')
    
    dynamic_vocab_sizes = {
        'buyer_id': 1000, 'sku_id': 15000, 'style_id': 3000, 'product_name': 2000, 
        'category': 50, 'sub_category': 100, 'season': 10, 'series': 50,
        'band': 50, 'size_id': 50, 'color_id': 100, 'month': 13
    }
    if os.path.exists(encoder_path):
        print(f"[*] Reading real vocabulary boundary from {encoder_path}...")
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
            for col, le in encoders.items():
                if col in dynamic_vocab_sizes:
                    dynamic_vocab_sizes[col] = len(le.classes_) + 5
    else:
        print("[!] No encoder pkl found, using default static dimension limits.")
    
    num_numeric_feats = len(static_order_keys) - len(dynamic_vocab_sizes)  # 14 - 12 = 2
    model = EnhancedTwoTowerLSTM(
        dyn_feat_dim=7,
        lstm_hidden=config['model']['hidden_size'],
        lstm_layers=config['model']['num_layers'],
        static_vocab_sizes=dynamic_vocab_sizes,
        static_emb_dim=16,
        num_numeric_feats=num_numeric_feats,
        dropout=config['model']['dropout']
    )
    
    # [方案3] torch.compile: PyTorch 2.0+ 图优化，首次 epoch 多花几分钟编译，之后每 batch 快 ~20%
    try:
        model = torch.compile(model)
        print("[*] torch.compile: ON (PyTorch 2.0+ 图优化已开启)")
    except Exception as e:
        print(f"[!] torch.compile 不可用 ({e})，跳过")
    
    # 5. 启动训练
    save_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(save_dir, exist_ok=True)
    
    best_model_path = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_monitor_loader,  # 训练监控用 20% 子集
        config_epochs=config['train']['epochs'],
        learning_rate=config['train']['learning_rate'],
        device=device,
        static_order_keys=static_order_keys,
        save_dir=save_dir,
        val_every=config['train'].get('val_every', 1)
    )
    
    print(f"🎯 训练完成. Best checkpoint => {best_model_path}")

if __name__ == "__main__":
    main()

