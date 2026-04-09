"""
B2B 补货系统 Phase 4A — 周粒度训练主脚本 (Weekly Training)
============================================================
与 run_training_v2.py 的核心差异：
  - 从 data/processed_weekly/ 读取周频张量
  - 读取 data/artifacts_weekly/label_encoders_weekly.pkl 和 meta_weekly.json
  - 模型保存到 models_weekly/（不覆盖任何现有模型）
  - 默认 Batch=4096，充分利用周频数据量小带来的显存余裕

控制变量 (Phase 4A)：
  - 与 Phase 2/3 一致：3 层 LSTM, hidden=256, dropout=0.3
  - 只改变数据粒度（周 vs 日），架构保持不变
  - 通过对比 F1 验证"周粒度聚合"是否真正带来提升

保持不变：
  - EnhancedTwoTowerLSTM 架构（双塔 LSTM）
  - trainer.py 的 fit_model 流程
  - 所有环境变量注入机制（EXP_HIDDEN/LAYERS/BATCH 等）
"""
import os
import json
import sys
import time
import pickle

# Windows GBK 终端兼容
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import yaml
import torch
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.train.dataset import create_lazy_dataloaders
from src.models.enhanced_model import (
    EnhancedTwoTowerLSTM, TwoTowerGRU, TwoTowerBiLSTM, TwoTowerLSTMWithAttn
)
from src.train.trainer import fit_model

MODEL_REGISTRY = {
    'lstm':   EnhancedTwoTowerLSTM,
    'gru':    TwoTowerGRU,
    'bilstm': TwoTowerBiLSTM,
    'attn':   TwoTowerLSTMWithAttn,
}


class DummyLE:
    classes_ = np.arange(13)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("🗓️  B2B Replenishment - Phase 4A 周粒度训练")
    print("=" * 60)

    # --------------------------------------------------
    # 1. 全局配置
    # --------------------------------------------------
    config_path = os.path.join(PROJECT_ROOT, 'config', 'model_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 设备: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print(f"[*] cuDNN benchmark mode: ON")
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1024**3
        print(f"[*] GPU: {props.name} ({vram_gb:.1f} GB)")

    # --------------------------------------------------
    # 2. 读取周频元信息（支持环境变量覆盖目录）
    # --------------------------------------------------
    # 挂机脚本会读入 WEEKLY_PROC_DIR/ART_DIR/SAVE_DIR 来实现不同回看窗口的自动切换
    artifacts_dir = os.environ.get(
        'WEEKLY_ART_DIR', os.path.join(PROJECT_ROOT, 'data', 'artifacts_weekly'))
    processed_dir = os.environ.get(
        'WEEKLY_PROC_DIR', os.path.join(PROJECT_ROOT, 'data', 'processed_weekly'))
    meta_path     = os.path.join(artifacts_dir, 'meta_weekly.json')
    encoder_path  = os.path.join(artifacts_dir, 'label_encoders_weekly.pkl')

    if not os.path.exists(meta_path):
        print("❌ 未找到 meta_weekly.json，请先运行:")
        print("   python src/features/build_features_weekly_sku.py")
        return
    if not os.path.exists(processed_dir):
        print("❌ 未找到 processed_weekly/ 目录，请先运行特征工程。")
        return

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    print(f"[*] 读取周频元信息:")
    print(f"    Train 样本: {meta['train_cnt']:,} | Val 样本: {meta['val_cnt']:,}")
    print(f"    Train 正样本率: {meta['pos_train']/max(meta['train_cnt'],1):.2%}")
    print(f"    Val   正样本率: {meta['pos_val']/max(meta['val_cnt'],1):.2%}")
    print(f"    序列长度: {meta['lookback_weeks']} 周 × {meta['dyn_feat_dim']} 维")

    # --------------------------------------------------
    # 3. 静态特征顺序（与 V2 保持一致）
    # --------------------------------------------------
    static_cat_cols = meta['static_cat_cols']
    static_num_cols = meta['static_num_cols']
    static_order_keys = static_cat_cols + ['month'] + static_num_cols
    print(f"[*] 静态特征顺序 ({len(static_order_keys)} 维): {static_order_keys}")

    # --------------------------------------------------
    # 4. 构建 DataLoader
    # 周频数据量小（428K），Batch 默认提升到 4096 充分利用显存
    # --------------------------------------------------
    exp_model_type = os.environ.get('EXP_MODEL_TYPE', 'lstm').lower()
    exp_hidden  = int(os.environ.get('EXP_HIDDEN',  256))
    exp_layers  = int(os.environ.get('EXP_LAYERS',  3))
    exp_dropout = float(os.environ.get('EXP_DROPOUT', 0.3))
    # 周频版本 Batch 默认 4096（显存占用极小，可大幅提升吞吐量）
    exp_batch   = int(os.environ.get('EXP_BATCH',   4096))
    exp_id      = os.environ.get('EXP_ID', 'w4a_weekly_baseline')

    print("=" * 60)
    print(f"  [架构注入] 类型={exp_model_type.upper()} | hidden={exp_hidden} | "
          f"layers={exp_layers} | dropout={exp_dropout} | batch={exp_batch}")
    print(f"  [实验 ID ] {exp_id}")
    print("=" * 60)

    print(f"[{time.strftime('%H:%M:%S')}] 构建 DataLoader (周频)...")
    train_loader, val_loader = create_lazy_dataloaders(
        processed_dir,
        batch_size=exp_batch,
        num_workers=0,
        use_sampler=False
    )

    # --------------------------------------------------
    # 5. 构建模型（控制变量：与 Phase 2/3 一致）
    # --------------------------------------------------
    ModelClass = MODEL_REGISTRY.get(exp_model_type, EnhancedTwoTowerLSTM)
    print(f"[*] 构建模型: {ModelClass.__name__} "
          f"(hidden={exp_hidden}, layers={exp_layers}, dropout={exp_dropout})")

    # 词典大小：从 weekly encoder 读取实际大小
    vocab_sizes = {
        'sku_id': 15000, 'style_id': 5000, 'product_name': 3000,
        'category': 50, 'sub_category': 150, 'season': 10,
        'series': 100, 'band': 50, 'size_id': 50, 'color_id': 200, 'month': 13
    }
    if os.path.exists(encoder_path):
        print(f"[*] 从 {encoder_path} 读取实际词典大小...")
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
        for col, le in encoders.items():
            if col in vocab_sizes:
                vocab_sizes[col] = len(le.classes_) + 5

    num_numeric = len(static_num_cols)
    model = ModelClass(
        dyn_feat_dim=7,
        lstm_hidden=exp_hidden,
        lstm_layers=exp_layers,
        static_vocab_sizes=vocab_sizes,
        static_emb_dim=16,
        num_numeric_feats=num_numeric,
        dropout=exp_dropout
    )

    # torch.compile 加速
    try:
        model = torch.compile(model)
        print("[*] torch.compile: ON")
    except Exception as e:
        print(f"[!] torch.compile 不可用: {e}")

    # --------------------------------------------------
    # 6. 启动训练（支持环境变量覆盖保存目录）
    # --------------------------------------------------
    save_dir = os.environ.get(
        'WEEKLY_SAVE_DIR', os.path.join(PROJECT_ROOT, 'models_weekly'))
    os.makedirs(save_dir, exist_ok=True)

    train_lr = float(os.environ.get('EXP_LR', config['train']['learning_rate']))
    patience = int(os.environ.get('EXP_PATIENCE', config['train']['early_stopping_patience']))
    epochs   = int(os.environ.get('EXP_EPOCHS', config['train']['epochs']))

    print(f"[*] 学习率={train_lr} | patience={patience} | epochs={epochs}")

    best_model_path = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config_epochs=epochs,
        learning_rate=train_lr,
        device=device,
        static_order_keys=static_order_keys,
        save_dir=save_dir,
        val_every=1,
        patience_limit=patience
    )

    print(f"\n🎯 Phase 4A 训练完成! Best checkpoint => {best_model_path}")


if __name__ == "__main__":
    main()
