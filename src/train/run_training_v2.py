"""
B2B 补货系统 V2.0 — SKU 级全国汇总 LSTM 训练主脚本
====================================================
与 V1 (run_training.py) 的核心差异：
  - 静态特征去掉 buyer_id，只保留 SKU 商品画像
  - 从 data/processed_v2/ 读取张量（不覆盖 V1）
  - 模型保存到 models_v2/（不覆盖 V1）
  - 读取 data/artifacts_v2/label_encoders_v2.pkl

保持不变：
  - EnhancedTwoTowerLSTM 架构（仍然是双塔 LSTM）
  - trainer.py 的 fit_model 流程
  - trainer.py 的验证 Ratio 逻辑
"""
import os
import json
import sys
import time
import pickle
import random

# Windows GBK 终端兼容: 必须在任何输出之前强制使用 UTF-8
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.train.dataset import create_lazy_dataloaders
from src.models.enhanced_model import (
    EnhancedTwoTowerLSTM, TwoTowerGRU, TwoTowerBiLSTM, TwoTowerLSTMWithAttn,
    TwoTowerLSTMPool, TwoTowerBiLSTMPool
)
from src.train.trainer import fit_model

# 模型类型路由表
MODEL_REGISTRY = {
    'lstm':    EnhancedTwoTowerLSTM,
    'gru':     TwoTowerGRU,
    'bilstm':  TwoTowerBiLSTM,
    'attn':    TwoTowerLSTMWithAttn,
    'lstm_pool': TwoTowerLSTMPool,
    'bilstm_pool': TwoTowerBiLSTMPool,
}


class DummyLE:
    classes_ = np.arange(13)


def maybe_set_seed():
    seed_env = os.environ.get('EXP_SEED')
    if seed_env is None or str(seed_env).strip() == '':
        return None

    seed = int(seed_env)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    deterministic = os.environ.get('EXP_DETERMINISTIC', '1').lower() not in {'0', 'false', 'no'}
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

    print("=" * 60)
    print("🚀 B2B Replenishment System V2.0 - SKU全国汇总训练")
    print("=" * 60)

    # --------------------------------------------------
    # 1. 读取配置（沿用 V1 的 model_config.yaml）
    # --------------------------------------------------
    config_path = os.path.join(PROJECT_ROOT, 'config', 'model_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 设备: {device}")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            torch.backends.cudnn.allow_tf32 = True
        print("[*] cuDNN benchmark mode: ON")

    exp_seed = maybe_set_seed()
    if exp_seed is not None:
        deterministic = os.environ.get('EXP_DETERMINISTIC', '1').lower() not in {'0', 'false', 'no'}
        print(f"[*] Seed: {exp_seed} | deterministic={deterministic}")

    # --------------------------------------------------
    # 2. 读取 meta 信息（支持 V3/V5 双版本）
    # --------------------------------------------------
    # EXP_VERSION=v5 使用 10 维新特征，默认 v3 使用原始 7 维
    exp_version    = os.environ.get('EXP_VERSION', 'v3').lower()
    if exp_version == 'v5':
        artifacts_dir = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v5')
        processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_v5')
        meta_path     = os.path.join(artifacts_dir, 'meta_v5.json')
        encoder_path  = os.path.join(artifacts_dir, 'label_encoders_v5.pkl')
        save_dir      = os.path.join(PROJECT_ROOT, 'models_v5')
        print(f"[*] 数据版本: V5 (10维新特征，含期货信号)")
    elif exp_version == 'v5_lite':
        artifacts_dir = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v5_lite')
        processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_v5_lite')
        meta_path     = os.path.join(artifacts_dir, 'meta_v5_lite.json')
        encoder_path  = os.path.join(artifacts_dir, 'label_encoders_v5_lite.pkl')
        save_dir      = os.path.join(PROJECT_ROOT, 'models_v5_lite')
        print(f"[*] 数据版本: V5-lite (6维: 补货3维 + 期货3维)")
    elif exp_version == 'v5_lite_cov':
        artifacts_dir = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v5_lite_cov')
        processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_v5_lite_cov')
        meta_path     = os.path.join(artifacts_dir, 'meta_v5_lite_cov.json')
        encoder_path  = os.path.join(artifacts_dir, 'label_encoders_v5_lite_cov.pkl')
        save_dir      = os.path.join(PROJECT_ROOT, 'models_v5_lite_cov')
        print(f"[*] 鏁版嵁鐗堟湰: V5-lite-cov (6缁? sparse sequence + buyer coverage)")
    elif exp_version == 'v3_filtered':
        artifacts_dir = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v3_filtered')
        processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_v3_filtered')
        meta_path     = os.path.join(artifacts_dir, 'meta_v3_filtered.json')
        encoder_path  = os.path.join(artifacts_dir, 'label_encoders_v3_filtered.pkl')
        save_dir      = os.path.join(PROJECT_ROOT, 'models_v3_filtered')
        print(f"[*] 数据版本: V3-filtered (7维原始特征，同 V5-lite SKU 宇宙)")
    else:
        artifacts_dir = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v3')
        processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_v3')
        meta_path     = os.path.join(artifacts_dir, 'meta_v2.json')
        encoder_path  = os.path.join(artifacts_dir, 'label_encoders_v2.pkl')
        save_dir      = os.path.join(PROJECT_ROOT, 'models_v2')
        print(f"[*] 数据版本: V3 (7维原始特征)")

    artifacts_dir = os.environ.get('EXP_ARTIFACTS_DIR', artifacts_dir)
    processed_dir = os.environ.get('EXP_PROCESSED_DIR', processed_dir)
    meta_path = os.environ.get('EXP_META_PATH', meta_path)
    encoder_path = os.environ.get('EXP_ENCODER_PATH', encoder_path)
    save_dir = os.environ.get('EXP_MODEL_DIR', save_dir)

    if not os.path.exists(meta_path):
        print(f"❌ 未找到元数据: {meta_path}")
        return

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    print(f"[*] 读取 V2 元信息:")
    print(f"    Train 样本: {meta['train_cnt']:,} | Val 样本: {meta['val_cnt']:,}")
    print(f"    Train 正样本率: {meta['pos_train']/max(meta['train_cnt'],1):.2%}")
    print(f"    Val   正样本率: {meta['pos_val']/max(meta['val_cnt'],1):.2%}")

    # --------------------------------------------------
    # 3. V2/V5 静态特征顺序（不变）
    # --------------------------------------------------
    static_cat_cols = meta['static_cat_cols']
    static_num_cols = meta['static_num_cols']
    static_order_keys = static_cat_cols + ['month'] + static_num_cols
    lookback = meta.get('lookback', 90)
    # 动态特征维度从 meta 读取，不硬编码
    dyn_feat_dim = meta.get('dyn_feat_dim', 7)

    print(f"[*] 静态特征顺序 ({len(static_order_keys)} 维): {static_order_keys}")
    print(f"[*] 动态特征形状: lookback={lookback} × dyn_feat_dim={dyn_feat_dim}")

    # --------------------------------------------------
    # 4. 构建 DataLoader
    # --------------------------------------------------
    # [V2.2 结构超参注入] 从环境变量读取，允许挂机脚本动态切换
    exp_model_type = os.environ.get('EXP_MODEL_TYPE', 'lstm').lower()
    exp_hidden  = int(os.environ.get('EXP_HIDDEN',  config['model']['hidden_size']))
    exp_layers  = int(os.environ.get('EXP_LAYERS',  config['model']['num_layers']))
    exp_dropout = float(os.environ.get('EXP_DROPOUT', config['model']['dropout']))
    exp_batch   = int(os.environ.get('EXP_BATCH',   config['train']['batch_size']))
    exp_workers = max(0, int(os.environ.get('EXP_WORKERS', '2')))
    exp_monitor = os.environ.get('EXP_MONITOR_METRIC', 'loss').lower()
    
    print("=" * 60)
    print(f"  [架构注入] 模型类型={exp_model_type.upper()} | hidden={exp_hidden} | layers={exp_layers} | dropout={exp_dropout} | batch={exp_batch} | workers={exp_workers}")
    print(f"  [选模口径] monitor={exp_monitor}")
    print("=" * 60)
    
    print(f"[{time.strftime('%H:%M:%S')}] 构建 V2 DataLoader...")
    train_loader, val_loader = create_lazy_dataloaders(
        processed_dir,
        batch_size=exp_batch,
        num_workers=exp_workers,
        use_sampler=False,
        lookback=lookback,
        dyn_feat_dim=dyn_feat_dim
    )

    # --------------------------------------------------
    # 5. 构建模型 (双塔 LSTM，右塔去掉 buyer_id)
    # --------------------------------------------------
    ModelClass = MODEL_REGISTRY.get(exp_model_type, EnhancedTwoTowerLSTM)
    print(f"[*] 构建模型: {ModelClass.__name__} (hidden={exp_hidden}, layers={exp_layers}, dropout={exp_dropout})")

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
        dyn_feat_dim=dyn_feat_dim,   # ★ 从 meta 动态读取，支持 7/10 维
        lstm_hidden=exp_hidden,
        lstm_layers=exp_layers,
        static_vocab_sizes=vocab_sizes,
        static_emb_dim=16,
        num_numeric_feats=num_numeric,
        dropout=exp_dropout
    )

    # torch.compile 加速 (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("[*] torch.compile: ON")
    except Exception as e:
        print(f"[!] torch.compile 不可用: {e}")

    # --------------------------------------------------
    # 6. 启动训练
    # --------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    
    # [V3.0 自动化挂机] 拦截配置文件学习率
    train_lr = config['train']['learning_rate']
    if 'EXP_LR' in os.environ:
        train_lr = float(os.environ['EXP_LR'])
        print(f"[*] 实验环境注入: 学习率 (learning_rate) = {train_lr}")

    # [Phase 5] 支持 epochs 和 patience 环境变量覆盖
    train_epochs   = int(os.environ.get('EXP_EPOCHS',   str(config['train']['epochs'])))
    train_patience = int(os.environ.get('EXP_PATIENCE', str(config['train']['early_stopping_patience'])))
    if 'EXP_EPOCHS' in os.environ:
        print(f"[*] 实验环境注入: epochs={train_epochs}, patience={train_patience}")

    best_model_path = fit_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config_epochs=train_epochs,
        learning_rate=train_lr,
        device=device,
        static_order_keys=static_order_keys,
        save_dir=save_dir,
        val_every=1,   
        patience_limit=train_patience
    )

    print(f"\n🎯 V2.0 训练完成! Best checkpoint => {best_model_path}")


if __name__ == "__main__":
    main()
