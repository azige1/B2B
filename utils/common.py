import yaml
import random
import numpy as np
import torch

def set_seed(seed):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_yaml(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_yaml(config, config_path):
    """保存配置到 YAML 文件"""
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

def ensure_directory(directory):
    """确保目录存在"""
    import os
    os.makedirs(directory, exist_ok=True)