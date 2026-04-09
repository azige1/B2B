"""
高并发惰性时序数据集引擎 (Lazy Sequence Dataset)

根据 1.5 铁律，18万 buyer-sku 组合 120 天的数据张量会使内存完全枯竭。
此类强制使用 np.load(..., mmap_mode='r')，让海量张量保持在磁盘，
仅在 __getitem__ 中按 batch 获取并送入显存。
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ReplenishSparseDataset(Dataset):
    def __init__(self, x_dyn_path, x_static_path, y_cls_path, y_reg_path,
                 dyn_feat_dim=7, lookback=None):
        import os
        # 严苛地遵守内存铁律：使用大小除法通过 mmap_mode 提取尺寸
        self.length = os.path.getsize(y_cls_path) // 4
        if self.length <= 0:
            raise ValueError(f"标签文件为空或样本数无效: {y_cls_path}")
        
        # 反推静态特征有多少列 (字节数 / 4 / 样本数)
        total_static_floats = os.path.getsize(x_static_path) // 4
        if total_static_floats % self.length != 0:
            raise ValueError(
                f"静态特征文件大小与样本数不整除: {x_static_path} | "
                f"floats={total_static_floats}, samples={self.length}"
            )
        static_dim = total_static_floats // self.length
        
        # 动态特征优先使用显式契约；未传入时退回旧逻辑推断 lookback
        total_dyn_floats = os.path.getsize(x_dyn_path) // 4
        if lookback is not None:
            expected_dyn_floats = self.length * lookback * dyn_feat_dim
            if total_dyn_floats != expected_dyn_floats:
                raise ValueError(
                    f"动态特征文件大小与显式契约不匹配: {x_dyn_path} | "
                    f"expected={expected_dyn_floats}, actual={total_dyn_floats}, "
                    f"samples={self.length}, lookback={lookback}, dyn_feat_dim={dyn_feat_dim}"
                )
        else:
            divisor = self.length * dyn_feat_dim
            if divisor <= 0 or total_dyn_floats % divisor != 0:
                raise ValueError(
                    f"无法根据文件大小推断动态特征形状: {x_dyn_path} | "
                    f"floats={total_dyn_floats}, samples={self.length}, dyn_feat_dim={dyn_feat_dim}"
                )
            lookback = total_dyn_floats // divisor
        
        self.lookback = lookback
        self.dyn_feat_dim = dyn_feat_dim
        self.static_dim = static_dim
        
        self.X_dyn = np.memmap(x_dyn_path, dtype=np.float32, mode='r', shape=(self.length, lookback, dyn_feat_dim))
        self.X_static = np.memmap(x_static_path, dtype=np.float32, mode='r', shape=(self.length, static_dim))
        
        self.Y_cls = np.memmap(y_cls_path, dtype=np.float32, mode='r', shape=(self.length,))
        self.Y_reg = np.memmap(y_reg_path, dtype=np.float32, mode='r', shape=(self.length,))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # 实时抽取单个张量并切换进内存 Float32 控制
        # NumPy array 到 Tensor
        x_d = torch.from_numpy(self.X_dyn[idx].copy()).float()
        # X_static 如今包含了分类序号与纯首单标价等连续数值，因此作为 float 统一流出给模型
        # (enhanced_model.py 内部会把分类特征动态视作为 long())
        x_s = torch.from_numpy(self.X_static[idx].copy()).float() 
        
        y_c = torch.tensor(self.Y_cls[idx], dtype=torch.float32).unsqueeze(-1)
        y_r = torch.tensor(self.Y_reg[idx], dtype=torch.float32).unsqueeze(-1)
        
        return x_d, x_s, y_c, y_r

def create_lazy_dataloaders(processed_dir, batch_size=256, num_workers=4, use_sampler=False,
                            dyn_feat_dim=7, lookback=None):
    """构建安全的训练和验证 DataLoader"""
    import os
    from torch.utils.data import WeightedRandomSampler
    
    train_dataset = ReplenishSparseDataset(
        x_dyn_path=os.path.join(processed_dir, 'X_train_dyn.bin'),
        x_static_path=os.path.join(processed_dir, 'X_train_static.bin'),
        y_cls_path=os.path.join(processed_dir, 'y_train_cls.bin'),
        y_reg_path=os.path.join(processed_dir, 'y_train_reg.bin'),
        dyn_feat_dim=dyn_feat_dim,
        lookback=lookback
    )
    
    val_dataset = ReplenishSparseDataset(
        x_dyn_path=os.path.join(processed_dir, 'X_val_dyn.bin'),
        x_static_path=os.path.join(processed_dir, 'X_val_static.bin'),
        y_cls_path=os.path.join(processed_dir, 'y_val_cls.bin'),
        y_reg_path=os.path.join(processed_dir, 'y_val_reg.bin'),
        dyn_feat_dim=dyn_feat_dim,
        lookback=lookback
    )
    
    # [V4.0] 平衡采样逻辑：目标 1:2 (POS 33%)，防止训练集高正样本率造成的过度积极
    sampler = None
    shuffle = True
    if use_sampler:
        print("[*] V4.0 正在计算平衡采样权重 (目标比例 POS:NEG = 1:2)...")
        y_train = train_dataset.Y_cls[:]
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        # 目标是 1:2 分布 (POS 33%)，防止正样本率 72% 带来的分布偏移
        target_pos_ratio = 0.33
        target_counts = np.array([1.0 - target_pos_ratio, target_pos_ratio])
        weight = target_counts / class_sample_count
        samples_weight = np.array([weight[int(t)] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        shuffle = False
        print(f"    - 采样完成: NEG权重={weight[0]:.6f}, POS权重={weight[1]:.6f}")

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
    }
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=shuffle,
        sampler=sampler,
        **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )
    
    return train_loader, val_loader
