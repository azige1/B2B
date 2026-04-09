# MODEL_ARCHITECTURE.md — 模型架构文档

## 1. 模型主入口

**文件**: `src/models/enhanced_model.py`（349行，15.7KB）

所有模型共享同一入口文件，通过 `MODEL_REGISTRY` 路由：

```python
# src/train/run_training_v2.py:L44-49
MODEL_REGISTRY = {
    'lstm':    EnhancedTwoTowerLSTM,    # 主力模型
    'gru':     TwoTowerGRU,              # 替代架构
    'bilstm':  TwoTowerBiLSTM,           # 双向变体
    'attn':    TwoTowerLSTMWithAttn,     # 注意力变体
}
```

选择方式: `os.environ.get('EXP_MODEL_TYPE', 'lstm')`

---

## 2. 核心模型类: EnhancedTwoTowerLSTM

### 整体架构

```
输入:
  x_dyn:    (Batch, 90, D)        D=7(V3) 或 10(V5)
  x_static: (Batch, 13)           10个分类ID + month + 2个数值
  static_order_keys: list[str]    列名列表，用于路由 Embedding

   ┌─────────────────────┐    ┌──────────────────────┐
   │   左塔 (时序/动态)    │    │   右塔 (静态/画像)     │
   │                     │    │                      │
   │  LSTM(D → H, L层)   │    │  Embedding×11 (各 emb_dim)│
   │  取最后时间步          │    │  + BN + Dense(num→8)  │
   │  Linear(H → 64)     │    │  cat → Linear(total→64)│
   │  + BN + ReLU + Drop  │    │  + BN + ReLU + Drop   │
   └─────────┬───────────┘    └──────────┬───────────┘
             │                           │
             └─────── cat(64+64=128) ────┘
                        │
                  shared_fc(128→128)
                  + BN + ReLU + Drop
                        │
                 ┌──────┴──────┐
                 │             │
            head_cls(128→1)  head_reg(128→1)
            (分类 logits)    (回归 log域)
```

### 构造函数参数

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `dyn_feat_dim` | 7 | 动态特征维度 (V3=7, V5=10) |
| `lstm_hidden` | 128 | LSTM 隐层大小 (实验用 256) |
| `lstm_layers` | 2 | LSTM 层数 (Phase2最优=3) |
| `static_vocab_sizes` | dict | 各分类特征的词表大小 |
| `static_emb_dim` | 16 | Embedding 维度基准 (被自适应公式覆盖) |
| `num_numeric_feats` | 2 | 数值特征数 (qty_first_order, price_tag) |
| `dropout` | 0.2 | Dropout 率 |

### 自适应 Embedding 维度

```python
def get_emb_dim(vocab_size):
    return max(4, min(128, int(vocab_size ** 0.5) * 4))
# 例: size_id(50) → 28维, sku_id(15000) → 128维
```

---

## 3. 模型变体

### 3.1 TwoTowerGRU (`gru`)

- **继承**: `EnhancedTwoTowerLSTM`
- **差异**: 用 `nn.GRU` 替换 `nn.LSTM`，参数少 ~25%
- **forward**: 重写了左塔（GRU 不返回 cell state），右塔完全复用父类代码
- **实验结论**: Phase 1 e11 测试，F1 不如 LSTM

### 3.2 TwoTowerBiLSTM (`bilstm`)

- **继承**: `nn.Module`（独立实现，不继承父类）
- **差异**: `nn.LSTM(bidirectional=True)`，输出维度 = `hidden_size * 2`
- **关键**: `dyn_fc` 输入维度 = `bi_hidden = lstm_hidden * 2`（不是 `lstm_hidden`）
- **实验结论**: Phase 2 L2 测试 F1=0.479，Phase 5 首次测试 L3 版本
- **⚠️ 独立实现**: 右塔代码完全复制自父类（代码重复，见 RISK_AND_TECH_DEBT.md）

### 3.3 TwoTowerLSTMWithAttn (`attn`)

- **继承**: `EnhancedTwoTowerLSTM`
- **差异**: 加入时序注意力层 `nn.Linear(lstm_hidden, 1)`
- **forward 改动**:
  ```python
  # 替代简单的 lstm_out[:, -1, :]
  scores = self.attn_score(lstm_out)        # (B, T, 1)
  weights = torch.softmax(scores, dim=1)    # 注意力权重
  seq_repr = (lstm_out * weights).sum(dim=1) # 加权求和
  ```
- **实验结论**: Phase 1 e13 测试，F1 与基线相当

---

## 4. 损失函数: TwoStageMaskedLoss

**文件**: `src/models/loss.py`（139行）

### 组成

```
TwoStageMaskedLoss
  ├── FocalLossWithLogits(gamma)     ← 分类：对难样本加权
  ├── SoftF1Loss()                   ← 分类：直接优化 F1 指标
  └── HuberLoss(delta)               ← 回归：对大单异常值鲁棒

总 Loss = cls_weight × (focal + soft_f1_coeff × soft_f1):
        + reg_weight × (masked_huber_pos + fp_penalty × huber_neg)
```

### 关键参数（全部通过环境变量注入）

| 参数 | 默认值 | 环境变量 |
|------|:------:|----------|
| pos_weight | 5.85 | EXP_POS_WEIGHT |
| fp_penalty | 0.15 | EXP_FP_PENALTY |
| gamma | 2.0 | EXP_GAMMA |
| reg_weight | 0.5 | EXP_REG_WEIGHT |
| soft_f1_coeff | 0.5 | EXP_SOFT_F1 |
| huber_delta | 1.5 | EXP_HUBER |
| label_smoothing | 0.0 | EXP_LABEL_SMOOTH |

### 遮罩机制

回归 Loss 仅在 `targets_cls > 0.5` 的正样本上计算（避免零膨胀样本拉低梯度）。负样本的回归 Loss 被 `fp_penalty=0.15` 抑制（惩罚对不需要补货的 SKU 的虚报）。

### Label Smoothing (Phase 5 新增)

```python
if self.label_smoothing > 0:
    targets_cls = targets_cls * (1 - α) + 0.5 * α
    # label=1 → 1-α/2, label=0 → α/2
```

---

## 5. 权重加载逻辑

```python
# 标准流程 (evaluate.py:L105-108)
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
```

**必须 `weights_only=False`**: checkpoint 由 `torch.save(model.state_dict(), ...)` 保存，标准格式。但 `weights_only=True` 在某些 PyTorch 版本下行为不一致。

**必须清理 `_orig_mod.`**: `torch.compile()` 编译后的模型，state_dict 所有 key 会被加上此前缀。

---

## 6. 经常改动区 vs 高风险区

| 区域 | 改动频率 | 修改风险 |
|------|:--------:|:--------:|
| `TwoStageMaskedLoss` 参数 | 高 | 中（参数调整安全，结构不动） |
| 模型超参 (hidden/layers/dropout) | 高 | 低（通过环境变量，不改代码） |
| `trainer.py` 验证逻辑 | 中 | 🔴 高（影响所有实验的指标计算） |
| `forward()` 路径 | 低 | 🔴 极高（破坏 checkpoint 兼容性） |
| 新增模型变体 | 低 | 中（在 MODEL_REGISTRY 注册即可） |

---

## 7. 理解模型最少要读的文件

1. `src/models/enhanced_model.py:L11-159` — `EnhancedTwoTowerLSTM` 类
2. `src/models/loss.py:L74-138` — `TwoStageMaskedLoss` 类
3. `src/train/trainer.py:L11-47` — `train_one_epoch` (前向+反向)
4. `src/train/trainer.py:L50-134` — `validate` (推断解码逻辑)

---

## Confidence Notes

| 内容 | 确认度 |
|------|:------:|
| 4个模型变体和继承关系 | ✅ 已确认（逐行审查） |
| BiLSTM 使用 hidden*2 作为 dyn_fc 输入 | ✅ 已确认 (L245) |
| BiLSTM 独立实现（不继承父类） | ✅ 已确认 |
| Loss 组成和参数注入 | ✅ 已确认 |
| Label Smoothing 实现 | ✅ 已确认 (loss.py:L104-105) |
| 遮罩机制 mask_pos 使用 > 0.5 阈值 | ✅ 已确认 (loss.py:L115) |
| torch.compile 产生 _orig_mod. 前缀 | ✅ 已确认 |
| Attention 变体的实验表现 | 高概率推断（基于历史实验总结） |
