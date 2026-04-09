# AGENTS.md — Codex 执行手册

> 本文件面向后续 AI coding agent（如 Codex），帮助其快速理解本仓库并安全执行任务。

---

## 1. 仓库概览

**一句话**: 服装 B2B 供应链补货预测系统，用双塔 LSTM 预测每个 SKU 未来 30 天的补货总量。

- **输入**: `data/gold/wide_table_sku.csv`（SKU×日期的宽表）
- **输出**: `reports/daily_orders_{DATE}.csv`（30天补货指令）
- **核心模型**: `EnhancedTwoTowerLSTM`（左塔时序LSTM + 右塔Embedding静态特征）
- **训练粒度**: SKU 级（全国买手聚合），日频90天回看，30天预测窗口
- **当前阶段**: Phase 5 实验期（12组实验矩阵，V3/V5 双特征版本并行）

---

## 2. 核心链路文件（按优先级排序）

| 优先级 | 文件 | 作用 | 修改风险 |
|:------:|------|------|:-------:|
| ★★★ | `src/train/trainer.py` | 训练主循环、Loss实例化、优化器、验证、checkpoint | 🔴 极高 |
| ★★★ | `src/models/enhanced_model.py` | 模型架构定义（4个变体类） | 🔴 极高 |
| ★★★ | `src/models/loss.py` | TwoStageMaskedLoss (Focal + SoftF1 + Huber) | 🔴 极高 |
| ★★★ | `src/train/dataset.py` | memmap Dataset + DataLoader 工厂 | 🔴 极高 |
| ★★☆ | `src/train/run_training_v2.py` | 训练入口脚本（V2主入口，环境变量注入） | 🟡 中等 |
| ★★☆ | `src/features/build_features_v5_sku.py` | V5 特征工程（10维，含期货信号） | 🟡 中等 |
| ★★☆ | `src/features/build_features_v2_sku.py` | V3 特征工程（7维原始版） | 🟡 中等 |
| ★★☆ | `evaluate.py` | 6维度评估脚本 | 🟡 中等 |
| ★★☆ | `evaluate_agg.py` | SKU级聚合Ratio分析 + 冷启动诊断 | 🟡 中等 |
| ★☆☆ | `scripts/runners/phase5/run_phase5_experiments.py` | 12组挂机实验Runner | 🟢 低 |
| ★☆☆ | `config/model_config.yaml` | 超参数配置（唯一真相源） | 🟡 中等 |

---

## 3. 修改区域规则

### ✅ 允许修改的区域
- `scripts/runners/phase5/run_phase5_experiments.py` — 实验定义、Runner 逻辑
- `evaluate_agg.py` — 评估分析逻辑
- `src/evaluate/*.py` — 评估脚本
- `scripts/`, `diagnostics/` — 工具脚本
- `config/model_config.yaml` — 超参调整（需说明理由）
- 新建分析脚本到项目根目录

### ⚠️ 需谨慎修改的区域
- `src/train/run_training_v2.py` — 改动需确保环境变量注入链路不断
- `src/features/build_features_*.py` — 改动需确保 .bin 张量格式不变
- `evaluate.py` — 共享模型加载逻辑，改动影响评估结果

### 🚫 不应轻易修改的区域
- `src/train/trainer.py` — 训练核心循环，任何改动都影响所有实验
- `src/models/enhanced_model.py` — 模型结构，改动破坏目前所有checkpoint
- `src/models/loss.py` — 损失函数，影响训练行为
- `src/train/dataset.py` — 数据加载，**内含 dyn_feat_dim=7 硬编码（已知技术债）**
- `data/processed_*/` — 训练张量，不要手动修改
- `models_v2/*.pth` — 已有 23 个 checkpoint，不要删除

---

## 4. 隐式约定（必须遵守）

### 4.1 DummyLE 兼容类
```python
class DummyLE:
    classes_ = np.arange(13)
```
**任何调用 `pickle.load(label_encoders.pkl)` 的脚本都必须先定义 `DummyLE`。**
原因：encoder 中的 `month` 键使用了 DummyLE 实例而非 LabelEncoder。如果不定义，pickle.load 会报 `AttributeError`。

- **已确认**: `run_training.py:L16`, `run_training_v2.py:L52`, `evaluate.py:L67`, `build_features_v2_sku.py:L50`

### 4.2 _orig_mod. 前缀清理
```python
state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
```
**所有加载 checkpoint 的代码都必须做此清理。**
原因：`torch.compile()` 会给所有权重 key 加上 `_orig_mod.` 前缀。

- **已确认**: `evaluate.py:L106`, `generate_daily_inference.py:L148`

### 4.3 数据版本隔离
| 版本 | 张量目录 | 编码器目录 | 动态维度 | Meta 文件 |
|:----:|---------|-----------|:-------:|----------|
| V3 | `data/processed_v3/` | `data/artifacts_v3/` | 7 | `meta_v2.json` |
| V5 | `data/processed_v5/` | `data/artifacts_v5/` | 10 | `meta_v5.json` |

**切换版本通过环境变量 `EXP_VERSION=v3|v5`**，在 `run_training_v2.py:L81-95` 实现路由。

### 4.4 dataset.py 的 dyn_feat_dim=7 硬编码
`dataset.py:L24` 硬编码 `dyn_feat_dim = 7`。V5 的 10 维数据加载时，这个值被文件字节大小自动推导覆盖：
```
total_dyn_floats // self.length // dyn_feat_dim → lookback
```
当 dyn_feat_dim 与实际不匹配时，lookback 计算结果错误但 reshape 后总字节数一致，**可能静默产生错误数据但不报错**。

**实际状态**：V5 的 `.bin` 文件 per-sample 大小 = 90×10×4 = 3600 bytes，除以 dyn_feat_dim=7 → lookback=128（错误！应该是90）。但由于 numpy memmap 的 shape 参数覆盖了自动推导，实际上 `lookback=total//length//7` 的结果不正确，但 memmap 的 reshape 会基于文件总大小做校验。

**⚠️ 这是已知技术债，V5 实验需验证此处不会出问题。**

### 4.5 数量空间与 log1p 变换
- **训练目标**: `y_reg = log1p(target_qty)`，回归头直接输出 log 域值
- **推断解码**: `pred_qty = expm1(preds)`
- **静态数值**: `qty_first_order` 和 `price_tag` 在特征工程中已做 `log1p`
- **动态特征**: V3/V5 均在构建时已做 `log1p` 再存入 `.bin`

### 4.6 环境变量注入体系
训练超参通过环境变量覆盖 `model_config.yaml` 默认值：

```
EXP_ID            — 实验标识符（影响日志文件名、checkpoint 备份名）
EXP_VERSION       — v3 | v5（切换数据路径）
EXP_MODEL_TYPE    — lstm | bilstm | gru | attn
EXP_HIDDEN        — LSTM hidden size
EXP_LAYERS        — LSTM layers
EXP_DROPOUT       — dropout rate
EXP_BATCH         — batch size
EXP_LR            — learning rate
EXP_EPOCHS        — max epochs [Phase5新增]
EXP_PATIENCE      — early stopping patience [Phase5新增]
EXP_POS_WEIGHT    — FocalLoss 正样本权重
EXP_FP_PENALTY    — 误报惩罚系数
EXP_GAMMA         — Focal gamma
EXP_REG_WEIGHT    — 回归损失权重
EXP_SOFT_F1       — SoftF1 系数
EXP_HUBER         — Huber delta
EXP_LABEL_SMOOTH  — Label Smoothing [Phase5新增]
EXP_WEIGHT_DECAY  — AdamW weight decay [Phase5新增]
```

---

## 5. 任务执行检查流程

### 开始前
1. 阅读本文件 (`AGENTS.md`)
2. 确认任务涉及哪些核心链路文件
3. 查看 `TRAINING_FLOW.md` 了解调用链
4. 检查 `RISK_AND_TECH_DEBT.md` 中是否有相关风险

### 执行时
1. **最小 diff 原则** — 只改需要改的代码
2. **不修改接口签名** — 函数参数、返回值格式不要变
3. **保持版本隔离** — V3/V5 数据不要混用
4. **不动 checkpoint 路径** — `best_enhanced_model.pth` 是约定名
5. **不动 .bin 张量格式** — (N, 90, D) float32 memmap 格式固定
6. **改 trainer/model/loss 时说明影响范围** — 这三个文件牵一发动全身

### 完成后
1. 说明修改了哪些文件
2. 列出运行命令和结果摘要
3. 标注不确定性（如果有）
4. 运行 `python -m py_compile <file>` 验证语法

---

## 6. 深度学习特别规则

1. **不要随意改 checkpoint/data/output 路径** — 23 个历史 checkpoint 路径已固化
2. **不要无依据调整 config schema** — config 键名被多个脚本引用
3. **训练脚本 ≠ 评测脚本** — evaluate.py 有自己的解码参数（prob>0.15, ×1.2），与 trainer.py 的验证逻辑（prob>0.45, ×1.0）不同
4. **修改 trainer 时**: 影响所有 12 组实验
5. **修改 model 时**: 破坏所有已有 checkpoint 的 load_state_dict 兼容性
6. **修改 loss 时**: 改变模型训练行为，需重新训练验证
7. **局部修改 > 全仓库重构** — 始终优先最小改动

---

## 7. 常用命令速查

```bash
# 特征工程（V3/V5）
python -m src.features.build_features_v2_sku    # V3 (7维)
python -m src.features.build_features_v5_sku    # V5 (10维)

# 训练（单次）
python -m src.train.run_training_v2             # 使用 config 默认值
EXP_VERSION=v5 EXP_MODEL_TYPE=bilstm python -m src.train.run_training_v2  # 指定版本

# 挂机实验（12组）
python scripts/runners/phase5/run_phase5_experiments.py

# 评估
python evaluate.py                             # 6维度完整评估
python evaluate_agg.py --exp e54_bilstm_l3_v5  # SKU聚合分析

# 推断
python -m src.inference.generate_daily_inference  # 生产推断

# 语法检查
python -m py_compile src/train/trainer.py
```

---

## Confidence Notes

| 内容 | 确认度 |
|------|:------:|
| 训练主链 (run_training_v2 → trainer.fit_model → train_one_epoch/validate) | ✅ 已确认 |
| 模型架构 (4个变体，双塔结构) | ✅ 已确认 |
| Loss 组成 (Focal + SoftF1 + Huber) | ✅ 已确认 |
| 环境变量注入体系 | ✅ 已确认 |
| DummyLE 兼容要求 | ✅ 已确认 |
| _orig_mod. 清理要求 | ✅ 已确认 |
| dataset.py dyn_feat_dim=7 硬编码问题 | ✅ 已确认（技术债） |
| V3/V5 数据路径隔离 | ✅ 已确认 |
| evaluate.py vs trainer.py 解码参数不同 | ✅ 已确认 |
| generate_daily_inference.py 使用 V1 artifacts (data/artifacts/) | ✅ 已确认（与V2/V3/V5不同） |
