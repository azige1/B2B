# DEVELOPER_QUICKSTART.md — 开发者快速上手

## 1. 快速环境准备

```bash
# 激活 Python 虚拟环境 (Windows)
.\venv\Scripts\activate

# 验证核心依赖
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

---

## 2. 三步启动 Phase 5 实验

### 第一步：生成 V5 数据（10维特征）
```bash
python -m src.features.build_features_v5_sku
```
*检查产出*: `data/processed_v5/` 目录下应有 8 个 `.bin` 文件。

### 第二步：运行挂机 Runner
```bash
python scripts/runners/phase5/run_phase5_experiments.py
```
*说明*: 脚本会按顺序执行 12 组实验。

### 第三步：查看结果聚合分析
```bash
# 自动寻找最新的验证结果明细进行 SKU 聚合分析
python evaluate_agg.py
```

---

## 3. 日常开发常用命令

### 3.1 单次模型训练 (手动调试)
```bash
# 运行 BiLSTM + V5 数据
EXP_VERSION=v5 EXP_MODEL_TYPE=bilstm python -m src.train.run_training_v2
```

### 3.2 运行完整评估表
```bash
# 评估最新生成的 V5 模型
EXP_VERSION=v5 python evaluate.py
```

### 3.3 仓库健康检查 (代码检查点)
```bash
python check_phase5_code.py
```

---

## 4. 核心文件导航

- **改架构**: `src/models/enhanced_model.py`
- **改 Loss**: `src/models/loss.py`
- **改训练细节**: `src/train/trainer.py`
- **改特征定义**: `src/features/build_features_v5_sku.py`

---

## 5. 开发者军规 (Rule of Thumb)

1. **不要硬编码维度**: 始终使用 `meta.get('dyn_feat_dim', 7)`。
2. **遵守 DummyLE 兼容**: 加载 encoder 前申明 `DummyLE` 类。
3. **清理 `_orig_mod.`**: 加载权重时替换 key 字符串。
4. **日志查看**: 训练实时轨迹在 `reports/history_{EXP_ID}.csv`。
5. **显存溢出**: 如果 OOM，在 `model_config.yaml` 调低 `batch_size`。

---

## Confidence Notes

| 内容 | 确认度 |
|------|:------:|
| 三步走实验流程 | ✅ 已确认 |
| 环境变量注入方式 | ✅ 已确认 |
| 生产推断与研发评估的区别 | ✅ 已确认 |
| 核心文件位置 | ✅ 已确认 |
