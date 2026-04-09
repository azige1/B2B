# TRAINING_FLOW.md — 训练主链路文档

## 1. 训练启动命令

```bash
# 当前主入口（V2 版本，支持环境变量注入）
python -m src.train.run_training_v2

# 挂机实验模式（Phase 5 Runner 注入环境变量后调用同一入口）
python scripts/runners/phase5/run_phase5_experiments.py
# Runner 内部调用: subprocess.Popen([sys.executable, "src/train/run_training_v2.py"], env=env)
```

> ⚠️ `src/train/run_training.py` (无 _v2) 是 **V1 历史入口**，使用 `data/artifacts/` 路径和 `buyer_id` 维度。V2+ 不再使用。

---

## 2. 完整调用链

```
scripts/runners/phase5/run_phase5_experiments.py (Phase 5 Runner)
  └─ subprocess.Popen → src/train/run_training_v2.py::main()
      │
      ├─ [1] 配置加载
      │   └─ yaml.safe_load(config/model_config.yaml)
      │
      ├─ [2] 数据版本路由 (EXP_VERSION 环境变量)
      │   ├─ v3 → data/processed_v3/ + data/artifacts_v3/meta_v2.json
      │   └─ v5 → data/processed_v5/ + data/artifacts_v5/meta_v5.json
      │
      ├─ [3] 环境变量热注入（覆盖 config 默认值）
      │   └─ EXP_MODEL_TYPE, EXP_HIDDEN, EXP_LAYERS, EXP_DROPOUT, EXP_BATCH, EXP_LR, EXP_EPOCHS, EXP_PATIENCE
      │
      ├─ [4] DataLoader 构建
      │   └─ dataset.py::create_lazy_dataloaders(processed_dir, batch_size, ...)
      │       ├─ ReplenishSparseDataset(X_train_dyn.bin, X_train_static.bin, y_train_cls.bin, y_train_reg.bin)
      │       └─ ReplenishSparseDataset(X_val_dyn.bin, X_val_static.bin, y_val_cls.bin, y_val_reg.bin)
      │
      ├─ [5] 模型构建
      │   ├─ MODEL_REGISTRY 路由: lstm→EnhancedTwoTowerLSTM, bilstm→TwoTowerBiLSTM, gru→TwoTowerGRU, attn→TwoTowerLSTMWithAttn
      │   ├─ vocab_sizes 从 label_encoders_v{n}.pkl 读取
      │   ├─ dyn_feat_dim 从 meta.json 读取 (7 或 10)
      │   └─ torch.compile(model)  ← 加 _orig_mod. 前缀
      │
      └─ [6] 启动训练
          └─ trainer.py::fit_model(model, train_loader, val_loader, ...)
              │
              ├─ AMP GradScaler 创建
              ├─ LR Warmup 配置 (3 epochs, 下限保护 max(base*0.1, 1e-5))
              ├─ Loss 实例化: TwoStageMaskedLoss(pos_weight, fp_penalty, label_smoothing, ...)
              ├─ Optimizer 选择: env_weight_decay > 0 → AdamW, else → Adam
              ├─ Scheduler: ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-6)
              │
              └─ for epoch in range(1, config_epochs+1):
                  │
                  ├─ Warmup LR 调整 (epoch ≤ 3)
                  │
                  ├─ train_one_epoch(model, train_loader, optimizer, criterion, device, static_order_keys, amp_scaler)
                  │   ├─ model.train()
                  │   ├─ for x_dyn, x_static, y_cls, y_reg in dataloader:
                  │   │   ├─ torch.amp.autocast('cuda')
                  │   │   ├─ logits, preds = model(x_dyn, x_static, static_order_keys)  ← forward
                  │   │   ├─ loss, loss_cls, loss_reg = criterion(logits, preds, y_cls, y_reg)
                  │   │   ├─ amp_scaler.scale(loss).backward()
                  │   │   ├─ amp_scaler.unscale_(optimizer)
                  │   │   ├─ clip_grad_norm_(model.parameters(), max_norm=1.0)
                  │   │   ├─ amp_scaler.step(optimizer)
                  │   │   └─ amp_scaler.update()
                  │   └─ return avg_loss, avg_cls, avg_reg
                  │
                  ├─ validate(model, val_loader, criterion, device, static_order_keys)
                  │   ├─ model.eval(), torch.no_grad()
                  │   ├─ 收集全量 pred_qty/true_qty (expm1 解码)
                  │   ├─ 计算 MAE, 中位 Ratio, WMAPE
                  │   ├─ F1/Precision/Recall (阈值搜索 0.30~0.78)
                  │   └─ return (v_loss, v_cls, v_reg, mae, ratio, ..., wmape)
                  │
                  ├─ History CSV 写入 → reports/history_{EXP_ID}.csv
                  │
                  ├─ Best checkpoint 保存 (val_loss 最低)
                  │   └─ torch.save(model.state_dict(), best_enhanced_model.pth)
                  │
                  ├─ Early Stopping (patience_counter >= patience_limit)
                  │
                  └─ ReduceLROnPlateau.step(v_loss)
```

---

## 3. 最小可运行训练路径

```bash
# 前置条件：data/gold/wide_table_sku.csv 已存在
# Step 1: 特征工程 (V3)
python -m src.features.build_features_v2_sku

# Step 2: 训练 (使用 config 默认值)
python -m src.train.run_training_v2

# 预期产出:
# - models_v2/best_enhanced_model.pth
# - models_v2/last_enhanced_model.pth
# - reports/history_default_exp.csv
```

---

## 4. 训练时最容易改错的 10 个点

| # | 风险点 | 文件:行号 | 说明 |
|:-:|--------|----------|------|
| 1 | **dataset.py dyn_feat_dim=7 硬编码** | dataset.py:L24 | V5 的 10 维数据加载时此值不正确，依赖字节大小自动推导 |
| 2 | **evaluate.py 解码参数 ≠ trainer.py** | evaluate.py:L153 vs trainer.py:L82 | eval 用 prob>0.15 ×1.2（激进），train 用 prob>0.45 ×1.0（保守） |
| 3 | **model_config.yaml 的 num_layers=2 与实验 L3=3 冲突** | config:L56 | config 写 2，但 Phase2+ 最优是 L3。环境变量 EXP_LAYERS 覆盖 |
| 4 | **DummyLE 必须在 pickle.load 之前定义** | 多个脚本 | 否则 unpickle 时报 AttributeError |
| 5 | **_orig_mod. 前缀在 load_state_dict 时必须清理** | evaluate.py:L106 | torch.compile 产物，不清理会 key mismatch |
| 6 | **train_lr 和 config LR 的覆盖链路** | run_training_v2.py:L187-190 | EXP_LR 环境变量 > config 值，但 warmup 用 base_lr |
| 7 | **warmup 跳过验证** | trainer.py:L249 | epoch < 3 时跳过验证，不会触发 early stopping 和 checkpoint 保存 |
| 8 | **V3 meta 文件名是 meta_v2.json（不是 meta_v3.json）** | run_training_v2.py:L92 | 历史命名，V3 数据目录叫 processed_v3 但 meta 文件叫 meta_v2 |
| 9 | **Focal Loss 的 label_smoothing 作用在 target 上** | loss.py:L104-105 | 先平滑再算 Focal，但 mask_pos 仍使用 > 0.5 阈值，平滑后标签 ≠ 0/1 |
| 10 | **evaluate.py 硬编码 dyn_feat_dim=7** | evaluate.py:L96 | V5 实验评估时需要修改此值，否则模型推断结果错误 |

---

## 5. Forward 路径详解

```python
# EnhancedTwoTowerLSTM.forward(x_dyn, x_static, static_order_keys)

# 左塔 (时序)
lstm_out, (hn, cn) = self.lstm(x_dyn)        # (B, 90, H) → LSTM 编码
seq_repr = lstm_out[:, -1, :]                 # (B, H) 取最后时间步
dyn_vector = self.dyn_fc(seq_repr)            # (B, 64) Linear+BN+ReLU+Drop

# 右塔 (静态)
for i, col_name in enumerate(static_order_keys):
    if col_name in self.embeddings:
        emb = self.embeddings[col_name](x_static[:, i].long())  # Embedding lookup
    else:
        num_list.append(x_static[:, i].float().unsqueeze(1))    # 数值特征收集
num_repr = self.num_bn(num_list_cat)          # BN 归一化
num_out = self.num_dense(num_repr)            # (B, 8) Dense
static_repr = cat(emb_list + num_out)         # 拼接所有 embedding + 数值
static_vector = self.static_fc(static_repr)   # (B, 64)

# 融合
combined = cat([dyn_vector, static_vector])   # (B, 128)
shared = self.shared_fc(combined)             # (B, 128) 共享层
logits = self.head_cls(shared)                # (B, 1) 分类 logits
preds = self.head_reg(shared)                 # (B, 1) 回归值 (log 域)
return logits, preds
```

---

## Confidence Notes

| 内容 | 确认度 |
|------|:------:|
| run_training_v2.py → fit_model 调用链 | ✅ 已确认（逐行追踪） |
| 环境变量注入到 fit_model 参数的映射 | ✅ 已确认 |
| AMP + GradScaler 持久化 across epochs | ✅ 已确认 (trainer.py:L164) |
| Warmup 前 3 轮跳过验证 | ✅ 已确认 (trainer.py:L249) |
| Early Stopping 由 val_loss 驱动 | ✅ 已确认 (trainer.py:L334) |
| dataset.py dyn_feat_dim=7 硬编码 | ✅ 已确认（技术债） |
| F1 阈值搜索范围 0.30~0.78 | ✅ 已确认 (trainer.py:L306) |
| evaluate.py 使用不同的解码参数 | ✅ 已确认 (prob>0.15, ×1.2) |
