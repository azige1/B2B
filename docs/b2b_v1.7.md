
# Antigravity 项目配置: B2B 智能补货系统 (v1.7 精准预测升级版)

> **系统指令**: 输出用中文，代码备注使用中文，执行过程也用中文输出。
> **版本说明**: v1.7 在 v1.6 实战交付版的基础上，核心目标从"能跑通"升级为"**预测更准**"。明确了唯一的业务交付形态：**站在今天（T），预测接下来 30 天的各门店补货需求量**。

---

## 0. 核心业务目标的最终定义（唯一真相）

```
给定条件: 今天是 T 日
模型输入: 每个 (买手, SKU) 组合在 [T-120, T] 区间的全量历史行为
模型输出: 该 (买手, SKU) 组合在 [T+1, T+30] 区间的预计补货总量

交付文件 1 (系统底表): reports/daily_orders_{T}.csv (粒度: 门店明细)
交付文件 2 (甲方最终需求): reports/daily_sku_replenish_{T}.csv (粒度: SKU 全国总采)
交付字段 (大盘表): prediction_anchor_date | sku_id | national_total_qty_30d
过滤规则: 仅保留 ai_budget_qty_30d > 0 的有效补货指令行
```

这是系统**唯一的核心产出**。验证集回测（`generate_strategy_report.py`）仅用于衡量模型性能，不是业务交付物。

---

## 1. 当前模型 v1.6 的已知瓶颈（审查结论）

| 瓶颈类型 | 表现 | 根本原因 |
|---------|------|---------|
| **爆款低估** | 真实大爆 1.5万件，AI 只敢给 4500 件 | 过去 90 天没有历史先兆，时序无法预见突变 |
| **换季高估** | AI 给了 2万件，真实只卖 6500 件 | 时序斜率上行但换季信号未建模 |
| **全量一样的输出** | 某些推断返回同一个值 | `clamp(8.5)` 封顶触碰，说明对这批 SKU 模型极度不确定 |
| **数值特征量纲偏移** | 推断输出极端的 130203 件（触击封顶） | `qty_first_order`, `price_tag` 训练张量已做 log1p，但推断脚本喂入了数千元原始值导致量纲爆炸 |

---

## 2. 项目核心定义 (Project Overview)

* **项目背景**: 服装供应链 B2B 补货预测。
* **预测目标**: 未来 **30 天** 的各买手门店补货总量。
* **回看窗口**: 过去 **90 天**（向前多取 30 天缓冲，共 120 天）。
* **预测粒度**: `(买手, SKU)` 维度预测，SKU 聚合后用于评估和库存管理。

---

## 3. 数据工程铁律 (v1.7 更新)

### 3.1 源数据与采样（不变）
* 极度稀疏零膨胀分布，严禁全量笛卡尔积。
* 正样本全提取，训练负样本 `neg_step=40`，验证 `neg_step=1`。
* 120 天缓冲池，预测目标 `[T+1, T+30]` 严格零穿越。
* 张量以 `.bin` 格式直写 SSD，读取使用 `np.memmap`。

### 3.2 ★ 数值特征量纲修正（v1.7 核心修复）

v1.6 的问题症结在于：**训练张量在构建后（mmap 阶段）实际上被静默执行了 log1p 压缩，但推断接口（如 `generate_daily_inference.py`）未同步对这些连续大数值进行 log1p 变换**，导致输入数据的量纲极大，推测致使模型输出爆表并触击 `clamp(8.5)` 天花板（例如全表预测 13 万件）。v1.7 已同步修正了**所有推断脚本**：

```python
# generate_daily_inference.py 修正（推断时）
for col in static_num_cols:
    val = getattr(row, col)
    val = float(val) if pd.notnull(val) else 0.0
    arr.append(np.log1p(max(0.0, val)))  # ★ 修正: 同步对齐
```

> ⚠️ **重要**: 由于训练集的 `.bin` 张量本身已经是健康的 log1p 分布，因此**无须重新执行耗时的特征工程**，也**无须重新训练模型**。推断脚本加上上述代码即可完美验证。

### 3.3 特征维度（v1.7 不变，沿用 v1.6 的 7+14 维）

```python
# 动态特征（7维，均经过 log1p）
dyn_cols = [
    'qty_replenish', 'roll_repl_7', 'roll_repl_30',
    'qty_debt', 'qty_shipped', 'roll_ship_7', 'qty_inbound'
]

# 静态特征顺序（14维，所有脚本必须保持一致）
static_order_keys = [
    'buyer_id', 'sku_id', 'style_id', 'product_name', 'category',
    'sub_category', 'season', 'series', 'band', 'size_id', 'color_id',
    'month',           # 分类 Embedding（索引 0~11）
    'qty_first_order', # 数值 Dense（索引 12，传入时做 log1p）
    'price_tag'        # 数值 Dense（索引 13，传入时做 log1p）
]
```

---

## 4. 模型架构规范 (v1.7 不变，沿用 v1.6)

`EnhancedTwoTowerLSTM`，位于 `src/models/enhanced_model.py`：

```
左塔 (LSTM 2层, Hidden=128) → dyn_fc(64)
                                        ↘
                                    shared_fc(128) → head_cls / head_reg
                                        ↗
右塔 (Embedding × 12 + BN Dense × 2) → static_fc(64)
```

Loss 函数：`TwoStageMaskedLoss`（FocalLoss + 掩码 HuberLoss），`pos_weight=4.0`。

---

## 5. 推断策略（四层防护，v1.7 不变）

```python
# 层1: 死区掩码
dead_mask = (x_dyn.abs().sum(dim=(1, 2)) < 1e-4)
prob[dead_mask] = 0.0

# 层2: 数值封顶（单买手单SKU 30天约4000件上限）
preds = torch.clamp(preds, max=8.5)

# 层3: 概率阈值 + 保守系数
pred_qty = torch.expm1(preds) * (prob > 0.20).float() * 0.8

# 层4: inf/nan 安全退化
pred_np = np.nan_to_num(pred_qty.cpu().numpy().flatten(), nan=0.0, posinf=0.0, neginf=0.0)
```

---

## 6. 核心交付脚本：`generate_daily_inference.py`

这是 v1.7 唯一需要在生产环境每日运行的脚本：

```
步骤1: 从 data/gold/wide_table_sku.csv 加载全量历史
步骤2: 自动识别最新日期 T = df['date'].max()
步骤3: 截取 [T-119, T] 共 120 天历史切片
步骤4: 对每个 (buyer_id, sku_id) 组合:
        - 用 LabelEncoder 编码分类特征
        - 构建 (90, 7) 动态张量（log1p）
        - 构建 (14,) 静态张量（分类编码 + log1p 数值）
步骤5: 批量推断（batch_size=4096）
步骤6: 应用四层防护策略解码为物理件数
步骤7: 过滤 ai_budget_qty_30d = 0 的无效行
步骤8: 打印数值健康摘要（中位数/90分位/最大值）
步骤9: 输出 reports/daily_orders_{T:%Y%m%d}.csv
```

**日常使用**: 每天用最新生产数据覆盖 `wide_table_sku.csv`，执行 `python generate_daily_inference.py`，即可获得最新预算排产指令。

---

## 7. 评估与验收标准（v1.7）

| 指标 | v1.6 现状 | v1.7 目标 |
|------|---------|---------|
| MAE（全量） | 0.08 件/样本 | ≤ 0.06 件/样本 |
| Ratio（大盘） | 2.14 | **1.20 ~ 1.50** |
| SKU 聚合 MAE | 256 件/款 | ≤ 150 件/款 |
| SKU 聚合 RMSE | 752 件/款 | ≤ 500 件/款 |

---

## 8. v1.7 升级路线图（精简版）

```
阶段一 ★ 立即执行：修复量纲问题（1天工作量）
  → build_features_final.py: qty_first_order, price_tag 改为 log1p 存入
  → generate_daily_inference.py: 推断时同步 log1p
  → 重新跑特征工程 + 重新训练模型
  → 效果预期: BN 层输入分布正常化，Ratio 从 2.14 向 1.5 靠近

阶段二 ✅ 已完成：打通单日推断交付
  → generate_daily_inference.py: 自动识别 T，截取 120 天，输出 CSV
  → 四层防护策略对齐（clamp + nan_to_num + 统计摘要）
  → 每日运行: python generate_daily_inference.py
```

> 阶段3+（注意力机制、新特征、系数微调）列入未来迭代，当前版本不做。

---

## 9. 技术规范（不变，沿用 v1.6）

* **路径管理**: `os.path.join(PROJECT_ROOT, ...)` 相对路径，严禁硬编码。
* **内存安全**: `np.memmap(..., mode='r')`，禁止全量加载 `.bin` 文件。
* **权重加载**: `torch.load(path, map_location=device, weights_only=False)` + 去除 `_orig_mod.` 前缀。
* **日志**: 所有循环使用 `tqdm`；每 Epoch 打印 `Train_Loss / Val_Loss / Val_MAE / Ratio`。
* **DummyLE**: 所有调用 `pickle.load(label_encoders.pkl)` 的脚本必须先 `from run_training import DummyLE`。

---

## 10. 项目目录（v1.7 不改变目录结构，沿用 v1.6）

```
B2B_Replenishment_System/
├── run_training.py
├── evaluate_dashboard.py
├── generate_daily_inference.py     ← 每日派单主脚本（生产核心）
├── generate_strategy_report.py     ← 回测评估（研发工具）
├── analyze_csv_agg.py              ← SKU 聚合分析
├── config/model_config.yaml        ← 唯一超参来源（output_dir: ./models）
├── data/gold/wide_table_sku.csv    ← 每日更新此文件 = 系统自动更新预测
├── data/processed_fast/*.bin       ← 训练张量（.bin 格式）
├── data/artifacts/label_encoders.pkl
├── models/best_enhanced_model.pth  ← 每次重训后自动覆盖
└── reports/daily_orders_{DATE}.csv ← 最终业务派单文件
```
