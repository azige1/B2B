# DATA_AND_CONFIG_FLOW.md — 数据流与配置流文档

## 1. 数据全生命周期

```
[Oracle 数据库 / CSV 原始文件]
  │   ↓  src/etl/extract_manager.py + src/etl/clean_data.py
  │
data/raw/                     ← 原始下载
  │   ↓  src/etl/clean_data.py (去重/填0/日期标准化)
  │
data/silver/                  ← 清洗后
  │   ↓  src/etl/build_wide_table.py (多表 JOIN)
  │
data/gold/wide_table_sku.csv  ← ★ 核心宽表（唯一数据入口）
  │   ↓  src/features/build_features_v2_sku.py  或  build_features_v5_sku.py
  │
data/processed_v3/            ← V3 张量 (7维, .bin memmap)
data/processed_v5/            ← V5 张量 (10维, .bin memmap)
  │   ↓  src/train/dataset.py::ReplenishSparseDataset (memmap 懒加载)
  │
DataLoader                    ← 训练/验证批次
  │   ↓  src/train/trainer.py::train_one_epoch / validate
  │
模型训练
```

---

## 2. 宽表结构 (`wide_table_sku.csv`)

每行 = 一个 `(sku_id, date)` 组合，关键字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| sku_id | str | SKU 唯一标识 |
| date | date | 日期 |
| qty_replenish | float | 当日补货量 |
| qty_future | float | 当日期货/预订量（V5 新增使用） |
| qty_debt | float | 欠货量（全 0，已废弃） |
| qty_shipped | float | 发货量（全 0，已废弃） |
| qty_inbound | float | 入库量（全 0，已废弃） |
| style_id, product_name, category, sub_category, season, series, band, size_id, color_id | str | 商品静态属性 |
| qty_first_order | float | 首单配货量 |
| price_tag | float | 吊牌价 |

**⚠️ 已确认**: qty_debt, qty_shipped, qty_inbound, roll_ship_7 等 4 个字段全表为 0，V5 已用期货特征替代。

---

## 3. 特征工程流程（以 V5 为例）

```python
# build_features_v5_sku.py 核心流程

1. 读取 wide_table_sku.csv
2. 聚合为 (sku_id, date) → (qty_replenish, qty_future)
3. LabelEncoder 编码 10 个分类特征 → 保存 label_encoders_v5.pkl
4. 构建致密日历矩阵 (n_sku × 365天 × 2列)
5. 计算 10 维时序特征池 (365天 × 10维)
   [0-2] 补货核心: qty, roll_7, roll_30  (log1p)
   [3-5] 期货信号: qty_future, roll_7, roll_30  (log1p)
   [6]   加速度: 7日均速 - 30日均速
   [7]   期货/补货比: roll_fut_7 / (roll_r7 + 1)
   [8]   波动率: std_7 / (mean_7 + 1)
   [9]   距最近补货天数: gap / 30
6. 暴力滑窗截取样本:
   - for sku in all_skus:
     - for day in range(LOOKBACK-1, end):
       - target = sum(qty_replenish[day+1:day+31])
       - window = ts_feat[sku, day-89:day+1]  → (90, 10)
       - static = [编码ID×10, month, log1p(qty_first_order), log1p(price_tag)]  → (13,)
       - y_cls = 1 if target > 0 else 0
       - y_reg = log1p(target)
       - 写入 .bin 文件
7. MinMaxScaler 归一化动态特征（在 .bin 上原地修改）
8. 保存 meta_v5.json + val_keys.csv
```

### 训练/验证分割

```
时间分割点: SPLIT_DATE = '2025-12-01'
  训练集: anchor_date < 2025-12-01 (1月~11月)
  验证集: anchor_date >= 2025-12-01 (12月)
  
负样本策略: NEG_STEP = 1 (全量保留，不降采样)
```

---

## 4. .bin 张量文件格式

所有张量以裸 float32 格式存储，通过 `np.memmap` 懒加载：

| 文件名 | Shape | 说明 |
|--------|-------|------|
| `X_train_dyn.bin` | (N_train, 90, D) | 动态特征，D=7(V3)或10(V5) |
| `X_train_static.bin` | (N_train, 13) | 静态特征 |
| `y_train_cls.bin` | (N_train,) | 分类标签，0.0 或 1.0 |
| `y_train_reg.bin` | (N_train,) | 回归标签，log1p(补货量) |
| `X_val_*.bin`, `y_val_*.bin` | 同上 | 验证集 |

**dataset.py 的形状推导逻辑**:
```python
self.length = os.path.getsize(y_cls_path) // 4        # 样本数
static_dim = os.path.getsize(x_static_path) // 4 // self.length  # 静态维度
dyn_feat_dim = 7  # ⚠️ 硬编码！V5时此值错误但被忽略
lookback = total_dyn_floats // self.length // dyn_feat_dim  # V5时=128(错误)
# ⚠️ 实际 memmap shape 可能不正确，需验证
```

---

## 5. 配置系统

### 5.1 配置文件层

```
config/
├── model_config.yaml    ← 模型超参 + 训练超参（唯一被代码读取的配置文件）
└── data_config.yaml     ← ETL 数据源配置（Oracle 连接、表结构定义）
                           仅被 src/etl/ 使用，不影响训练
```

### 5.2 model_config.yaml 关键字段

| 字段 | 默认值 | 影响范围 | 可被环境变量覆盖？ |
|------|:------:|---------|:--:|
| `model.hidden_size` | 256 | LSTM 隐层大小 | ✅ EXP_HIDDEN |
| `model.num_layers` | 2 | LSTM 层数 | ✅ EXP_LAYERS |
| `model.dropout` | 0.3 | Dropout 率 | ✅ EXP_DROPOUT |
| `train.batch_size` | 1536 | DataLoader batch | ✅ EXP_BATCH |
| `train.learning_rate` | 0.0002 | 初始学习率 | ✅ EXP_LR |
| `train.epochs` | 20 | 最大 epoch 数 | ✅ EXP_EPOCHS |
| `train.early_stopping_patience` | 5 | 早停耐心值 | ✅ EXP_PATIENCE |
| `train.val_every` | 1 | 每 N 轮验证 | ❌ |
| `paths.dataset_dir` | `data/processed_v3` | 数据路径 | ✅ EXP_VERSION 间接 |
| `paths.output_dir` | `models_v2` | 模型输出 | ✅ EXP_VERSION 间接 |

### 5.3 配置覆盖优先级

```
环境变量 (EXP_*) > model_config.yaml 默认值
```

**无继承/层叠机制**。没有 Hydra、argparse、dataclass 等复杂系统。就是 `yaml.safe_load` + `os.environ.get`。

### 5.4 容易造成 silent bug 的配置字段

| 字段 | 风险 |
|------|------|
| `model.num_layers=2` | config 里写 2，但 Phase2 最优实验用 L3 (=3)。实验依赖 EXP_LAYERS 覆盖。如果忘了传环境变量，会用错层数 |
| `train.batch_size=1536` | config 写 1536，但实验 Runner 注入 1024。不一致 |
| `train.learning_rate=0.0002` | config 默认 2e-4，但 Phase2 最优是 5e-5。由 EXP_LR 覆盖 |
| `data.dynamic_cols` 列表 | config 里列了 7 维动态列，但 V5 用 10 维。config 中此列表实际**未被代码读取**，仅作文档用 |

---

## 6. 数据流与配置流的汇合点

```
run_training_v2.py::main()
  │
  ├─ config ← yaml.safe_load(model_config.yaml)
  ├─ meta   ← json.load(meta_v{n}.json)   ← 由特征工程产出
  ├─ env    ← os.environ.get(EXP_*)       ← 由实验 Runner 注入
  │
  ├─ 最终参数 = env > config
  │
  ├─ DataLoader ← dataset.py(processed_dir 由 EXP_VERSION 决定)
  ├─ Model      ← enhanced_model.py(dyn_feat_dim 从 meta 读取)
  └─ Trainer    ← trainer.py(epochs/lr/patience 从 env 或 config)
```

---

## Confidence Notes

| 内容 | 确认度 |
|------|:------:|
| wide_table_sku.csv 是唯一数据入口 | ✅ 已确认 |
| ETL 链路 raw → silver → gold | ✅ 已确认（clean_data.py 代码验证） |
| V3/V5 特征维度差异 (7 vs 10) | ✅ 已确认 |
| .bin 文件 float32 memmap 格式 | ✅ 已确认 |
| dataset.py dyn_feat_dim=7 硬编码 | ✅ 已确认 |
| 环境变量 > config 覆盖优先级 | ✅ 已确认 |
| data_config.yaml 仅被 ETL 使用 | ✅ 已确认 |
| config 中的 dynamic_cols 列表未被训练代码读取 | 高概率推断（features 脚本硬编码自己的列表） |
| MinMaxScaler 原地修改 .bin 文件 | ✅ 已确认 (build_features_v5_sku.py:L307-326) |
