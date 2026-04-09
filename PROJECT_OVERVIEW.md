# PROJECT_OVERVIEW.md — 项目总览

## 一句话定义

服装 B2B 供应链 SKU 级 30 天补货量预测系统，基于双塔 LSTM 时序模型。

## 主要用途

1. **每日补货指令生成**: 站在今天 T，预测每个 SKU 在 [T+1, T+30] 的总补货件数
2. **交付物**: `reports/daily_orders_{DATE}.csv`，含 `buyer_id, sku_id, ai_replenish_prob_30d, ai_budget_qty_30d`
3. **业务场景**: 甲方按此表安排仓库备货

## 核心能力列表

| 能力 | 状态 | 说明 |
|------|:----:|------|
| 特征工程 (V3/V5) | ✅ | 7维/10维时序特征 + 13维静态特征 |
| 模型训练 | ✅ | 双塔 LSTM (4个架构变体)，支持挂机实验 |
| 验证评估 | ✅ | 6维度评估 + SKU聚合Ratio + 冷启动诊断 |
| 生产推断 | ✅ | 每日推断脚本，自动识别最新日期 |
| 实验管理 | ✅ | 环境变量注入 + 12组实验Runner |

## 代码库主要组成部分

```
B2B_Replenishment_System/
├── src/                    ← 核心源码包
│   ├── train/              ← 训练管线（入口+训练器+数据集）
│   ├── models/             ← 模型架构 + 损失函数
│   ├── features/           ← 特征工程脚本 (V2/V5/Weekly)
│   ├── etl/                ← 数据抽取清洗（Oracle → CSV）
│   ├── evaluate/           ← 评估分析脚本（历史版本）
│   └── inference/          ← 生产推断脚本
├── config/                 ← YAML 配置文件
├── data/                   ← 数据资产（宽表、张量、编码器）
├── models_v2/              ← V2/V3 模型权重（23个checkpoint）
├── reports/                ← 评估报告、推断结果
├── evaluate.py             ← ★ 6维度完整评估（当前主入口）
├── evaluate_agg.py         ← ★ SKU聚合Ratio分析 + 冷启动
├── scripts/runners/phase5/run_phase5_experiments.py ← ★ Phase5 挂机实验Runner
└── diagnostics/            ← 调试诊断工具
```

## 核心技术栈

- **语言**: Python 3.x (Windows 环境)
- **深度学习**: PyTorch 2.0+ (torch.compile, AMP, GradScaler)
- **数据处理**: Pandas, NumPy (memmap 懒加载)
- **特征编码**: scikit-learn LabelEncoder + MinMaxScaler
- **时序模型**: LSTM / GRU / BiLSTM / LSTM+Attention
- **损失函数**: FocalLoss + SoftF1Loss + HuberLoss (多任务)
- **配置**: YAML + 环境变量

## 各子系统职责划分

| 子系统 | 主入口 | 产出 |
|--------|--------|------|
| ETL | `src/etl/clean_data.py` | `data/silver/*.csv` → `data/gold/wide_table_sku.csv` |
| 特征工程 | `src/features/build_features_v{n}_sku.py` | `data/processed_v{n}/*.bin` + `data/artifacts_v{n}/` |
| 训练 | `src/train/run_training_v2.py` | `models_v{n}/best_enhanced_model.pth` |
| 评估 | `evaluate.py` + `evaluate_agg.py` | `reports/eval_report*.txt` + `reports/phase5/` |
| 推断 | `src/inference/generate_daily_inference.py` | `reports/daily_orders_{DATE}.csv` |
| 实验管理 | `scripts/runners/phase5/run_phase5_experiments.py` | 自动化12组实验 |

## 新人第一次应该先看的文件

1. **`AGENTS.md`** — 修改规则和隐式约定
2. **`TRAINING_FLOW.md`** — 训练调用链
3. **`config/model_config.yaml`** — 超参数定义
4. **`src/train/run_training_v2.py`** — 训练入口
5. **`src/models/enhanced_model.py`** — 模型结构
6. **`src/train/trainer.py`** — 训练循环

## 多阶段项目说明

```
V1.0 (历史) → V1.5 → V1.6 → V1.7 → V2.0 (SKU级) → Phase 2 (架构搜索)
→ Phase 4 (周频实验，已放弃) → Phase 5 (当前，12组实验)

版本并存关系:
  V1 入口: src/train/run_training.py    ← 已废弃，仅保留兼容
  V2+ 入口: src/train/run_training_v2.py ← 当前主入口

数据版本:
  V3 (7维): data/processed_v3/ + artifacts_v3/  ← 基线
  V5 (10维): data/processed_v5/ + artifacts_v5/ ← 新特征（含期货信号）
  Weekly: data/processed_weekly*/               ← 周频实验（已放弃）
```

## What Codex Should Know First

1. **唯一的训练入口**是 `src/train/run_training_v2.py`（不是 `run_training.py`）
2. **模型架构有 4 个变体**，通过 `MODEL_REGISTRY` 路由，环境变量 `EXP_MODEL_TYPE` 选择
3. **任何加载 pickle encoder 的脚本必须先定义 `DummyLE` 类**
4. **加载 checkpoint 时必须清理 `_orig_mod.` 前缀**
5. **evaluate.py 和 trainer.py 使用不同的解码参数**（prob 阈值和缩放系数不同）
6. **dataset.py 中 `dyn_feat_dim=7` 是硬编码**，V5 的 10 维数据依赖字节自动推导
7. **model_config.yaml 的默认值被环境变量覆盖**，实验时以环境变量为准
8. **不要删除 `models_v2/` 下的 23 个 checkpoint**——它们是历史实验的备份

---

## Confidence Notes

| 内容 | 确认度 |
|------|:------:|
| 项目用途和业务目标 | ✅ 已确认 |
| 代码目录结构 | ✅ 已确认（ls 验证） |
| 技术栈 | ✅ 已确认（import 追踪） |
| V1 入口已废弃 | ✅ 已确认（V1 用 buyer_id 维度） |
| 数据版本隔离 | ✅ 已确认 |
| Weekly 实验已放弃 | 高概率推断（Phase 4 F1=0.43 远低于日频 F1=0.57） |
