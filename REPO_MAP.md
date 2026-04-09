# REPO_MAP.md — 仓库目录地图

## 目录树（仅重要目录和文件）

```
B2B_Replenishment_System/
│
├── src/                               ← 核心源码包
│   ├── train/                         ← 🔴 训练管线（高频改动区）
│   │   ├── run_training_v2.py         ← ★ 当前训练主入口
│   │   ├── run_training.py            ← ⚠️ V1 历史入口（已废弃）
│   │   ├── trainer.py                 ← ★ 训练核心循环（fit_model）
│   │   └── dataset.py                 ← ★ memmap Dataset + DataLoader
│   │
│   ├── models/                        ← 🔴 模型定义（危险修改区）
│   │   ├── enhanced_model.py          ← ★ 4个模型变体类
│   │   ├── loss.py                    ← ★ TwoStageMaskedLoss
│   │   └── best_model.pth             ← ⚠️ 遗留在源码目录的旧权重（应忽略）
│   │
│   ├── features/                      ← 🟡 特征工程（偶尔修改）
│   │   ├── build_features_v2_sku.py   ← V3 特征（7维）
│   │   ├── build_features_v5_sku.py   ← V5 特征（10维，含期货）
│   │   ├── build_features_weekly_sku.py ← 周频特征（已放弃）
│   │   └── build_features_final.py    ← V1 特征（历史，已废弃）
│   │
│   ├── etl/                           ← 🟢 数据管道（低频修改）
│   │   ├── clean_data.py              ← 数据清洗主脚本（611行）
│   │   ├── build_wide_table.py        ← 宽表构建
│   │   └── extract_manager.py         ← Oracle 数据抽取
│   │
│   ├── evaluate/                      ← 🟢 评估脚本（历史版本）
│   │   ├── evaluate_dashboard.py      ← 仪表盘式评估
│   │   ├── evaluate_business_value.py ← 业务价值评估
│   │   ├── evaluate_pure_ml_metrics.py ← 纯 ML 指标
│   │   ├── generate_strategy_report.py ← 策略回测报告
│   │   └── analyze_csv_agg.py         ← CSV 聚合分析
│   │
│   ├── inference/                     ← 🟡 生产推断
│   │   └── generate_daily_inference.py ← ★ 每日推断主脚本
│   │
│   ├── utils/                         ← 工具函数
│   │   ├── db.py                      ← Oracle 连接
│   │   └── inspect_parquet.py         ← Parquet 检查
│   │
│   └── analysis/                      ← 数据分析脚本（一次性）
│       ├── analyze_correlation.py
│       ├── analyze_qtyspo.py
│       └── ... (8个分析脚本)
│
├── config/                            ← 配置文件
│   ├── model_config.yaml              ← ★ 唯一超参数配置
│   └── data_config.yaml               ← Oracle/ETL 数据源配置
│
├── data/                              ← 数据资产（不进版本控制）
│   ├── gold/wide_table_sku.csv        ← ★ 核心宽表（唯一数据入口）
│   ├── processed_v3/                  ← V3 张量 (7维 .bin)
│   ├── processed_v5/                  ← V5 张量 (10维 .bin)
│   ├── processed_weekly*/             ← 周频张量（多个windows，已放弃）
│   ├── artifacts_v3/                  ← V3 编码器 + meta
│   ├── artifacts_v5/                  ← V5 编码器 + meta
│   ├── raw/                           ← 原始数据
│   └── silver/                        ← 清洗数据
│
├── models/                            ← V1 模型权重（6个文件）
├── models_v2/                         ← ★ V2/V3 模型权重（23个 checkpoint）
├── models_weekly/                     ← 周频模型权重（已放弃）
│
├── reports/                           ← 评估报告和推断结果
│   ├── phase5/                        ← Phase 5 实验结果
│   ├── history_*.csv                  ← 各实验训练曲线
│   └── daily_orders_*.csv             ← 推断输出
│
├── diagnostics/                       ← 调试工具（5个脚本）
├── scripts/                           ← 运行脚本集合
│   ├── runners/                       ← 批量运行器
│   ├── eval/                          ← 评估脚本
│   └── archive/                       ← 归档
│
├── utils/                             ← 顶层工具包
│   ├── common.py                      ← load_yaml 等工具函数
│   └── logger.py                      ← 日志工具
│
├── evaluate.py                        ← ★ 6维度完整评估（高频入口）
├── evaluate_agg.py                    ← ★ SKU聚合Ratio分析
├── scripts/runners/phase5/run_phase5_experiments.py          ← ★ Phase 5 挂机Runner
├── run_training_weekly.py             ← 周频训练（已放弃）
├── run_weekly_experiments.py          ← 周频实验Runner（已放弃）
├── scripts/diagnostic/check_phase5_code.py     ← Phase 5 代码检查
├── scripts/analysis/analyze_daily_vs_weekly.py ← 日频vs周频对比分析
├── scripts/analysis/analyze_phase2_timing.py   ← Phase 2 耗时分析
├── scripts/analysis/analyze_phase4_results.py  ← Phase 4 结果分析
│
├── AGENTS.md                          ← ★ Codex 执行手册
├── TRAINING_FLOW.md                   ← ★ 训练主链路
├── DATA_AND_CONFIG_FLOW.md            ← ★ 数据流与配置流
├── PROJECT_OVERVIEW.md                ← ★ 项目总览
└── readme.md                          ← 原始 README
```

## 标注说明

| 标记 | 含义 |
|:----:|------|
| ★ | 高频入口文件 |
| 🔴 | 危险修改区（改动影响全局） |
| 🟡 | 需谨慎修改 |
| 🟢 | 相对安全修改 |
| ⚠️ | 历史遗留 / 已废弃 |

## 建议阅读顺序

```
1. AGENTS.md            ← 先读执行规则
2. PROJECT_OVERVIEW.md   ← 理解项目全貌
3. config/model_config.yaml  ← 理解超参数
4. TRAINING_FLOW.md      ← 理解训练链路
5. src/train/run_training_v2.py  ← 训练入口代码
6. src/models/enhanced_model.py  ← 模型结构
7. src/train/trainer.py  ← 训练循环
8. src/models/loss.py    ← 损失函数
9. DATA_AND_CONFIG_FLOW.md  ← 数据流
10. RISK_AND_TECH_DEBT.md  ← 风险点
```

---

## Confidence Notes

| 内容 | 确认度 |
|------|:------:|
| 目录结构 | ✅ 已确认（ls 验证） |
| 文件用途标注 | ✅ 已确认（代码审查） |
| 历史/废弃标记 | 高概率推断（基于代码注释和版本号） |
| scripts/ 内部结构 | 尚未深入确认（仅看到子目录名） |
