# B2B Replenishment System

B2B 服装补货预测项目。当前仓库已经从早期的 LSTM 主线，演进到以树模型为正式主线的研究与交付工作区。

## 当前状态

- 当前官方阶段：`phase7`
- 当前官方状态：`frozen`
- 当前官方主线：`tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`
- 当前官方树模型家族：`LightGBM`
- 当前 phase8 方向：`event + inventory`，但仍处于 shadow / exploratory 阶段

当前正式入口优先看：

- `PROJECT_INDEX.md`
- `reports/current/current_mainline.json`
- `reports/current/current_freeze_summary.md`
- `reports/current/current_model_compare_summary.md`
- `reports/current/phase8_direction_note.md`

## 仓库结构

```text
B2B_Replenishment_System/
├── src/                 # 核心代码：ETL、特征、训练、评估、推理、分析
├── config/              # 模型与数据配置
├── docs/                # 项目说明与工作规范
├── reports/             # 当前结论、阶段总结、历史实验结论
├── scripts/             # 诊断脚本与历史 runner
├── data/                # 数据资产索引与本地实验资产映射
├── data_warehouse/      # 原始抽取与快照数据（默认不纳入 Git）
└── models/              # 本地模型产物（默认不纳入 Git）
```

## 常用入口

当前项目更偏“阶段 runner + 报告归档”的工作方式，常用入口包括：

```bash
# 查看当前官方状态
python run_phase7_freeze.py

# 生成当前官方 December compare
python run_phase7i_full_model_compare.py

# phase8 准备分析
python run_phase8a_prep.py

# phase8 库存约束分析
python run_phase8f_inventory_constraint_pack.py

# 仓库卫生检查
python scripts/diagnostic/check_git_hygiene.py
```

如果是阅读而不是重跑，优先直接看 `reports/current/`。

## 数据与版本管理

- Git 主要跟踪源码、配置、关键文档、阶段结论。
- 原始数据、处理产物、模型权重、导出型报表默认不纳入版本库。
- Git 规则见 `docs/GIT_WORKFLOW.md`。

## 当前工作重点

- 保持 `phase7` 官方主线稳定可追溯
- 基于 `event + inventory` 做 phase8 方向验证
- 等待客户确认 `V_IRS_ORDERFTP` 语义与生命周期表后，再进入正式 phase8
