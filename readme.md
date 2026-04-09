# B2B Replenishment System

服装 B2B 补货预测研究与交付工作区。

## Current Status

- 当前官方阶段: `phase7`
- 当前官方状态: `frozen`
- 当前官方主线: `tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`
- 当前官方树模型家族: `LightGBM`
- 当前 `phase8` 工作方向: `event + inventory`
- 当前 `phase8` 最优探索基线: `event_inventory_zero_split`

当前仓库已经不是早期的 LSTM 主线仓库。LSTM 相关内容主要保留为历史实验与对照参考，当前官方主线是树模型。

## Start Here

如果只想快速了解当前状态，按这个顺序看：

1. `DOCS_INDEX.md`
2. `PROJECT_INDEX.md`
3. `reports/current/current_mainline.json`
4. `reports/current/current_freeze_summary.md`
5. `reports/current/phase8_restart_playbook_20260409.md`
6. `RUNNERS_INDEX.md`

## Repository Layout

```text
B2B_Replenishment_System/
├── src/                    # 核心代码: etl / features / train / analysis / inference
├── scripts/
│   ├── runners/            # phase5 ~ phase8 runner
│   ├── analysis/           # 历史分析辅助脚本
│   ├── diagnostic/         # 仓库与实验自检脚本
│   └── eval/               # 辅助评估脚本
├── reports/                # 当前结论、阶段报告、历史实验报告
├── data/                   # 当前数据资产映射与本地处理产物
├── data_warehouse/         # 原始抽取与快照数据
├── models/                 # 当前官方模型资产
├── config/                 # 配置文件
└── docs/                   # 其他说明文档
```

## Common Entry Points

```bash
# 刷新当前官方 freeze 摘要
python scripts/runners/phase7/run_phase7_freeze.py

# 刷新当前官方 compare 页面
python scripts/runners/phase7/run_phase7i_full_model_compare.py

# 运行 phase8 准备分析
python scripts/runners/phase8/run_phase8a_prep.py

# 运行 phase8 库存约束分析
python scripts/runners/phase8/run_phase8f_inventory_constraint_pack.py

# 仓库卫生检查
python scripts/diagnostic/check_git_hygiene.py
```

如果只是阅读当前结论，优先直接看 `reports/current/`，不要先从历史 phase 报告开始。

## Documentation Rules

- 当前状态判断以 `PROJECT_INDEX.md` 和 `reports/current/` 为准。
- 执行入口以 `RUNNERS_INDEX.md` 为准。
- 历史阶段报告主要在 `reports/phase5*/`, `reports/phase6*/`, `reports/phase7*/`。
- 历史 runner 主要在 `scripts/runners/phase5/` 到 `scripts/runners/phase8/`。
- 旧的 LSTM 背景说明可以作为历史上下文阅读，但不代表当前官方主线。

## Versioning Rules

- Git 主要跟踪源码、配置、关键结论文档和当前入口文档。
- 大部分原始数据、处理中间产物、模型二进制和导出报表默认不纳入版本库。
- 当前官方参考入口已经收敛到 `DOCS_INDEX.md`, `PROJECT_INDEX.md`, `RUNNERS_INDEX.md`, `reports/current/`。
