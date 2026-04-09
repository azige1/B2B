# Phase8 探索进度摘要（2026-04-09）

## 一句话结论

- 正式主线仍然是 `phase7 frozen`
- `phase8` 当前已经有明确方向，但仍处于探索验证阶段
- 当前最优先方向是：`event + inventory`

## 当前定位

`phase8` 现在的含义是：

- exploratory shadow validation
- residual error analysis
- schema / feature-line preparation

`phase8` 现在**还不意味着**：

- 官方主线替换
- 正式 phase8 freeze
- 开放式超参搜索

## 已做实验与结论

### 1. Phase8a Prep

目标：

- 先从当前 `phase7` 正式主线出发，看残差缺口到底在哪里
- 判断 phase8 应该优先补哪类信号

关键文件：

- `reports/phase8a_prep/phase8_residual_gap_summary.md`
- `reports/phase8a_prep/phase8_data_coverage_audit.md`
- `reports/phase8a_prep/phase8_shap_summary.md`
- `reports/phase8a_prep/phase8_shadow_experiment_policy.md`

关键结论：

- 当前 `phase7` 主线在四锚点正式指标上复算一致，没有口径漂移
- 主要残差仍集中在：
  - weak-signal blockbuster
  - `repl0_fut0`
  - `repl1_fut0`
  - zero-true false positives
- 这一步主要是确定“phase8 该补什么”，不是替换主线

### 2. Event-Only Shadow

目标：

- 先验证 `event` 信号本身有没有方向性价值

关键文件：

- `reports/phase8_event_shadow/phase8_event_shadow_summary.md`

范围：

- `2025-10/11/12`

关键结果：

- `global_wmape: 0.6678 -> 0.6350`
- `4_25_under_wape: 0.3741 -> 0.3604`
- `blockbuster_under_wape: 0.3853 -> 0.3469`
- `blockbuster_sku_p50: 0.5774 -> 0.6156`

结论：

- `event` 有正向方向性证据
- 但由于事件覆盖从 `2025-09-18` 才开始，这条证据只够做 shadow evidence，不足以直接替换主线

### 3. Event + Inventory Shadow

目标：

- 验证库存信号加入后，是否能进一步改善当前正式主线最弱的部分

关键文件：

- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_summary.md`
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_detail_summary.md`

范围：

- `2026-02-15`
- `2026-02-24`

两锚点平均结果：

- `global_wmape: 0.8614 -> 0.6667`
- `4_25_under_wape: 0.4384 -> 0.3452`
- `4_25_sku_p50: 0.5966 -> 0.6818`
- `ice_4_25_sku_p50: 0.5234 -> 0.6212`
- `blockbuster_under_wape: 0.4934 -> 0.3773`
- `blockbuster_sku_p50: 0.4735 -> 0.6326`
- `rank_corr_positive_skus: 0.6918 -> 0.7522`

结论：

- 这是目前最强的 phase8 正向证据
- 改善集中出现在当前 `phase7` 仍然最弱的区域
- 当前工作方向因此明确收敛到 `event + inventory`

### 4. Extended Signal Audit

目标：

- 在 2026 数据上看 `event / preorder / inventory` 的实际覆盖与可用性

关键文件：

- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_summary.md`

关键结论：

- inventory 匹配率较高：`0.8601`
- event 有一定覆盖：`0.1243`
- preorder 基本不可用：`0.0000`

进一步判断：

- `event + inventory` 有继续做特征线的现实基础
- `preorder` 目前只适合当辅助证据，不适合做主方向

### 5. Inventory Constraint Analysis

目标：

- 判断当前库存影子线到底是在识别“真实缺货约束”，还是只是在利用“有库存支撑”的正信号

关键文件：

- `reports/phase8_extended_signal_2026/phase8_inventory_constraint_summary.md`

关键结论：

- 当前影子线已经能利用正库存信号改善预测
- 但还不能严格识别“明确缺货导致未补货”
- 根因是当前库存特征逻辑没有保留“有快照但库存为 0”的独立状态

这意味着：

- 库存方向有效
- 但库存特征工程本身还有一轮必要修正

## 当前总判断

按当前证据强弱排序：

1. `event + inventory`：当前 phase8 主方向
2. `event only`：有正向证据，但不如 event+inventory 完整
3. `preorder`：目前仅保留为次级辅助线，不作为主方向

## 现在该怎么理解项目进度

当前不是“phase8 还没开始”，而是：

- `phase8` 已经完成了方向探索
- 当前主方向已经收敛
- 当前还没有进入正式替换主线阶段

因此更准确的说法是：

- `phase7` 已定档并冻结
- `phase8` 正在做定向探索
- 当前探索结论是继续走 `event + inventory`

## 暂缓项

在甲方新数据或语义确认到位前，先不推进这些动作：

- 正式替换 `phase7` 主线
- 修订最终标签定义
- 启动正式 phase8 freeze

## 你现在如果只想记一句

截至 `2026-04-09`，`phase8` 的核心结论不是“还在乱试”，而是：

- **主方向已经定成 `event + inventory`**
- **证据是正向的**
- **但正式 phase8 仍然先等甲方数据 / 语义确认后再推进**
