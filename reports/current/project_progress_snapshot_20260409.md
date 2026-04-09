# 项目进度快照（2026-04-09）

## 说明

- 本快照基于当前工作区文件现状整理，不依赖 Git 提交历史。
- 当前工作区是项目快照，不含 `.git/` 元数据；进度判断以 `reports/current/`、phase 报告、runner 和模型资产为准。
- 当前官方参考入口仍应优先使用：
  - `reports/current/current_mainline.json`
  - `reports/current/current_freeze_summary.md`
  - `reports/current/current_model_compare_summary.md`
  - `reports/current/phase8_direction_note.md`

## 一句话状态

项目已经完成正式主线冻结，当前处于“`phase7` 已冻结、`phase8` 已完成方向验证但尚未正式启动”的状态。

## 已完成里程碑

### 1. 主链路已成型

以下核心链路都已具备并有对应代码目录：

- ETL：`src/etl/`
- 特征工程：`src/features/`
- 训练：`src/train/`
- 评估：`src/evaluate/`
- 推理：`src/inference/`
- 诊断/分析：`src/analysis/`

这说明项目早已超过“基础搭建”阶段，已具备完整研究与交付主链路。

### 2. 历史实验已从 phase5 持续推进到 phase7

当前仓库中保留了从 `phase5` 到 `phase7` 的 runner、汇总脚本和阶段报告，包括：

- `scripts/runners/phase5/run_phase5*`
- `scripts/runners/phase6/run_phase6*`
- `scripts/runners/phase7/run_phase7*`
- `reports/phase5*/`
- `reports/phase6*/`
- `reports/phase7*/`

这部分工作说明模型路线已经经历了多轮树模型构建、校准、稳定化、验证和比较，不是单次试验性结果。

### 3. 官方主线已经冻结

截至 `2026-04-09`，官方状态仍是：

- 当前官方阶段：`phase7`
- 当前官方主线：`tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`
- 当前官方树模型家族：`LightGBM`
- 当前状态：`frozen`

当前主线相对上一版官方树模型和当前最佳 LSTM 参考线都有明确提升，当前正式结论已经落地到：

- `reports/current/current_freeze_summary.md`
- `reports/current/current_mainline.json`
- `reports/current/current_model_compare_summary.md`
- `models/current_phase7_mainline/`

### 4. 当前官方模型资产和数据入口已经落地

当前工作区内保留了：

- 官方模型目录：`models/current_phase7_mainline/`
- 当前资产映射：`data/current_assets.json`
- 当前正式评估与展示材料：`reports/current/current_model_compare.html`、`reports/current/current_model_compare.csv`

这表明项目已经具备明确的“当前正式版本”定义，而不是仍在多个候选之间摇摆。

## 当前进行中的工作

### 1. phase8 预备分析已完成

`phase8a_prep` 已经产出覆盖审计、残差分析、SHAP、影子实验策略等准备材料，包括：

- `reports/phase8a_prep/phase8_data_coverage_audit.md`
- `reports/phase8a_prep/phase8_residual_gap_summary.md`
- `reports/phase8a_prep/phase8_shap_summary.md`
- `reports/phase8a_prep/phase8_shadow_experiment_policy.md`

### 2. phase8 当前工作方向已明确

截至 `2026-04-09`，当前方向结论已经明确记录为：

- 官方赢家仍保持 `phase7`
- `phase8` 主方向为 `event + inventory`
- `preorder` 暂时只作为次级证据

对应文件：

- `reports/current/phase8_direction_note.md`

### 3. event + inventory 影子线已经做出正向证据

当前已经完成 `2026-02-15` 和 `2026-02-24` 两个锚点的 `event + inventory` 影子实验，结果显示该方向在以下问题上有改善：

- `4_25 / Ice 4_25`
- weak-signal blockbuster
- zero-true false positives
- ranking / allocation

对应文件：

- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_summary.md`
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_detail_summary.md`

### 4. 最新分析已推进到库存约束问题

截至 `2026-04-09`，仓库中最新的 phase8 工作已继续推进到库存约束分析：

- `scripts/runners/phase8/run_phase8f_inventory_constraint_pack.py`
- `src/analysis/generate_phase8f_inventory_constraint_pack.py`
- `reports/phase8_extended_signal_2026/phase8_inventory_constraint_summary.md`

当前结论是：

- `event + inventory` 已经能利用正库存信号改善预测
- 但现有特征还不能严格识别“明确缺货导致未补货”

## 当前未完成 / 阻塞项

### 1. 正式 phase8 还没有开始

虽然 phase8 的方向验证已经做了不少，但当前仍属于：

- exploratory shadow validation
- residual / case analysis
- schema / feature-line preparation

当前还不是：

- 官方主线替换
- 正式 phase8 定向优化
- 标签定义调整后的新一轮正式 freeze

### 2. 客户侧语义确认仍是主要阻塞

正式 phase8 当前明确被以下事项阻塞：

- `V_IRS_ORDERFTP` 的 `TYPE` 语义
- `V_IRS_ORDERFTP` 中 `QTY < 0` 的业务含义
- `V_IRS_ORDERFTP` 重复行的业务含义
- 生命周期表是否可提供，以及字段粒度和口径

对应文件：

- `reports/current/phase8_client_data_request.md`
- `reports/current/client_phase8_data_request_external.md`
- `reports/current/v_irs_orderftp_client_questions.md`

### 3. 库存特征逻辑还有一处关键缺口

当前库存特征没有保留“有快照但库存为 0”的独立状态，因此：

- 现在可以部分区分“有库存支撑的需求”和“无库存信号样本”
- 但还不能严格回答“没有补货到底是因为真没需求，还是因为缺货”

这会直接影响 phase8 是否能把库存约束问题做成正式特征线。

## 建议的下一步

### 1. 外部等待项

优先等待并整理客户回复：

- `V_IRS_ORDERFTP` 语义确认
- 生命周期表可用性确认

### 2. 内部可继续做的事

在客户未回复前，可以继续做但应控制边界：

- 继续 `event + inventory` 的影子验证
- 继续 residual / case 分析
- 修库存特征生成逻辑，保留“快照存在性”和“0 库存状态”
- 不启动开放式超参搜索
- 不提前替换当前官方 `phase7` 主线

### 3. 客户回复后再做的事

客户回复到位后，再进入正式 phase8：

- 修订标签清洗口径
- 引入生命周期特征
- 在 `event + inventory` 方向上做正式定向优化
- 评估是否满足替换当前 `phase7` 主线的门槛

## 当前结论

截至 `2026-04-09`，这个项目的真实进度可以概括为：

- 主链路已经完成
- `phase7` 官方主线已经冻结
- 当前正式版本已经具备可交付基线
- `phase8` 方向验证已经做出正向信号
- 但正式 phase8 仍被客户侧数据语义和生命周期表阻塞
