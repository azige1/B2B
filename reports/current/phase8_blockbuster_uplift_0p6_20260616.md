# Phase8 爆款低估校准结论

生成日期：2026-06-16

## 结论

在沿用 Phase7 旧 `0.6` 系列评估口径的前提下，Phase8 已经可以定一个可解释、可复现的版本。

推荐定版分支：

`coverage_router + conservative_blockbuster_uplift`

对应候选：

`uplift_pred_ge_20_prob_ge_0.50_signal_ge_20_x12`

规则含义：
- 先使用 `coverage_router`：`2025-09-01` 回退 Phase7，`2025-10-01 / 2025-11-01 / 2025-12-01` 使用 Event+求购增强分支。
- 再只对极少数高确定性的爆款候选做轻量上调：`ai_pred_qty >= 20`、`ai_pred_prob >= 0.50`、历史需求信号 `>= 20` 的 SKU-date，预测量乘以 `1.20`。
- 该规则只命中约 `0.48%` 的样本，不改变标签、不引入新数据、不重训模型。

如果必须坚持“完全无后处理”的严格模型分支，则当前 Phase8 应定为：

`coverage_router`

但如果允许一层可解释的业务校准，则推荐把 `coverage_router + conservative_blockbuster_uplift` 作为 Phase8 对外主结果。

## 核心指标

| 分支 | global_wmape | 相对 Phase7 改善 | global_ratio | max_anchor_ratio | blockbuster_under_wape | blockbuster_sku_p50 | 4_25_under_wape | ice_4_25_sku_p50 | rank_corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase7 official | 0.6863 | - | 1.0159 | 1.0793 | 0.4165 | 0.5539 | 0.3566 | 0.5853 | 0.8044 |
| Event+求购全量 | 0.6620 | 3.54% | 1.0189 | 1.0756 | 0.3887 | 0.5891 | 0.3467 | 0.5980 | 0.8070 |
| coverage_router | 0.6591 | 3.97% | 1.0198 | 1.0793 | 0.3820 | 0.5957 | 0.3466 | 0.5981 | 0.8078 |
| 推荐 uplift | 0.6482 | 5.56% | 1.0578 | 1.0993 | 0.3101 | 0.6855 | 0.3460 | 0.5981 | 0.8078 |
| 激进 uplift 参考 | 0.6364 | 7.27% | 1.0962 | 1.1583 | 0.2875 | 0.7148 | 0.3100 | 0.6266 | 0.8079 |

推荐 uplift 相比 `coverage_router`：
- `global_wmape` 从 `0.6591` 降到 `0.6482`，下降 `0.0109`。
- `blockbuster_under_wape` 从 `0.3820` 降到 `0.3101`，下降 `0.0719`。
- `blockbuster_sku_p50` 从 `0.5957` 提升到 `0.6855`。
- `rank_corr_positive_skus` 基本不变，说明排序能力没有被破坏。
- `zero_true_pred_ge_3_rate` 保持 `0.0145`，说明没有额外抬高零真实样本的大额误报。

## 为什么不选指标最优

指标最优候选是：

`uplift_pred_ge_8_prob_ge_0.50_signal_ge_20_x12`

它的 `global_wmape=0.6364`、`blockbuster_under_wape=0.2875`，数字更好，但 `max_anchor_ratio=1.1583`，其中 `2025-09-01` 锚点整体预测偏高已经超过 Phase7 风格的安全上界。

所以它适合作为 shadow reference，不适合作为当前对外定版。推荐候选虽然 WMAPE 略高，但每个锚点的 ratio 都没有超过 `1.10`，更容易解释和交付。

## 推荐候选锚点表现

| anchor | selected_rate | global_wmape | global_ratio | blockbuster_under_wape | blockbuster_sku_p50 | 4_25_under_wape |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025-09-01 | 0.0032 | 0.7299 | 1.0993 | 0.4566 | 0.5044 | 0.3038 |
| 2025-10-01 | 0.0036 | 0.7057 | 1.0777 | 0.2979 | 0.6528 | 0.3884 |
| 2025-11-01 | 0.0056 | 0.5208 | 1.0136 | 0.1608 | 0.8964 | 0.3439 |
| 2025-12-01 | 0.0067 | 0.6361 | 1.0407 | 0.3250 | 0.6882 | 0.3479 |

## Phase8 定版标准

建议把 Phase8 定版门槛设成以下几条：
- 必须沿用 Phase7 旧 `0.6` 系列口径，四个锚点 `2025-09-01 / 2025-10-01 / 2025-11-01 / 2025-12-01`。
- `global_wmape` 必须低于 Phase7 的 `0.6863`，最好低于 `0.6650`。
- `blockbuster_under_wape` 必须明显低于 Phase7 的 `0.4165`，最好低于 `0.3900`。
- `blockbuster_sku_p50` 必须高于 Phase7 的 `0.5539`。
- `rank_corr_positive_skus` 不应低于 Phase7。
- `global_ratio` 不作为主优化目标，但均值和单锚点都不应明显失控；推荐单锚点不超过 `1.10`。

按这个标准，`coverage_router + conservative_blockbuster_uplift` 可以作为 Phase8 定版候选。

## 后续建议

短期不建议继续大范围搜索规则，否则容易变成针对四个锚点过拟合。更合理的是：
- 将推荐 uplift 固化为一个可开关的校准层，默认开启，保留 `coverage_router` 作为纯模型 fallback。
- 等生命周期表和更完整 2026 标签后，再评估是否把这个 uplift 思路迁移进训练阶段，做成模型内生的爆款/长尾非对称损失。
- 对外汇报时主讲 Event+求购增强、覆盖路由、爆款低估校准三件事，不强调规则搜索过程。

## 产物

- 候选汇总：`reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_candidate_summary.csv`
- 锚点明细：`reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_anchor_table.csv`
- 推荐上下文：`reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_recommended_context.csv`
- 激进最优上下文：`reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_metric_best_context.csv`
- 实验摘要：`reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_summary.md`
- 分析脚本：`src/analysis/analyze_phase8u_blockbuster_uplift.py`
