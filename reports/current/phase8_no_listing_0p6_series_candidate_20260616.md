# Phase8 不依赖上市日期的 0.6 系列候选结论

生成日期：2026-06-16

## 结论

在沿用 Phase7 当时的旧 0.6 系列评估口径下，不使用上市日期、不做 listing-eligible 过滤，本轮新增的 `Event + 求购` 组合候选是当前最适合对外主打的 Phase8 no-listing 结果。

四个官方锚点 `2025-09-01 / 2025-10-01 / 2025-11-01 / 2025-12-01` 上：

| 模型/候选 | 说明 | mean global_wmape | 相对 Phase7 改善 |
| --- | --- | ---: | ---: |
| Phase7 current | 原 0.6 系列主线 | 0.6863 | - |
| purchase_request_plus | 只加求购信号 | 0.6770 | 1.36% |
| event_request_plus | 四锚点统一使用 Event+求购特征集，9月 Event 无有效历史覆盖 | 0.6620 | 3.54% |

其中 `event_request_plus` 相对 `purchase_request_plus` 又多改善约 `2.22%`。

## Event 覆盖锚点

Event 数据从 2025-09-18 后才开始覆盖，因此更公平的 Event 覆盖比较是 `2025-10-01 / 2025-11-01 / 2025-12-01` 三个锚点。

| 模型/候选 | mean global_wmape | 相对 Phase7 改善 |
| --- | ---: | ---: |
| Phase7 current | 0.6678 | - |
| event_intent_plus | 0.6350 | 4.91% |
| event_request_plus | 0.6314 | 5.44% |

`Event+求购` 相对 pure Event 的增量约 `0.55%`，说明主要提升来自 Event，求购信号提供小幅补充。

## 关键指标

四锚点 `event_request_plus` 候选：

| 指标 | Phase7 current | event_request_plus | 变化 |
| --- | ---: | ---: | ---: |
| global_ratio | 1.0159 | 1.0189 | +0.0030 |
| global_wmape | 0.6863 | 0.6620 | -0.0243 |
| blockbuster_under_wape | 0.4165 | 0.3887 | -0.0278 |
| blockbuster_sku_p50 | 0.5539 | 0.5891 | +0.0352 |
| rank_corr_positive_skus | 0.8044 | 0.8070 | +0.0026 |
| top20_true_volume_capture | 0.6494 | 0.6518 | +0.0024 |

Event 覆盖三锚点：

| 指标 | Phase7 current | event_request_plus | 变化 |
| --- | ---: | ---: | ---: |
| global_ratio | 0.9948 | 1.0000 | +0.0053 |
| global_wmape | 0.6678 | 0.6314 | -0.0363 |
| blockbuster_under_wape | 0.3853 | 0.3393 | -0.0460 |
| blockbuster_sku_p50 | 0.5774 | 0.6331 | +0.0557 |
| rank_corr_positive_skus | 0.8094 | 0.8140 | +0.0046 |
| top20_true_volume_capture | 0.6597 | 0.6638 | +0.0041 |

## 对外口径建议

建议主讲：

1. Phase8 在不依赖上市日期的情况下，已经把旧 0.6 系列四锚点 WMAPE 从 `0.6863` 降到 `0.6620`，相对改善 `3.54%`。
2. 在 Event 有覆盖的 10-12 月，WMAPE 从 `0.6678` 降到 `0.6314`，相对改善 `5.44%`。
3. 爆款低估也同步改善，四锚点 `blockbuster_under_wape` 从 `0.4165` 降到 `0.3887`；Event 覆盖三锚点从 `0.3853` 降到 `0.3393`。
4. 上市日期不是当前 0.6 系列结果的前提条件；它更适合作为下一阶段解决冷启动、生命周期分层、短周期商品总量预测和盈亏分析的关键字段。

不建议主讲：

1. 不建议把 formal 1.0 系列 WMAPE 与旧 0.6 系列直接混在同一张表里，它们的样本过滤和时间切分口径不同。
2. 不建议说求购信号是主贡献；当前数据看，Event 是主提升来源，求购是补充增益。
3. 不建议把 2026 两锚点探索结果当作正式结论，它可以作为后续潜力说明，但不是当前对外主线。

## 产物

- 汇总表：`reports/phase8_event_request_shadow/phase8_event_request_shadow_anchor_table.csv`
- 汇总报告：`reports/phase8_event_request_shadow/phase8_event_request_shadow_summary.md`
- 结果 JSON：`reports/phase8_event_request_shadow/phase8_event_request_shadow_result.json`
- Runner：`scripts/runners/phase8/run_phase8t_event_request_shadow.py`
- Summary：`src/analysis/summarize_phase8t_event_request_shadow_results.py`
