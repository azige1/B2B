# Phase8 Event 覆盖路由优化结论

生成日期：2026-06-16

## 结论

在旧 `0.6` 可比口径下，当前最合理的 Phase8 优化不是继续堆特征，而是做 `Event 覆盖感知路由`：

- `2025-09-01`：Event 没有有效历史覆盖，回退 Phase7。
- `2025-10-01 / 2025-11-01 / 2025-12-01`：Event 已有覆盖，使用 `Event+求购`。

这个路由不重新训练模型，只组合现有预测结果；因此工程改动小、解释性强、风险低。

## 四锚点结果

| 候选 | 规则 | mean global_wmape | 相对 Phase7 改善 | global_ratio | blockbuster_under_wape | rank_corr_positive_skus |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Phase7 current | 原主线 | 0.6863 | - | 1.0159 | 0.4165 | 0.8044 |
| purchase_request_plus | 全部用求购 | 0.6770 | 1.36% | 1.0111 | 0.4131 | 0.8025 |
| event_request_plus | 全部用 Event+求购 | 0.6620 | 3.54% | 1.0189 | 0.3887 | 0.8070 |
| coverage_router | 9月回退 Phase7，10-12月用 Event+求购 | 0.6591 | 3.97% | 1.0198 | 0.3820 | 0.8078 |

## 为什么选择简单锚点路由

本轮同时试了样本级 Event 覆盖阈值，例如：

- `event_any_30 >= 1`
- `event_active_buyers_30 >= 1`
- `event_strong_30 >= 1`
- `event_any_30 + recency` 组合条件

结果这些样本级规则都没有超过锚点级路由。当前最优仍是：

`router_phase7_anchor_event_covered`

原因是 Event 在 9 月前整体不可用，问题主要不是某些 SKU 覆盖不足，而是整个锚点的 Event 历史窗口不足。按锚点回退比按样本阈值更稳定，也更容易向甲方解释。

## 业务解释

这版可以这样对外说：

> Phase8 引入 Event 和求购信号后，在 Event 有有效历史覆盖的月份启用增强模型；如果 Event 历史覆盖不足，则自动回退到原 Phase7 主线，避免新信号缺失时拖累预测。

这比“所有月份强行用 Event+求购”更合理。

## 当前推荐

建议把 `coverage_router` 作为旧 `0.6` 体系下新的 Phase8 no-listing 最优探索分支。

后续再优化时，优先方向不是继续调 Event 覆盖阈值，而是：

1. 在 `coverage_router` 基础上做爆款低估校准。
2. 在 `coverage_router` 基础上迁移 long-zero / zero-split 训练策略。
3. 用同一口径扩展到 2026 锚点，验证稳定性。

## 产物

- 候选汇总：`reports/phase8_event_request_router/phase8_event_request_router_candidate_summary.csv`
- 锚点明细：`reports/phase8_event_request_router/phase8_event_request_router_anchor_table.csv`
- 最优上下文：`reports/phase8_event_request_router/phase8_event_request_router_best_context.csv`
- 路由报告：`reports/phase8_event_request_router/phase8_event_request_router_summary.md`
- 分析脚本：`src/analysis/analyze_phase8t_event_request_router.py`
