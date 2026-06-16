# Phase8 最终定版建议报告

日期：2026-06-16

## 结论

Phase8 现在可以阶段定版。建议把 `legacy_combo_listing_eligible` 定为 Phase8 当前 formal 主线候选，用于后续汇报和工程收口；`rich_combo_listing_eligible` 保留为备选；`asym_mild_cls_listing_targeted_reg` 保留为后续 zero-split 探索方向，不直接作为正式主线。

推荐定版口径：

> Phase8 相比 Phase7 的核心提升，不是单纯调参，而是接入并验证了上市日期/生命周期、Event、库存、求购等新业务信号，并用 2025-09 到 2026-05 的多锚点时间外评估证明新分支整体优于 Phase7。当前可将 `legacy_combo_listing_eligible` 作为 Phase8 formal 候选主线冻结；6 月完整标签到齐后只做后验监控，不作为当前定版前置条件。

## 当前主线

| 项目 | 决定 |
| --- | --- |
| Phase8 formal 主线 | `legacy_combo_listing_eligible` |
| 对照基线 | `phase7_listing_eligible` / Phase7 official |
| 预测目标 | 锚点后未来 30 天 SKU 级正向补货量 |
| 主指标 | `global_wmape` |
| 样本粒度 | `sku_id + anchor_date` |
| 正式评估方式 | 按时间锚点评估，只用锚点前信息预测锚点后未来 30 天 |
| 6 月处理 | 暂不正式评估，等待完整 30 天标签后做 post-freeze monitoring |

## 核心指标

### 1. 旧官方 2025 四锚点

Phase7 official 四锚点为 `2025-09-01 / 2025-10-01 / 2025-11-01 / 2025-12-01`。

| 口径 | anchor 数 | mean global_wmape | mean global_ratio | blockbuster_under_wape | rank_corr_positive_skus |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase7 official | 4 | 0.6863 | 1.0159 | 0.4165 | 0.8044 |
| purchase_request_plus | 4 | 0.6770 | 1.0111 | 0.4131 | 0.8025 |
| event_intent_plus | 3 | 0.6350 | 1.0003 | 0.3469 | 0.8180 |

解释：

- 求购信息在完整四锚点上有小幅收益：`0.6863 -> 0.6770`，改善约 `1.36%`。
- Event 因为从 `2025-09-18` 才开始，只能覆盖 2025 年后三个官方锚点；在可覆盖锚点上从 `0.6678 -> 0.6350`，改善约 `4.91%`。
- 这部分证明 Phase8 新信号在 2025 不是无效噪声。

### 2. 2026 旧口径延伸

`phase8_2026_out_of_time` 是最接近 Phase7 原评估方式的 2026 延伸实验：单锚点、同类 LightGBM、`seed=2028`、`qty_gate=0.27`、同指标。

| 分支 | anchor 数 | mean global_wmape | mean global_ratio | 相对 Phase7 改善 |
| --- | ---: | ---: | ---: | ---: |
| phase7_base | 3 | 1.0581 | 1.0915 | 0.00% |
| listing_stage_interactions | 3 | 0.9614 | 1.0023 | 9.14% |
| peer_prior_46_90 | 3 | 0.9662 | 1.0085 | 8.69% |

解释：

- 上市日期/生命周期阶段是 2026 中最明确的新信号。
- `global_ratio` 从 `1.0915` 拉到 `1.0023`，说明 Phase7 在 2026 的整体高估被明显校正。

### 3. Phase8 formal stage2 多锚点评估

formal stage2 覆盖 14 个锚点：

`2025-09-01 / 2025-10-01 / 2025-11-01 / 2025-12-01 / 2026-01-01 / 2026-01-15 / 2026-02-01 / 2026-02-15 / 2026-03-01 / 2026-03-15 / 2026-04-01 / 2026-04-15 / 2026-05-01 / 2026-05-14`

全 14 锚点均值：

| 分支 | mean global_wmape | 相比 Phase7 | 相对改善 | mean global_ratio | blockbuster_under_wape | zero_true_pred_ge_3_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| phase7_listing_eligible | 1.1176 | 0.0000 | 0.00% | 0.8140 | 0.7540 | 0.0324 |
| legacy_combo_listing_eligible | 1.0750 | -0.0426 | 3.81% | 0.8177 | 0.7126 | 0.0274 |
| rich_combo_listing_eligible | 1.0795 | -0.0381 | 3.41% | 0.8144 | 0.7177 | 0.0288 |

分时间段：

| 时间段 | Phase7 WMAPE | legacy_combo WMAPE | rich_combo WMAPE | legacy 相对改善 |
| --- | ---: | ---: | ---: | ---: |
| 2025 四锚点 | 1.1824 | 1.1699 | 1.1709 | 1.05% |
| 2026 十锚点 | 1.0916 | 1.0370 | 1.0430 | 5.00% |
| 2026 Jan-Mar 六锚点 | 1.1210 | 1.0838 | 1.0871 | 3.32% |
| 2026 Apr-May 四锚点 | 1.0476 | 0.9669 | 0.9767 | 7.71% |

解释：

- `legacy_combo_listing_eligible` 在 broad-anchor formal 评估里是当前最稳的分支。
- 2026 的改善明显强于 2025，说明 Phase8 新信号更适合新数据环境。
- `rich_combo` 也优于 Phase7，但略弱于 `legacy_combo`，说明不是特征越多越好。

## 相比 Phase7 多做了什么

### 1. 数据源和语义

- Phase7 主要使用原有订单、商品、埋点等数据链。
- Phase8 明确把 `V_IRS_ORDERFTP` 作为 canonical 订单流水源。
- 甲方已确认 `QTY < 0` 是冲单，负数总是对应原始正订单。
- 甲方已确认重复行代表多次下单操作，不应简单去重。
- `TYPE` 缺失属于历史脏数据，已做逻辑覆盖，后续不会再新增为空。
- 新接入 `2026-06-14` 数据快照，订单覆盖到 `2026-06-14`。

### 2. 生命周期/上市日期

- 新商品表提供 `listing_date`。
- Phase8 构造上市天数、上市阶段、冷启动阶段、生命周期交互特征。
- 这个方向在 2026 out-of-time 上收益最明显：`1.0581 -> 0.9614`。

### 3. Event/埋点

- Phase8 将用户行为和商品热度信号作为 SKU/SKC 级辅助特征。
- Event 在 2025 可覆盖三锚点上改善明显：`0.6678 -> 0.6350`。
- Event 不能覆盖 2025-09-01，因为 Event 起始日期为 `2025-09-18`。

### 4. 库存快照

- Phase8 接入每日库存快照。
- 库存不只是提升 WMAPE，更重要的是帮助解释缺货、long-zero、库存为 0 但真实需求存在的问题。
- 库存快照从 2026-01 下旬开始覆盖，因此更适合作为 2026 后续稳定信号。

### 5. 求购信息

- Phase8 加入求购/request 相关特征。
- 2025 完整四锚点上有小幅提升：`0.6863 -> 0.6770`。
- robust 评估里，求购和库存/Event 组合对压误报和减少 blockbuster under 有帮助。

### 6. 零膨胀处理

- Phase7 已经是两阶段思路：先判断未来 30 天是否正向补货，再预测补货量。
- Phase8 进一步探索 `zero_split`，把短期零和长期零样本分开看。
- 两锚点探索中，`event_inventory_zero_split` 从 `0.6667` 进一步到 `asym_mild_cls_listing_targeted_reg` 的 `0.6584`，并把 long-zero 误报从 `0.7394` 降到 `0.6408`。
- 该方向有效，但目前只作为探索，不作为 formal 主线。

### 7. 评估体系

- Phase7 主要围绕 2025 官方四锚点。
- Phase8 扩展到 2025、2026 Jan-Mar、2026 Apr-May、two-anchor exploration、robust 复核等多组评估。
- 评估方式强调按时间锚点预测未来 30 天，而不是普通随机划分，避免未来信息泄漏。

## 为什么不等 6 月

Phase8 定版不需要等 6 月完整标签。

原因：

- 当前已经覆盖 2025-09 到 2026-05 的正式和半正式评估。
- 2026 Jan-Mar 与 2026 Apr-May 都显示 Phase8 优于 Phase7。
- 6 月当前只到 `2026-06-14`，无法形成完整未来 30 天标签。
- 强行评估 6 月会系统性低估真实补货量，反而不公平。

6 月后续处理：

- `2026-06-01` 锚点需要真实订单覆盖到 `2026-07-01`。
- `2026-06-14` 锚点需要真实订单覆盖到 `2026-07-14`。
- 等数据完整后，只做定版后的监控验证；除非出现严重反向结果，否则不影响当前 Phase8 freeze。

## 分支处理建议

| 分支 | 当前处理 |
| --- | --- |
| `legacy_combo_listing_eligible` | 定为 Phase8 formal 主线候选 |
| `rich_combo_listing_eligible` | 保留为备选，不优先定主线 |
| `listing_stage_interactions` | 作为上市日期有效性的关键证据 |
| `purchase_request_plus/core` | 作为求购信息有效性证据 |
| `event_intent_plus` | 作为 Event 有效性证据 |
| `event_inventory_request_core` | 作为 robust 稳健性证据 |
| `asym_mild_cls_listing_targeted_reg` | 后续探索方向，不直接替代 formal 主线 |

## 对外汇报口径

建议直接讲：

> Phase8 相比 Phase7 主要完成了数据源更新、生命周期/上市日期接入、库存快照接入、Event 和求购信号验证，以及更严格的多锚点评估。旧官方四锚点上，求购分支有小幅提升；Event 在可覆盖锚点上提升明显。把评估延伸到 2026 后，上市日期/生命周期特征使 Jan-Mar WMAPE 从 1.0581 降到 0.9614，改善约 9.14%；formal 多锚点评估中，当前主线 `legacy_combo_listing_eligible` 在 14 个锚点上相对 Phase7 改善 3.81%，其中 2026 十锚点改善 5.00%，4/5 月四锚点改善 7.71%。因此 Phase8 可以作为当前阶段最优候选主线定版。

## 后续工作

- 工程上冻结 Phase8 formal 候选：`legacy_combo_listing_eligible`。
- 汇报材料统一使用本报告中的三层证据：2025 official shadow、2026 out-of-time、formal stage2 broad-anchor。
- 6 月完整标签到齐后，补做 post-freeze monitoring。
- 如果甲方继续提供生命周期末期、计划下架、生产周期、成本售价等字段，再启动 Phase9 或盈亏分析增强，不再混入 Phase8 定版。

## 数据来源

- `reports/phase8a_prep/phase8_current_mainline_anchor_eval.csv`
- `reports/phase8_purchase_request_shadow/phase8_purchase_request_shadow_anchor_table.csv`
- `reports/phase8_event_shadow/phase8_event_shadow_anchor_table.csv`
- `reports/phase8_2026_out_of_time/phase8_2026_out_of_time_mean_compare.csv`
- `reports/phase8_formal_stage2_20260614/phase8_stage2_anchor_metrics.csv`
- `reports/phase8_formal_stage2_20260614/phase8_stage2_summary.csv`
- `reports/phase8_listing_zero_split_shadow_2026/phase8_listing_zero_split_candidate_summary.csv`
- `reports/current/phase8_official_style_2026_evaluation_20260616.md`
