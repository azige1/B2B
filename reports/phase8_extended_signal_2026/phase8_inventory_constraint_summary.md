# Phase8 Inventory Constraint 2026 Summary

- Status: `analysis_only`
- Source: `phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_row_compare.csv`
- Purpose: assess whether current inventory-aware shadow can distinguish `true=0` rows that may be stock-constrained rather than true no-demand.

## Bottom Line

- 当前正式 `phase7` 主线仍然不能识别库存约束，因为它没有使用库存特征。
- `event + inventory` 影子线已经能利用**正库存信号**改善预测，但还不能严格识别“明确缺货导致未补货”。
- 原因是当前库存特征表没有保留“存在快照但库存为 0”的独立状态；`snapshot_zero_stock` 在影子明细里为 0 行。

## Zero-True by Stock State

| stock_state | rows | base_fp_rate | shadow_fp_rate | fp_rate_delta | event_strong_rate | lookback_repl_pos_rate | qfo_ge_25_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_stock_signal | 8669 | 0.1459 | 0.0816 | -0.0644 | 0.0239 | 0.2779 | 0.4237 |
| snapshot_positive_stock | 7624 | 0.2635 | 0.2529 | -0.0106 | 0.0353 | 0.3342 | 0.6616 |

- `true=0` 且 `no_stock_signal` 的行共有 `8669`，`base_fp_rate=0.1459`，`shadow_fp_rate=0.0816`。
- `true=0` 且 `snapshot_positive_stock` 的行共有 `7624`，`base_fp_rate=0.2635`，`shadow_fp_rate=0.2529`。

## Positive-True Under-Predict by Stock State

| stock_state | rows | true_sum | base_under_wape | shadow_under_wape | under_wape_delta | event_strong_rate | mean_total_stock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| snapshot_positive_stock | 3104 | 13453.0001 | 0.3497 | 0.2571 | -0.0926 | 0.2368 | 21.3611 |
| no_stock_signal | 883 | 2915.0000 | 0.5508 | 0.5706 | 0.0198 | 0.1427 | 0.0000 |

## Inventory Source Audit

- inventory_daily_rows: `562552`
- inventory_distinct_days: `76`
- duplicate `date+sku` rows in inventory_daily_features: `66749`
- inventory rows with `qty_storage_stock = 0`: `72994`
- inventory rows with `qty_b2b_hq_stock = 0`: `383239`
- wide `qty_stock > 0` rate after `2026-01-23`: `0.8252`
- wide `is_real_stock > 0` rate after `2026-01-23`: `0.8252`

## Interpretation

- `snapshot_positive_stock` 可以被识别，这也是 `event + inventory` 影子线在 2026 两锚点上明显改善的原因之一。
- `no_stock_signal` 目前不能直接解释成缺货。它可能是：没有快照、快照漏匹配、或真实 0 库存被当前 presence 逻辑吞掉。
- 因为 `snapshot_zero_stock` 为 0 行，当前还不能回答“明确有快照且库存为 0 时模型怎么判断”。

## Practical Answer

- 现在模型**能部分区分**“有库存支撑的需求”和“无库存信号的样本”。
- 现在模型**还不能严格区分**“没补货是因为缺货”与“没补货是因为真没需求”。
- 如果要把这件事做严，需要下一步修库存特征生成逻辑，保留**原始快照存在性**与**0 库存状态**，而不是把 presence 直接定义成 `qty > 0`。

## Output Files

- `reports/phase8_extended_signal_2026/phase8_inventory_constraint_zero_true_table.csv`
- `reports/phase8_extended_signal_2026/phase8_inventory_constraint_positive_true_table.csv`
- `reports/phase8_extended_signal_2026/phase8_inventory_constraint_candidate_cases.csv`
