# Phase 5.5 Decision Table

Business priority:
1. `4-25`
2. `Ice / Ice 4-25`
3. `global_ratio / global_wmape`
4. `AUC / F1`

- baseline_sequence: `p527_lstm_l3_v5_lite_s2027`
- main_line: `p531_tree_hard_core`
- anchor_wins: `{'p531_tree_hard_core': 1, 'p535_tree_hard_cov_activity': 1}`

| anchor_date | exp_id | track | base_exp | auc | f1 | global_ratio | global_wmape | 4_25_ratio | 4_25_wmape_like | 4_25_sku_p50 | ice_sku_p50 | ice_4_25_sku_p50 | blockbuster_sku_p50 | eligible_vs_p527 | status | elapsed_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2025-09-01 | p55_20250901_p531_tree_hard_core_s2026 | event_tree | p531_tree_hard_core | 0.9406 | 0.7389 | 1.2511 | 1.1100 | 0.6975 | 0.4536 | 0.6403 | 0.9983 | 0.5824 | 0.2873 | False | success | 0.4800 |
| 2025-09-01 | p55_20250901_p535_tree_hard_cov_activity_s2026 | event_tree | p535_tree_hard_cov_activity | 0.9411 | 0.7352 | 1.2738 | 1.1211 | 0.7115 | 0.4484 | 0.6584 | 0.9899 | 0.5853 | 0.2905 | False | success | 0.5000 |
| 2025-09-01 | p55_20250901_p527_lstm_l3_v5_lite_s2027_s2026 | sequence | p527_lstm_l3_v5_lite_s2027 | 0.7965 | 0.5519 | 0.7543 | 1.2122 | 0.2751 | 0.7309 | 0.2724 | 0.1996 | 0.1203 | 0.0443 | False | success | 28.5300 |
| 2025-10-01 | p55_20251001_p531_tree_hard_core_s2026 | event_tree | p531_tree_hard_core | 0.9390 | 0.6896 | 1.3818 | 1.1688 | 0.6152 | 0.4929 | 0.5809 | 0.9608 | 0.4888 | 0.4263 | False | success | 0.5000 |
| 2025-10-01 | p55_20251001_p535_tree_hard_cov_activity_s2026 | event_tree | p535_tree_hard_cov_activity | 0.9376 | 0.6890 | 1.3754 | 1.1717 | 0.6128 | 0.4928 | 0.5634 | 0.9566 | 0.4942 | 0.4049 | False | success | 8.2400 |
| 2025-10-01 | p55_20251001_p527_lstm_l3_v5_lite_s2027_s2026 | sequence | p527_lstm_l3_v5_lite_s2027 | 0.7363 | 0.5164 | 1.2329 | 1.6077 | 0.2732 | 0.7391 | 0.2286 | 0.3296 | 0.1287 | 0.1455 | False | success | 34.1600 |
| 2025-11-01 | p55_20251101_p531_tree_hard_core_s2026 | event_tree | p531_tree_hard_core | 0.9561 | 0.7781 | 0.9982 | 0.8105 | 0.5814 | 0.5168 | 0.5654 | 0.7954 | 0.4901 | 0.4343 | True | success | 0.5800 |
| 2025-11-01 | p55_20251101_p535_tree_hard_cov_activity_s2026 | event_tree | p535_tree_hard_cov_activity | 0.9562 | 0.7782 | 0.9957 | 0.7989 | 0.5799 | 0.5095 | 0.5598 | 0.7926 | 0.4843 | 0.4545 | True | success | 9.7600 |
| 2025-11-01 | p55_20251101_p527_lstm_l3_v5_lite_s2027_s2026 | sequence | p527_lstm_l3_v5_lite_s2027 | 0.6966 | 0.5140 | 0.9988 | 1.4186 | 0.2376 | 0.7752 | 0.1874 | 0.0000 | 0.0000 | 0.1618 | False | success | 52.2300 |
| 2025-12-01 | p55_20251201_p531_tree_hard_core_s2026 | event_tree | p531_tree_hard_core | 0.9390 | 0.7081 | 1.0559 | 1.0374 | 0.5950 | 0.5174 | 0.5600 | 0.8039 | 0.4445 | 0.2776 | True | success | 0.6300 |
| 2025-12-01 | p55_20251201_p535_tree_hard_cov_activity_s2026 | event_tree | p535_tree_hard_cov_activity | 0.9377 | 0.7063 | 1.0715 | 1.0541 | 0.5904 | 0.5197 | 0.5642 | 0.8030 | 0.4396 | 0.2746 | True | success | 0.6900 |
| 2025-12-01 | p54_p527_lstm_l3_v5_lite_s2027_s2026 | sequence | p527_lstm_l3_v5_lite_s2027 | 0.7334 | 0.5023 | 1.1944 | 1.6556 | 0.2718 | 0.7483 | 0.2433 | 0.1532 | 0.0970 | 0.0486 | False | fallback_phase54 |  |