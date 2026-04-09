# Phase 5.3 Decision Table

Business priority:
1. `4-25`
2. `Ice / Ice 4-25`
3. `global_ratio / global_wmape`
4. `auc / f1`

| exp_id | track | version | config | auc | f1 | global_ratio | global_wmape | 4_25_ratio | 4_25_wmape_like | 4_25_sku_p50 | ice_sku_p50 | ice_4_25_sku_p50 | blockbuster_sku_p50 | eligible_vs_p527 | status | elapsed_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p531_tree_hard_core | event_tree | v6_event | hard_core | 0.9390 | 0.7081 | 1.0559 | 1.0374 | 0.5950 | 0.5174 | 0.5600 | 0.8039 | 0.4445 | 0.2776 | True | success | 0.72 |
| p533_tree_hard_cov | event_tree | v6_event | hard_cov | 0.9388 | 0.7089 | 1.0685 | 1.0457 | 0.5961 | 0.5182 | 0.5555 | 0.8058 | 0.4529 | 0.2861 | True | success | 0.6 |
| p535_tree_hard_cov_activity | event_tree | v6_event | hard_cov_activity | 0.9377 | 0.7063 | 1.0715 | 1.0541 | 0.5904 | 0.5197 | 0.5642 | 0.8030 | 0.4396 | 0.2746 | True | success | 0.65 |
| p532_tree_soft_core | event_tree | v6_event | soft_core | 0.9390 | 0.7081 | 0.8659 | 0.9306 | 0.5456 | 0.5581 | 0.4989 | 0.5236 | 0.3718 | 0.2405 | False | success | 0.69 |
| p534_tree_soft_cov | event_tree | v6_event | soft_cov | 0.9388 | 0.7089 | 0.8731 | 0.9329 | 0.5440 | 0.5603 | 0.4999 | 0.5384 | 0.3739 | 0.2482 | False | success | 0.73 |
| p536_tree_soft_cov_activity | event_tree | v6_event | soft_cov_activity | 0.9377 | 0.7063 | 0.8760 | 0.9449 | 0.5379 | 0.5626 | 0.5005 | 0.5198 | 0.3699 | 0.2313 | False | success | 0.67 |
| p539_attn_v5_lite_cov | sequence | v5_lite_cov | attn | 0.7125 | 0.4539 | 1.0683 | 1.5393 | 0.2832 | 0.7682 | 0.2217 | 0.1516 | 0.0950 | 0.0421 | False | success | 41.25 |
| p537_lstm_pool_v5_lite_cov | sequence | v5_lite_cov | lstm_pool | 0.7374 | 0.4900 | 1.1377 | 1.5878 | 0.2876 | 0.7519 | 0.2386 | 0.1686 | 0.1069 | 0.0503 | False | success | 45.8 |

## Winners

```json
{
  "baseline_sequence": "p527_lstm_l3_v5_lite_s2027",
  "main_line": "p535_tree_hard_cov_activity",
  "event_tree_candidates": [
    "p535_tree_hard_cov_activity",
    "p531_tree_hard_core"
  ],
  "sequence_keep": [
    "p527_lstm_l3_v5_lite_s2027"
  ],
  "promote_phase54": [
    "p535_tree_hard_cov_activity",
    "p531_tree_hard_core",
    "p527_lstm_l3_v5_lite_s2027"
  ],
  "decision": "shift_to_event_tree"
}
```