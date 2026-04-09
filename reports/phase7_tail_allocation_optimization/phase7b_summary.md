# Phase7b Tail Feature Sweep

- winner_action: `freeze_current_tree_mainline`

## Raw Four-Anchor Compare

| candidate_key | track | raw_stage_pass | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_raw_g025 | current_raw_mainline | False | 1.0449 | 0.4130 | 0.6123 | 0.5186 | 0.5405 | 0.4149 | 0.6213 | 0.7365 |
| qfo_plus | cov_activity_qfo | False | 0.9636 | 0.4512 | 0.5898 | 0.4586 | 0.6642 | 0.3073 | 0.6754 | 0.6957 |
| style_category_priors | cov_activity_priors | False | 0.9636 | 0.4512 | 0.5898 | 0.4586 | 0.6642 | 0.3073 | 0.6754 | 0.6957 |
| tail_full | cov_activity_tail_full | False | 0.9636 | 0.4512 | 0.5898 | 0.4586 | 0.6642 | 0.3073 | 0.6754 | 0.6957 |
| tail_peak | cov_activity_tail | False | 0.9636 | 0.4512 | 0.5898 | 0.4586 | 0.6642 | 0.3073 | 0.6754 | 0.6957 |

## Calibrated Final Compare

| candidate_key | track | replacement_gate | anchor_passes | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_sep098_oct093 | current_calibrated_mainline | False | 4 | 1.0188 | 0.4236 | 0.5986 | 0.5069 | 0.5507 | 0.4048 | 0.6213 | 0.7365 |
| qfo_plus | cov_activity_qfo | False | 3 | 0.9419 | 0.4613 | 0.5765 | 0.4483 | 0.6705 | 0.3004 | 0.6754 | 0.6957 |
| style_category_priors | cov_activity_priors | False | 3 | 0.9419 | 0.4613 | 0.5765 | 0.4483 | 0.6705 | 0.3004 | 0.6754 | 0.6957 |
| tail_full | cov_activity_tail_full | False | 3 | 0.9419 | 0.4613 | 0.5765 | 0.4483 | 0.6705 | 0.3004 | 0.6754 | 0.6957 |
| tail_peak | cov_activity_tail | False | 3 | 0.9419 | 0.4613 | 0.5765 | 0.4483 | 0.6705 | 0.3004 | 0.6754 | 0.6957 |