# STAGE_S0 Summary

- baseline_mainline: `p57_covact_lr005_l63_hard_g025 + sep098_oct093`

## Raw Compare

| candidate_key | track | raw_stage_pass | tail_gain_like | allocation_gain_like | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_raw_g025 | current_raw_mainline | False | False | False | 1.0449 | 0.4130 | 0.6123 | 0.5186 | 0.5405 | 0.4149 | 0.6213 | 0.7365 |
| qfo_plus_lr005_l63_g025_n400_s2026 | cov_activity_qfo | False | False | True | 0.9742 | 0.4447 | 0.5844 | 0.4629 | 0.6626 | 0.3038 | 0.6774 | 0.6951 |

## Calibrated Compare

| candidate_key | track | replacement_gate_like | potential_gate_like | anchor_passes | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus | 1_3_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_sep098_oct093 | current_calibrated_mainline | False | False | 4 | 1.0188 | 0.4236 | 0.5986 | 0.5069 | 0.5507 | 0.4048 | 0.6213 | 0.7365 | 1.3801 |
| qfo_plus_lr005_l63_g025_n400_s2026 | cov_activity_qfo | False | False | 1 | 0.9742 | 0.4447 | 0.5844 | 0.4629 | 0.6626 | 0.3038 | 0.6774 | 0.6951 | 1.4744 |