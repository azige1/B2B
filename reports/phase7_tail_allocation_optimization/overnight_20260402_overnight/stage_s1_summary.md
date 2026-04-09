# STAGE_S1 Summary

- baseline_mainline: `p57_covact_lr005_l63_hard_g025 + sep098_oct093`

## Raw Compare

| candidate_key | track | raw_stage_pass | tail_gain_like | allocation_gain_like | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_raw_g025 | current_raw_mainline | False | False | False | 1.0449 | 0.4130 | 0.6123 | 0.5186 | 0.5405 | 0.4149 | 0.6213 | 0.7365 |
| qfo_plus_lr005_l63_g025_n400_s2026 | cov_activity_qfo | False | False | False | 1.0386 | 0.4101 | 0.6155 | 0.5188 | 0.5419 | 0.4014 | 0.6247 | 0.7435 |
| style_category_priors_lr005_l63_g025_n400_s2026 | cov_activity_priors | True | True | False | 1.0802 | 0.3887 | 0.6258 | 0.5658 | 0.4860 | 0.4760 | 0.6322 | 0.7486 |
| tail_full_lr005_l63_g025_n400_s2026 | cov_activity_tail_full | True | True | False | 1.0799 | 0.3871 | 0.6276 | 0.5644 | 0.4841 | 0.4670 | 0.6335 | 0.7538 |
| tail_peak_lr005_l63_g025_n400_s2026 | cov_activity_tail | False | False | False | 1.0398 | 0.4152 | 0.6087 | 0.5214 | 0.5413 | 0.4093 | 0.6228 | 0.7339 |

## Calibrated Compare

| candidate_key | track | replacement_gate_like | potential_gate_like | anchor_passes | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus | 1_3_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_sep098_oct093 | current_calibrated_mainline | False | False | 4 | 1.0188 | 0.4236 | 0.5986 | 0.5069 | 0.5507 | 0.4048 | 0.6213 | 0.7365 | 1.3801 |
| qfo_plus_lr005_l63_g025_n400_s2026 | cov_activity_qfo | False | False | 4 | 1.0127 | 0.4208 | 0.6017 | 0.5073 | 0.5517 | 0.3917 | 0.6247 | 0.7435 | 1.3749 |
| style_category_priors_lr005_l63_g025_n400_s2026 | cov_activity_priors | False | False | 2 | 1.0534 | 0.3996 | 0.6117 | 0.5536 | 0.4966 | 0.4652 | 0.6322 | 0.7486 | 1.3723 |
| tail_full_lr005_l63_g025_n400_s2026 | cov_activity_tail_full | False | False | 2 | 1.0530 | 0.3981 | 0.6134 | 0.5523 | 0.4949 | 0.4563 | 0.6335 | 0.7538 | 1.3734 |
| tail_peak_lr005_l63_g025_n400_s2026 | cov_activity_tail | False | False | 4 | 1.0138 | 0.4259 | 0.5948 | 0.5096 | 0.5512 | 0.3993 | 0.6228 | 0.7339 | 1.3790 |