# STAGE_S3 Summary

- baseline_mainline: `p57_covact_lr005_l63_hard_g025 + sep098_oct093`

## Raw Compare

| candidate_key | track | raw_stage_pass | tail_gain_like | allocation_gain_like | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_raw_g025 | current_raw_mainline | False | False | False | 1.0449 | 0.4130 | 0.6123 | 0.5186 | 0.5405 | 0.4149 | 0.6213 | 0.7365 |
| style_category_priors_lr003_l63_g027_n800_s2026 | cov_activity_priors | True | True | True | 1.0567 | 0.3792 | 0.6434 | 0.5747 | 0.4711 | 0.4903 | 0.6388 | 0.7631 |
| style_category_priors_lr003_l63_g027_n800_s2027 | cov_activity_priors | True | True | True | 1.0514 | 0.3786 | 0.6413 | 0.5860 | 0.4670 | 0.4905 | 0.6360 | 0.7629 |
| style_category_priors_lr003_l63_g027_n800_s2028 | cov_activity_priors | True | True | True | 1.0525 | 0.3798 | 0.6381 | 0.5723 | 0.4657 | 0.4900 | 0.6405 | 0.7629 |
| style_category_priors_lr005_l63_g025_n800_s2026 | cov_activity_priors | True | True | True | 1.0598 | 0.3480 | 0.6677 | 0.6050 | 0.4152 | 0.5623 | 0.6479 | 0.7966 |
| style_category_priors_lr005_l63_g025_n800_s2027 | cov_activity_priors | True | True | True | 1.0619 | 0.3447 | 0.6764 | 0.6115 | 0.4107 | 0.5522 | 0.6476 | 0.7977 |
| style_category_priors_lr005_l63_g025_n800_s2028 | cov_activity_priors | True | True | True | 1.0532 | 0.3481 | 0.6701 | 0.6131 | 0.4288 | 0.5401 | 0.6480 | 0.7970 |
| style_category_priors_lr005_l63_g027_n800_s2026 | cov_activity_priors | True | True | True | 1.0426 | 0.3480 | 0.6677 | 0.6050 | 0.4167 | 0.5623 | 0.6473 | 0.7964 |
| style_category_priors_lr005_l63_g027_n800_s2027 | cov_activity_priors | True | True | True | 1.0442 | 0.3448 | 0.6764 | 0.6115 | 0.4107 | 0.5522 | 0.6476 | 0.7973 |
| style_category_priors_lr005_l63_g027_n800_s2028 | cov_activity_priors | True | True | True | 1.0352 | 0.3481 | 0.6701 | 0.6131 | 0.4288 | 0.5401 | 0.6480 | 0.7962 |
| tail_full_lr003_l63_g027_n800_s2026 | cov_activity_tail_full | True | True | True | 1.0524 | 0.3756 | 0.6372 | 0.5836 | 0.4657 | 0.4888 | 0.6396 | 0.7669 |
| tail_full_lr003_l63_g027_n800_s2027 | cov_activity_tail_full | True | True | True | 1.0565 | 0.3771 | 0.6373 | 0.5741 | 0.4661 | 0.4988 | 0.6395 | 0.7654 |
| tail_full_lr003_l63_g027_n800_s2028 | cov_activity_tail_full | True | True | True | 1.0501 | 0.3755 | 0.6367 | 0.5709 | 0.4689 | 0.4754 | 0.6413 | 0.7687 |
| tail_full_lr005_l63_g025_n800_s2026 | cov_activity_tail_full | True | True | True | 1.0630 | 0.3417 | 0.6740 | 0.6154 | 0.4077 | 0.5603 | 0.6482 | 0.8020 |
| tail_full_lr005_l63_g025_n800_s2027 | cov_activity_tail_full | True | True | True | 1.0590 | 0.3417 | 0.6724 | 0.6099 | 0.4118 | 0.5535 | 0.6496 | 0.8020 |
| tail_full_lr005_l63_g025_n800_s2028 | cov_activity_tail_full | True | True | True | 1.0593 | 0.3446 | 0.6736 | 0.5982 | 0.4047 | 0.5665 | 0.6494 | 0.8040 |
| tail_full_lr005_l63_g027_n800_s2026 | cov_activity_tail_full | True | True | True | 1.0444 | 0.3417 | 0.6740 | 0.6154 | 0.4077 | 0.5603 | 0.6482 | 0.8028 |
| tail_full_lr005_l63_g027_n800_s2027 | cov_activity_tail_full | True | True | True | 1.0398 | 0.3417 | 0.6724 | 0.6099 | 0.4118 | 0.5535 | 0.6496 | 0.8022 |
| tail_full_lr005_l63_g027_n800_s2028 | cov_activity_tail_full | True | True | True | 1.0410 | 0.3446 | 0.6736 | 0.5982 | 0.4047 | 0.5665 | 0.6494 | 0.8044 |

## Calibrated Compare

| candidate_key | track | replacement_gate_like | potential_gate_like | anchor_passes | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus | 1_3_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_sep098_oct093 | current_calibrated_mainline | False | False | 4 | 1.0188 | 0.4236 | 0.5986 | 0.5069 | 0.5507 | 0.4048 | 0.6213 | 0.7365 | 1.3801 |
| style_category_priors_lr003_l63_g027_n800_s2026 | cov_activity_priors | True | True | 4 | 1.0306 | 0.3906 | 0.6289 | 0.5623 | 0.4817 | 0.4793 | 0.6388 | 0.7631 | 1.3378 |
| style_category_priors_lr003_l63_g027_n800_s2027 | cov_activity_priors | True | True | 4 | 1.0258 | 0.3898 | 0.6270 | 0.5735 | 0.4779 | 0.4799 | 0.6360 | 0.7629 | 1.3463 |
| style_category_priors_lr003_l63_g027_n800_s2028 | cov_activity_priors | True | True | 4 | 1.0264 | 0.3912 | 0.6238 | 0.5598 | 0.4766 | 0.4788 | 0.6405 | 0.7629 | 1.3430 |
| style_category_priors_lr005_l63_g025_n800_s2026 | cov_activity_priors | True | True | 4 | 1.0341 | 0.3599 | 0.6529 | 0.5918 | 0.4270 | 0.5498 | 0.6479 | 0.7966 | 1.3051 |
| style_category_priors_lr005_l63_g025_n800_s2027 | cov_activity_priors | True | True | 4 | 1.0363 | 0.3566 | 0.6615 | 0.5987 | 0.4216 | 0.5402 | 0.6476 | 0.7977 | 1.3103 |
| style_category_priors_lr005_l63_g025_n800_s2028 | cov_activity_priors | True | True | 4 | 1.0277 | 0.3600 | 0.6553 | 0.6001 | 0.4401 | 0.5283 | 0.6480 | 0.7970 | 1.3090 |
| style_category_priors_lr005_l63_g027_n800_s2026 | cov_activity_priors | True | True | 4 | 1.0173 | 0.3599 | 0.6529 | 0.5918 | 0.4284 | 0.5498 | 0.6473 | 0.7964 | 1.2993 |
| style_category_priors_lr005_l63_g027_n800_s2027 | cov_activity_priors | True | True | 4 | 1.0191 | 0.3567 | 0.6615 | 0.5987 | 0.4216 | 0.5402 | 0.6476 | 0.7973 | 1.3054 |
| style_category_priors_lr005_l63_g027_n800_s2028 | cov_activity_priors | True | True | 4 | 1.0102 | 0.3600 | 0.6553 | 0.6001 | 0.4401 | 0.5283 | 0.6480 | 0.7962 | 1.3011 |
| tail_full_lr003_l63_g027_n800_s2026 | cov_activity_tail_full | True | True | 4 | 1.0264 | 0.3869 | 0.6230 | 0.5711 | 0.4765 | 0.4781 | 0.6396 | 0.7669 | 1.3490 |
| tail_full_lr003_l63_g027_n800_s2027 | cov_activity_tail_full | False | False | 3 | 1.0303 | 0.3883 | 0.6232 | 0.5619 | 0.4771 | 0.4877 | 0.6395 | 0.7654 | 1.3470 |
| tail_full_lr003_l63_g027_n800_s2028 | cov_activity_tail_full | False | False | 3 | 1.0243 | 0.3865 | 0.6224 | 0.5587 | 0.4797 | 0.4648 | 0.6413 | 0.7687 | 1.3463 |
| tail_full_lr005_l63_g025_n800_s2026 | cov_activity_tail_full | True | True | 4 | 1.0371 | 0.3535 | 0.6589 | 0.6022 | 0.4201 | 0.5478 | 0.6482 | 0.8020 | 1.3079 |
| tail_full_lr005_l63_g025_n800_s2027 | cov_activity_tail_full | True | True | 4 | 1.0334 | 0.3534 | 0.6576 | 0.5970 | 0.4242 | 0.5410 | 0.6496 | 0.8020 | 1.3053 |
| tail_full_lr005_l63_g025_n800_s2028 | cov_activity_tail_full | True | True | 4 | 1.0335 | 0.3566 | 0.6588 | 0.5853 | 0.4165 | 0.5539 | 0.6494 | 0.8040 | 1.3049 |
| tail_full_lr005_l63_g027_n800_s2026 | cov_activity_tail_full | True | True | 4 | 1.0191 | 0.3535 | 0.6589 | 0.6022 | 0.4201 | 0.5478 | 0.6482 | 0.8028 | 1.3017 |
| tail_full_lr005_l63_g027_n800_s2027 | cov_activity_tail_full | True | True | 4 | 1.0147 | 0.3534 | 0.6576 | 0.5970 | 0.4242 | 0.5410 | 0.6496 | 0.8022 | 1.2988 |
| tail_full_lr005_l63_g027_n800_s2028 | cov_activity_tail_full | True | True | 4 | 1.0159 | 0.3566 | 0.6588 | 0.5853 | 0.4165 | 0.5539 | 0.6494 | 0.8044 | 1.2999 |