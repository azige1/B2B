# Phase 5.7 Summary

- baseline_sequence: `p527_lstm_l3_v5_lite_s2027`
- confirmed_tree_main: `p535_tree_hard_cov_activity`
- confirmed_tree_backup: `p531_tree_hard_core`
- recommended_candidate: `p57_covact_lr005_l63_hard_g025`
- delivery_ready: `False`
- next_stage: `phase6_tree_stabilization`

| candidate_key | track | anchor_passes | global_ratio | global_wmape | 4_25_sku_p50 | 4_25_under_wape | ice_4_25_sku_p50 | 1_3_ratio | blockbuster_sku_p50 | blockbuster_under_wape | top20_true_volume_capture | rank_corr_positive_skus | auc | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p531 | event_tree_backup | 2 | 1.1717 | 1.0317 | 0.5866 | 0.4365 | 0.5015 | 1.4981 | 0.3564 | 0.5794 | 0.6081 | 0.6918 | 0.9437 | 0.7287 |
| p535 | event_tree_confirmed | 2 | 1.1791 | 1.0364 | 0.5864 | 0.4345 | 0.5009 | 1.5038 | 0.3561 | 0.5791 | 0.6080 | 0.6913 | 0.9432 | 0.7272 |
| p57_covact_lr005_l63_hard_g025 | event_tree_tuned | 2 | 1.0449 | 0.8593 | 0.6123 | 0.4130 | 0.5186 | 1.4096 | 0.4149 | 0.5405 | 0.6213 | 0.7365 | 0.9552 | 0.7558 |
| p57_covact_lr005_l63_hard_g020 | event_tree_tuned | 2 | 1.1067 | 0.9157 | 0.6123 | 0.4127 | 0.5186 | 1.4324 | 0.4149 | 0.5405 | 0.6208 | 0.7359 | 0.9552 | 0.7558 |
| p527 | sequence_baseline | 0 | 1.0451 | 1.4735 | 0.2329 | 0.7419 | 0.0865 | 0.9109 | 0.1000 | 0.8649 | 0.3868 | 0.2669 | 0.7407 | 0.5212 |