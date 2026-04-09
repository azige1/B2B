# Phase6f Tree Family Compare

- current_mainline: `sep098_oct093`
- freeze_current_tree_mainline: `True`
- promoted_backend: `None`

## Stage A

- syntax_ok: `True`
- `catboost` import/backend_smoke: `True` / `True`
- `xgboost` import/backend_smoke: `True` / `True`

## Stage B Single-Anchor Smoke

| candidate_key | backend | smoke_gate_pass | forced_to_stage_c | advance_to_stage_c | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_raw_g025 | lightgbm | True | False | True | 0.9636 | 0.4512 | 0.5898 | 0.4586 | 0.6642 | 0.3073 | 0.6754 | 0.6957 |
| catboost_raw_g025 | catboost | False | False | False | 1.0852 | 0.4646 | 0.5801 | 0.4417 | 0.7087 | 0.2468 | 0.6291 | 0.6125 |
| xgboost_raw_g025 | xgboost | False | True | True | 1.0237 | 0.4474 | 0.5914 | 0.4578 | 0.6614 | 0.2973 | 0.6781 | 0.6771 |

## Stage C Raw Four-Anchor Compare

| candidate_key | backend | raw_stage_pass | forced_to_stage_d | advance_to_stage_d | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_raw_g025 | lightgbm | True | False | True | 1.0449 | 0.4130 | 0.6123 | 0.5186 | 0.5405 | 0.4149 | 0.6213 | 0.7365 |
| xgboost_raw_g025 | xgboost | False | True | True | 1.1148 | 0.4205 | 0.6084 | 0.5121 | 0.5440 | 0.3941 | 0.6172 | 0.7166 |

## Stage D Calibrated Final Compare

| candidate_key | backend | replacement_gate | anchor_passes | global_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm_sep098_oct093 | lightgbm | False | 4 | 1.0188 | 0.4236 | 0.5986 | 0.5069 | 0.5507 | 0.4048 | 0.6213 | 0.7365 |
| xgboost_sep098_oct093 | xgboost | False | 2 | 1.0862 | 0.4310 | 0.5945 | 0.5004 | 0.5543 | 0.3847 | 0.6172 | 0.7166 |