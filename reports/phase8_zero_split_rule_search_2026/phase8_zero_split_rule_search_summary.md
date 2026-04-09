# Phase8 Zero-Split Rule Search Summary

- Status: `analysis_only_shadow`
- Scope: `2026-02-15 / 2026-02-24`
- Base variant for this search: `event_inventory_zero_split`
- Method: asymmetric suppressive post-processing on `short_zero / long_zero` rows

## Best Rule

- `short_zero_scale = 0.80`
- `short_zero_signal_scale = 0.90`
- `long_zero_scale = 0.40`
- `long_zero_signal_scale = 0.75`

## Baseline vs Best Rule

| metric | zero_split_baseline | best_rule | delta |
| --- | --- | --- | --- |
| mean_global_wmape | 0.6667 | 0.6640 | -0.0026 |
| mean_4_25_under_wape | 0.3312 | 0.3315 | +0.0003 |
| mean_blockbuster_under_wape | 0.3635 | 0.3635 | +0.0000 |
| mean_rank_corr_positive_skus | 0.7642 | 0.7638 | -0.0005 |
| mean_top20_true_volume_capture | 0.5725 | 0.5722 | -0.0003 |
| long_zero_zero_true_fp_rate | 0.7394 | 0.7394 | +0.0000 |
| long_zero_positive_true_under_wape | 0.1612 | 0.3567 | +0.1955 |
| short_zero_zero_true_fp_rate | 0.5000 | 0.5000 | +0.0000 |
| short_zero_positive_true_under_wape | 0.2117 | 0.2620 | +0.0503 |

## Best Rule Anchor Metrics

| anchor_date | global_wmape | 4_25_under_wape | blockbuster_under_wape | rank_corr_positive_skus | top20_true_volume_capture |
| --- | --- | --- | --- | --- | --- |
| 2026-02-15 | 0.6547 | 0.2863 | 0.2899 | 0.7463 | 0.6012 |
| 2026-02-24 | 0.6734 | 0.3766 | 0.4371 | 0.7812 | 0.5431 |

## Best Rule State Metrics

| stock_state | rows | zero_true_rows | zero_true_fp_rate | positive_true_rows | positive_true_under_wape |
| --- | --- | --- | --- | --- | --- |
| long_zero | 234 | 142 | 0.7394 | 92 | 0.3567 |
| other | 20038 | 16149 | 0.1679 | 3889 | 0.2986 |
| short_zero | 8 | 2 | 0.5000 | 6 | 0.2620 |

## Interpretation

- This experiment only tests whether a lightweight rule can improve the best current zero-split shadow.
- If the best rule improves both long-zero false positives and long-zero positive-true under-predict, the next step should be to operationalize it more cleanly.
- If gains are tiny or unstable, keep the rule as a diagnostic only.
