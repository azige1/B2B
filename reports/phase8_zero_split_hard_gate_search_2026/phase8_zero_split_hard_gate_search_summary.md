# Phase8 Zero-Split Hard Gate Search Summary

- Status: `analysis_only_shadow`
- Scope: `2026-02-15 / 2026-02-24`
- Base variant for this search: `event_inventory_zero_split`
- Method: asymmetric hard-gate post-processing on `short_zero / long_zero` rows

## Best Rule

- `short_zero_gate_qty = 0.50`
- `long_zero_gate_qty = 1.00`
- `long_zero_signal_gate_qty = 0.00`
- `long_zero_min_streak = 21`

## Baseline vs Best Rule

| metric | zero_split_baseline | best_rule | delta |
| --- | --- | --- | --- |
| mean_global_wmape | 0.6667 | 0.6668 | +0.0002 |
| mean_4_25_under_wape | 0.3312 | 0.3312 | +0.0000 |
| mean_blockbuster_under_wape | 0.3635 | 0.3635 | +0.0000 |
| mean_rank_corr_positive_skus | 0.7642 | 0.7643 | +0.0001 |
| mean_top20_true_volume_capture | 0.5725 | 0.5725 | +0.0000 |
| gate_rule_rate | 0.0018 | 0.0018 | +0.0000 |
| long_zero_zero_true_fp_rate | 0.7394 | 0.7324 | -0.0070 |
| long_zero_positive_true_under_wape | 0.1612 | 0.1883 | +0.0271 |
| short_zero_zero_true_fp_rate | 0.5000 | 0.5000 | +0.0000 |
| short_zero_positive_true_under_wape | 0.2117 | 0.2117 | +0.0000 |

## Best Rule Anchor Metrics

| anchor_date | global_wmape | 4_25_under_wape | blockbuster_under_wape | rank_corr_positive_skus | top20_true_volume_capture |
| --- | --- | --- | --- | --- | --- |
| 2026-02-15 | 0.6587 | 0.2861 | 0.2899 | 0.7461 | 0.6014 |
| 2026-02-24 | 0.6749 | 0.3763 | 0.4371 | 0.7825 | 0.5435 |

## Best Rule State Metrics

| stock_state | rows | zero_true_rows | zero_true_fp_rate | positive_true_rows | positive_true_under_wape |
| --- | --- | --- | --- | --- | --- |
| long_zero | 234 | 142 | 0.7324 | 92 | 0.1883 |
| other | 20038 | 16149 | 0.1679 | 3889 | 0.2986 |
| short_zero | 8 | 2 | 0.5000 | 6 | 0.2117 |

## Interpretation

- This experiment is useful only if it actually changes the zero-true false-positive rate on `long_zero` rows.
- If it reduces false positives but destroys positive-true rows, it should remain diagnostic only.
