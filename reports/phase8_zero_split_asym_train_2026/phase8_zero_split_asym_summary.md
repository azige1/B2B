# Phase8 Zero-Split Asymmetric Training Summary

- Status: `analysis_only_shadow`
- Baseline: `event_inventory_zero_split`
- Scope: `2026-02-15 / 2026-02-24`
- Method: model-internal asymmetric weighting for `short_zero / long_zero` only
- No new data, no label change, no additional post-processing

## Decision Rule

- mean `global_wmape <= 0.6667`
- mean `blockbuster_under_wape <= 0.3635`
- `long_zero zero_true_fp_rate < 0.7361`
- `long_zero positive_true_under_wape <= 0.1712`

## Candidate Table

| candidate | mean_global_wmape | mean_blockbuster_under_wape | long_zero_zero_true_fp_rate | long_zero_positive_true_under_wape | passes_all |
| --- | ---: | ---: | ---: | ---: | --- |
| event_inventory_zero_split | 0.6667 | 0.3635 | 0.7394 | 0.1588 | False |
| zero_split_asym_mild | 0.6663 | 0.3688 | 0.6408 | 0.1570 | False |
| zero_split_asym_balanced | 0.6806 | 0.3667 | 0.7394 | 0.1580 | False |
| zero_split_asym_strong | 0.6722 | 0.3624 | 0.5563 | 0.1599 | False |

## Best Candidate

- `candidate = zero_split_asym_strong`
- `passes_all = False`
- `mean_global_wmape = 0.6722`
- `mean_blockbuster_under_wape = 0.3624`
- `long_zero_zero_true_fp_rate = 0.5563`
- `long_zero_positive_true_under_wape = 0.1599`

## Recommendation

- Do not promote the asymmetric variant yet.
- Keep `event_inventory_zero_split` as the current best exploratory baseline.
- Pause further optimization on this line until client-side data semantics are clarified.
