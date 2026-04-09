# Phase8 Snapshot Zero Stock Case Pack

- Status: `analysis_only`
- Scope: `2026-02-15 / 2026-02-24`
- Source: `phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_row_compare.csv`
- Purpose: isolate rows with `snapshot_present=1` and `stock_zero=1` to see whether the inventory line is learning more than a pure positive-stock signal.

## Bottom Line

- `snapshot_zero_stock` zero-true rows: `144`; false-positive rate improves from `0.9306` to `0.7569`.
- `snapshot_zero_stock` positive-true rows: `98`; under-WAPE moves from `0.0649` to `0.1689`.
- For comparison, `snapshot_positive_stock` positive-true rows improve from `0.3483` to `0.2408`.

## Anchor Mix

| anchor_date | true_bin | stock_state | rows | base_pred_mean | shadow_pred_mean | event_strong_rate | lookback_repl_pos_rate | qfo_ge_25_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | positive_true | no_snapshot | 276 | 1.0156 | 0.8008 | 0.1087 | 0.3659 | 0.1377 |
| 2026-02-15 | positive_true | snapshot_positive_stock | 1530 | 3.3144 | 3.7743 | 0.1686 | 0.5876 | 0.5725 |
| 2026-02-15 | positive_true | snapshot_zero_stock | 50 | 2.5454 | 1.5017 | 0.1000 | 0.7000 | 0.5200 |
| 2026-02-15 | zero_true | no_snapshot | 4359 | 0.2273 | 0.1085 | 0.0181 | 0.2787 | 0.4074 |
| 2026-02-15 | zero_true | snapshot_positive_stock | 3853 | 0.3970 | 0.4356 | 0.0189 | 0.3496 | 0.6647 |
| 2026-02-15 | zero_true | snapshot_zero_stock | 72 | 3.3316 | 1.4202 | 0.0000 | 0.9167 | 0.8611 |
| 2026-02-24 | positive_true | no_snapshot | 509 | 2.3616 | 2.2189 | 0.1513 | 0.2456 | 0.0884 |
| 2026-02-24 | positive_true | snapshot_positive_stock | 1574 | 3.1329 | 3.6194 | 0.3030 | 0.5693 | 0.6042 |
| 2026-02-24 | positive_true | snapshot_zero_stock | 48 | 2.2516 | 1.4369 | 0.2917 | 0.6875 | 0.5208 |
| 2026-02-24 | zero_true | no_snapshot | 4166 | 0.2611 | 0.1025 | 0.0298 | 0.2552 | 0.4256 |
| 2026-02-24 | zero_true | snapshot_positive_stock | 3771 | 0.5705 | 0.5597 | 0.0520 | 0.3185 | 0.6584 |
| 2026-02-24 | zero_true | snapshot_zero_stock | 72 | 3.3236 | 0.9099 | 0.0556 | 0.9028 | 0.8611 |

## Zero-True Compare

| stock_state | rows | base_fp_rate | shadow_fp_rate | fp_rate_delta | base_pred_mean | shadow_pred_mean | lookback_repl_pos_rate | qfo_ge_25_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_snapshot | 8525 | 0.1347 | 0.0622 | -0.0725 | 0.2438 | 0.1056 | 0.2672 | 0.4163 |
| snapshot_positive_stock | 7624 | 0.2642 | 0.3107 | 0.0466 | 0.4828 | 0.4970 | 0.3342 | 0.6616 |
| snapshot_zero_stock | 144 | 0.9306 | 0.7569 | -0.1736 | 3.3276 | 1.1651 | 0.9097 | 0.8611 |

## Positive-True Compare

| stock_state | rows | true_sum | base_under_wape | shadow_under_wape | under_wape_delta | base_pred_mean | shadow_pred_mean | event_strong_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| snapshot_positive_stock | 3104 | 13453.0001 | 0.3483 | 0.2408 | -0.1075 | 3.2223 | 3.6957 | 0.2368 |
| no_snapshot | 785 | 2772.0000 | 0.5707 | 0.5973 | 0.0267 | 1.8883 | 1.7203 | 0.1363 |
| snapshot_zero_stock | 98 | 143.0000 | 0.0649 | 0.1689 | 0.1040 | 2.4015 | 1.4699 | 0.1939 |

## Interpretation

- If `snapshot_zero_stock` keeps reducing false positives but worsens positive-true under-predict, it is acting more like a cautious suppression signal than a complete stock-constraint solution.
- If future work wants to exploit this state better, the next features should be time-based inventory state features rather than another new data source.
- Focus cases exported: `120`.

## Output Files

- `reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_anchor_mix.csv`
- `reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_zero_true_compare.csv`
- `reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_positive_true_compare.csv`
- `reports/phase8_snapshot_zero_stock_2026/phase8_snapshot_zero_stock_cases.csv`
