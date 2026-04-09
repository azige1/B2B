# Phase8 Inventory Constraint 2026 Summary

- Status: `analysis_only`
- Source: `phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_row_compare.csv`
- Purpose: inspect whether the 2026 event+inventory shadow behaves differently across `no_snapshot`, `snapshot_zero_stock`, and `snapshot_positive_stock`.

## Bottom Line

- This pack does not change the official phase7 mainline.
- Inventory states are now evaluated explicitly instead of inferring zero stock from presence flags.
- The practical question is whether `snapshot_zero_stock` behaves differently from `no_snapshot` after the semantic fix.

## Zero-True by Stock State

| stock_state | rows | base_fp_rate | shadow_fp_rate | fp_rate_delta | event_strong_rate | lookback_repl_pos_rate | qfo_ge_25_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_snapshot | 8525 | 0.1347 | 0.0622 | -0.0725 | 0.0238 | 0.2672 | 0.4163 |
| snapshot_positive_stock | 7624 | 0.2642 | 0.3107 | 0.0466 | 0.0353 | 0.3342 | 0.6616 |
| snapshot_zero_stock | 144 | 0.9306 | 0.7569 | -0.1736 | 0.0278 | 0.9097 | 0.8611 |

- `true=0` rows with `no_snapshot`: `8525`, `base_fp_rate=0.1347`, `shadow_fp_rate=0.0622`.
- `true=0` rows with `snapshot_zero_stock`: `144`, `base_fp_rate=0.9306`, `shadow_fp_rate=0.7569`.
- `true=0` rows with `snapshot_positive_stock`: `7624`, `base_fp_rate=0.2642`, `shadow_fp_rate=0.3107`.

## Positive-True Under-Predict by Stock State

| stock_state | rows | true_sum | base_under_wape | shadow_under_wape | under_wape_delta | event_strong_rate | mean_total_stock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| snapshot_positive_stock | 3104 | 13453.0001 | 0.3483 | 0.2408 | -0.1075 | 0.2368 | 21.3611 |
| no_snapshot | 785 | 2772.0000 | 0.5707 | 0.5973 | 0.0267 | 0.1363 | 0.0000 |
| snapshot_zero_stock | 98 | 143.0000 | 0.0649 | 0.1689 | 0.1040 | 0.1939 | 0.0000 |

## Inventory Source Audit

- inventory_daily_rows: `562552`
- inventory_distinct_days: `76`
- duplicate `date+sku` rows in inventory_daily_features: `66749`
- inventory rows with `qty_storage_stock = 0`: `72994`
- inventory rows with `qty_b2b_hq_stock = 0`: `383239`
- inventory rows with `snapshot_present = 1`: `562552`
- inventory rows with `stock_zero = 1`: `8661`
- wide `qty_stock > 0` rate after `2026-01-23`: `0.8252`
- wide `is_real_stock > 0` rate after `2026-01-23`: `0.8361`
- wide `snapshot_present = 1` rate after `2026-01-23`: `0.8361`
- wide `stock_zero = 1` rate after `2026-01-23`: `0.0110`

## Interpretation

- `snapshot_positive_stock` captures rows with explicit inventory support.
- `snapshot_zero_stock` captures rows where a snapshot exists but total stock is zero or below.
- `no_snapshot` now means missing snapshot evidence, not zero stock by default.

## Practical Answer

- If `snapshot_zero_stock` now shows a distinct error pattern from `no_snapshot`, the inventory line is carrying more than a pure positive-stock signal.
- If the gains still sit almost entirely in `snapshot_positive_stock`, most of the lift is coming from positive-stock evidence.
- Candidate `true=0` demand-risk rows exported for follow-up: `200`.

## Output Files

- `reports/phase8_extended_signal_2026/phase8_inventory_constraint_zero_true_table.csv`
- `reports/phase8_extended_signal_2026/phase8_inventory_constraint_positive_true_table.csv`
- `reports/phase8_extended_signal_2026/phase8_inventory_constraint_candidate_cases.csv`
