# Phase8 Data Coverage Audit

## Summary

- Official label logic remains frozen before client reply.
- This audit only evaluates data coverage and feature-table feasibility.
- Support status is defined against a 90-day lookback requirement.

## Source Coverage

| source | source_file | rows | date_min | date_max | distinct_days | duplicate_rows_on_key | sku_mapping_rate | style_mapping_rate | buyer_mapping_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inventory_daily_features | snapshot_inventory/*.csv | 562552 | 2026-01-23 | 2026-04-08 | 76 | 66749 | 0.9640 | 0.9640 |  |
| preorder_daily_features | V_IRS_PREORDER.csv | 333 | 2024-12-19 | 2026-04-08 | 28 | 0 | 0.9880 | 0.9880 | 1.0000 |
| event_intent_daily_features | V_IRS_EVENT1.csv | 141912 | 2025-09-18 | 2026-04-08 | 203 | 0 |  | 0.9429 | 1.0000 |

## Anchor Support

| source | 20250901_available_days | 20250901_support | 20251001_available_days | 20251001_support | 20251101_available_days | 20251101_support | 20251201_available_days | 20251201_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inventory_daily_features | 0 | unsupported | 0 | unsupported | 0 | unsupported | 0 | unsupported |
| preorder_daily_features | 256 | full | 286 | full | 317 | full | 347 | full |
| event_intent_daily_features | 0 | unsupported | 13 | partial | 44 | partial | 74 | partial |

## Current Default Judgement

- Inventory snapshots are 2026-only in the current workspace, so they do not support the official 2025 four-anchor replacement compare.
- Preorder snapshots are sparse and 2026-heavy in the current workspace, so they do not support the official 2025 four-anchor replacement compare.
- Event source currently resolves to `V_IRS_EVENT1.csv`.
- Event data starts on 2025-09-18, so it is only shadow-experiment eligible for `2025-10/11/12`, and still partial against a 90-day lookback.