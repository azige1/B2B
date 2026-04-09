# Phase8 Inventory Zero-Split Shadow 2026 Summary

- Status: `analysis_only_shadow`
- Scope: `2026-02-15 / 2026-02-24`
- Compare: base vs current event+inventory shadow vs zero-split shadow
- This compare does not participate in official phase7 replacement

## Three-Way Compare

| metric | base_tail_full | event_inventory_plus | event_inventory_zero_split | zero_split_minus_plus |
| --- | --- | --- | --- | --- |
| global_ratio | 1.0999 | 1.0935 | 1.0777 | -0.0159 |
| global_wmape | 0.8623 | 0.6874 | 0.6667 | -0.0208 |
| 4_25_under_wape | 0.4371 | 0.3331 | 0.3312 | -0.0019 |
| 4_25_sku_p50 | 0.5966 | 0.6999 | 0.6975 | -0.0024 |
| ice_4_25_sku_p50 | 0.5251 | 0.6303 | 0.6312 | +0.0009 |
| blockbuster_under_wape | 0.4934 | 0.3697 | 0.3635 | -0.0062 |
| blockbuster_sku_p50 | 0.4735 | 0.6513 | 0.6340 | -0.0173 |
| top20_true_volume_capture | 0.5603 | 0.5693 | 0.5725 | +0.0032 |
| rank_corr_positive_skus | 0.6963 | 0.7538 | 0.7642 | +0.0105 |
| 1_3_ratio | 1.1495 | 1.0955 | 1.0980 | +0.0025 |

## Interpretation

- The zero-split variant is useful only if it improves the current event+inventory shadow, not merely the base line.
- Focus on whether the new variant improves `snapshot_zero_stock` behavior without giving back too much on global error.
- No result here is allowed to replace the official phase7 winner.
