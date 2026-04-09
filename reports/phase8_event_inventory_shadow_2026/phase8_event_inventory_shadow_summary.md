# Phase8 Event+Inventory Shadow 2026 Summary

- Status: `analysis_only_shadow`
- Scope: `2026-02-15 / 2026-02-24`
- Compare: base `cov_activity_tail_full` vs extended `cov_activity_tail_full_event` with event+inventory features in the event group
- This compare does not participate in official phase7 replacement

## Base vs Event+Inventory Shadow

| metric | base_tail_full | event_inventory_shadow | delta |
| --- | --- | --- | --- |
| global_ratio | 1.0999 | 1.0935 | -0.0064 |
| global_wmape | 0.8623 | 0.6874 | -0.1749 |
| 4_25_under_wape | 0.4371 | 0.3331 | -0.1040 |
| 4_25_sku_p50 | 0.5966 | 0.6999 | +0.1033 |
| ice_4_25_sku_p50 | 0.5251 | 0.6303 | +0.1052 |
| blockbuster_under_wape | 0.4934 | 0.3697 | -0.1237 |
| blockbuster_sku_p50 | 0.4735 | 0.6513 | +0.1778 |
| top20_true_volume_capture | 0.5603 | 0.5693 | +0.0090 |
| rank_corr_positive_skus | 0.6963 | 0.7538 | +0.0575 |
| 1_3_ratio | 1.1495 | 1.0955 | -0.0540 |

## Interpretation

- This is a 2026-only exploratory line.
- It is useful for deciding whether inventory should join the event shadow line in later phase8 work.
- No result here is allowed to replace the official phase7 winner.
