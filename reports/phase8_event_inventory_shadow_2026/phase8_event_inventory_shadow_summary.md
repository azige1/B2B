# Phase8 Event+Inventory Shadow 2026 Summary

- Status: `analysis_only_shadow`
- Scope: `2026-02-15 / 2026-02-24`
- Compare: base `cov_activity_tail_full` vs extended `cov_activity_tail_full_event` with event+inventory features in the event group
- This compare does not participate in official phase7 replacement

## Base vs Event+Inventory Shadow

| metric | base_tail_full | event_inventory_shadow | delta |
| --- | --- | --- | --- |
| global_ratio | 1.0946 | 1.0476 | -0.0471 |
| global_wmape | 0.8614 | 0.6667 | -0.1947 |
| 4_25_under_wape | 0.4384 | 0.3452 | -0.0932 |
| 4_25_sku_p50 | 0.5966 | 0.6818 | +0.0852 |
| ice_4_25_sku_p50 | 0.5234 | 0.6212 | +0.0978 |
| blockbuster_under_wape | 0.4934 | 0.3773 | -0.1161 |
| blockbuster_sku_p50 | 0.4735 | 0.6326 | +0.1591 |
| top20_true_volume_capture | 0.5606 | 0.5766 | +0.0161 |
| rank_corr_positive_skus | 0.6918 | 0.7522 | +0.0604 |
| 1_3_ratio | 1.1421 | 1.0749 | -0.0671 |

## Interpretation

- This is a 2026-only exploratory line.
- It is useful for deciding whether inventory should join the event shadow line in later phase8 work.
- No result here is allowed to replace the official phase7 winner.
