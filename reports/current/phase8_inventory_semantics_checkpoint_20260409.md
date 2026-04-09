# Phase8 Inventory Semantics Checkpoint

- Date: `2026-04-09`
- Status: `analysis_only_shadow`
- Scope: inventory semantics fix + rerun for `2026-02-15 / 2026-02-24`

## What Changed

- Inventory state is now represented explicitly as:
  - `snapshot_present`
  - `stock_positive`
  - `stock_zero`
- The state columns were propagated through:
  - `data/phase8a_prep/inventory_daily_features.csv`
  - `data/gold/wide_table_sku.csv`
  - `phase8 event+inventory shadow` feature generation
  - detail packs and inventory-constraint analysis

## Rerun Result

- The event+inventory shadow remains stronger than the 2026 base line after the semantic fix.
- `2026-02-15`
  - `global_wmape: 0.8538 -> 0.6431`
  - `blockbuster_under_wape: 0.4291 -> 0.2883`
- `2026-02-24`
  - `global_wmape: 0.8690 -> 0.6903`
  - `blockbuster_under_wape: 0.5576 -> 0.4663`

## Current Interpretation

- The inventory line is still valid after the semantic fix.
- Most of the current lift still comes from `snapshot_positive_stock`.
- `snapshot_zero_stock` now exists as an explicit state in the 2026 shadow outputs.
- Current evidence suggests `snapshot_zero_stock` behaves more like a cautious suppression signal than a fully learned stock-constraint signal.

## Immediate Next Focus

- Use `phase8g snapshot_zero_stock case pack` to inspect:
  - `true=0 + snapshot_zero_stock` false positives
  - `true>0 + snapshot_zero_stock` under-predict regressions
- If this state remains unstable, the next feature work should focus on inventory-state timing features rather than adding a new data source.
