# Phase8 Zero-Split Follow-Up

- Date: `2026-04-09`
- Status: `analysis_only_shadow`
- Scope: compare `event_inventory_zero_split` against the current `event_inventory_plus`

## Headline

- `zero_split` is a real improvement over the current `event_inventory_plus`.
- It is still not a promotion candidate, but it is the best inventory shadow variant so far in the current workspace.

## Three-Way Summary

- Mean across `2026-02-15 / 2026-02-24`
- `global_wmape`
  - base: `0.8623`
  - event_inventory_plus: `0.6874`
  - event_inventory_zero_split: `0.6667`
- `4_25_under_wape`
  - base: `0.4371`
  - event_inventory_plus: `0.3331`
  - event_inventory_zero_split: `0.3312`
- `blockbuster_under_wape`
  - base: `0.4934`
  - event_inventory_plus: `0.3697`
  - event_inventory_zero_split: `0.3635`
- `rank_corr_positive_skus`
  - base: `0.6963`
  - event_inventory_plus: `0.7538`
  - event_inventory_zero_split: `0.7642`

## Stock-State Direct Compare

- `snapshot_zero_stock`
  - zero-true false-positive rate: `0.7569 -> 0.7361`
  - positive-true under-WAPE: `0.1689 -> 0.1662`
- `snapshot_positive_stock`
  - zero-true false-positive rate: `0.3107 -> 0.2858`
  - positive-true under-WAPE: `0.2408 -> 0.2405`
- `no_snapshot`
  - zero-true false-positive rate: `0.0622 -> 0.0625`
  - positive-true under-WAPE: `0.5973 -> 0.5806`

## Practical Judgement

- The split between `short_zero` and `long_zero` is useful.
- This variant improves the exact slice that was still weak in the previous round: `snapshot_zero_stock`.
- The gain is incremental rather than decisive.
- If the inventory line keeps moving, the next logical step is not another broad feature dump; it is to treat `short_zero` and `long_zero` asymmetrically at training or gating time.
