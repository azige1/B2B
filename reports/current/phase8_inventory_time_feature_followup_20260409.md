# Phase8 Inventory Time Feature Follow-Up

- Date: `2026-04-09`
- Status: `analysis_only_shadow`
- Scope: add inventory-state timing features to the `2026-02-15 / 2026-02-24` event+inventory shadow

## Added Feature Types

- recent `snapshot_present` counts: `7 / 14 / 30`
- recent `stock_positive` counts: `7 / 14 / 30`
- recent `stock_zero` counts: `7 / 14 / 30`
- days since last snapshot
- days since last positive stock
- days since last zero stock
- zero-stock streak length
- positive-to-zero switch flag

## Rerun Result

### 2026-02-15

- base
  - `global_wmape = 0.8450`
  - `4_25_under_wape = 0.3971`
  - `blockbuster_under_wape = 0.4291`
- event+inventory + time features
  - `global_wmape = 0.6632`
  - `4_25_under_wape = 0.2827`
  - `blockbuster_under_wape = 0.2786`

### 2026-02-24

- base
  - `global_wmape = 0.8796`
  - `4_25_under_wape = 0.4771`
  - `blockbuster_under_wape = 0.5576`
- event+inventory + time features
  - `global_wmape = 0.7117`
  - `4_25_under_wape = 0.3835`
  - `blockbuster_under_wape = 0.4608`

## Interpretation

- The time-feature variant remains stronger than the base line on both anchors.
- The extra inventory-state timing features continue to help the two business-critical under-predict slices:
  - `4_25`
  - `blockbuster`
- But the gain pattern is still dominated by `snapshot_positive_stock`.
- On the `snapshot_zero_stock` slice, the model becomes even more suppressive:
  - zero-true false positives still go down versus base
  - positive-true rows become more under-predicted than in the previous semantic-fix-only run

## Practical Judgement

- This variant is useful as evidence that inventory-state timing has signal.
- It is not yet a clean promotion candidate.
- Current behavior suggests the timing features are reinforcing a conservative zero-stock prior more than they are learning a balanced stock-constraint rule.

## Next Best Move

- Keep this variant as an exploratory branch.
- If continuing on inventory:
  - inspect `snapshot_zero_stock` cases with long zero streaks but later positive replenish
  - distinguish short-term zero-stock transitions from long flat zero-stock runs
  - consider asymmetric handling so zero-stock timing suppresses false positives without over-killing positive-true rows
