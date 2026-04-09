# Phase8 Behavior Control Follow-Up

- Date: `2026-04-09`
- Status: `analysis_only_shadow`
- Scope: `event_inventory_zero_split` plus lightweight post-processing experiments

## Current Best Variant

- `event_inventory_zero_split` is the best inventory shadow variant in the current workspace.
- Relative to `event_inventory_plus`, it improves:
  - `global_wmape`
  - `blockbuster_under_wape`
  - `rank_corr_positive_skus`
- It also improves the target slice that motivated the work:
  - `snapshot_zero_stock` rows

## Post-Processing Experiments

### Scale-only rule search

- Conclusion: not useful enough
- Reason:
  - scaling can reduce predicted quantity
  - but it usually does not change the binary `pred > 0` decision
  - so false-positive rate barely moves

### Hard-gate rule search

- Best rule found:
  - `short_zero_gate_qty = 0.5`
  - `long_zero_gate_qty = 1.0`
  - `long_zero_signal_gate_qty = 0.0`
  - `long_zero_min_streak = 21`
- Effect:
  - `long_zero` zero-true false-positive rate: `0.7394 -> 0.7324`
  - `long_zero` positive-true under-WAPE: `0.1612 -> 0.1883`
  - mean `global_wmape`: `0.6667 -> 0.6668`

## Practical Judgement

- Pure post-processing can move behavior a little.
- But the gain is too small to justify treating hard-gate rules as the next main path.
- The useful part is the diagnosis:
  - `long_zero` can be suppressed a bit
  - but over-suppression immediately hurts positive-true rows

## Recommended Next Step

- Keep `event_inventory_zero_split` as the current best exploratory inventory variant.
- If continuing on this line, prefer model-level asymmetric handling over more hand-written post-processing:
  - stronger treatment for `long_zero`
  - softer treatment for `short_zero`
  - avoid using a blanket hard gate as the main solution
