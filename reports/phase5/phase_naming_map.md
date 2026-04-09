# Phase Naming Map

## Historical to Semantic Mapping

- `phase5_4` = `phase5_tree_seed_confirm`
- `phase5_5` = `phase5_tree_anchor_backtest`
- `phase5_6` = `phase5_tree_sweep`
- `phase5_7` = `phase5_tree_delivery_calibration`

## Executed Semantic Stages

- `phase5_tree_anchor_calibration`
- `phase6_tree_stabilization`
- `phase6_tree_monthaware_refit`
- `phase6d_tree_micro_calibration`
- `phase6e_tree_validation_pack`
- `phase6f_tree_family_compare`
- `phase6h_december_readable_report`
- `phase7_tail_allocation_optimization`
- `phase7h_december_readable_report`
- `phase7i_full_model_compare`

## Current Official State

- `phase6` is frozen and remains the previous official tree baseline.
- `phase7` is frozen and is the current official phase.
- Current official mainline:
  - `tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`
- Previous official mainline:
  - `p57_covact_lr005_l63_hard_g025 + sep098_oct093`
- `LightGBM` remains the current official tree family.
- The old label `phase6_delivery_validation` is not an executed phase name and should not be used.

## Current Official Entry Points

- `reports/current/current_mainline.json`
- `reports/current/current_freeze_summary.md`
- `reports/current/current_model_compare.html`

## Next Planned Stage

- `phase8_delivery_readiness` (not started)

## Phase7 Meaning

- Focus:
  - `>25 / blockbuster`
  - allocation
  - `repl0_fut0 + ice + high first-order quantity`
- Do not reopen:
  - `sequence` expansion
  - tree family compare
  - large post-processing sweeps
