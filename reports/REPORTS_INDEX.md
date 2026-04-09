# Reports Index

## Current Official Reports

- `current/`
  - Canonical entry point for the current official model state.
  - Preferred for internal references, handoff, and external reporting.

## Current Official Source Directories

- `phase7/`
  - Official phase freeze documents for the current mainline.
- `phase7i_full_model_compare/`
  - Official full December comparison source for the current mainline.

## Historical Phase Results

- `phase5*/`
  - Tree build-up, anchor backtests, sweeps, and delivery-calibration history.
- `phase6*/`
  - Tree stabilization, validation, family compare, and prior readable-report history.

## Analysis And Experiment Outputs

- `phase7_tail_allocation_optimization/`
  - Tail/allocation experiments, including overnight runs and sweep outputs.
- `phase8a_prep/`
  - Pre-client-reply phase8 preparation outputs: coverage audit, standardized feature tables, SHAP, and residual-gap diagnostics.
- `*_readable_report`
  - Single-period readable reports and dashboards.
- `*_full_model_compare`
  - Rich comparison pages and CSV exports.
- `overnight_*`
  - Long-run experiment artifacts and logs.

## Historical Reference Only

- `phase6h_december_readable_report`
- `phase7h_december_readable_report`

These remain valid history, but they are no longer the default report entry points.

## Default Reading Order

1. `current/`
2. `phase7/`
3. `phase7i_full_model_compare/`
4. Historical phase or experiment directories only when needed
