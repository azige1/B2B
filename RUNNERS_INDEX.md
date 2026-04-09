# Runners Index

## Current Official

- `scripts/runners/phase7/run_phase7_freeze.py`
  - Freezes the current official phase7 mainline and writes canonical phase7 freeze documents.
- `scripts/runners/phase7/run_phase7i_full_model_compare.py`
  - Builds the current official full December comparison page and CSV.

## Historical Phase Runners

- `scripts/runners/phase5/run_phase5_experiments.py`
- `scripts/runners/phase5/run_phase5_1_experiments.py`
- `scripts/runners/phase5/run_phase5_2_experiments.py`
- `scripts/runners/phase5/run_phase5_3_experiments.py`
- `scripts/runners/phase5/run_phase5_4_local_confirm.py`
- `scripts/runners/phase5/run_phase5_5_local_anchors.py`
- `scripts/runners/phase5/run_phase5_6_tree_sweep.py`
- `scripts/runners/phase5/run_phase5_7a_tuned_anchors.py`
- `scripts/runners/phase5/run_phase5_7b_low_demand_calibration.py`
- `scripts/runners/phase5/run_phase5_tree_anchor_calibration.py`
- `scripts/runners/phase5/run_phase5_overnight_chain.py`
- `scripts/runners/phase6/run_phase6_tree_stabilization.py`
- `scripts/runners/phase6/run_phase6_tree_monthaware_refit.py`
- `scripts/runners/phase6/run_phase6d_tree_micro_calibration.py`
- `scripts/runners/phase6/run_phase6e_tree_validation_pack.py`
- `scripts/runners/phase6/run_phase6f_tree_family_compare.py`
- `scripts/runners/phase7/run_phase7a_tail_gap_pack.py`
- `scripts/runners/phase7/run_phase7_tail_feature_sweep.py`
- `scripts/runners/phase7/run_phase7_overnight_chain.py`

These are retained for reproduction of historical or intermediate phase results.

## One-Off Analysis And Readable Reports

- `scripts/runners/phase6/run_phase6h_december_readable_report.py`
- `scripts/runners/phase7/run_phase7h_december_readable_report.py`
- `scripts/runners/phase8/run_phase8a_prep.py`
  - Builds pre-client-reply phase8 preparation outputs: coverage audit, standardized feature tables, SHAP, and residual-gap diagnostics.

These are analysis/reporting entry points, not the preferred current official entry points.

## Working Rule

- Prefer the current official runners for current references.
- Use historical or one-off runners only when reproducing a specific phase artifact.
- Root-level phase runners have been consolidated under `scripts/runners/phase5` to `scripts/runners/phase8`.
