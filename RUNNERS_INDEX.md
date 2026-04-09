# Runners Index

## Current Official

- `run_phase7_freeze.py`
  - Freezes the current official phase7 mainline and writes canonical phase7 freeze documents.
- `run_phase7i_full_model_compare.py`
  - Builds the current official full December comparison page and CSV.

## Historical Phase Runners

- `run_phase5_experiments.py`
- `run_phase5_1_experiments.py`
- `run_phase5_2_experiments.py`
- `run_phase5_3_experiments.py`
- `run_phase5_4_local_confirm.py`
- `run_phase5_5_local_anchors.py`
- `run_phase5_6_tree_sweep.py`
- `run_phase5_7a_tuned_anchors.py`
- `run_phase5_7b_low_demand_calibration.py`
- `run_phase5_tree_anchor_calibration.py`
- `run_phase5_overnight_chain.py`
- `run_phase6_tree_stabilization.py`
- `run_phase6_tree_monthaware_refit.py`
- `run_phase6d_tree_micro_calibration.py`
- `run_phase6e_tree_validation_pack.py`
- `run_phase6f_tree_family_compare.py`
- `run_phase7a_tail_gap_pack.py`
- `run_phase7_tail_feature_sweep.py`
- `run_phase7_overnight_chain.py`

These are retained for reproduction of historical or intermediate phase results.

## One-Off Analysis And Readable Reports

- `run_phase6h_december_readable_report.py`
- `run_phase7h_december_readable_report.py`
- `run_phase8a_prep.py`
  - Builds pre-client-reply phase8 preparation outputs: coverage audit, standardized feature tables, SHAP, and residual-gap diagnostics.

These are analysis/reporting entry points, not the preferred current official entry points.

## Working Rule

- Prefer the current official runners for current references.
- Use historical or one-off runners only when reproducing a specific phase artifact.
