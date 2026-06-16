# Models Index

## Current Interpretation

The repository contains both frozen historical models and many Phase8 experiment model directories. Do not treat every `models_phase8_*` directory as a current candidate.

Current recommended Phase8 result is selected from report/context artifacts, not from a single new monolithic model directory:

- Base branch: `coverage_router`
- Recommended calibration: `conservative_blockbuster_uplift`
- Main report: `reports/current/phase8_blockbuster_uplift_0p6_20260616.md`
- Recommended context: `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_recommended_context.csv`

## Historical Frozen Baseline

- `models/current_phase7_mainline/`
  - Frozen Phase7 reference model assets.
  - Use for Phase7 reproduction only.

Phase7 report references:

- `reports/current/current_mainline.json`
- `reports/current/current_freeze_summary.md`
- `reports/phase7_tail_allocation_optimization/`

## Phase8 Experiment Model Directories

These directories are retained for reproducibility and audit:

- `models_phase8_event_request_shadow/`
- `models_phase8_purchase_request_shadow/`
- `models_phase8_event_core_robust_oot/`
- `models_phase8_robust_oot/`
- `models_phase8_2026_out_of_time/`
- `models_phase8_listing_zero_split_shadow_2026/`
- `models_phase8_lifecycle_shadow/`
- `models_phase8_lifecycle_peer_priors/`
- `models_phase8_listing_date_targeted/`
- `models_phase8_listing_date_stage_ablation/`
- `models_phase8_peer_prior_46_90/`
- `models_phase8_formal_stage1_20260614/`
- `models_phase8_formal_stage1b_20260614/`
- `models_phase8_formal_stage2_20260614/`

Older Phase8 directories may exist for April inventory/event/zero-split explorations. Treat them as historical unless a current report explicitly references them.

## Cleanup Rule

- Do not delete model directories during documentation cleanup.
- Do not rename model directories unless all runner/report references are updated.
- If disk cleanup is needed later, first create a manifest with directory name, purpose, related report, and regeneration command.
- Generated model binaries should generally stay out of git unless they are a deliberately frozen small reference artifact.
