# Docs Index

## Default Reading Order

For current project status:

1. `PROJECT_INDEX.md`
2. `reports/current/phase8_blockbuster_uplift_0p6_20260616.md`
3. `reports/current/server_whitelist_refresh_20260616.md`
4. `reports/current/server_daily_oracle_snapshot_automation_20260616.md`
5. `RUNNERS_INDEX.md`
6. `data/DATA_INDEX.md`
7. `MODELS_INDEX.md`

For the profit-analysis module:

1. `modules/profit_analysis/README.md`
2. `modules/profit_analysis/docs/profit_analysis_module_v1_detailed_design_20260522.md`
3. `modules/profit_analysis/docs/profit_analysis_module_delivery_20260612.md`
4. `modules/profit_analysis/docs/profit_analysis_v0_client_feedback_rules_20260515.md`

## Current Canonical Documents

Prediction:

- `reports/current/phase8_blockbuster_uplift_0p6_20260616.md`
- `reports/current/phase8_event_coverage_router_0p6_20260616.md`
- `reports/current/phase8_no_listing_0p6_series_candidate_20260616.md`
- `reports/current/phase8_final_freeze_report_20260616.md`

Data:

- `data/current_assets.json`
- `data/manifests/phase8_data_snapshot_20260616.json`
- `reports/current/server_whitelist_refresh_20260616.md`
- `reports/current/server_daily_oracle_snapshot_automation_20260616.md`
- `reports/current/client_source_table_registry.md`
- `reports/current/phase8_data_semantics_20260614.md`

Profit analysis:

- `modules/profit_analysis/README.md`
- `modules/profit_analysis/docs/profit_analysis_module_v1_detailed_design_20260522.md`
- `modules/profit_analysis/docs/profit_analysis_module_delivery_20260612.md`

## Document Roles

- `readme.md`: short project landing page for GitHub and handoff.
- `PROJECT_INDEX.md`: top-level current status and module boundaries.
- `DOCS_INDEX.md`: canonical document routing.
- `RUNNERS_INDEX.md`: executable entry points and reproduction commands.
- `MODELS_INDEX.md`: model artifact map and current-vs-experiment distinction.
- `data/DATA_INDEX.md`: data source, snapshot, and downstream rebuild status.
- `reports/REPORTS_INDEX.md`: report directory map.

## Historical Material

Use these only for reproduction or audit:

- `reports/phase1/` to `reports/phase7*/`
- older `reports/phase8_*` experiments not referenced by `PROJECT_INDEX.md`
- `scripts/runners/phase5/`, `scripts/runners/phase6/`, `scripts/runners/phase7/`
- old root-level training scripts retained for historical context

## What Not To Assume

- Do not assume LSTM docs describe the current mainline.
- Do not assume Phase7 is still the best current result; it is the historical frozen baseline.
- Do not assume all Phase8 experiment directories are current candidates.
- Do not assume the latest raw `6_16` data has rebuilt silver/gold or retrained models.
- Do not treat `V_IRS_ORDER` as a label source.
