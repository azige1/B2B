# Project Index

## Current State

- Current project stage: Phase8 staged conclusion and engineering consolidation.
- Historical frozen baseline: Phase7 tree mainline.
- Current recommended Phase8 branch: `coverage_router + conservative_blockbuster_uplift`.
- Current model family: tree model stack with two-stage replenishment prediction and explainable calibration.
- Current data source of truth: `V_IRS_ORDERFTP` for order flow and labels.
- Current raw-order checkpoint: `data_warehouse/fact_orders/V_IRS_ORDERFTP_6_16.csv`.
- Current profit-analysis status: standalone V1 module under `modules/profit_analysis`.

## Phase8 Result

Under the old Phase7 `0.6` four-anchor evaluation style:

| Branch | global_wmape | blockbuster_under_wape | Note |
| --- | ---: | ---: | --- |
| Phase7 baseline | 0.6863 | 0.4165 | Historical frozen reference |
| Phase8 coverage router | 0.6591 | 0.3820 | Pure routing/model-side branch |
| Phase8 recommended uplift | 0.6482 | 0.3101 | Recommended staged final candidate |

Current conclusion:

- If strict no-postprocess is required, use `coverage_router`.
- If an explainable calibration layer is acceptable, use `coverage_router + conservative_blockbuster_uplift`.
- Do not keep searching arbitrary rules before new lifecycle labels or fuller 2026 validation data arrive.

## Read First

- Root README: `readme.md`
- Docs index: `DOCS_INDEX.md`
- Runner index: `RUNNERS_INDEX.md`
- Reports index: `reports/REPORTS_INDEX.md`
- Data index: `data/DATA_INDEX.md`
- Models index: `MODELS_INDEX.md`
- Artifact policy: `ARTIFACTS_INDEX.md`

## Current Canonical Reports

- Phase8 recommendation: `reports/current/phase8_blockbuster_uplift_0p6_20260616.md`
- Phase8 event coverage router: `reports/current/phase8_event_coverage_router_0p6_20260616.md`
- Phase8 no-listing candidate: `reports/current/phase8_no_listing_0p6_series_candidate_20260616.md`
- Phase8 freeze context: `reports/current/phase8_final_freeze_report_20260616.md`
- Server whitelist refresh: `reports/current/server_whitelist_refresh_20260616.md`
- Server daily Oracle automation: `reports/current/server_daily_oracle_snapshot_automation_20260616.md`
- Source table registry: `reports/current/client_source_table_registry.md`

## Current Data Scope

- Raw order baseline: `data_warehouse/fact_orders/V_IRS_ORDERFTP_6_16.csv`
- Raw event baseline: `data_warehouse/fact_events/V_IRS_EVENT_20260616.csv`
- Product dimension with listing date: `data_warehouse/dim_product/product_info_20260616.csv`
- Asset map: `data/current_assets.json`
- Snapshot manifest: `data/manifests/phase8_data_snapshot_20260616.json`

Important limitation:

- The latest raw snapshot has not rebuilt `data/silver` or `data/gold`.
- Current Phase8 metrics are still based on the comparable old `0.6` evaluation assets, not a full retrain on `6_16`.

## Module Boundaries

Prediction module:

- Core code: `src/`
- Repro runners: `scripts/runners/`
- Main current reports: `reports/current/`
- Experiment outputs: `reports/phase8*/`
- Model artifacts: `models/` and `models_phase8_*/`

Profit analysis module:

- Code: `modules/profit_analysis/src/profit_analysis/`
- Scripts: `modules/profit_analysis/scripts/`
- Config templates: `modules/profit_analysis/config/`
- Docs: `modules/profit_analysis/docs/`
- Tests: `modules/profit_analysis/tests/`

Data operations:

- Server scripts: `scripts/data/`
- Server automation report: `reports/current/server_daily_oracle_snapshot_automation_20260616.md`
- Incoming snapshots: `data/incoming/`
- Warehouse snapshots: `data_warehouse/`

## Historical References

- Phase5 and earlier: early LSTM, aggregation, and tree-transition history.
- Phase6: tree stabilization and validation history.
- Phase7: frozen official baseline and old external-report baseline.
- Phase8: latest exploration, lifecycle/listing-date experiments, event/request/router/uplift work.

Use historical directories only when reproducing a named result. Do not start a new handoff from historical folders.

## Working Rule

- Current decisions live in `PROJECT_INDEX.md`, `DOCS_INDEX.md`, `RUNNERS_INDEX.md`, `data/current_assets.json`, and `reports/current/`.
- Historical reports and model directories are retained for reproducibility, not as default entry points.
- Any future cleanup that moves or deletes generated artifacts must include a manifest mapping old path to new path.
