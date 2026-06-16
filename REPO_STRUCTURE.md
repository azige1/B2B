# Repository Structure Guide

This guide explains how to navigate the repository without being distracted by historical experiment artifacts.

## Mental Model

The project has three active areas:

1. Replenishment prediction
2. Data intake and warehouse snapshots
3. Profit analysis

Everything else is either historical reproduction material or generated experiment output.

## Active Prediction Area

Code:

- `src/etl/`
- `src/features/`
- `src/train/`
- `src/analysis/`
- `src/inference/`
- `scripts/runners/`

Current Phase8 result:

- `reports/current/phase8_blockbuster_uplift_0p6_20260616.md`
- `reports/phase8_blockbuster_uplift_0p6/`
- `reports/phase8_event_request_router/`
- `reports/phase8_event_request_shadow/`

Current reproduction entry points:

- `src/analysis/analyze_phase8u_blockbuster_uplift.py`
- `src/analysis/analyze_phase8t_event_request_router.py`
- `scripts/runners/phase8/run_phase8t_event_request_shadow.py`

## Active Data Area

Source snapshots:

- `data/incoming/`
- `data_warehouse/`

Manifests and asset registry:

- `data/current_assets.json`
- `data/manifests/`
- `data/DATA_INDEX.md`

Server automation:

- `scripts/data/server_run_daily_oracle_snapshot.sh`
- `scripts/data/server_daily_oracle_snapshot.cron`
- `reports/current/server_daily_oracle_snapshot_automation_20260616.md`

Data source rule:

- `V_IRS_ORDERFTP` is the only official order-flow and label source.
- `V_IRS_ORDER` is backup only.

## Active Profit Analysis Area

Module root:

- `modules/profit_analysis/`

Useful subdirectories:

- `modules/profit_analysis/src/profit_analysis/`
- `modules/profit_analysis/scripts/`
- `modules/profit_analysis/config/`
- `modules/profit_analysis/docs/`
- `modules/profit_analysis/tests/`

Start here:

- `modules/profit_analysis/README.md`
- `modules/profit_analysis/docs/profit_analysis_module_v1_detailed_design_20260522.md`
- `modules/profit_analysis/docs/profit_analysis_module_delivery_20260612.md`

## Historical Areas

Historical reports:

- `reports/phase1/` to `reports/phase7*/`
- older `reports/phase8_*` directories not referenced by `PROJECT_INDEX.md`
- `reports/rolling_backtest/`
- `reports/eval_history/`

Historical runners:

- `scripts/runners/phase5/`
- `scripts/runners/phase6/`
- `scripts/runners/phase7/`
- older Phase8 runners listed under `RUNNERS_INDEX.md`

Historical/experiment models:

- `models/`
- `models_phase8_*/`

## What Not To Touch Casually

Do not casually move or delete:

- `data_warehouse/`
- `data/incoming/`
- `data/processed_*`
- `data/artifacts_*`
- `models_phase8_*`
- `reports/phase8_*`

These paths are referenced by experiment reports and runner scripts. If they need cleanup, create a manifest first.

## Cleanup Policy

Safe cleanup:

- Update indexes and current docs.
- Add manifests.
- Add README files to confusing areas.
- Delete only temporary logs or caches after confirming they are not referenced.

Requires manifest:

- Moving generated reports.
- Moving model directories.
- Moving data snapshots.
- Renaming feature asset directories.

Avoid:

- Rewriting historical reports.
- Reusing old report names for new metrics.
- Mixing new raw data with old feature assets in a metric claim.

## Current Handoff Rule

When handing this project to someone else, point them to:

1. `readme.md`
2. `PROJECT_INDEX.md`
3. `DOCS_INDEX.md`
4. `REPO_STRUCTURE.md`
5. `RUNNERS_INDEX.md`

They should not start from `reports/phase*/` unless they are auditing history.
