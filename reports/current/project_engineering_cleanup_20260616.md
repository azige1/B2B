# Project Engineering Cleanup Report (2026-06-16)

## Scope

This cleanup focused on project readability and handoff structure. It did not delete data, move model artifacts, or rewrite historical experiment outputs.

## Why This Was Needed

The repository contains many historical reports, feature assets, model directories, and Phase8 exploration outputs. Without clear entry points, a new reader can easily mistake an old Phase7/Phase8 experiment for the current conclusion.

Largest high-level areas by item count at cleanup time:

| Area | Approximate item count | Interpretation |
| --- | ---: | --- |
| `reports/` | 2,194 | Most historical and experiment outputs live here |
| `data/` | 2,154 | Processed features, artifacts, manifests, incoming snapshots |
| `data_warehouse/` | 773 | Raw/source snapshots |
| `models_phase8_*` | many directories | Experiment model artifacts |
| `src/` | 226 | Core code |
| `scripts/` | 163 | Runners, data scripts, demos |
| `modules/` | 81 | Profit-analysis module |

## Changes Made

Updated root/current entry documents:

- `readme.md`
- `PROJECT_INDEX.md`
- `DOCS_INDEX.md`
- `RUNNERS_INDEX.md`
- `MODELS_INDEX.md`
- `REPO_STRUCTURE.md`
- `data/DATA_INDEX.md`
- `reports/REPORTS_INDEX.md`

The new entry documents now distinguish:

- Phase7 historical frozen baseline
- Phase8 current recommended candidate
- Latest raw data snapshot and server automation
- Prediction module vs profit-analysis module
- Active reports vs historical reports
- Current runners vs historical runners
- Model artifacts vs report/context-selected Phase8 result

## Current Project Entry Points

Use this reading order:

1. `readme.md`
2. `PROJECT_INDEX.md`
3. `DOCS_INDEX.md`
4. `REPO_STRUCTURE.md`
5. `RUNNERS_INDEX.md`
6. `data/DATA_INDEX.md`
7. `reports/current/phase8_blockbuster_uplift_0p6_20260616.md`

## Current Module Boundaries

Prediction module:

- Code: `src/`
- Runners: `scripts/runners/`
- Data scripts: `scripts/data/`
- Current reports: `reports/current/`
- Phase8 outputs: `reports/phase8_blockbuster_uplift_0p6/`, `reports/phase8_event_request_router/`, `reports/phase8_event_request_shadow/`

Profit analysis module:

- Root: `modules/profit_analysis/`
- Docs: `modules/profit_analysis/docs/`
- Scripts: `modules/profit_analysis/scripts/`
- Tests: `modules/profit_analysis/tests/`

Data operations:

- Incoming snapshots: `data/incoming/`
- Snapshot manifests: `data/manifests/`
- Warehouse snapshots: `data_warehouse/`
- Server automation report: `reports/current/server_daily_oracle_snapshot_automation_20260616.md`

## What Was Deliberately Not Done

No large generated directories were moved. This is deliberate because reports and runners may reference exact paths.

Not moved or deleted:

- `reports/phase*/`
- `reports/phase8_*`
- `models_phase8_*`
- `data/processed_*`
- `data/artifacts_*`
- `data_warehouse/`
- `data/incoming/`

## Recommended Next Cleanup Step

If the user wants a deeper cleanup later, the safest next step is to create an archive manifest before moving anything:

```text
old_path,new_path,artifact_type,phase,keep_reason,regeneration_command
```

Only after that should generated artifacts be moved into an archive tree. The current cleanup intentionally stopped at documentation and index consolidation.
