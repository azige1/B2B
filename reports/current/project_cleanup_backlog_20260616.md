# Project Cleanup Backlog (2026-06-16)

## Current Cleanup State

The current handoff layer has been cleaned and pushed:

- Phase8 staged conclusion
- Server whitelist refresh
- Daily Oracle snapshot automation
- Root README and project indexes
- Repository structure guide

This pass added artifact policy and a workspace inventory so the remaining local changes can be reviewed without committing generated data or model binaries.

## What Is Now Ignored

`.gitignore` now excludes these generated/local categories:

- `data/incoming/`
- `data/phase8_*/`
- `models_phase*/`
- `*.pid`
- `*.tar.gz`
- `*.zip`
- `*.sha256`

This keeps server snapshots, large local experiment models, compressed exports, and run markers out of Git by default.

## Remaining Working-Tree Categories

### 1. Prediction Source Code Pending Review

These are real source-code changes and should be reviewed before any commit:

- `evaluate_tabular.py`
- `src/etl/build_wide_table.py`
- `src/etl/clean_data.py`
- `src/features/build_features_v5_lite_sku.py`
- `src/features/build_features_v6_event_inventory_shadow_sku.py`
- `src/features/build_features_v6_event_sku.py`
- `src/features/phase53_feature_utils.py`
- `src/train/train_tabular_v6.py`
- untracked Phase8 analysis/runner scripts not included in the last commit

Recommended next action:

- Inspect diffs by file.
- Compile changed Python files.
- Commit only if they are still needed for current Phase8 or reproducible historical experiments.

### 2. Profit Analysis Module Pending Separate Commit

The profit-analysis module has many pending code, config, doc, and test files. It should be handled as one focused module commit, not mixed with prediction cleanup.

Recommended next action:

- Run module tests/smoke checks.
- Review docs and source boundaries.
- Commit under a message such as `Organize profit analysis module`.

### 3. Current Reports Pending Curation

There are many untracked `reports/current/` documents from earlier work. Some are useful handoff documents, but they should not all be blindly committed.

Recommended next action:

- Keep current canonical reports referenced by `PROJECT_INDEX.md`.
- Commit only reports that explain active decisions or external communication.
- Leave obsolete or duplicate reports local unless an archive index references them.

### 4. Historical Experiment Reports

Many `reports/phase8_*` and `reports/profit_analysis_*` directories remain untracked. They are generated experiment outputs.

Recommended next action:

- Do not commit whole directories by default.
- If a result is current, commit only small summaries and metric tables.
- Otherwise record it in an artifact inventory or archive manifest.

### 5. Data Manifests

`data/manifests/phase8_data_snapshot_20260614.json` is still untracked.

Recommended next action:

- Commit it if `20260614` remains a referenced historical data checkpoint.
- Otherwise leave it local; `20260616` is already the current registered snapshot.

## Recommended Cleanup Order

1. Commit this artifact-policy cleanup.
2. Review and commit the profit-analysis module separately.
3. Review pending prediction source-code diffs separately.
4. Curate useful `reports/current/` leftovers.
5. Only after that, consider a physical archive move for large generated report/model directories.

## Do Not Do Yet

- Do not delete `data/`, `data_warehouse/`, or `models_phase*/`.
- Do not move report directories until an archive manifest exists.
- Do not commit large raw CSV, model binaries, or full prediction-context files.
- Do not claim new model metrics on `V_IRS_ORDERFTP_6_16.csv` until silver/gold and feature assets are rebuilt.
