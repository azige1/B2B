# Models Index

## Current Official Model Directory

- `models/current_phase7_mainline/`
  - Current official phase7 winner files are preserved here.
  - Current official raw model family:
    - `tail_full_lr005_l63_g027_n800_s2028`
  - Preserved scope:
    - four-anchor winner model files for `2025-09-01 / 2025-10-01 / 2025-11-01 / 2025-12-01`
    - classifier / regressor / hard-gate meta files

## Current Official Model Lineage

- The official winner was selected from:
  - `reports/phase7_tail_allocation_optimization/overnight_20260402_overnight/overnight_winner.json`
- The official frozen summary remains at:
  - `reports/current/current_mainline.json`
  - `reports/current/current_freeze_summary.md`

## Workspace Cleanup Rule

- Historical model directories and non-current intermediate model families have been pruned from the workspace to save disk space.
- If an old experiment needs to be revisited, its model artifacts should be regenerated from:
  - raw data in `data_warehouse/`
  - cleaned tables in `data/silver` and `data/gold`
  - current source code under `src/`

## Working Rule

- Treat `models/current_phase7_mainline/` as the only current official model directory in the workspace.
- Treat report files under `reports/current/` as the canonical current entrypoint.
- Do not assume deleted historical model directories still exist locally; rebuild them only when necessary.
