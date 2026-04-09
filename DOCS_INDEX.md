# Docs Index

## What To Read First

If you only need the current official state, read these in order:

1. `PROJECT_INDEX.md`
2. `reports/current/current_mainline.json`
3. `reports/current/current_freeze_summary.md`
4. `reports/current/phase8_restart_playbook_20260409.md`
5. `RUNNERS_INDEX.md`

These five files are the canonical current entry points.

## Current Official State

- Official phase: `phase7`
- Official status: `frozen`
- Official tree family: `LightGBM`
- Current exploratory direction: `event + inventory`
- Current best phase8 exploratory baseline: `event_inventory_zero_split`

## Current Working Rules

- Use `reports/current/` for current conclusions.
- Use `PROJECT_INDEX.md` as the top-level current status page.
- Use `RUNNERS_INDEX.md` for executable entry points.
- Use `DOCS_INDEX.md` when you need to know which document is canonical and which is only historical context.

## Current Execution Entry Points

- Official freeze refresh:
  - `python scripts/runners/phase7/run_phase7_freeze.py`
- Official compare refresh:
  - `python scripts/runners/phase7/run_phase7i_full_model_compare.py`
- Phase8 prep:
  - `python scripts/runners/phase8/run_phase8a_prep.py`
- Phase8 inventory constraint pack:
  - `python scripts/runners/phase8/run_phase8f_inventory_constraint_pack.py`

## Document Roles

- `PROJECT_INDEX.md`
  - top-level current project status
- `RUNNERS_INDEX.md`
  - runner entry points and reproduction paths
- `reports/current/current_freeze_summary.md`
  - current official freeze explanation
- `reports/current/phase8_restart_playbook_20260409.md`
  - phase8 restart conditions and replay entry points
- `PROJECT_OVERVIEW.md`
  - broader project background; useful, but not the current canonical state page
- `REPO_MAP.md`
  - structural map of the repository; useful for orientation, not for current-state decisions
- `AGENTS.md`
  - execution and repository working rules for agent work

## Historical Material

Use these only when reproducing or auditing history:

- `reports/phase5*/`
- `reports/phase6*/`
- `reports/phase7*/`
- `scripts/runners/phase5/`
- `scripts/runners/phase6/`
- `scripts/runners/phase7/`
- `scripts/runners/phase8/`

## What Not To Assume

- Do not assume the old LSTM documentation describes the current official mainline.
- Do not treat historical phase runners as current recommended entry points.
- Do not treat all root markdown files as equal; current-state authority lives in `PROJECT_INDEX.md` and `reports/current/`.
