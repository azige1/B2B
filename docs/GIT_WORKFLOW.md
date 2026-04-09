# Git Workflow

## Purpose

This repository uses Git to track:

- source code
- configs
- runner scripts
- key project documentation
- current decision records and phase summaries

Git is not the system of record for raw data, processed tensors, model weights, or large rendered outputs.

## What Should Be Tracked

- `src/`
- `config/`
- `docs/`
- root runner scripts such as `run_phase*.py`
- root evaluation and analysis scripts
- key reference docs such as `PROJECT_INDEX.md`
- curated summaries under `reports/current/`
- selected phase conclusions under `reports/phase*/`

## What Should Not Be Tracked

- `data/raw/`
- `data/silver/*.csv`
- `data/gold/*.csv`
- `data/processed_*/`
- `data/artifacts_*/`
- `data_warehouse/**/*.csv`
- `data_warehouse/**/*.parquet`
- `models/`
- rendered report exports such as `reports/**/*.csv` and `reports/**/*.html`
- local caches, notebooks checkpoints, and vendored runtime folders

If a large file is important for reproducibility, record its path and generation rule in a tracked markdown or json file instead of committing the file itself.

## Commit Rules

- Keep one logical change per commit.
- Separate code changes from report-only updates when practical.
- Do not mix `.gitignore` changes, model code changes, and experiment conclusions in one commit unless they are tightly coupled.
- Use commit messages that describe the actual change, not the activity. Prefer examples like:
  - `Freeze phase7 current mainline metadata`
  - `Add phase8 inventory constraint analysis pack`
  - `Tighten gitignore for generated report outputs`

## Branch Rules

- `main` should stay usable and readable.
- Use short-lived branches for isolated work.
- Prefer branch names with clear intent, for example:
  - `codex/phase8-inventory-features`
  - `codex/orderftp-label-audit`
  - `codex/report-cleanup`

## Reports Rules

- Treat `reports/current/` as the canonical current handoff area.
- Treat phase directories as historical evidence.
- Do not commit bulk evaluation logs by default.
- Only commit a report artifact when it records a conclusion you expect future readers to rely on.

## Data Change Rules

- If a code change depends on a new data assumption, document that assumption in a tracked file.
- If a feature source changes business meaning, update the relevant current note before or with the code change.
- If a raw extract is refreshed, do not commit the extract; commit only the code or docs that explain how it is used.

## First Checks Before Commit

1. Run `git status` and make sure only intended files are staged.
2. Check that no large generated artifacts were picked up by mistake.
3. Confirm that current-facing docs still point to the right official mainline.
4. If code changed, include a short note in the commit or adjacent doc about why.
