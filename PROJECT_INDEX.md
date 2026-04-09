# Project Index

## Current Official State

- Current phase: `phase7`
- Current official mainline: `tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`
- Current official tree family: `LightGBM`
- Current phase8 working direction: `event + inventory` (shadow / exploratory only; not yet an official replacement)
- Current objective status: the model is frozen as the official research mainline and is ready for delivery-readiness work.

## Current Official Entry Points

- Docs index: `DOCS_INDEX.md`
- Mainline metadata: `reports/current/current_mainline.json`
- Freeze summary: `reports/current/current_freeze_summary.md`
- Phase8 restart playbook: `reports/current/phase8_restart_playbook_20260409.md`
- Official December compare page: `reports/current/current_model_compare.html`
- Official December compare CSV: `reports/current/current_model_compare.csv`
- Official December compare summary: `reports/current/current_model_compare_summary.md`
- Current data assets map: `data/current_assets.json`

## Current Data Scope

- Canonical gold table: `data/gold/wide_table_sku.csv`
- Canonical current reporting scope: cleaned gold universe
- Cleaned gold rule: SKUs that appear in orders but do not exist in `data/silver/clean_products.csv` are excluded from the current official universe.

## Where To Look Next

- Reports index: `reports/REPORTS_INDEX.md`
- Data index: `data/DATA_INDEX.md`
- Models index: `MODELS_INDEX.md`
- Runner index: `RUNNERS_INDEX.md`
- Phase8 prep outputs: `reports/phase8a_prep/`

## Historical References

- Official phase freeze: `reports/phase7/`
- Official December model compare source: `reports/phase7i_full_model_compare/`
- Historical tree stabilization and validation: `reports/phase6*/`
- Historical tree construction and anchor work: `reports/phase5*/`

## Working Rule

- Use `reports/current/` and `data/current_assets.json` for current references.
- Use phase-specific directories only when reproducing or auditing historical work.
- If the official state changes, update `PROJECT_INDEX.md`, `readme.md`, and the corresponding files under `reports/current/` in the same commit.
