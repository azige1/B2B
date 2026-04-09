# Data Index

## Base Layers

- `raw/`
  - Imported source data before cleaned joins and modeling transforms.
- `silver/`
  - Cleaned operational tables such as products, orders, stores, and buyer profile.
- `gold/`
  - Canonical modeling-facing merged tables.
  - Current official gold table: `gold/wide_table_sku.csv`

## Current Feature Assets

- Current official feature family: `v6_event`
- Current official feature set: `cov_activity_tail_full`
- Current official phase7 anchor assets are the `p7b_*` processed/artifacts directories.
- Canonical map: `current_assets.json`

## Experiment Assets

- `processed_*`
- `artifacts_*`
- `phase8a_prep/`
  - Pre-client-reply standardized feature tables for phase8 preparation.

These directories are retained in place for compatibility. They are indexed in:

- `experiment_asset_inventory.csv`

## Legacy Assets

- Older `v3`, `v5`, `weekly`, and early `v6_event` asset directories remain available for audit and reproduction.
- Legacy assets are not the default reference unless explicitly needed.

## Current Universe Rule

- The current official universe uses cleaned gold.
- SKUs that exist in orders but do not exist in `silver/clean_products.csv` are excluded from the current official gold scope.

## Working Rule

- Do not move or rename base data paths in this cleanup step.
- Use `current_assets.json` for current canonical references.
- Use `experiment_asset_inventory.csv` for locating historical or experimental assets.
