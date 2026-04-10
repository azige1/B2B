# Profit Analysis Module

This directory contains the standalone profit-analysis module for the B2B replenishment project.

## Scope

The module is intentionally scoped as:

- `single SKU`
- `30-day horizon`
- `single replenishment decision`
- `profit simulation + candidate-plan recommendation`

It is designed to sit downstream of the current tree-based replenishment model.

## Structure

- `src/profit_analysis/`
  - core dataclasses, scenario construction, profit simulation, and plan recommendation
- `scripts/`
  - executable entry points for batch snapshot runs
- `config/`
  - normalized input CSV templates for prototype use
- `docs/`
  - module-specific design docs and data mapping notes

## Current Entry Points

- Chinese V1 design:
  - `modules/profit_analysis/docs/盈亏分析模块V1技术方案_20260410.md`
- English V1 design:
  - `modules/profit_analysis/docs/profit_analysis_module_v1_proposal_20260410.md`
- Data mapping:
  - `modules/profit_analysis/docs/profit_analysis_data_mapping_20260410.md`
- Prototype runner:
  - `python modules/profit_analysis/scripts/run_profit_analysis_snapshot.py`
- Input builder:
  - `python modules/profit_analysis/scripts/build_profit_analysis_inputs.py --prediction-csv <your_prediction_csv>`

## Current Workflow

1. Normalize or build three input snapshots through:
   - `python modules/profit_analysis/scripts/build_profit_analysis_inputs.py --prediction-csv <your_prediction_csv>`
2. Run the profit-analysis prototype on the normalized snapshots:
   - `python modules/profit_analysis/scripts/run_profit_analysis_snapshot.py`

## Notes

- Project-level status pages still live under `PROJECT_INDEX.md`, `DOCS_INDEX.md`, and `reports/current/`.
- This module folder is the canonical home for profit-analysis implementation assets.
