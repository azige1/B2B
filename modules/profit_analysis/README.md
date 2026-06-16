# Profit Analysis Module

This directory is the standalone home for the profit-analysis module of the B2B replenishment project.

All profit-analysis code, configuration templates, design documents, and source notes should live under this directory. Do not place profit-analysis implementation assets in the main replenishment module root or `reports/current`.

## Scope

The production decision flow is scoped as:

- `SKC-level decision` (`style_id` is the client-confirmed SKC key)
- `SKU-level size allocation and realized inventory simulation`
- configurable `30/45-day` analysis horizon
- `single replenishment decision`
- `daily inventory-flow profit simulation + candidate-plan recommendation`

It sits downstream of the current Phase7 LightGBM hurdle replenishment model.
The client business rule uses a 45-day lifecycle window. The repository daily
demand table covers both strict inventory anchors through their complete future
45-day windows, so the current leakage-safe experiment now uses actual 45-day
labels and daily demand curves. Phase7 quantity predictions are still 30-day
outputs and are linearly extended before applying empirically calibrated
45-day demand scenarios.

The implementation follows the 2026-04-30 feedback rules:

- production batch minimum is `100` units per SKC, with `10` unit increments;
- lifecycle residual value is `0` at the lifecycle end;
- if no explicit cost is available, unit cost defaults to `price_tag / 7`;
- target sell-through rate is `85%` within 45 days;
- prepayment and storage cost can stay at `0` unless a business config overrides them.

Core formula for each demand scenario and each SKU after SKC allocation:

```text
sales_qty_t = min(opening_inventory_t, demand_qty_t)
lost_sales_qty_t = max(demand_qty_t - sales_qty_t, 0)
ending_inventory_t = opening_inventory_t - sales_qty_t

sales_revenue = sum_t(sales_qty_t) * unit_price
terminal_value = ending_inventory_H * salvage_value_per_unit
replenish_cost = plan_qty * unit_cost
holding_cost = sum_t(ending_inventory_t * holding_cost_per_unit_per_day)
stockout_cost = sum_t(lost_sales_qty_t) * stockout_penalty_per_unit

profit = sales_revenue + terminal_value
         - replenish_cost - holding_cost - stockout_cost - other_fixed_cost
```

With the current client-feedback defaults, `terminal_value = 0` and `holding_cost = 0`.
The current product file provides a 90-day `PL_CYCLE`, but backtest evidence shows
continued replenishment after `launch_date + 90 days`. Therefore `PL_CYCLE` is
preserved as a soft source field and is not used as a hard cutoff by default.

## Structure

- `src/profit_analysis/`
  - core dataclasses, scenario construction, profit simulation, and plan recommendation
- `scripts/`
  - executable entry points for batch snapshot runs
- `config/`
  - normalized input CSV templates for prototype use
- `docs/`
  - module-specific design docs and data mapping notes
- `docs/source_materials/`
  - original client / advisor notes copied into the module for handoff

## Current Entry Points

- Current status and runnable example:
  - `modules/profit_analysis/docs/profit_analysis_current_status_20260616.md`
  - `modules/profit_analysis/examples/minimal_snapshot/`
- Current detailed Chinese design and experiment report:
  - `modules/profit_analysis/docs/profit_analysis_module_v1_detailed_design_20260522.md`
- Delivery and acceptance checklist:
  - `modules/profit_analysis/docs/profit_analysis_module_delivery_20260612.md`
- English V1 design:
  - `modules/profit_analysis/docs/profit_analysis_module_v1_proposal_20260410.md`
- Data mapping:
  - `modules/profit_analysis/docs/profit_analysis_data_mapping_20260410.md`
- Source notes:
  - `modules/profit_analysis/docs/source_materials/client_profit_analysis_note.md`
  - `modules/profit_analysis/docs/source_materials/profit_analysis_initial_concept.docx`
- Prototype runner:
  - `python modules/profit_analysis/scripts/run_profit_analysis_snapshot.py`
- Input builder:
  - `python modules/profit_analysis/scripts/build_profit_analysis_inputs.py --prediction-csv <your_prediction_csv>`
- Leakage-safe SKU diagnostic experiment:
  - `python modules/profit_analysis/scripts/run_real_data_experiment.py`
- SKC decision + SKU allocation real-data experiment:
  - `python modules/profit_analysis/scripts/run_skc_real_data_experiment.py`
- Production-style SKC snapshot decision:
  - `python modules/profit_analysis/scripts/run_skc_profit_snapshot.py --prediction-csv <prediction.csv> --inventory-csv <inventory.csv> --economics-csv <economics.csv>`

## Current Real-Data Evidence

The reproducible SKC experiment is under:

`reports/profit_analysis_skc_real_cost_h45_20260612/`

Phase7, balanced policy, tag-price 100% sensitivity:

- `2,682` complete SKC-anchor observations and `7,853` SKU-anchor observations
- direct model-gap ordering: `210` positive SKC plans, `204` harmful versus no replenishment
- profit module: `3` positive SKC plans, all beneficial versus no replenishment
- total module plan: `300` units
- proxy incremental profit versus no replenishment: `+130,980.00`
- proxy incremental profit versus direct ordering: `+4,045,606.24`

These amounts are proxy economics, not audited financial profit. Inventory,
future demand, daily timing, product identities, and tag prices are real project
data. The client cost workbook is joined exactly by `style_id`; unmatched SKCs
still use the confirmed fallback `price_tag / 7`. Actual transaction price,
production lead time, and confirmed inbound orders are not yet available.

Cost intake:

- Normalize the client workbook:
  - `python modules/profit_analysis/scripts/normalize_style_costs.py --source-xlsx <cost_workbook.xlsx>`
- Normalized cost table:
  - `data/incoming/profit_analysis/style_costs_2024_2026.csv`
- Cost audit:
  - `reports/current/profit_analysis_style_cost_audit_20260612.md`

## Current Workflow

Fast deterministic smoke example:

```powershell
python modules/profit_analysis/scripts/run_skc_profit_snapshot.py `
  --prediction-csv modules/profit_analysis/examples/minimal_snapshot/prediction.csv `
  --inventory-csv modules/profit_analysis/examples/minimal_snapshot/inventory.csv `
  --economics-csv modules/profit_analysis/examples/minimal_snapshot/economics.csv `
  --policy balanced `
  --horizon-days 45 `
  --run-id minimal_demo `
  --output-dir modules/profit_analysis/examples/minimal_snapshot/output
```

1. If your upstream file is still a raw eval/detail CSV, normalize it first:
   - `python modules/profit_analysis/scripts/normalize_prediction_snapshot.py --source-csv <your_raw_prediction_csv>`
2. Normalize or build three input snapshots through:
   - `python modules/profit_analysis/scripts/build_profit_analysis_inputs.py --prediction-csv <your_prediction_snapshot_csv>`
3. Run the profit-analysis prototype on the normalized snapshots:
   - `python modules/profit_analysis/scripts/run_profit_analysis_snapshot.py`
4. Backtest the strategy families against realized demand:
   - `python modules/profit_analysis/scripts/backtest_profit_analysis.py --source-csv <your_raw_prediction_csv> --horizon-days 45`
5. For the current client batch rule, run the SKC experiment rather than
   interpreting a SKU-level `100`-unit minimum:
   - `python modules/profit_analysis/scripts/run_skc_real_data_experiment.py`
6. For a production snapshot, run the strict quality-gated SKC interface:
   - `python modules/profit_analysis/scripts/run_skc_profit_snapshot.py --prediction-csv <prediction.csv> --inventory-csv <inventory.csv> --economics-csv <economics.csv>`

The production runner rejects duplicate keys, missing inventory/economics,
invalid probabilities or prices, absent `style_id`, and allocation-total
mismatches. It writes SKC recommendations, SKU allocations, candidate details,
a quality report, and a run manifest.

## Lifecycle-Aware Simulation

If `economics_config` includes `lifecycle_end_date`, the module caps the requested simulation window to the remaining lifecycle:

```text
effective_horizon_days = min(requested_horizon_days, lifecycle_end_date - snapshot_date + 1)
```

When production arrival is after `lifecycle_end_date`, the output flags `late_arrival_risk = 1`. This prevents a late replenishment plan from looking attractive when there is no remaining selling window.

The lifecycle source can be either:

- legacy product metadata with `NO`, `LISTING_DATE`, `PL_CYCLE`
- lifecycle V0 features from `src/analysis/build_launch_date_lifecycle_v0_features.py`, with `sku_id`, `launch_date`, `estimated_lifecycle_end_date`, and `lifecycle_days_assumption`

`requested_horizon_days` is the profit simulation window. A hard
`lifecycle_end_date` should come from a confirmed off-shelf or selling-end field.
The launch-date builder can derive a provisional hard end from `PL_CYCLE` only
when explicitly run with `--derive-hard-lifecycle-end`; this mode is disabled by
default until the client confirms the field meaning.

## Notes

- Project-level status pages still live under `PROJECT_INDEX.md`, `DOCS_INDEX.md`, and `reports/current/`.
- This module folder is the canonical home for profit-analysis implementation assets.
- If a new profit-analysis document or script is added, put it in this folder first and link it from project-level indexes only when necessary.
