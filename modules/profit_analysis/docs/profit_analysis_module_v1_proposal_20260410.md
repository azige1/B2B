# Profit Analysis Module V1 Proposal

- Date: `2026-04-10`
- Scope: proposal only
- Goal: define the most suitable profit-analysis module for the current tree-based replenishment model

## Bottom Line

The most suitable V1 module is:

- a `single-SKU`
- `30-day horizon`
- `replenishment-profit simulator and candidate-plan recommender`

It should answer one practical question:

- for a given SKU, how many units should we replenish now to maximize expected 30-day profit under the current prediction model

It should **not** start as:

- a multi-batch production scheduler
- a day-level production timing optimizer
- a gradient-based manufacturing planner

Those are mismatched with the current model output granularity.

## Why This Is The Right Fit

The current official tree model provides:

- probability that future replenish quantity is positive
- a 30-day replenish quantity point estimate
- business-slice strength on `4_25`, `ice_4_25`, `blockbuster`, and ranking

The current model does **not** provide:

- a stable day-level demand curve
- a full demand probability distribution
- a native production-batch schedule output

Therefore the profit module should consume:

- `30-day quantity forecast`
- `positive probability`
- `current inventory`
- `basic economics`

and convert them into:

- expected profit
- stockout risk
- leftover risk
- recommended replenish quantity

## Module Position In The Project

The project should be treated as two linked but separate parts:

1. replenishment prediction
   - predicts whether a SKU will replenish and how much over the next 30 days
2. profit analysis
   - evaluates candidate replenish plans using the prediction output plus inventory and economics

The prediction module remains the upstream source of demand information.
The profit module becomes a downstream decision layer.

## V1 Module Definition

### Core function 1

`assess_replenishment_plan(sku_id, demand_scenarios, inventory_state, economics, plan) -> profit_summary`

Purpose:

- evaluate one candidate replenish plan for one SKU over the next 30 days

### Core function 2

`recommend_replenishment_plans(sku_id, model_output, inventory_state, economics, policy) -> ranked_plans`

Purpose:

- generate a small set of candidate replenish quantities
- evaluate each candidate through `assess_replenishment_plan`
- return top recommendations

## Input Design

### 1. Current model output

The current tree model output should be normalized into:

- `pred_prob_positive`
- `pred_qty_30d`

Optional future extension:

- `bucket_probs_30d`
- `p50_qty_30d`
- `p90_qty_30d`

### 2. Inventory state

Minimum V1 fields:

- `current_inventory`
- `inventory_snapshot_date`
- `lead_time_days`

Optional:

- `min_batch_qty`
- `max_replenish_qty`

### 3. Economics

Minimum V1 fields:

- `unit_cost`
- `unit_price`
- `holding_cost_per_unit_per_day`
- `salvage_value_per_unit`

Optional:

- `stockout_penalty_per_unit`
- `other_fixed_cost`
- `lifecycle_end_date`

### 4. Candidate plan

For V1, the plan should be intentionally simple:

- `plan_qty`

Optional:

- `arrival_day`

V1 should assume a single replenish decision instead of multiple production batches.

## Demand Representation

The current model is not a full probabilistic forecaster, so V1 should use:

- `discrete demand scenarios`

instead of a continuous demand distribution.

### Recommended scenario construction

From:

- `p = pred_prob_positive`
- `q = pred_qty_30d`

construct:

- `zero`
  - demand = `0`
  - probability = `1 - p`
- `low`
  - demand = `0.6 * q`
- `base`
  - demand = `1.0 * q`
- `high`
  - demand = `1.5 * q`

The remaining positive probability mass should be distributed across `low/base/high`.

Default practical split:

- `zero`: `1 - p`
- `low`: `0.25 * p`
- `base`: `0.50 * p`
- `high`: `0.25 * p`

This is not the final probabilistic target of the research project.
It is the most practical bridge from the current point-forecast tree model to a decision module.

## Profit Simulation Logic

For each demand scenario:

- `available_qty = current_inventory + plan_qty`
- `sold_qty = min(available_qty, demand_qty)`
- `leftover_qty = max(available_qty - demand_qty, 0)`
- `lost_sales_qty = max(demand_qty - available_qty, 0)`

Then calculate:

- `sales_profit = sold_qty * (unit_price - unit_cost)`
- `salvage_profit = leftover_qty * salvage_value_per_unit`
- `holding_cost = leftover_qty * holding_cost_per_unit_per_day * avg_days_held`
- `replenish_cost = plan_qty * unit_cost`
- `stockout_cost = lost_sales_qty * stockout_penalty_per_unit`

Suggested V1 profit function:

`profit = sales_profit + salvage_profit - holding_cost - replenish_cost - stockout_cost`

If `stockout_penalty_per_unit` is not available, V1 can set it to `0` and expose it as a configurable sensitivity parameter.

## Candidate Plan Recommendation

V1 should use:

- `enumeration + simulation + ranking`

and not gradient optimization.

### Candidate quantity generation

Recommended default candidate set:

- `0`
- `min_batch_qty`
- `max(0, round(pred_qty_30d - current_inventory))`
- `round(0.8 * pred_qty_30d)`
- `round(1.0 * pred_qty_30d)`
- `round(1.2 * pred_qty_30d)`
- `round(1.5 * pred_qty_30d)`

If batch-size constraints exist, round all candidates to the nearest allowed batch multiple.

### Ranking outputs

Each candidate should return:

- `expected_profit`
- `profit_std` or scenario-spread proxy
- `stockout_rate`
- `expected_leftover_qty`
- `sell_through_rate`

The recommender should then return:

- `recommended_conservative_plan`
- `recommended_balanced_plan`
- `recommended_aggressive_plan`

## Output Design

### `profit_summary`

Recommended fields:

- `sku_id`
- `plan_qty`
- `expected_profit`
- `profit_variance`
- `stockout_rate`
- `expected_sold_qty`
- `expected_leftover_qty`
- `expected_lost_sales_qty`
- `sell_through_rate`
- `scenario_breakdown`

### `ranked_plans`

Recommended fields:

- `sku_id`
- `prediction_snapshot`
- `inventory_snapshot`
- `economics_snapshot`
- `ranked_candidates`
- `best_balanced_plan`
- `best_profit_plan`
- `lowest_risk_plan`

## Policy Layer

The policy input should be simple in V1.

Recommended policy modes:

- `conservative`
- `balanced`
- `aggressive`

Interpretation:

- `conservative`
  - prioritize lower leftover risk
- `balanced`
  - prioritize expected profit
- `aggressive`
  - allow higher leftover risk for higher upside

This is more suitable than introducing a large optimization parameter surface in the first version.

## What V1 Explicitly Does Not Try To Solve

V1 should not try to solve:

- true day-level production scheduling
- multi-batch continuous optimization
- exact stockout-censored demand recovery
- factory-capacity coordination across many SKUs

Those require either:

- richer demand-distribution output
- stronger lifecycle / supply data
- a separate operations-research layer

## Data Dependencies

V1 requires only a modest amount of new data beyond the current prediction stack:

- current inventory
- lead time
- unit cost
- unit price
- holding cost
- salvage value

Helpful but optional:

- lifecycle end date
- batch-size constraints
- stockout penalty proxy

## Recommended Implementation Order

### Step 1

Build:

- `assess_replenishment_plan`

for a single SKU and a single candidate quantity.

### Step 2

Build:

- scenario generator from current tree-model output

using `pred_prob_positive + pred_qty_30d`.

### Step 3

Build:

- candidate enumerate-and-rank recommender

for one SKU.

### Step 4

Add:

- policy presets
- sensitivity analysis
- basic report output

## Future Upgrade Path

If the prediction model later supports bucket probabilities or quantile outputs, the profit module can be upgraded naturally:

- replace heuristic scenarios with model-derived scenarios
- move from point-estimate-centered simulation to bucket-distribution simulation
- then consider multi-batch planning

This keeps V1 fully compatible with future model upgrades.

## Final Recommendation

For the current project state, the most suitable profit-analysis module is:

- a `30-day`
- `single-SKU`
- `single-replenish-decision`
- `scenario-based profit simulator and recommender`

This is the best match for the current official tree model, the cleanest engineering boundary, and the lowest-risk path to a usable second module.
