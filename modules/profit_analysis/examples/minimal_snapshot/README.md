# Minimal Profit-Analysis Snapshot Example

This folder contains a tiny, deterministic example for explaining the profit-analysis module without loading full project data.

## Run

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

## What This Example Shows

`AK1001` has two SKU rows, high positive-demand probabilities, zero current inventory, high price relative to cost, and therefore receives a positive SKC production recommendation.

`AK1002` has lower demand probability and already has 30 units of inventory. Even though the model still predicts some demand, the profit layer recommends `0` because producing the minimum batch is not economically justified.

Expected high-level result with the bundled calibration:

- input rows pass quality gates;
- `AK1001` receives a positive SKC plan;
- `AK1002` receives `0`;
- the `AK1001` SKC plan is allocated back to `AK1001-36` and `AK1001-38`.

The concrete output files are written under `output/` when the command is run.
