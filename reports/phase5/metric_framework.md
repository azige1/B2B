# Metric Framework

## Core Principle

This project predicts **future 30-day replenish quantity per SKU**.

No single metric is sufficient. Metrics are grouped by role:

- primary gate metrics: decide whether the current tree mainline is acceptable
- secondary formal metrics: enter fixed reports and ranking, but do not alone decide pass/fail
- diagnostic metrics: explain failure modes and stability

## Primary Gate Metrics

- `4_25_sku_p50`
  - typical SKU calibration in the main demand band
- `4_25_wmape_like`
  - aggregate error in the main demand band
- `ice_4_25_sku_p50`
  - cold-start performance inside the main demand band
- `global_ratio`
  - total-volume guardrail
- `1_3_ratio`
  - low-demand overprediction guardrail

## Secondary Formal Metrics

- `4_25_ratio`
- `4_25_under_wape`
- `blockbuster_ratio`
- `blockbuster_wmape_like`
- `blockbuster_sku_p50`
- `blockbuster_under_wape`
- `blockbuster_within_50pct_rate`
- `top20_true_volume_capture`
- `rank_corr_positive_skus`

`>25` is treated as a secondary formal slice: it must be tracked in every summary and ranking, but it does not currently override the main gate built around `4-25` and `Ice 4-25`.

## Distribution Metrics

For global and each slice (`1_3`, `4_10`, `11_25`, `4_25`, `ice`, `ice_4_25`, `blockbuster`), the framework now includes:

- `ratio`
- `wmape_like`
- `sku_p50`
- `sku_trimmed_mean_10`
- `within_20pct_rate`
- `within_30pct_rate`
- `within_50pct_rate`
- `catastrophic_under_rate`
- `catastrophic_over_rate`
- `under_wape`
- `over_wape`

Interpretation:

- `ratio` measures aggregate scale
- `sku_p50` measures the typical SKU
- `wmape_like` measures aggregate absolute error
- `under_wape` measures stockout-side risk
- `over_wape` measures overstock-side risk
- `within_*` measures usable coverage
- `catastrophic_*` measures extreme failures

## Ranking / Allocation Metrics

- `rank_corr_positive_skus`
  - Spearman correlation between predicted and true SKU totals
- `top10_true_volume_capture`
  - true replenish volume captured by the top 10% SKUs ranked by predicted quantity
- `top20_true_volume_capture`
  - true replenish volume captured by the top 20% SKUs ranked by predicted quantity

These do not replace calibration metrics. They measure whether the model allocates attention to the right SKUs.

## 4-25 Official Evaluation Bundle

`4-25` is never judged by `4_25_sku_p50` alone.

The official bundle is:

- `4_25_ratio`
- `4_25_sku_p50`
- `4_25_wmape_like`
- `4_25_under_wape`

Interpretation:

- `ratio`: overall volume for the band
- `sku_p50`: typical SKU in the band
- `wmape_like`: aggregate error in the band
- `under_wape`: shortage risk in the band

## Diagnostic Metrics

- `AUC`
- `F1`
- `false_zero_rate`
- `false_positive_rate_zero_true`
- `zero_true_pred_ge_3_rate`
- `true_gt_10_pred_le_1_rate`
- `category_worst5_ratio`
- `category_worst5_under_wape`
- `repl0_fut0 / repl0_fut1 / repl1_fut0 / repl1_fut1`
- `blockbuster_top10_true_volume_capture`
- `blockbuster_top20_true_volume_capture`

These are explanatory diagnostics. They do not directly control the current mainline gate.
