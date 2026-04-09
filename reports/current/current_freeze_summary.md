# Current Freeze Summary

## Official State

- Current official phase: `phase7`
- Current official mainline: `tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`
- Current official tree family: `LightGBM`
- Previous official mainline: `p57_covact_lr005_l63_hard_g025 + sep098_oct093`

## Why This Is The Official Mainline

- `phase7` improved the current tree mainline on:
  - `4_25`
  - `Ice 4_25`
  - `>25 / blockbuster`
  - allocation
- The winner passed the replacement gate and became the official frozen mainline.

## Official Metrics

- `4_25_under_wape = 0.3566`
- `4_25_sku_p50 = 0.6588`
- `ice_4_25_sku_p50 = 0.5853`
- `blockbuster_under_wape = 0.4165`
- `blockbuster_sku_p50 = 0.5539`
- `top20_true_volume_capture = 0.6494`
- `rank_corr_positive_skus = 0.8044`
- `global_ratio = 1.0159`
- `global_wmape = 0.6863`
- `1_3_ratio = 1.2999`

## Canonical Sources

- Freeze metadata: `reports/current/current_mainline.json`
- Phase freeze source: `reports/phase7/phase7_frozen_mainline.json`
- Official compare page: `reports/current/current_model_compare.html`
- Official compare CSV: `reports/current/current_model_compare.csv`

## Working Rule

- Use this directory as the default current reference.
- Use `reports/phase7/` and `reports/phase7i_full_model_compare/` only when you need the historical source files behind the current canonical copies.
