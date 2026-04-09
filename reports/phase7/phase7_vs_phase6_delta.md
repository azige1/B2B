# Phase7 vs Phase6 主线对照

| metric | direction | phase6 | phase7 | delta |
| --- | --- | ---: | ---: | ---: |
| `4_25_under_wape` | 越低越好 | 0.4236 | 0.3566 | -0.0670 |
| `4_25_sku_p50` | 越高越好 | 0.5986 | 0.6588 | +0.0602 |
| `ice_4_25_sku_p50` | 越高越好 | 0.5069 | 0.5853 | +0.0784 |
| `blockbuster_under_wape` | 越低越好 | 0.5507 | 0.4165 | -0.1342 |
| `blockbuster_sku_p50` | 越高越好 | 0.4048 | 0.5539 | +0.1491 |
| `top20_true_volume_capture` | 越高越好 | 0.6213 | 0.6494 | +0.0281 |
| `rank_corr_positive_skus` | 越高越好 | 0.7365 | 0.8044 | +0.0679 |
| `global_ratio` | 接近 1 更好 | 1.0188 | 1.0159 | -0.0029 |
| `global_wmape` | 越低越好 | 0.8526 | 0.6863 | -0.1663 |
| `1_3_ratio` | 当前仅 guardrail | 1.3801 | 1.2999 | -0.0802 |

## 判定

- `phase7` 新主线在核心业务段、cold-start、tail、allocation 上都优于 `phase6` 旧主线。
- 本轮不是单点指标偶然改善，而是结构性提升。
- `phase7` 现在可以作为新的正式主线引用。
