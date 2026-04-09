# Phase8 SHAP Summary

## Scope

- Model: current official phase7 LightGBM hurdle mainline
- Method: LightGBM native `pred_contrib=True` contribution outputs
- Classifier contributions are in raw-score space
- Regressor contributions are in log-quantity space
- Sep/Oct external calibration scales are not part of the additive SHAP decomposition

## Top Features: all

### Classifier

| feature | mean_abs_shap | mean_shap | sample_count |
| --- | --- | --- | --- |
| month | 0.5965 | 0.5234 | 40560 |
| band | 0.5506 | 0.0582 | 40560 |
| sku_id | 0.4948 | -0.0500 | 40560 |
| days_since_last_repl | 0.4531 | -0.2614 | 40560 |
| days_since_last_any_order | 0.3954 | -0.0331 | 40560 |
| qty_first_order | 0.3694 | 0.0145 | 40560 |
| style_id | 0.3428 | -0.0186 | 40560 |
| season | 0.3318 | 0.0312 | 40560 |
| qty_first_order_category_z | 0.2594 | -0.0071 | 40560 |
| style_sum_repl_30 | 0.2006 | 0.0118 | 40560 |

### Regressor

| feature | mean_abs_shap | mean_shap | sample_count |
| --- | --- | --- | --- |
| qty_first_order | 0.1329 | -0.0685 | 40560 |
| qty_first_order_x_repl0_fut0 | 0.0679 | -0.0016 | 40560 |
| season | 0.0676 | -0.0464 | 40560 |
| month | 0.0572 | -0.0482 | 40560 |
| days_since_last_any_order | 0.0495 | -0.0361 | 40560 |
| style_sum_repl_30 | 0.0486 | -0.0371 | 40560 |
| sku_id | 0.0453 | -0.0286 | 40560 |
| days_since_last_repl | 0.0423 | -0.0261 | 40560 |
| sum_repl_14 | 0.0422 | -0.0323 | 40560 |
| price_tag | 0.0415 | -0.0095 | 40560 |

## Top Features: 4_25

### Classifier

| feature | mean_abs_shap | mean_shap | sample_count |
| --- | --- | --- | --- |
| month | 1.3318 | 1.3317 | 2253 |
| days_since_last_any_order | 0.8355 | 0.7733 | 2253 |
| season | 0.5070 | 0.3343 | 2253 |
| style_sum_repl_30 | 0.4838 | 0.3878 | 2253 |
| band | 0.4500 | 0.3352 | 2253 |
| qty_first_order | 0.3815 | 0.3598 | 2253 |
| sku_id | 0.3434 | 0.2913 | 2253 |
| style_id | 0.2442 | 0.1225 | 2253 |
| qty_first_order_category_z | 0.2387 | 0.2079 | 2253 |
| days_since_last_repl | 0.2217 | 0.1845 | 2253 |

### Regressor

| feature | mean_abs_shap | mean_shap | sample_count |
| --- | --- | --- | --- |
| qty_first_order | 0.1287 | 0.0469 | 2253 |
| qty_first_order_x_repl0_fut0 | 0.0710 | 0.0463 | 2253 |
| style_sum_repl_30 | 0.0684 | 0.0186 | 2253 |
| sum_repl_14 | 0.0670 | 0.0196 | 2253 |
| season | 0.0637 | 0.0343 | 2253 |
| price_tag | 0.0492 | -0.0058 | 2253 |
| style_sum_repl_90 | 0.0477 | 0.0049 | 2253 |
| sku_id | 0.0470 | 0.0073 | 2253 |
| month | 0.0450 | 0.0044 | 2253 |
| qty_first_order_style_z | 0.0428 | 0.0209 | 2253 |

## Top Features: blockbuster

### Classifier

| feature | mean_abs_shap | mean_shap | sample_count |
| --- | --- | --- | --- |
| month | 1.6014 | 1.6014 | 279 |
| days_since_last_any_order | 0.7583 | 0.6973 | 279 |
| qty_first_order | 0.6780 | 0.6754 | 279 |
| style_sum_repl_30 | 0.6488 | 0.5565 | 279 |
| band | 0.4374 | 0.4276 | 279 |
| season | 0.3890 | 0.1007 | 279 |
| style_id | 0.3323 | 0.0036 | 279 |
| qty_first_order_category_z | 0.3298 | 0.3287 | 279 |
| sku_id | 0.2912 | 0.1960 | 279 |
| days_since_last_repl | 0.2709 | 0.2468 | 279 |

### Regressor

| feature | mean_abs_shap | mean_shap | sample_count |
| --- | --- | --- | --- |
| qty_first_order | 0.2772 | 0.2762 | 279 |
| qty_first_order_x_repl0_fut0 | 0.1826 | 0.1731 | 279 |
| sum_repl_14 | 0.1611 | 0.1354 | 279 |
| style_sum_repl_30 | 0.1433 | 0.0924 | 279 |
| sku_id | 0.0954 | 0.0758 | 279 |
| style_sum_repl_90 | 0.0755 | 0.0564 | 279 |
| price_tag | 0.0678 | 0.0444 | 279 |
| qty_first_order_category_z | 0.0654 | 0.0652 | 279 |
| season | 0.0599 | 0.0216 | 279 |
| style_sum_future_30 | 0.0566 | 0.0134 | 279 |

## Top Features: zero_true_fp

### Classifier

| feature | mean_abs_shap | mean_shap | sample_count |
| --- | --- | --- | --- |
| month | 0.6912 | 0.6738 | 5434 |
| days_since_last_any_order | 0.5101 | 0.3983 | 5434 |
| season | 0.4510 | 0.3618 | 5434 |
| band | 0.4121 | 0.3189 | 5434 |
| sku_id | 0.3255 | 0.3066 | 5434 |
| qty_first_order | 0.3251 | 0.2046 | 5434 |
| style_sum_repl_30 | 0.2665 | 0.1941 | 5434 |
| qty_first_order_category_z | 0.2295 | 0.0997 | 5434 |
| style_id | 0.2267 | 0.2096 | 5434 |
| days_since_last_repl | 0.1752 | -0.0000 | 5434 |

### Regressor

| feature | mean_abs_shap | mean_shap | sample_count |
| --- | --- | --- | --- |
| qty_first_order | 0.1328 | -0.0393 | 5434 |
| month | 0.0678 | -0.0567 | 5434 |
| season | 0.0531 | -0.0071 | 5434 |
| qty_first_order_x_repl0_fut0 | 0.0509 | -0.0196 | 5434 |
| style_sum_repl_90 | 0.0498 | -0.0246 | 5434 |
| days_since_last_any_order | 0.0492 | -0.0384 | 5434 |
| style_sum_repl_30 | 0.0427 | -0.0313 | 5434 |
| sum_repl_14 | 0.0417 | -0.0344 | 5434 |
| price_tag | 0.0390 | -0.0145 | 5434 |
| qty_first_order_style_z | 0.0373 | -0.0106 | 5434 |

## Output Files

- `reports/phase8a_prep/phase8_shap_global_summary.csv`
- `reports/phase8a_prep/phase8_shap_local_cases.csv`