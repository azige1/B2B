# Phase8 Residual Gap Summary

## Scope

- Source: current official phase7 mainline
- Evaluation base: recomputed four-anchor official mainline contexts with official Sep/Oct calibration scales
- Metric summary rule: per-anchor evaluation first, then four-anchor mean; combined all-anchor context is used only for residual case mining
- Formal winner status remains unchanged; this is a residual-gap analysis only

## Official Metrics vs Recomputed Check

| metric | official_phase7 | recomputed_all_anchors |
| --- | --- | --- |
| global_ratio | 1.0159 | 1.0159 |
| global_wmape | 0.6863 | 0.6863 |
| 4_25_under_wape | 0.3566 | 0.3566 |
| 4_25_sku_p50 | 0.6588 | 0.6588 |
| ice_4_25_sku_p50 | 0.5853 | 0.5853 |
| blockbuster_under_wape | 0.4165 | 0.4165 |
| blockbuster_sku_p50 | 0.5539 | 0.5539 |
| top20_true_volume_capture | 0.6494 | 0.6494 |
| rank_corr_positive_skus | 0.8044 | 0.8044 |
| 1_3_ratio | 1.2999 | 1.2999 |

## Weak-Signal Blockbuster Residuals

| signal_quadrant | rows | total_true | total_pred | ratio | under_wape |
| --- | --- | --- | --- | --- | --- |
| repl0_fut0 | 144 | 6745.0000 | 3314.8538 | 0.4915 | 0.5085 |
| repl1_fut0 | 30 | 1088.0000 | 402.3715 | 0.3698 | 0.6302 |

## Blockbuster Worst Categories

| category | rows | total_true | total_pred | ratio | under_wape |
| --- | --- | --- | --- | --- | --- |
| 棉服 | 3 | 192.0000 | 46.1554 | 0.2404 | 0.7596 |
| 西装 | 1 | 41.0000 | 10.6918 | 0.2608 | 0.7392 |
| 套装 | 1 | 26.0000 | 9.5984 | 0.3692 | 0.6308 |
| 毛衣开衫 | 10 | 373.0000 | 142.2932 | 0.3815 | 0.6185 |
| 卫衣 | 2 | 89.0000 | 40.2449 | 0.4522 | 0.5478 |
| 马夹 | 3 | 98.0000 | 45.2706 | 0.4619 | 0.5381 |
| 裙子 | 18 | 657.0000 | 340.4946 | 0.5183 | 0.4918 |
| 外套 | 20 | 961.0000 | 506.6325 | 0.5272 | 0.4728 |
| 毛衣 | 26 | 1311.0000 | 736.7593 | 0.5620 | 0.4428 |
| T恤 | 46 | 2403.0000 | 1431.1567 | 0.5956 | 0.4189 |

## Zero-True False Positives

- zero_true_rows: `33508`
- false_positive_rows: `5434`
- false_positive_rate_zero_true: `0.1622`
- pred_sum_on_zero_true: `10162.4551`
- pred_ge_3_rows: `531`
- pred_ge_5_rows: `123`
- pred_ge_10_rows: `12`

## Output Files

- `reports/phase8a_prep/phase8_current_mainline_eval_context_all_anchors.csv`
- `reports/phase8a_prep/phase8_current_mainline_anchor_eval.csv`
- `reports/phase8a_prep/phase8_residual_gap_cases.csv`