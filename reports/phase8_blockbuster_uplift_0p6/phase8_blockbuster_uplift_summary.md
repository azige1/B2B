# Phase8 Blockbuster Uplift Summary

- Status: `shadow_postprocess_exploration`
- Base branch: `coverage_router` under old Phase7 0.6-series evaluation.
- This run does not retrain the model. It tests business-safe uplift rules using only prediction-time signals.

## Key Metrics

| candidate | rule | mult | selected_rate | global_wmape | global_ratio | max_anchor_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_capture | rank_corr | 1_3_ratio | zero_true_ge3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| phase7_all | baseline | 1.00 | 0.0000 | 0.6863 | 1.0159 | 1.0793 | 0.3566 | 0.6588 | 0.5853 | 0.4165 | 0.5539 | 0.6494 | 0.8044 | 1.2999 | 0.0160 |
| event_request_all | baseline | 1.00 | 1.0000 | 0.6620 | 1.0189 | 1.0756 | 0.3467 | 0.6636 | 0.5980 | 0.3887 | 0.5891 | 0.6518 | 0.8070 | 1.2824 | 0.0141 |
| coverage_router | anchor_event_covered | 1.00 | 0.7500 | 0.6591 | 1.0198 | 1.0793 | 0.3466 | 0.6646 | 0.5981 | 0.3820 | 0.5957 | 0.6524 | 0.8078 | 1.2759 | 0.0145 |
| uplift_pred_ge_8_prob_ge_0.50_signal_ge_20_x12 | pred_ge_8_prob_ge_0.50_signal_ge_20 | 1.20 | 0.0215 | 0.6364 | 1.0962 | 1.1583 | 0.3100 | 0.6921 | 0.6266 | 0.2875 | 0.7148 | 0.6524 | 0.8079 | 1.2835 | 0.0145 |
| uplift_pred_ge_20_prob_ge_0.50_signal_ge_20_x12 | pred_ge_20_prob_ge_0.50_signal_ge_20 | 1.20 | 0.0048 | 0.6482 | 1.0578 | 1.0993 | 0.3460 | 0.6646 | 0.5981 | 0.3101 | 0.6855 | 0.6524 | 0.8078 | 1.2759 | 0.0145 |

## Interpretation

- Metric-best candidate: `uplift_pred_ge_8_prob_ge_0.50_signal_ge_20_x12`.
- Recommended final candidate: `uplift_pred_ge_20_prob_ge_0.50_signal_ge_20_x12`.
- Recommended candidate changes blockbuster under-WAPE by `-0.0719` and global WMAPE by `-0.0109` relative to `coverage_router`.
- Recommended candidate improves Phase7 WMAPE by `5.56%` and blockbuster under-WAPE by `25.55%`.
- Metric-best max anchor ratio is `1.1583`; recommended max anchor ratio is `1.0993`.
- Use the recommended candidate when Phase8 needs a defensible final branch. Use metric-best only as an aggressive shadow reference.

## Outputs

- Candidate summary: `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_candidate_summary.csv`
- Anchor metrics: `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_anchor_table.csv`
- Metric-best context: `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_metric_best_context.csv`
- Recommended context: `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_recommended_context.csv`
