# Phase8 Event Coverage Router Summary

- Status: `shadow_router_exploration`
- Evaluation style: old Phase7 0.6-series four anchors, no listing-date feature, no listing-eligible filter.
- Router only combines existing predictions; it does not retrain the model and does not use future labels as input.

## Key Candidates

| candidate | rule | fallback | event_use_rate | global_wmape | global_ratio | blockbuster_under_wape | rank_corr_positive_skus |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| phase7_all | baseline |  | 0.0000 | 0.6863 | 1.0159 | 0.4165 | 0.8044 |
| purchase_all | baseline |  | 0.0000 | 0.6770 | 1.0111 | 0.4131 | 0.8025 |
| event_request_all | baseline |  | 1.0000 | 0.6620 | 1.0189 | 0.3887 | 0.8070 |
| router_phase7_anchor_event_covered | anchor_event_covered | phase7 | 0.7500 | 0.6591 | 1.0198 | 0.3820 | 0.8078 |

## Interpretation

- `event_request_all` improves WMAPE from `0.6863` to `0.6620` (`3.54%`).
- Anchor-level route improves WMAPE to `0.6591` (`3.97%`) by falling back on `2025-09-01` where Event has no useful pre-anchor coverage.
- Best gated candidate is `router_phase7_anchor_event_covered`, WMAPE `0.6591`, blockbuster under-WAPE `0.3820`.
- If a row-level coverage rule does not beat anchor-level routing while preserving blockbuster metrics, prefer anchor-level routing because it is simpler and easier to explain.

## Outputs

- Candidate summary: `reports/phase8_event_request_router/phase8_event_request_router_candidate_summary.csv`
- Anchor metrics: `reports/phase8_event_request_router/phase8_event_request_router_anchor_table.csv`
- Best context: `reports/phase8_event_request_router/phase8_event_request_router_best_context.csv`
