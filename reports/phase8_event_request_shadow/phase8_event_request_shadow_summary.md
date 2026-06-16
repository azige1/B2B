# Phase8 Event + Purchase Request Shadow Summary

- Status: `shadow_only_no_replace`
- Evaluation style: old Phase7 official 0.6-series anchors, no listing-date feature, no listing-eligible filter
- Full four-anchor candidate: all anchors use `event_request_plus`; on `2025-09-01`, Event columns have no effective historical coverage, so the extra Event signal is naturally zero-like.
- Four-anchor global WMAPE improvement: `3.54%`
- Event-covered three-anchor global WMAPE improvement: `5.44%`

## Four-Anchor Event + Request Candidate

| metric | phase7_current_4anchors | event_request_plus_4anchors | delta |
| --- | --- | --- | --- |
| global_ratio | 1.0159 | 1.0189 | +0.0030 |
| global_wmape | 0.6863 | 0.6620 | -0.0243 |
| 4_25_under_wape | 0.3566 | 0.3467 | -0.0099 |
| 4_25_sku_p50 | 0.6588 | 0.6636 | +0.0048 |
| ice_4_25_sku_p50 | 0.5853 | 0.5980 | +0.0127 |
| blockbuster_under_wape | 0.4165 | 0.3887 | -0.0278 |
| blockbuster_sku_p50 | 0.5539 | 0.5891 | +0.0352 |
| top20_true_volume_capture | 0.6494 | 0.6518 | +0.0024 |
| rank_corr_positive_skus | 0.8044 | 0.8070 | +0.0026 |
| 1_3_ratio | 1.2999 | 1.2824 | -0.0175 |

## Event-Covered Three Anchors

| metric | phase7_current_3anchors | event_request_plus_3anchors | delta |
| --- | --- | --- | --- |
| global_ratio | 0.9948 | 1.0000 | +0.0053 |
| global_wmape | 0.6678 | 0.6314 | -0.0363 |
| 4_25_under_wape | 0.3741 | 0.3607 | -0.0134 |
| 4_25_sku_p50 | 0.6375 | 0.6453 | +0.0077 |
| ice_4_25_sku_p50 | 0.5599 | 0.5770 | +0.0172 |
| blockbuster_under_wape | 0.3853 | 0.3393 | -0.0460 |
| blockbuster_sku_p50 | 0.5774 | 0.6331 | +0.0557 |
| top20_true_volume_capture | 0.6597 | 0.6638 | +0.0041 |
| rank_corr_positive_skus | 0.8094 | 0.8140 | +0.0046 |
| 1_3_ratio | 1.2389 | 1.2068 | -0.0320 |

## Interpretation

- This is the clean no-listing-date route for the external 0.6-series story.
- The four-anchor number is conservative because September has no meaningful pre-anchor Event signal.
- Event is still the main contributor; purchase request features add a smaller incremental lift.
