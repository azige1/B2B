# Phase 5.2 Decision Table

| exp_id | version | model | seed | auc | f1 | global_ratio | global_wmape | sku_p50 | four_25_ratio | four_25_wmape_like | four_25_sku_p50 | ice_sku_p50 | ice_4_25_sku_p50 | blockbuster_sku_p50 | status | actual_epochs | elapsed_min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p522_bilstm_l3_v3_filtered_s2026 | v3_filtered | bilstm | 2026 | 0.7143 | 0.4810 | 0.9636 | 1.4802 | 0.5973 | 0.2479 | 0.7634 | 0.2076 | 0.1274 | 0.0847 | 0.0514 | success | 3 | 90.45 |
| p526_bilstm_l3_v3_filtered_s2027 | v3_filtered | bilstm | 2027 | 0.7519 | 0.4820 | 1.2751 | 1.6468 | 0.7930 | 0.3532 | 0.6866 | 0.3160 | 0.2597 | 0.1846 | 0.0799 | success | 4 | 68.34 |
| p521_lstm_l3_v3_filtered_s2026 | v3_filtered | lstm | 2026 | 0.8225 | 0.5368 | 1.0138 | 1.3906 | 0.8132 | 0.3512 | 0.7003 | 0.2899 | 0.2802 | 0.1556 | 0.0739 | success | 5 | 40.88 |
| p525_lstm_l3_v3_filtered_s2027 | v3_filtered | lstm | 2027 | 0.7052 | 0.4483 | 1.2562 | 1.7214 | 0.6401 | 0.2843 | 0.7377 | 0.2421 | 0.2017 | 0.1380 | 0.0600 | success | 3 | 23.43 |
| p524_bilstm_l3_v5_lite_s2026 | v5_lite | bilstm | 2026 | 0.7118 | 0.4440 | 1.0978 | 1.6037 | 0.6252 | 0.2413 | 0.7654 | 0.2181 | 0.1269 | 0.0880 | 0.0543 | success | 3 | 56.63 |
| p528_bilstm_l3_v5_lite_s2027 | v5_lite | bilstm | 2027 | 0.6675 | 0.3868 | 1.2919 | 1.7651 | 0.6484 | 0.2741 | 0.7458 | 0.2376 | 0.1854 | 0.1221 | 0.0603 | success | 3 | 58.61 |
| p523_lstm_l3_v5_lite_s2026 | v5_lite | lstm | 2026 | 0.7334 | 0.5023 | 1.1968 | 1.6584 | 0.7327 | 0.2721 | 0.7480 | 0.2433 | 0.1571 | 0.0970 | 0.0486 | success | 3 | 23.64 |
| p527_lstm_l3_v5_lite_s2027 | v5_lite | lstm | 2027 | 0.8627 | 0.5882 | 0.9361 | 1.2703 | 0.8710 | 0.3735 | 0.6827 | 0.3335 | 0.3013 | 0.1658 | 0.1002 | success | 5 | 34.4 |

## Frozen References

```json
{
  "v3_filtered": {
    "exp_id": "p521_lstm_l3_v3_filtered_s2026",
    "model": "lstm",
    "seed": "2026",
    "selection_mode": "fixed_override",
    "auc": 0.8224925563093939,
    "f1": 0.536839222130923,
    "global_ratio": 1.0137815147282947,
    "global_wmape": 1.3905519668857795,
    "sku_p50": 0.8131630660651719,
    "ice_sku_p50": 0.28022272884845734,
    "four_25_ratio": 0.35118237472878566,
    "four_25_wmape_like": 0.7003038817170698,
    "four_25_sku_p50": 0.2898963987827301,
    "ice_4_25_sku_p50": 0.1555603542174669,
    "blockbuster_sku_p50": 0.07388885850188819
  },
  "v5_lite": {
    "exp_id": "p527_lstm_l3_v5_lite_s2027",
    "model": "lstm",
    "seed": "2027",
    "selection_mode": "fixed_override",
    "auc": 0.8627485219063766,
    "f1": 0.5882352941176471,
    "global_ratio": 0.9360691786900472,
    "global_wmape": 1.270310701653166,
    "sku_p50": 0.870988130569458,
    "ice_sku_p50": 0.3013180047273636,
    "four_25_ratio": 0.3735186567267386,
    "four_25_wmape_like": 0.6826516833228998,
    "four_25_sku_p50": 0.33348938416482793,
    "ice_4_25_sku_p50": 0.1657968973569205,
    "blockbuster_sku_p50": 0.10017782900535446
  }
}
```