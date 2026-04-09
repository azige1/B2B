# Phase6e Tree Validation Pack

- selected_candidate: `sep098_oct093`
- freeze_current_tree_mainline: `True`
- phase6f_trigger: `True`
- lightgbm_mainline_stands: `True`
- structural_shortfall_present: `True`
- no_meaningful_lgbm_gain_left: `True`

## Candidate Summary

| candidate_key | track | anchor_passes | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_true_volume_capture | rank_corr_positive_skus | false_zero_rate | true_gt_10_pred_le_1_rate | global_ratio | global_wmape | 1_3_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sep098_oct093 | event_tree_mainline | 4 | 0.4236 | 0.5986 | 0.5069 | 0.5507 | 0.4048 | 0.6213 | 0.7365 | 0.0372 | 0.0010 | 1.0188 | 0.8526 | 1.3801 |
| sep095_oct090 | event_tree_conservative | 4 | 0.4310 | 0.5891 | 0.4986 | 0.5569 | 0.3988 | 0.6213 | 0.7365 | 0.0372 | 0.0010 | 1.0017 | 0.8485 | 1.3587 |
| p535 | event_tree_confirmed | 2 | 0.4345 | 0.5864 | 0.5009 | 0.5791 | 0.3561 | 0.6080 | 0.6913 | 0.0273 | 0.0010 | 1.1791 | 1.0364 | 1.5038 |
| p531 | event_tree_backup | 2 | 0.4365 | 0.5866 | 0.5015 | 0.5794 | 0.3564 | 0.6081 | 0.6918 | 0.0287 | 0.0010 | 1.1717 | 1.0317 | 1.4981 |
| p527 | sequence_baseline | 0 | 0.7419 | 0.2329 | 0.0865 | 0.8649 | 0.1000 | 0.3868 | 0.2669 | 0.1060 | 0.2225 | 1.0451 | 1.4735 | 0.9109 |

## Selected Candidate Diagnostics

- `category_worst5_ratio`: `[{"category": "连体裤", "ratio": 6.954764869333334, "total_true": 6.0, "total_pred": 41.728589216}, {"category": "背心", "ratio": 3.8152585236945415, "total_true": 168.000002, "total_pred": 640.9634396112}, {"category": "连衣裙", "ratio": 2.2072496097291205, "total_true": 381.9999975, "total_pred": 843.1693453984}, {"category": "皮草", "ratio": 2.1878973923076925, "total_true": 13.0, "total_pred": 28.4426661}, {"category": "皮衣", "ratio": 1.943035086427257, "total_true": 40.9999995, "total_pred": 79.664437572}]`
- `category_worst5_under_wape`: `[{"category": "连体裤", "under_wape": 0.594536864, "total_true": 6.0, "total_pred": 41.728589216}, {"category": "棉服", "under_wape": 0.5603808305396504, "total_true": 454.0000105, "total_pred": 264.299135962}, {"category": "卫衣", "under_wape": 0.5305889052357713, "total_true": 360.000007, "total_pred": 280.441810654}, {"category": "毛衣", "under_wape": 0.4542613984936462, "total_true": 4077.0000245, "total_pred": 4047.6835723821}, {"category": "毛衣开衫", "under_wape": 0.43980126616257803, "total_true": 1839.000015, "total_pred": 1776.907510581}]`
- `repl0_fut0_ratio / under_wape`: `0.7271` / `0.5525`
- `repl0_fut1_ratio / under_wape`: `1.4359` / `0.2135`
- `repl1_fut0_ratio / under_wape`: `2.1525` / `0.3690`
- `repl1_fut1_ratio / under_wape`: `1.1456` / `0.2698`
- `blockbuster_top10_true_volume_capture`: `0.1816`
- `blockbuster_top20_true_volume_capture`: `0.3070`