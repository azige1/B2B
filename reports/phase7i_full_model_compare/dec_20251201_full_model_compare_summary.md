# 2025-12 详细模型对照页摘要

- 输出 CSV: `E:\LSTM\B2B\B2B_Replenishment_System\reports\phase7i_full_model_compare\dec_20251201_full_model_compare.csv`
- 输出 HTML: `E:\LSTM\B2B\B2B_Replenishment_System\reports\phase7i_full_model_compare\dec_20251201_full_model_compare.html`
- 当前最强版本: `tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`
- 中间版本: `p57_covact_lr005_l63_hard_g025 + sep098_oct093`
- 最佳 LSTM: `p527_lstm_l3_v5_lite_s2027_s2027`

## 模型来源

| 版本            | 模型                                                | 口径             | 来源                                                                                                                                                                                                                                            |
|:----------------|:----------------------------------------------------|:-----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Phase7 当前最强 | tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093 | 2025-12 raw 视图 | E:\LSTM\B2B\B2B_Replenishment_System\reports\phase7_tail_allocation_optimization\overnight_20260402_overnight\stage_s3\20251201\phase5\eval_context_p7ov_20260402_overnight_stage_s3_20251201_tail_full_lr005_l63_g027_n800_s2028_hard_g027.csv |
| Phase6 冻结主线 | p57_covact_lr005_l63_hard_g025 + sep098_oct093      | 2025-12 raw 视图 | E:\LSTM\B2B\B2B_Replenishment_System\reports\phase5_7\20251201\phase5\eval_context_p57a_20251201_covact_lr005_l63_s2026_hard_g025.csv                                                                                                           |
| 最佳 LSTM       | p527_lstm_l3_v5_lite_s2027_s2027                    | 2025-12 raw 视图 | E:\LSTM\B2B\B2B_Replenishment_System\reports\phase5_4\phase5\eval_context_p54_p527_lstm_l3_v5_lite_s2027_s2027.csv                                                                                                                              |

## 核心指标

| metric           |   phase7 |   phase6 |     lstm |   delta_vs_phase6 |   delta_vs_lstm |
|:-----------------|---------:|---------:|---------:|------------------:|----------------:|
| Global WMAPE     | 0.694244 | 0.923712 | 1.24012  |        -0.229468  |      -0.54588   |
| Global Ratio     | 0.996162 | 0.963673 | 0.904648 |         0.0324881 |       0.0915131 |
| 4-25 Under WAPE  | 0.353523 | 0.451188 | 0.654818 |        -0.0976651 |      -0.301295  |
| 4-25 SKU P50     | 0.668282 | 0.589785 | 0.333489 |         0.0784971 |       0.334792  |
| Ice 4-25 SKU P50 | 0.596415 | 0.45864  | 0.165797 |         0.137775  |       0.430618  |
| >25 Under WAPE   | 0.451576 | 0.664198 | 0.873083 |        -0.212623  |      -0.421507  |
| >25 SKU P50      | 0.529206 | 0.307322 | 0.100178 |         0.221884  |       0.429029  |
| Top20 Capture    | 0.700051 | 0.675493 | 0.487418 |         0.0245579 |       0.212633  |
| Rank Corr        | 0.782174 | 0.695541 | 0.419764 |         0.0866329 |       0.36241   |
| 1-3 Ratio        | 1.32243  | 1.45381  | 1.21058  |        -0.13138   |       0.111853  |
