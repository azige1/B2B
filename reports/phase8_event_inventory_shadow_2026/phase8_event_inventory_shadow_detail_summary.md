# Phase8 Event+Inventory Shadow 2026 Detail Summary

- Status: `analysis_only_shadow`
- Scope: `2026-02-15 / 2026-02-24`
- Purpose: inspect where event+inventory helps or hurts relative to the 2026 base line.
- This detail pack does not participate in official phase7 replacement.

## Weak-Signal Blockbuster Slice

| anchor_date | signal_quadrant | true_sum | pred_sum_base | pred_sum_shadow | ratio_base | ratio_shadow | under_wape_base | under_wape_shadow | under_wape_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | repl0_fut0 | 218.0000 | 83.8027 | 145.8684 | 0.3844 | 0.6691 | 0.6156 | 0.3309 | -0.2847 |
| 2026-02-15 | repl1_fut0 | 627.0000 | 333.9706 | 476.4048 | 0.5326 | 0.7598 | 0.4678 | 0.2594 | -0.2084 |
| 2026-02-24 | repl1_fut0 | 604.0000 | 253.3460 | 354.0076 | 0.4194 | 0.5861 | 0.5806 | 0.4162 | -0.1643 |
| 2026-02-24 | repl0_fut0 | 398.0000 | 92.0014 | 122.4476 | 0.2312 | 0.3077 | 0.7688 | 0.6923 | -0.0765 |

## Weak-Signal Slice Regressions

| anchor_date | signal_quadrant | true_sum | pred_sum_base | pred_sum_shadow | ratio_base | ratio_shadow | under_wape_base | under_wape_shadow | under_wape_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-24 | repl0_fut0 | 398.0000 | 92.0014 | 122.4476 | 0.2312 | 0.3077 | 0.7688 | 0.6923 | -0.0765 |
| 2026-02-24 | repl1_fut0 | 604.0000 | 253.3460 | 354.0076 | 0.4194 | 0.5861 | 0.5806 | 0.4162 | -0.1643 |
| 2026-02-15 | repl1_fut0 | 627.0000 | 333.9706 | 476.4048 | 0.5326 | 0.7598 | 0.4678 | 0.2594 | -0.2084 |
| 2026-02-15 | repl0_fut0 | 218.0000 | 83.8027 | 145.8684 | 0.3844 | 0.6691 | 0.6156 | 0.3309 | -0.2847 |

## Zero-True False Positive

| anchor_date | zero_true_rows | false_positive_rows_base | false_positive_rate_base | false_positive_rows_shadow | false_positive_rate_shadow | false_positive_rate_delta | pred_ge_5_rows_base | pred_ge_5_rows_shadow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | 8284 | 1435 | 0.1732 | 1414 | 0.1707 | -0.0025 | 35 | 17 |
| 2026-02-24 | 8009 | 1861 | 0.2324 | 1594 | 0.1990 | -0.0333 | 28 | 13 |

## Blockbuster Category Improvements

| anchor_date | category | rows | true_sum | pred_sum_base | pred_sum_shadow | under_wape_base | under_wape_shadow | under_wape_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | 裤类 | 5 | 164.0000 | 59.9099 | 121.9829 | 0.6347 | 0.2562 | -0.3785 |
| 2026-02-15 | 裙子 | 4 | 185.0000 | 67.9301 | 112.4776 | 0.6328 | 0.3920 | -0.2408 |
| 2026-02-15 | 外套 | 6 | 220.0000 | 104.5081 | 141.5426 | 0.5250 | 0.3566 | -0.1683 |
| 2026-02-15 | T恤 | 5 | 151.0000 | 92.7897 | 115.9303 | 0.3855 | 0.2556 | -0.1299 |
| 2026-02-15 | 毛衣 | 4 | 125.0000 | 96.2692 | 101.0714 | 0.2307 | 0.1914 | -0.0392 |
| 2026-02-15 | 毛衣开衫 | 6 | 310.0000 | 236.2671 | 234.3520 | 0.2793 | 0.2440 | -0.0353 |
| 2026-02-24 | 裤类 | 7 | 221.0000 | 54.8275 | 99.9805 | 0.7519 | 0.5476 | -0.2043 |
| 2026-02-24 | T恤 | 7 | 205.0000 | 92.7808 | 127.2596 | 0.5474 | 0.3792 | -0.1682 |
| 2026-02-24 | 外套 | 9 | 328.0000 | 126.6570 | 170.4718 | 0.6139 | 0.4803 | -0.1336 |
| 2026-02-24 | 毛衣 | 6 | 192.0000 | 82.5214 | 107.8338 | 0.5702 | 0.4457 | -0.1245 |
| 2026-02-24 | 裙子 | 4 | 180.0000 | 44.0814 | 61.0214 | 0.7551 | 0.6610 | -0.0941 |
| 2026-02-24 | 毛衣开衫 | 5 | 270.0000 | 198.3856 | 182.1579 | 0.2652 | 0.3253 | 0.0601 |

## Blockbuster Category Regressions

| anchor_date | category | rows | true_sum | pred_sum_base | pred_sum_shadow | under_wape_base | under_wape_shadow | under_wape_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | 毛衣开衫 | 6 | 310.0000 | 236.2671 | 234.3520 | 0.2793 | 0.2440 | -0.0353 |
| 2026-02-15 | 毛衣 | 4 | 125.0000 | 96.2692 | 101.0714 | 0.2307 | 0.1914 | -0.0392 |
| 2026-02-15 | T恤 | 5 | 151.0000 | 92.7897 | 115.9303 | 0.3855 | 0.2556 | -0.1299 |
| 2026-02-15 | 外套 | 6 | 220.0000 | 104.5081 | 141.5426 | 0.5250 | 0.3566 | -0.1683 |
| 2026-02-24 | 毛衣开衫 | 5 | 270.0000 | 198.3856 | 182.1579 | 0.2652 | 0.3253 | 0.0601 |
| 2026-02-24 | 裙子 | 4 | 180.0000 | 44.0814 | 61.0214 | 0.7551 | 0.6610 | -0.0941 |
| 2026-02-24 | 毛衣 | 6 | 192.0000 | 82.5214 | 107.8338 | 0.5702 | 0.4457 | -0.1245 |
| 2026-02-24 | 外套 | 9 | 328.0000 | 126.6570 | 170.4718 | 0.6139 | 0.4803 | -0.1336 |

## Top Improved Weak-Signal Cases

| anchor_date | sku_id | category | signal_quadrant | true_replenish_qty | base_pred_qty | shadow_pred_qty | event_daily_order_success_30 | inv_total_stock | stock_positive | abs_error_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | AL1150374236000 | 裙子 | repl1_fut0 | 67.0000 | 28.6434 | 46.9207 |  | 168.0000 | 1.0000 | 18.2773 |
| 2026-02-15 | AL1130163538903 | 裤类 | repl0_fut0 | 35.0000 | 12.7439 | 27.6360 |  | 85.0000 | 1.0000 | 14.8921 |
| 2026-02-15 | AL1130163536903 | 裤类 | repl0_fut0 | 32.0000 | 12.1689 | 25.5281 |  | 101.0000 | 1.0000 | 13.3592 |
| 2026-02-24 | AL1130163536903 | 裤类 | repl0_fut0 | 30.0000 | 6.7584 | 19.5308 | 1.0000 | 101.0000 | 1.0000 | 12.7724 |
| 2026-02-15 | AL1130163138902 | 裤类 | repl1_fut0 | 37.0000 | 12.1628 | 24.3981 |  | 31.0000 | 1.0000 | 12.2353 |
| 2026-02-15 | AL1010576136560 | 外套 | repl1_fut0 | 41.0000 | 18.4140 | 29.8017 |  | 95.0000 | 1.0000 | 11.3877 |
| 2026-02-15 | AL1130162338721 | 裤类 | repl0_fut0 | 29.0000 | 10.4497 | 21.7941 |  | 29.0000 | 1.0000 | 11.3444 |
| 2026-02-24 | AL1100152038121 | T恤 | repl1_fut0 | 31.0000 | 14.5666 | 25.5425 |  | 27.0000 | 1.0000 | 10.9760 |
| 2026-02-15 | AL1150374238000 | 裙子 | repl1_fut0 | 54.0000 | 19.5658 | 30.2232 |  | 136.0000 | 1.0000 | 10.6575 |
| 2026-02-15 | AL1010777640560 | 毛衣开衫 | repl0_fut0 | 28.0000 | 8.8220 | 19.2492 |  | 61.0000 | 1.0000 | 10.4272 |

## Top Worsened Weak-Signal Cases

| anchor_date | sku_id | category | signal_quadrant | true_replenish_qty | base_pred_qty | shadow_pred_qty | event_daily_order_success_30 | inv_total_stock | stock_zero | abs_error_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | AK1080153738043 | 单衣 | repl1_fut0 | 40.0000 | 40.2509 | 45.2800 | 11.0000 | 35.0000 | 0.0000 | -5.0291 |
| 2026-02-24 | AL1070266238601 | 马夹 | repl0_fut0 | 34.0000 | 15.5396 | 12.8931 | 1.0000 | 46.0000 | 0.0000 | -2.6465 |
| 2026-02-24 | AK1080153738043 | 单衣 | repl1_fut0 | 45.0000 | 37.7072 | 35.1662 | 15.0000 | 26.0000 | 0.0000 | -2.5410 |
| 2026-02-24 | RK3010524536560 | 外套 | repl0_fut0 | 31.0000 | 7.8025 | 5.5240 |  |  |  | -2.2785 |
| 2026-02-24 | RK3010524538560 | 外套 | repl0_fut0 | 33.0000 | 7.4387 | 5.6276 |  |  |  | -1.8111 |
| 2026-02-24 | AK3130667436610 | 裤类 | repl0_fut0 | 31.0000 | 2.9178 | 2.6627 |  |  |  | -0.2551 |
| 2026-02-24 | AL1010777640560 | 毛衣开衫 | repl0_fut0 | 27.0000 | 12.4250 | 12.4054 | 1.0000 | 61.0000 | 0.0000 | -0.0196 |
| 2026-02-24 | AK3100152338700 | T恤 | repl0_fut0 | 27.0000 | 3.5583 | 3.6245 |  |  |  | 0.0662 |

## Output Files

- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_row_compare.csv`
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_focus_cases.csv`
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_weak_signal_table.csv`
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_zero_true_table.csv`
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_category_delta_table.csv`
