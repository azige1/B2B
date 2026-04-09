# Phase8 Event+Inventory Shadow 2026 Detail Summary

- Status: `analysis_only_shadow`
- Scope: `2026-02-15 / 2026-02-24`
- Purpose: inspect where event+inventory helps or hurts relative to the 2026 base line.
- This detail pack does not participate in official phase7 replacement.

## Weak-Signal Blockbuster Slice

| anchor_date | signal_quadrant | true_sum | pred_sum_base | pred_sum_shadow | ratio_base | ratio_shadow | under_wape_base | under_wape_shadow | under_wape_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | repl0_fut0 | 218.0000 | 83.8027 | 136.8031 | 0.3844 | 0.6275 | 0.6156 | 0.3725 | -0.2431 |
| 2026-02-24 | repl1_fut0 | 604.0000 | 253.3460 | 368.0352 | 0.4194 | 0.6093 | 0.5806 | 0.3907 | -0.1899 |
| 2026-02-15 | repl1_fut0 | 627.0000 | 333.9706 | 467.7487 | 0.5326 | 0.7460 | 0.4678 | 0.2817 | -0.1860 |
| 2026-02-24 | repl0_fut0 | 398.0000 | 92.0014 | 106.1043 | 0.2312 | 0.2666 | 0.7688 | 0.7334 | -0.0354 |

## Weak-Signal Slice Regressions

| anchor_date | signal_quadrant | true_sum | pred_sum_base | pred_sum_shadow | ratio_base | ratio_shadow | under_wape_base | under_wape_shadow | under_wape_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-24 | repl0_fut0 | 398.0000 | 92.0014 | 106.1043 | 0.2312 | 0.2666 | 0.7688 | 0.7334 | -0.0354 |
| 2026-02-15 | repl1_fut0 | 627.0000 | 333.9706 | 467.7487 | 0.5326 | 0.7460 | 0.4678 | 0.2817 | -0.1860 |
| 2026-02-24 | repl1_fut0 | 604.0000 | 253.3460 | 368.0352 | 0.4194 | 0.6093 | 0.5806 | 0.3907 | -0.1899 |
| 2026-02-15 | repl0_fut0 | 218.0000 | 83.8027 | 136.8031 | 0.3844 | 0.6275 | 0.6156 | 0.3725 | -0.2431 |

## Zero-True False Positive

| anchor_date | zero_true_rows | false_positive_rows_base | false_positive_rate_base | false_positive_rows_shadow | false_positive_rate_shadow | false_positive_rate_delta | pred_ge_5_rows_base | pred_ge_5_rows_shadow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | 8284 | 1469 | 0.1773 | 1191 | 0.1438 | -0.0336 | 35 | 14 |
| 2026-02-24 | 8009 | 1805 | 0.2254 | 1444 | 0.1803 | -0.0451 | 28 | 9 |

## Blockbuster Category Improvements

| anchor_date | category | rows | true_sum | pred_sum_base | pred_sum_shadow | under_wape_base | under_wape_shadow | under_wape_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | 裤类 | 5 | 164.0000 | 59.9099 | 111.6477 | 0.6347 | 0.3192 | -0.3155 |
| 2026-02-15 | 裙子 | 4 | 185.0000 | 67.9301 | 99.3141 | 0.6328 | 0.4632 | -0.1696 |
| 2026-02-15 | 外套 | 6 | 220.0000 | 104.5081 | 138.7687 | 0.5250 | 0.3692 | -0.1557 |
| 2026-02-15 | 毛衣开衫 | 6 | 310.0000 | 236.2671 | 262.0201 | 0.2793 | 0.1761 | -0.1031 |
| 2026-02-15 | T恤 | 5 | 151.0000 | 92.7897 | 104.9916 | 0.3855 | 0.3047 | -0.0808 |
| 2026-02-15 | 毛衣 | 4 | 125.0000 | 96.2692 | 99.8671 | 0.2307 | 0.2011 | -0.0296 |
| 2026-02-24 | 外套 | 9 | 328.0000 | 126.6570 | 178.4092 | 0.6139 | 0.4561 | -0.1578 |
| 2026-02-24 | 裙子 | 4 | 180.0000 | 44.0814 | 66.5068 | 0.7551 | 0.6305 | -0.1246 |
| 2026-02-24 | 裤类 | 7 | 221.0000 | 54.8275 | 82.2225 | 0.7519 | 0.6280 | -0.1240 |
| 2026-02-24 | T恤 | 7 | 205.0000 | 92.7808 | 117.4820 | 0.5474 | 0.4269 | -0.1205 |
| 2026-02-24 | 毛衣 | 6 | 192.0000 | 82.5214 | 105.6000 | 0.5702 | 0.4500 | -0.1202 |
| 2026-02-24 | 毛衣开衫 | 5 | 270.0000 | 198.3856 | 185.3693 | 0.2652 | 0.3134 | 0.0482 |

## Blockbuster Category Regressions

| anchor_date | category | rows | true_sum | pred_sum_base | pred_sum_shadow | under_wape_base | under_wape_shadow | under_wape_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | 毛衣 | 4 | 125.0000 | 96.2692 | 99.8671 | 0.2307 | 0.2011 | -0.0296 |
| 2026-02-15 | T恤 | 5 | 151.0000 | 92.7897 | 104.9916 | 0.3855 | 0.3047 | -0.0808 |
| 2026-02-15 | 毛衣开衫 | 6 | 310.0000 | 236.2671 | 262.0201 | 0.2793 | 0.1761 | -0.1031 |
| 2026-02-15 | 外套 | 6 | 220.0000 | 104.5081 | 138.7687 | 0.5250 | 0.3692 | -0.1557 |
| 2026-02-24 | 毛衣开衫 | 5 | 270.0000 | 198.3856 | 185.3693 | 0.2652 | 0.3134 | 0.0482 |
| 2026-02-24 | 毛衣 | 6 | 192.0000 | 82.5214 | 105.6000 | 0.5702 | 0.4500 | -0.1202 |
| 2026-02-24 | T恤 | 7 | 205.0000 | 92.7808 | 117.4820 | 0.5474 | 0.4269 | -0.1205 |
| 2026-02-24 | 裤类 | 7 | 221.0000 | 54.8275 | 82.2225 | 0.7519 | 0.6280 | -0.1240 |

## Top Improved Weak-Signal Cases

| anchor_date | sku_id | category | signal_quadrant | true_replenish_qty | base_pred_qty | shadow_pred_qty | event_daily_order_success_30 | qty_b2b_hq_stock | abs_error_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-24 | AL1010576136560 | 外套 | repl1_fut0 | 39.0000 | 10.0370 | 26.0541 | 1.0000 | 53.0000 | 16.0171 |
| 2026-02-24 | AL1010677338720 | 毛衣开衫 | repl1_fut0 | 42.0000 | 14.9377 | 28.9921 | 1.0000 | 50.0000 | 14.0544 |
| 2026-02-15 | AL1150374238000 | 裙子 | repl1_fut0 | 54.0000 | 19.5658 | 31.3598 |  | 85.0000 | 11.7941 |
| 2026-02-15 | AL1130163538903 | 裤类 | repl0_fut0 | 35.0000 | 12.7439 | 24.1607 |  | 50.0000 | 11.4168 |
| 2026-02-15 | AL1010777640560 | 毛衣开衫 | repl0_fut0 | 28.0000 | 8.8220 | 19.7484 |  | 30.0000 | 10.9264 |
| 2026-02-15 | AL1010677338720 | 毛衣开衫 | repl1_fut0 | 42.0000 | 23.4064 | 34.0799 |  | 50.0000 | 10.6735 |
| 2026-02-15 | AL1010576136560 | 外套 | repl1_fut0 | 41.0000 | 18.4140 | 28.7051 |  | 53.0000 | 10.2911 |
| 2026-02-15 | AL1130163138902 | 裤类 | repl1_fut0 | 37.0000 | 12.1628 | 22.4011 |  | 0.0000 | 10.2383 |
| 2026-02-15 | AL1130162338721 | 裤类 | repl0_fut0 | 29.0000 | 10.4497 | 20.6230 |  | 0.0000 | 10.1733 |
| 2026-02-15 | AL1130162438640 | 裤类 | repl1_fut0 | 31.0000 | 12.3846 | 22.5576 |  | 45.0000 | 10.1729 |

## Top Worsened Weak-Signal Cases

| anchor_date | sku_id | category | signal_quadrant | true_replenish_qty | base_pred_qty | shadow_pred_qty | event_daily_order_success_30 | qty_b2b_hq_stock | abs_error_gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-15 | AK1080153738043 | 单衣 | repl1_fut0 | 40.0000 | 40.2509 | 56.1549 | 11.0000 | 0.0000 | -15.9041 |
| 2026-02-24 | RK3010524536560 | 外套 | repl0_fut0 | 31.0000 | 7.8025 | 4.4635 |  |  | -3.3390 |
| 2026-02-24 | AL1070266238601 | 马夹 | repl0_fut0 | 34.0000 | 15.5396 | 12.6220 | 1.0000 | 0.0000 | -2.9176 |
| 2026-02-24 | RK3010524538560 | 外套 | repl0_fut0 | 33.0000 | 7.4387 | 4.6593 |  |  | -2.7794 |
| 2026-02-15 | AK1060163938043 | 毛衣 | repl1_fut0 | 33.0000 | 22.2221 | 21.5392 | 2.0000 | 0.0000 | -0.6829 |
| 2026-02-24 | AK3130667436610 | 裤类 | repl0_fut0 | 31.0000 | 2.9178 | 2.5653 |  |  | -0.3525 |
| 2026-02-24 | AK3130261336500 | 裤类 | repl0_fut0 | 34.0000 | 3.2034 | 3.0898 |  |  | -0.1136 |
| 2026-02-24 | AK3100152338700 | T恤 | repl0_fut0 | 27.0000 | 3.5583 | 3.9205 |  |  | 0.3622 |

## Output Files

- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_row_compare.csv`
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_focus_cases.csv`
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_weak_signal_table.csv`
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_zero_true_table.csv`
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_category_delta_table.csv`
