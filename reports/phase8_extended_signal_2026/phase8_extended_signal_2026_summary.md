# Phase8 Extended Signal 2026 Summary

## Scope

- This is an exploratory 2026 analysis pack only.
- It does not participate in official phase7 replacement.
- Inventory is joined same-day on `sku_id + order_date`.
- Event and preorder signals are joined as prior-day features to avoid same-day leakage in the analysis view.

## Coverage

- rows: `28719`
- order_date range: `2026-01-23 ~ 2026-03-26`
- inventory_match_rate: `0.8601`
- preorder_match_rate: `0.0000`
- event_match_rate: `0.1243`

## Monthly Coverage and Replenish Rates

| month | rows | inventory_match_rate | preorder_match_rate | event_match_rate | replenish_positive_rate | replenish_gt10_rate | replenish_gt25_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01 | 2748 | 0.9785 | 0.0000 | 0.2471 | 0.9971 | 0.0142 | 0.0007 |
| 2026-02 | 10322 | 0.6867 | 0.0000 | 0.1228 | 0.5723 | 0.0060 | 0.0010 |
| 2026-03 | 15649 | 0.9537 | 0.0000 | 0.1037 | 0.3388 | 0.0022 | 0.0000 |

## Event Bucket

| event_bucket | rows | replenish_positive_rate | replenish_gt10_rate | replenish_gt25_rate | mean_replenish_qty | total_replenish_qty |
| --- | --- | --- | --- | --- | --- | --- |
| order_or_pay_30 | 2193 | 0.9799 | 0.0128 | 0.0000 | 1.7091 | 3748.0000 |
| view_30 | 389 | 0.8123 | 0.0103 | 0.0000 | 1.4807 | 576.0000 |
| cart_30 | 785 | 0.9809 | 0.0089 | 0.0000 | 1.5936 | 1251.0000 |
| no_event_30 | 25149 | 0.4234 | 0.0039 | 0.0005 | 0.6872 | 17283.0000 |
| click_only_30 | 203 | 0.3202 | 0.0000 | 0.0000 | 0.3448 | 70.0000 |

## Preorder Bucket

| preorder_bucket | rows | replenish_positive_rate | replenish_gt10_rate | replenish_gt25_rate | mean_replenish_qty | total_replenish_qty |
| --- | --- | --- | --- | --- | --- | --- |
| preorder_missing | 28719 | 0.4857 | 0.0047 | 0.0004 | 0.7984 | 22928.0000 |

## Inventory Bucket

| inventory_bucket | rows | replenish_positive_rate | replenish_gt10_rate | replenish_gt25_rate | mean_replenish_qty | total_replenish_qty |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 13031 | 0.5077 | 0.0024 | 0.0002 | 0.7748 | 10096.0000 |
| 1-10 | 7387 | 0.2883 | 0.0014 | 0.0000 | 0.4302 | 3178.0000 |
| 11-50 | 6608 | 0.5689 | 0.0071 | 0.0006 | 0.9926 | 6559.0000 |
| 50+ | 1693 | 0.8529 | 0.0284 | 0.0030 | 1.8281 | 3095.0000 |

## Joint Signal Bucket

| joint_signal_bucket | rows | replenish_positive_rate | replenish_gt10_rate | replenish_gt25_rate | mean_replenish_qty | total_replenish_qty |
| --- | --- | --- | --- | --- | --- | --- |
| event_strong_only | 2978 | 0.9802 | 0.0118 | 0.0000 | 1.6786 | 4999.0000 |
| neither | 25741 | 0.4285 | 0.0039 | 0.0005 | 0.6965 | 17929.0000 |

## Quick Read

- Highest event-led replenish>10 bucket: `order_or_pay_30` with `replenish_gt10_rate=0.0128`.
- Strongest joint bucket: `event_strong_only` with `replenish_gt10_rate=0.0118`.
- Focus cases exported: `62` rows.

## Output Files

- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_base.csv`
- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_month_table.csv`
- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_event_bucket_table.csv`
- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_preorder_table.csv`
- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_inventory_table.csv`
- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_joint_signal_table.csv`
- `reports/phase8_extended_signal_2026/phase8_extended_signal_2026_focus_cases.csv`
