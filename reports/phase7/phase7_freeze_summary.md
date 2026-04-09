# Phase7 冻结总结

`phase7` 的定义是：**Tail / Allocation 定向优化阶段**。

## 结论

- 过夜长实验已完成，当前 winner 已满足替换主线门槛。
- `phase7` 新主线正式升级为 `tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`。
- 本轮改进不是只优化一边，而是同时改善了 `4-25 / Ice 4-25 / blockbuster / allocation`。
- `phase6` 冻结主线 `p57_covact_lr005_l63_hard_g025 + sep098_oct093` 退为旧主线参考。

## 新主线

- 主线候选：`tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`
- raw 模型：`tail_full_lr005_l63_g027_n800_s2028`
- 校准规则：`sep098_oct093`
- 树模型家族：`LightGBM`
- 特征组：`cov_activity_tail_full`

## 关键证据

- `reports/phase7_tail_allocation_optimization/overnight_20260402_overnight/overnight_summary.md`
- `reports/phase7_tail_allocation_optimization/overnight_20260402_overnight/overnight_winner.json`
- `reports/phase7h_december_readable_report/dec_20251201_dashboard.html`
- `reports/phase7h_december_readable_report/dec_20251201_static_feature_audit.md`

## 主指标变化

- `4_25_under_wape`: 0.4236 -> 0.3566 (-0.0670)
- `4_25_sku_p50`: 0.5986 -> 0.6588 (+0.0602)
- `ice_4_25_sku_p50`: 0.5069 -> 0.5853 (+0.0784)
- `blockbuster_under_wape`: 0.5507 -> 0.4165 (-0.1342)
- `blockbuster_sku_p50`: 0.4048 -> 0.5539 (+0.1491)
- `top20_true_volume_capture`: 0.6213 -> 0.6494 (+0.0281)
- `rank_corr_positive_skus`: 0.7365 -> 0.8044 (+0.0679)
- `global_ratio`: 1.0188 -> 1.0159 (-0.0029)
- `global_wmape`: 0.8526 -> 0.6863 (-0.1663)
- `1_3_ratio`: 1.3801 -> 1.2999 (-0.0802)

