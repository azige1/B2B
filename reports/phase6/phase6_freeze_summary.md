# Phase6 冻结总结

`phase6` 的定义是：**Tree 主线稳定化与家族验证阶段**。

## 结论

- `event/tree` 已取代 `sequence` 成为研发主线。
- 当前最优树模型家族仍为 `LightGBM`。
- `CatBoost/XGBoost` 已完成复核，不替换当前主线。
- `2025-12-01` 的业务可读报表和静态特征审计已完成。

## 冻结主线

- 主线候选：`p57_covact_lr005_l63_hard_g025 + sep098_oct093`
- raw 模型：`p57_covact_lr005_l63_hard_g025`
- 校准规则：`sep098_oct093`
- 树模型家族：`LightGBM`
- sequence 备线：`p527_lstm_l3_v5_lite_s2027`

## Phase6 固定判断

- 当前主线在四锚点上 `anchor_passes = 4`
- 当前主线不再继续做 `LightGBM` 小校准
- 当前主线不再继续做 `CatBoost/XGBoost` 家族对比
- 当前主线不再重新打开 `sequence` 扩展

## 关键证据

- `reports/phase6e_tree_validation_pack/phase6e_tree_validation_pack_summary.md`
- `reports/phase6f_tree_family_compare/phase6f_tree_family_compare_summary.md`
- `reports/phase6h_december_readable_report/dec_20251201_dashboard.html`
- `reports/phase6h_december_readable_report/dec_20251201_static_feature_audit.md`

## 下一阶段入口

- 下一阶段：`phase7_tail_allocation_optimization`
- 下一阶段目标不是交付，而是继续补：
  - `>25 / blockbuster`
  - allocation
  - `repl0_fut0 + ice + 高首单量`
