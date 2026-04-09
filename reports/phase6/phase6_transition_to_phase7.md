# Phase6 到 Phase7 的切换定义

## Phase7 名称

- `phase7_tail_allocation_optimization`

## Phase7 目标

- 提升 `blockbuster_under_wape`
- 提升 `blockbuster_sku_p50`
- 提升 `top20_true_volume_capture`
- 提升 `rank_corr_positive_skus`

## Phase7 主要优化方向

- `qty_first_order` 更强表达
- `style/category` 历史先验
- `tail peak / high-qty event` 特征
- `buyer concentration`

## Phase7 明确不做

- `sequence` 扩展
- 树模型家族对比
- 继续大量后处理微调

## Phase7 入口约束

- 继续以当前冻结主线为基线：
  - `p57_covact_lr005_l63_hard_g025 + sep098_oct093`
- 新候选必须在不伤 `4_25 / Ice 4_25` 的前提下，证明对 tail 或 allocation 有净收益
