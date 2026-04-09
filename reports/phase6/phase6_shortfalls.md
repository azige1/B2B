# Phase6 主要短板

## 1. `>25 / blockbuster` 仍明显少补

- `blockbuster_under_wape = 0.5507`
- `blockbuster_sku_p50 = 0.4048`
- `blockbuster_top20_true_volume_capture = 0.3070`

当前主线已经明显优于 baseline，但高需求 SKU 仍然没有拿到足够的量，tail 仍然是主短板。

## 2. 高价值 SKU 分配能力还不够尖锐

- `top20_true_volume_capture = 0.6213`
- `rank_corr_positive_skus = 0.7365`

整体分配能力已经起来了，但对最重要那批 SKU 的排序和集中分配还不够强。

## 3. 结构性失衡集中在 `repl0_fut0 + ice + 高首单量`

- `repl0_fut0_ratio / under_wape = 0.7271 / 0.5525`
- `2025-12` 中 `repl0_fut0 + ice + true_blockbuster` 的均值：
  - `true = 49.12`
  - `pred = 11.96`
  - `ratio = 0.274`
  - `qty_first_order = 233.5`

当前最大的漏补不是随机发生，而是集中在弱动态信号、冷启动、且首单量很大的 SKU 上。

## 静态特征结论

- `sku_id/style_id/category/...` 当前是 `LabelEncoder`
- `qty_first_order`、`price_tag`、`month` 保留数值语义
- `qty_first_order` 已证明对冷启动和 tail 有显著信息量

## 不再优先做的事项

- 不再扩 `sequence`
- 不再继续 `CatBoost/XGBoost` 家族对比
- 不再继续 `sep095/sep098` 小校准
- `1_3_ratio` 仅做 guardrail
