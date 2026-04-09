# Phase5.2 Strategy Deep Dive

## Summary
- Reference context: `eval_context_p512_bilstm_l3_v5_lite_recheck.csv`
- Validation universe: `10240` rows, `1734` positive rows (`16.93%`)
- Fair-universe meta: `train_cnt=2,508,800`, `val_cnt=10,240`, `split_date=2025-12-01`
- Structural sparsity in the fair-universe 2025 calendar: `future != 0` on `0.2904%` of SKU-day cells, `replenish != 0` on `1.7659%`

结论：当前数据更像“稀疏事件 + 分段校准”问题，不像标准 dense sequence 问题。`future` 有价值，但它是局部高价值信号，不是高覆盖主信号。

## 1. Future 分层价值
- `repl0_fut1` 只有 `90` 行，但 `pos_rate=0.5111`，说明 `future-only` 是有效领先信号，只是覆盖很小。
- `repl1_fut0` 有 `2458` 行且 `ratio=1.8500`，这是当前最明显的过预测区。
- `repl1_fut1` 的 `ratio=0.8308`，说明补货和期货同时出现时，当前模型反而没有把这部分高价值区域吃透。

| signal_quadrant | rows | pos_rate | true_qty_sum | pred_qty_sum | ratio | wmape_like |
| --- | --- | --- | --- | --- | --- | --- |
| repl0_fut0 | 6563 | 0.0954 | 5605.0002 | 4801.0229 | 0.8566 | 1.6466 |
| repl0_fut1 | 90 | 0.5111 | 54.0000 | 86.7572 | 1.6066 | 1.0121 |
| repl1_fut0 | 2458 | 0.2229 | 1609.0000 | 2976.5783 | 1.8500 | 1.8614 |
| repl1_fut1 | 1129 | 0.4553 | 2628.0000 | 2183.2848 | 0.8308 | 0.9261 |

按品类看，`future` 不是全局同权信号。只保留了有足够样本和未来历史支撑的品类：

| category | rows | pos_rate_with_future | pos_rate_without_future | future_lift | ratio_with_future | recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| 连衣裙 | 40 | 0.3000 | 0.0307 | 9.7688 | 3.0686 | Future history adds meaningful lift here; preserve and interact with it. |
| 套装 | 28 | 0.2857 | 0.0426 | 6.7143 | 0.8828 | Future history adds meaningful lift here; preserve and interact with it. |
| 马夹 | 18 | 0.2778 | 0.0505 | 5.5051 | 4.3093 | Future history adds meaningful lift here; preserve and interact with it. |
| 衬衣 | 24 | 0.4167 | 0.0920 | 4.5312 | 2.0300 | Future history adds meaningful lift here; preserve and interact with it. |
| T恤 | 101 | 0.4851 | 0.1092 | 4.4440 | 1.2089 | Future history adds meaningful lift here; preserve and interact with it. |
| 裙子 | 152 | 0.4145 | 0.1062 | 3.9021 | 1.2556 | Future history adds meaningful lift here; preserve and interact with it. |
| 裤类 | 293 | 0.4949 | 0.1323 | 3.7412 | 0.7091 | Future history adds meaningful lift here; preserve and interact with it. |
| 单衣 | 41 | 0.4146 | 0.1202 | 3.4487 | 1.0869 | Future history adds meaningful lift here; preserve and interact with it. |

## 2. Buyer Coverage
- `future_buyer_count=3+` 的正样本率是 `0.4943`，而 `future_buyer_count=0` 只有 `0.1301`。
- 但 `future_buyer_count=3+` 的 `ratio=0.8045`，说明高覆盖未来需求目前被低估；这不是噪声，而是结构信息丢失。

| future_buyer_count_90 | rows | pos_rate | true_qty_sum | pred_qty_sum | ratio | wmape_like | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 9021 | 0.1301 | 7214.0002 | 7777.6012 | 1.0781 | 1.6945 | Keep as baseline reference. |
| 1 | 113 | 0.2389 | 71.0000 | 132.0587 | 1.8600 | 1.5255 | Over-predicted segment; lower gate intensity or add segment-specific calibration. |
| 2 | 54 | 0.2407 | 13.0000 | 47.9272 | 3.6867 | 2.9753 | Over-predicted segment; lower gate intensity or add segment-specific calibration. |
| 3+ | 1052 | 0.4943 | 2598.0000 | 2090.0562 | 0.8045 | 0.9013 | Include buyer coverage features; high-coverage future demand is currently under-predicted. |

判断：`phase5.3` 必须引入 buyer coverage 特征，不应继续只用 SKU 聚合总量。

## 3. 冷启动细分
- `ice` 桶有 `6653` 行，`sku_p50=0.2055`，共享主模型无法单独解决这部分。
- `no_repl_yes_future` 有 `90` 行，虽然规模小，但它是最纯净的 `future-led` 冷启动口袋，值得单独保留策略。

| cold_type | rows | pos_rate | true_qty_sum | pred_qty_sum | ratio | wmape_like | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| both | 1129 | 0.4553 | 2628.0000 | 2183.2848 | 0.8308 | 0.9261 | High-signal segment; keep explicitly visible in Phase5.3 features. |
| no_repl_no_future | 6563 | 0.0954 | 5605.0002 | 4801.0229 | 0.8566 | 1.6466 | Keep as baseline reference. |
| no_repl_yes_future | 90 | 0.5111 | 54.0000 | 86.7572 | 1.6066 | 1.0121 | Keep a dedicated future-led fallback path; this is the cleanest cold-start pocket. |
| repl_only | 2458 | 0.2229 | 1609.0000 | 2976.5783 | 1.8500 | 1.8614 | Over-predicted segment; lower gate intensity or add segment-specific calibration. |

判断：冷启动不是边角问题，而是主问题。`phase5.3` 需要明确的 cold-start fallback，而不是继续只调主模型。

## 4. 爆款前兆
- 爆款样本数 `85`，但 `ratio=0.0797`，当前 quantity 路线对大需求款完全没接住。
- 爆款在过去 90 天并非无信号：它们通常有更高的 `repl_sum_90`、`future_sum_90` 和 `future_buyer_count_90`。

| blockbuster_flag | rows | pos_rate | true_qty_sum | pred_qty_sum | ratio | wmape_like | recommendation |
| --- | --- | --- | --- | --- | --- | --- | --- |
| blockbuster | 85 | 1.0000 | 4089.0001 | 325.7728 | 0.0797 | 0.9203 | Add blockbuster weighting or a dedicated quantity calibration path. |
| non_blockbuster | 10155 | 0.1624 | 5807.0000 | 9721.8705 | 1.6742 | 1.8856 | Over-predicted segment; lower gate intensity or add segment-specific calibration. |

判断：`phase5.3` 需要对爆款做单独权重或单独 quantity 校准，否则整体 WMAPE 改善也不会转化成关键 SKU 改善。

## 5. 单锚点稳定性判断

更可能稳定的结论：
- `future` is a sparse but high-value specialist signal. In the fair-universe 2025 calendar it is active on only 0.2904% of SKU-day cells, while in the real validation universe `repl0_fut1` still reaches `pos_rate=0.5111`.
- `buyer coverage` carries real structure. `future_buyer_count=3+` has `pos_rate=0.4943` versus `0.1301` at `future_buyer_count=0`.
- `cold-start` is a first-order problem. The `ice` bucket has `6653` rows with `sku_p50=0.2055`.

明显依赖单锚点的结论：
- Exact category ordering and which categories are most over/under-predicted still depend on the single `2025-12-01` anchor.
- Exact ratio values by bucket should be re-checked under rolling backtest before turning them into production policy thresholds.
- Model-family conclusions (`sequence` vs `event/tree`) should still be validated on `phase5.2` outputs and then rolling anchors.

## 6. Phase5.3 决策含义
- 默认主路线应转向 `event/tree`，sequence 保留为对照基线。
- `buyer coverage` 必须加入下一版特征。
- `cold-start fallback` 需要作为显式设计，而不是后处理补丁。
- `blockbuster` 需要单独权重或单独 quantity 路径。
- 原始 `V5` 的 dense 派生维度不应继续保留为主路线输入。
