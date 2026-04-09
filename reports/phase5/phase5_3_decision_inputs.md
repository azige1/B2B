# Phase5.3 决策输入

- 基准上下文：`eval_context_p512_bilstm_l3_v5_lite_recheck.csv`

## 决策

1. 是否必须上 `event/tree`：**是**
   - 证据：`repl0_fut1` 只有 `90` 行但 `pos_rate=0.5111`，说明 `future` 是局部高价值信号；`repl1_fut0` 的 `ratio=1.8500`，说明当前共享 sequence 路线在大块区域校准错误。

2. 是否必须加入 `buyer coverage`：**是**
   - 证据：`future_buyer_count=3+` 的 `pos_rate=0.4943`，而 `future_buyer_count=0` 只有 `0.1301`；高覆盖组仍然 `ratio=0.8045`，说明这层结构必须显式建模。

3. 是否需要冷启动 fallback：**是**
   - 证据：`ice` 桶有 `6653` 行，`sku_p50=0.2055`；共享主模型不能把这部分拉回健康区间。

4. 是否需要爆款单独权重/单独头：**是**
   - 证据：爆款只有 `85` 行，但 `ratio=0.0797`，属于系统性低估，不是随机波动。

5. 哪些原始 V5 派生特征明确不再保留：
   - `repl_velocity`
   - `days_since_last`（原 dense 逐日版本）
   - `repl_volatility`（原 dense 逐日版本）
   - `fut2repl_ratio`（原 dense 逐日版本）
   - 理由：这些特征在原始 `V5(10维)` 中与稀疏事件信号混合，已被 `phase5.1` 和既有审计证明会扰乱校准，而不是稳定增益。

## Phase5.3 默认方向
- 主线：`event/rolling features + hurdle tree baseline`
- 对照：保留 `V3-filtered / V5-lite` sequence 结果，不再继续扩原始 `V5(10维)`
- 先做单锚点 `2025-12-01`，赢了再上 rolling backtest 和服务器多 seed
