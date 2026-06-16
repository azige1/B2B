# 盈亏分析模块完整设计与真实数据实验

更新时间：2026-06-12

## 1. 当前结论

盈亏分析模块已经从“单 SKU 利润计算原型”升级为：

1. 使用当前官方 Phase7 LightGBM hurdle 模型输出；
2. 用历史真实标签校准补货发生概率和需求场景；
3. 在 SKC 层决定是否启动生产批次以及生产多少；
4. 将 SKC 总量分配到尺码 SKU；
5. 按真实逐日需求、库存和到货时间模拟销售、缺货、剩余库存与利润；
6. 对“不补货、直接按预测补货、盈亏模块推荐、事后真实缺口量”进行同口径回测。

当前模块已经具备可运行代码、测试、真实 SKC 成本接入和真实数据实验，但利润金额仍不是审计财务利润。原因是实际成交价、正式生产周期和确认在途订单尚未提供；未被成本表覆盖的 SKC 仍需使用成本兜底规则。

## 2. 模块定位

预测模型回答：

> 未来会不会发生补货需求，以及发生时大约需要多少件？

盈亏分析模块回答：

> 考虑库存、批量约束、成本、售价、到货时间和需求不确定性后，现在生产多少件最合算？

预测数量是需求信号，推荐数量是经营决策，二者不能直接画等号。

```text
Phase7 SKU预测
  -> 概率校准与需求区间校准
  -> SKU输出聚合到SKC
  -> 生成SKC候选生产量
  -> 各需求场景逐日库存流模拟
  -> 计算期望利润与风险
  -> 选择SKC推荐量
  -> 推荐量分配到尺码SKU
  -> 输出决策明细和回测指标
```

## 3. 决策粒度与时间口径

### 3.1 决策粒度

- 生产批量约束属于 SKC；
- 甲方确认 `style_id` 就是 SKC 主键；
- 推荐总量在 SKC 层产生；
- 到货后的库存和销售在 SKU 层模拟；

不能把“每个 SKC 最少 100 件”错误地应用到每个 SKU。真实实验已经证明，这会把少量需求放大成大量过补。

### 3.2 30 天与 45 天

- Phase7 当前数量输出和真实标签是未来 30 天；
- 甲方当前业务生命周期规则是 45 天；
- 引擎支持任意分析天数，默认业务配置为 45 天；
- `wide_table_sku.csv` 的逐日需求覆盖到 2026-04-16，足够为 2026-02-15 和 2026-02-24 两个库存锚点构造严格的未来 45 天真实标签和逐日曲线；
- 当前正式回测已经使用 45 天真实需求，不再把 30 天真实量直接当作 45 天结果。

若临时使用 30 天模型估算 H 天需求，当前公式为：

```text
q_H = q_30 * H / 30
```

当前 Phase7 预测输入仍通过上式扩展到 45 天，但需求场景倍数使用真实 45 天标签重新校准，回测结果也使用真实 45 天逐日需求。长期最合理方案仍是直接训练 45 天数量目标，或建立逐日需求曲线模型。

## 4. 输入数据

### 4.1 三类核心输入

预测输入：

- `sku_id`
- `snapshot_date`
- `pred_prob_positive`：未来窗口需求大于 0 的原始概率
- `pred_qty_30d`：在需求发生条件下的 30 天数量预测
- `prediction_version`

库存与供应约束：

- 当前库存 `I_i`
- 已确认在途 `W_i`
- 生产周期 `L`
- 最小批量 `B = 100`
- 递增粒度 `Delta = 10`
- 最大补货量、已下单量、安全库存等可选字段

经济参数：

- 单件成本 `c`
- 单件实际售价 `p`
- 单件每日持有成本 `h`
- 生命周期末残值 `v`
- 缺货惩罚 `u`
- 固定成本 `F`
- 目标售罄率 `tau = 85%`

### 4.2 本次实验哪些是真实数据

真实项目数据：

- Phase7 和 Phase8 辅助预测输出；
- 未来 45 天真实补货数量；
- 未来 45 天逐日真实补货节奏；
- 锚点当天真实库存快照；
- SKU、`style_id`、品类、吊牌价、上市日期；
- 历史 Phase7 锚点标签，用于概率和数量场景校准。

真实成本接入情况：

- 成本工作表按“款号 = style_id”精确关联；
- 2,537 个 SKC 有有效正成本；
- 项目全量 5,523 个 SKC 中精确匹配 2,224 个，覆盖率 40.27%；
- 本次回测每个锚点约 69.5% 的 SKC 有真实成本；
- 同一 style_id 有多条不同成本时，保守取最大值；
- 未匹配项继续使用 `c = 吊牌价 / 7`。

当前代理或默认值：

- 实际售价分别取吊牌价的 20%、30%、50%、100% 做敏感性；
- 默认生产周期 7 天；
- 残值为 0；
- V0 持有成本和缺货成本为 0，另增加持有成本敏感性；
- 没有真实确认在途时按 0 处理。

### 4.3 实际成交价现状

当前仓库没有可直接用于收入计算的逐交易或逐 SKC 实际成交价：

- `V_IRS_ORDERFTP` 只有款号、SKU、客户、日期、数量和订单类型；
- `V_IRS_ORDER_2025` 有吊牌价 `PRICELIST`，没有实收金额；
- `V_SALE03181334` 有发货数量，没有成交单价或成交金额；
- `avg_discount_rate` 是买手级平均折扣率，只覆盖约 35% 的买手，不是逐 SKC 或逐交易价格。

因此当前不能把 `price_tag * avg_discount_rate` 直接称为真实成交价。它最多可作为后续增加的一种买手级价格敏感性方案，正式收入仍需要销售金额、含税成交单价或可确认的统一结算折扣。

## 5. 预测校准公式

设 SKU i 的原始 hurdle 输出为：

```text
p_i_raw = P(D_i > 0)
q_i_pos = E[D_i | D_i > 0]
```

其中：

- `D_i`：未来窗口 SKU i 的需求量；
- `p_i_raw`：需求是否发生的原始概率；
- `q_i_pos`：需求发生条件下的数量预测。

### 5.1 概率校准

使用历史 Phase7 数据拟合 isotonic regression：

```text
p_i = Calibrate(p_i_raw)
```

它学习一个单调函数，使“预测为某个概率的样本”更接近真实发生比例。

本次校准结果：

```text
总样本数 = 42,844
真实正样本数 = 8,910
原始 Brier Score = 0.07978
校准后 Brier Score = 0.07320
```

Brier Score 越低越好。45 天口径下校准后下降约 8.24%，校准概率均值也从 24.46% 修正为与真实正样本率一致的 20.80%。

### 5.2 正需求数量场景校准

对历史正样本计算：

```text
r_j = ActualQty_j / PredictedConditionalQty_j
```

再取分位数作为低、中、高场景倍数。本次真实历史数据得到：

```text
m_low  = 0.4580
m_mid  = 0.6670
m_high = 1.1280
```

这些倍数作用在 `q_45 = q_30 * 1.5` 上。若换算成相对原始 `q_30` 的有效倍数，约为 `0.6871、1.0006、1.6920`。

场景内部权重为：

```text
w_low  = 0.25
w_mid  = 0.50
w_high = 0.25
```

因此不再人工固定使用 0.6、1.0、1.5，而是由历史误差分布估计。

## 6. 从 SKU 聚合到 SKC

设一个 SKC g 包含 n 个尺码 SKU。

SKC 发生正需求的概率：

```text
p_g = 1 - product_i(1 - p_i)
```

当前公式使用 SKU 事件条件独立近似。含义是：只要该 SKC 下至少一个 SKU 出现需求，就认为 SKC 有正需求。

SKC 无条件期望需求量：

```text
E[D_g] = sum_i(p_i * q_i_pos)
```

SKC 条件需求量：

```text
q_g_pos = E[D_g] / p_g,  当 p_g > 0
```

因此：

```text
p_g * q_g_pos = E[D_g]
```

这保证聚合前后的期望需求量一致。后续可以直接训练 SKC 概率与数量模型，替换条件独立假设。

## 7. 需求场景

对每个 SKC 建立四个场景：

```text
场景0：D_g,0 = 0
概率：Pr_0 = 1 - p_g

场景1：D_g,1 = q_g_pos * m_low
概率：Pr_1 = p_g * w_low

场景2：D_g,2 = q_g_pos * m_mid
概率：Pr_2 = p_g * w_mid

场景3：D_g,3 = q_g_pos * m_high
概率：Pr_3 = p_g * w_high
```

满足：

```text
sum_s Pr_s = 1
```

场景不是四次重复预测，而是把预测误差和零需求风险显式放进决策。

## 8. 候选生产量

设：

```text
I_g = SKC当前库存
W_g = 分析窗口内已确认在途
D_hat_g = 分析窗口条件需求预测
tau = 目标售罄率
```

基础缺口：

```text
Gap_g = max(D_hat_g - I_g - W_g, 0)
```

达到目标售罄率时允许的总供给：

```text
TargetSupply_g = D_hat_g / tau
```

对应生产量：

```text
Q_sellthrough = max(TargetSupply_g - I_g - W_g, 0)
```

候选原始数量包括：

```text
0
最小批量
0.5 * Gap_g
0.8 * Gap_g
1.0 * Gap_g
Q_sellthrough
1.2 * D_hat_g - I_g - W_g
1.5 * D_hat_g - I_g - W_g
```

批量归一化公式：

```text
RoundBatch(x) =
    0,                               x <= 0
    Delta * ceil(max(x, B) / Delta), x > 0
```

当前：

```text
B = 100
Delta = 10
```

所以可选生产量是：

```text
0, 100, 110, 120, ...
```

## 9. 单场景逐日库存模拟

对候选生产量 Q 和需求场景 s，在每天 t 模拟库存流。

设：

- `d_s,t`：场景 s 在第 t 天的需求；
- `A_t`：第 t 天到货量，到货日为 Q，其他日期为 0；
- `I_s,t,open`：当天销售前可用库存；
- `S_s,t`：当天实际销量；
- `L_s,t`：当天缺货损失量；
- `I_s,t,end`：当天结束库存。

公式为：

```text
I_s,t,open = I_s,t-1,end + A_t

S_s,t = min(I_s,t,open, d_s,t)

L_s,t = max(d_s,t - S_s,t, 0)

I_s,t,end = I_s,t,open - S_s,t
```

初始库存：

```text
I_s,0,end = I_g + W_g
```

生产量在生产周期 L 后到货：

```text
A_t =
    Q, t = L
    0, 其他日期
```

因此，到货前的需求可能形成真实缺货，不会把未来到货错误地当成今天可卖库存。

## 10. 单场景利润公式

场景 s 的总销量、总缺货和期末库存：

```text
Sold_s = sum_t(S_s,t)
Lost_s = sum_t(L_s,t)
Left_s = I_s,H,end
```

销售收入：

```text
Revenue_s = Sold_s * p
```

期末残值：

```text
TerminalValue_s = Left_s * v
```

生产成本：

```text
ProductionCost(Q) = Q * c
```

库存持有成本：

```text
HoldingCost_s = sum_t(I_s,t,end * h)
```

缺货成本：

```text
StockoutCost_s = Lost_s * u
```

总成本：

```text
TotalCost_s =
      ProductionCost(Q)
    + HoldingCost_s
    + StockoutCost_s
    + F
```

单场景利润：

```text
Profit_s(Q) =
      Revenue_s
    + TerminalValue_s
    - TotalCost_s
```

按当前甲方 V0 设置：

```text
v = 0
h = 0
u = 0
F = 0
```

基础口径简化为：

```text
Profit_s(Q) = Sold_s * p - Q * c
```

现有库存的历史采购成本属于沉没成本，因此策略比较只扣本次新增生产量 Q 的成本。所有策略都拥有相同初始库存，比较的是新增决策的增量收益。

## 11. 概率加权指标

每个候选生产量 Q 都要在所有需求场景下计算，再按概率加权。

```text
ExpectedProfit(Q)
    = sum_s(Pr_s * Profit_s(Q))

ExpectedRevenue(Q)
    = sum_s(Pr_s * Revenue_s)

ExpectedTerminalValue(Q)
    = sum_s(Pr_s * TerminalValue_s)

ExpectedHoldingCost(Q)
    = sum_s(Pr_s * HoldingCost_s)

ExpectedStockoutCost(Q)
    = sum_s(Pr_s * StockoutCost_s)

ExpectedTotalCost(Q)
    = Q * c
    + ExpectedHoldingCost(Q)
    + ExpectedStockoutCost(Q)
    + F
```

期望销量、剩余量和缺货量：

```text
ExpectedSoldQty(Q)
    = sum_s(Pr_s * Sold_s)

ExpectedLeftoverQty(Q)
    = sum_s(Pr_s * Left_s)

ExpectedLostSalesQty(Q)
    = sum_s(Pr_s * Lost_s)
```

利润方差：

```text
ProfitVariance(Q)
    = sum_s(Pr_s * Profit_s(Q)^2)
    - ExpectedProfit(Q)^2
```

期望售罄率：

```text
SellThroughRate(Q)
    = ExpectedSoldQty(Q) / (I_g + W_g + Q)
```

所有场景指标必须用同一组概率加权，不能只加权利润而使用另一个场景的库存或缺货量。

## 12. 最终推荐评分

定义：

```text
RiskPenalty(Q)
    = sqrt(ProfitVariance(Q))

LeftoverPenalty(Q)
    = ExpectedLeftoverQty(Q) * max(c - v, 0)

StockoutPenalty(Q)
    = ExpectedLostSalesQty(Q) * p

SellThroughGap(Q)
    = max(tau - SellThroughRate(Q), 0)

SellThroughPenalty(Q)
    = SellThroughGap(Q) * ExpectedSupplyQty(Q) * c
```

平衡策略：

```text
Score_balanced(Q) =
      ExpectedProfit(Q)
    - 0.25 * RiskPenalty(Q)
    - 0.35 * LeftoverPenalty(Q)
    - 0.20 * StockoutPenalty(Q)
    - 0.20 * SellThroughPenalty(Q)
```

保守策略：

```text
Score_conservative(Q) =
      ExpectedProfit(Q)
    - 0.50 * RiskPenalty(Q)
    - 0.70 * LeftoverPenalty(Q)
    - 0.10 * StockoutPenalty(Q)
    - 0.30 * SellThroughPenalty(Q)
```

激进策略：

```text
Score_aggressive(Q) =
      ExpectedProfit(Q)
    - 0.15 * RiskPenalty(Q)
    - 0.10 * LeftoverPenalty(Q)
    - 0.45 * StockoutPenalty(Q)
    - 0.10 * SellThroughPenalty(Q)
```

最终推荐量：

```text
Q_recommended = argmax_Q Score_policy(Q)
```

候选集中必须保留 `Q = 0`。当所有正生产量都不划算时，模块可以明确推荐“不补货”。

## 13. SKC 总量分配到 SKU

推荐得到 SKC 总量 `Q_g` 后，先计算 SKU 的校准需求缺口分数：

```text
mu = sum_k(w_k * m_k)

g_i = max(p_i * q_i_pos * mu - I_i, 0)
```

若 `sum_i(g_i) > 0`，理想分配量为：

```text
a_i_raw = Q_g * g_i / sum_j(g_j)
```

先取整数下界：

```text
a_i_floor = floor(a_i_raw)
```

剩余件数：

```text
R = Q_g - sum_i(a_i_floor)
```

按 `a_i_raw - a_i_floor` 从大到小，把 R 件逐件分配。最终保证：

```text
a_i >= 0
sum_i(a_i) = Q_g
```

## 14. 真实数据回测设计

### 14.1 防止数据泄漏

库存只能使用：

```text
inventory_date <= snapshot_date
```

正式主样本进一步要求：

```text
inventory_date = snapshot_date
inventory_snapshot_present = 1
```

一个 SKC 下所有参与预测的 SKU 都必须有当天库存，避免只保留有库存的尺码造成选择偏差。

### 14.2 回测锚点

真实库存覆盖允许严格回测的锚点为：

```text
2026-02-15
2026-02-24
```

Phase7 的 2025 年锚点可用于校准概率和数量误差，但库存快照不覆盖这些日期，不能用于库存利润回测。

### 14.3 对比策略

```text
no_replenishment:
    Q = 0

model_direct:
    Q = RoundBatch(max(模型最终预测量 - 当前库存 - 在途, 0))

profit_module:
    Q = 风险调整期望利润评分最高的候选量

hindsight_qty:
    Q = RoundBatch(max(真实未来需求 - 当前库存 - 在途, 0))
```

`hindsight_qty` 使用未来真实需求，只作为上限诊断，不是可上线策略。它仍受 100 件最小批量约束，因此即使知道未来，也可能因为批量过大而亏损。

### 14.4 实际收益计算

回测使用未来 45 天真实逐日需求曲线，并在 SKU 层执行库存流公式。这样可以识别生产到货前已经发生的缺货。

## 15. 真实实验结果

### 15.1 样本与缺口

Phase7 主样本：

```text
完整 SKC-锚点样本 = 2,682
SKU-锚点样本 = 7,853
未来45天真实需求量 = 15,406
锚点库存量 = 169,376
```

需求缺口分布：

```text
真实需求大于库存的 SKC = 97
缺口达到100件的 SKC = 0
全部正缺口合计 = 981件
最大单SKC缺口 = 99件
```

当前样本的核心问题不是“补多少”，而是“是否值得为了不足 100 件的缺口启动一个 100 件批次”。

### 15.2 Phase7、平衡策略、售价等于吊牌价

| 策略 | 正生产SKC数 | 总生产量 | 相对不补货有利 | 相对不补货有害 | 缺货率 | 相对不补货实验利润 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 不补货 | 0 | 0 | 0 | 0 | 9.15% | 0 |
| 直接按模型缺口 | 210 | 21,000 | 6 | 204 | 5.50% | -3,914,626.24 |
| 盈亏模块 | 3 | 300 | 3 | 0 | 7.85% | +130,980.00 |
| 事后真实缺口 | 97 | 9,700 | 15 | 82 | 2.44% | -769,284.18 |

盈亏模块相对直接按模型下单：

```text
实验利润改善 = +4,045,606.24
少生产 = 20,700件
```

最重要的结果不是绝对金额，而是决策结构：

- 直接下单 210 个 SKC，其中 204 个不如不补货；
- 盈亏模块只推荐 3 个 SKC，三次都优于不补货；
- 盈亏模块没有为了降低少量缺货率而接受大规模积压。

### 15.3 两个锚点一致性

```text
2026-02-15：
盈亏模块生产200件，相对不补货 +92,695.00

2026-02-24：
盈亏模块生产100件，相对不补货 +38,285.00
```

两个锚点都推荐 SKC `AK10801537`，并在 2026-02-15 额外推荐 `AK10601651`：

```text
AK10801537：
2026-02-15：真实需求138，库存66，45天预测140.30，相对不补货 +30,025
2026-02-24：真实需求121，库存42，45天预测131.25，相对不补货 +38,285
真实单件成本：71.45元

AK10601651：
2026-02-15：真实需求68，库存8，45天预测38.69，相对不补货 +62,670
真实单件成本：207.30元
```

### 15.4 价格敏感性

| 实际售价假设 | 盈亏模块生产量 | 相对不补货实验利润 |
| --- | ---: | ---: |
| 吊牌价100% | 300 | +130,980.00 |
| 吊牌价50% | 200 | +27,010.00 |
| 吊牌价50%且有持有成本 | 200 | +26,622.03 |
| 吊牌价30% | 100 | +6,484.00 |
| 吊牌价20% | 0 | 0 |

模块会随毛利下降自动减少生产，而不是固定输出同一个数量。

### 15.5 风险策略

| 策略 | 生产量 | 相对不补货实验利润 |
| --- | ---: | ---: |
| 保守 | 200 | +68,310.00 |
| 平衡 | 300 | +130,980.00 |
| 激进 | 300 | +130,980.00 |

平衡和激进策略在当前样本中选择了相同计划；保守策略少启动一个批次，利润增益较低但风险更小。

### 15.6 Phase8 辅助输入

Phase8 zero-split 辅助预测在相同设置下：

```text
生产量 = 340
相对不补货实验利润 = +134,612.00
```

比 Phase7 的 `+130,980.00` 略高，但 Phase7 仍是当前官方主线。该结果只能视为辅助对照，不能据此直接替换生产模型。

## 16. 利润金额的解释边界

当前输出适合解释为：

```text
在真实成本优先、缺失成本兜底、售价敏感性假设下，不同策略的相对经济效果
```

不适合解释为：

```text
公司真实会计利润或可以直接入账的金额
```

相对比较比绝对金额更可信，因为各策略共用同一套售价、成本和库存假设。

## 17. 工程实现与产物

核心代码：

- `src/profit_analysis/core.py`：场景、库存流、利润、评分和推荐；
- `src/profit_analysis/calibration.py`：概率和数量场景校准；
- `src/profit_analysis/horizon.py`：任意窗口真实需求、逐日曲线和校准标签构造；
- `src/profit_analysis/grouping.py`：SKU hurdle 输出聚合到 SKC；
- `src/profit_analysis/allocation.py`：SKC 总量整数分配到 SKU；
- `src/profit_analysis/decision.py`：生产快照的 SKC 推荐、SKU 分配和质量门禁；
- `src/profit_analysis/builders.py`：预测、库存、产品和经济参数构建；
- `src/profit_analysis/io.py`：输入读取与按快照日期连接。

真实实验命令：

```text
python modules/profit_analysis/scripts/run_real_data_experiment.py
python modules/profit_analysis/scripts/run_skc_real_data_experiment.py
python modules/profit_analysis/scripts/run_skc_profit_snapshot.py --prediction-csv <prediction.csv> --inventory-csv <inventory.csv> --economics-csv <economics.csv>
```

主要结果目录：

```text
reports/profit_analysis_real_data_20260612/
reports/profit_analysis_skc_real_cost_20260612/  # 历史30天对照
reports/profit_analysis_skc_real_cost_h45_20260612/
```

关键结果文件：

- `skc_real_data_experiment_report.md`
- `skc_experiment_summary.csv`
- `skc_experiment_aggregate.csv`
- `skc_paired_decision_summary.csv`
- `skc_actual_gap_summary.csv`
- `skc_plan_detail.csv`
- `skc_allocation_detail.csv`
- `demand_scenario_calibration.json`
- 生产快照输出：`skc_recommendations_*.csv`、`sku_allocations_*.csv`、`quality_report_*.json`

测试命令：

```text
python -m unittest discover -s modules/profit_analysis/tests -p "test_*.py" -v
```

当前共有 31 个测试，覆盖：

- 批量取整；
- 单场景利润；
- 到货前缺货；
- 真实逐日需求节奏；
- 生命周期截断；
- 历史库存防泄漏；
- 多快照日期连接；
- 概率与数量场景校准；
- SKC hurdle 聚合；
- SKC 总量整数分配；
- 45 天真实标签和逐日曲线；
- 生产快照输入质量门禁与分配守恒。

## 18. 当前完成度与下一步

已经完成：

- 核心公式与代码；
- 甲方 V0 规则配置；
- 30/45 天可配置引擎；
- SKC 决策与 SKU 分配；
- 历史概率和数量场景校准；
- 严格 45 天真实标签和逐日曲线；
- 历史库存 as-of 防泄漏；
- 真实逐日需求回放；
- 多策略、多价格、多风险偏好实验；
- 明细、汇总和可复现报告；
- 生产级 SKC 快照入口和质量报告；
- 单元测试。

仍需甲方数据：

- 实际成交价或折扣；
- SKC 真实生产周期；
- 确认在途和生产订单；
- 实际采用推荐后的在线决策日志。

下一步优先级：

1. 接入实际成交价、生产周期和在途订单，重跑同一套实验；
2. 直接训练 45 天或 SKC 层需求概率与数量模型，替换 30 天线性扩展；
3. 小范围影子运行，保存系统推荐、人工决策、实际到货和后续需求；
4. 用在线决策日志持续重校准概率、需求场景和风险策略。

## 19. 最终判断

当前盈亏分析模块已经不是只写公式的概念方案，而是有真实库存、真实需求、真实预测输入和可复现实验的决策模块。

实验支持的核心结论是：

> 在当前库存普遍充足、每个 SKC 最少生产 100 件的条件下，预测到少量缺口并不等于应该生产。盈亏模块的主要价值是过滤不经济的补货批次，只在预期销售收益足以覆盖整批成本和风险时启动生产。

上线前最大的剩余问题不是算法代码和 45 天评估口径，而是甲方实际成交价、生产周期、确认在途数据以及影子运行验证。
