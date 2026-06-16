# 盈亏分析模块交付与验收说明

更新时间：2026-06-12

## 1. 交付状态

模块当前已完成以下闭环：

```text
Phase7 SKU hurdle预测
  -> 历史概率校准
  -> 45天需求场景校准
  -> SKU聚合到SKC
  -> SKC候选生产量
  -> 逐日库存与利润模拟
  -> 风险调整后的最终推荐
  -> SKC总量分配到SKU
  -> 质量报告、推荐明细和运行清单
```

`style_id` 按甲方确认作为 SKC 主键。生产约束为每个 SKC 最少 100 件、10 件递增。

## 2. 核心公式

SKU 模型输出：

```text
p_i = P(D_i > 0)
q_i = E[D_i | D_i > 0]
```

历史校准后的 SKC 概率和条件数量：

```text
p_g = 1 - product_i(1 - Calibrate(p_i))

E[D_g] = sum_i(Calibrate(p_i) * q_i)

q_g = E[D_g] / p_g
```

Phase7 的数量输出是 30 天条件数量，分析窗口为 H 天：

```text
q_g,H = q_g * H / 30
```

45 天需求场景：

```text
zero: D_0 = 0
low:  D_1 = q_g,45 * 0.4580
mid:  D_2 = q_g,45 * 0.6670
high: D_3 = q_g,45 * 1.1280

Pr_0 = 1 - p_g
Pr_1 = p_g * 0.25
Pr_2 = p_g * 0.50
Pr_3 = p_g * 0.25
```

这些倍数由 42,844 条历史锚点样本和真实 45 天需求估计，不再人工固定为 `0.6、1.0、1.5`。

对每个场景、候选生产量和每一天：

```text
opening_t = ending_(t-1) + arrival_t
sold_t = min(opening_t, demand_t)
lost_t = max(demand_t - opening_t, 0)
ending_t = opening_t - sold_t
```

单场景利润：

```text
Revenue_s(Q) = sum_t(sold_t) * UnitPrice
TerminalValue_s(Q) = ending_H * SalvageValue
ProductionCost(Q) = Q * UnitCost
HoldingCost_s(Q) = sum_t(ending_t * HoldingCostPerUnitPerDay)
StockoutCost_s(Q) = sum_t(lost_t) * StockoutPenaltyPerUnit

Profit_s(Q) =
    Revenue_s(Q)
  + TerminalValue_s(Q)
  - ProductionCost(Q)
  - HoldingCost_s(Q)
  - StockoutCost_s(Q)
  - FixedCost
```

期望利润：

```text
ExpectedProfit(Q) = sum_s(Pr_s * Profit_s(Q))
```

最终推荐不是只取期望利润最大值，而是在期望利润基础上扣除利润波动、剩余库存、缺货和售罄目标偏差的惩罚。候选集合始终保留 `Q = 0`，所以模块可以明确推荐不补货。

## 3. 45 天真实回测

主样本：

```text
严格库存锚点：2026-02-15、2026-02-24
完整SKC-锚点：2,682
SKU-锚点：7,853
未来45天真实需求：15,406
锚点库存：169,376
```

Phase7、平衡策略、售价按吊牌价 100% 的结果：

| 策略 | 正计划数 | 总生产量 | 有利 | 有害 | 相对不补货实验利润 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 不补货 | 0 | 0 | 0 | 0 | 0 |
| 直接按预测缺口 | 210 | 21,000 | 6 | 204 | -3,914,626.24 |
| 盈亏模块 | 3 | 300 | 3 | 0 | +130,980.00 |
| 事后真实缺口 | 97 | 9,700 | 15 | 82 | -769,284.18 |

盈亏模块相对直接按预测缺口下单，少生产 20,700 件，实验利润改善 `+4,045,606.24`。

推荐的三个 SKC-锚点计划：

| 锚点 | style_id | 计划量 | 相对不补货实际增益 |
| --- | --- | ---: | ---: |
| 2026-02-15 | AK10601651 | 100 | +62,670 |
| 2026-02-15 | AK10801537 | 100 | +30,025 |
| 2026-02-24 | AK10801537 | 100 | +38,285 |

这些金额用于同口径策略比较，不是审计会计利润。

## 4. 概率与价格敏感性

45 天概率校准：

```text
原始Brier Score = 0.07978
校准后Brier Score = 0.07320
改善 = 8.24%
```

价格敏感性：

| 售价假设 | 盈亏模块生产量 | 相对不补货实验利润 |
| --- | ---: | ---: |
| 吊牌价100% | 300 | +130,980.00 |
| 吊牌价50% | 200 | +27,010.00 |
| 吊牌价50%且有持有成本 | 200 | +26,622.03 |
| 吊牌价30% | 100 | +6,484.00 |
| 吊牌价20% | 0 | 0 |

## 5. 生产运行

先构建标准输入：

```powershell
python modules/profit_analysis/scripts/build_profit_analysis_inputs.py `
  --prediction-csv <prediction.csv>
```

再运行 SKC 决策：

```powershell
python modules/profit_analysis/scripts/run_skc_profit_snapshot.py `
  --prediction-csv <prediction_snapshot.csv> `
  --inventory-csv <inventory_snapshot.csv> `
  --economics-csv <economics_config.csv> `
  --policy balanced `
  --horizon-days 45 `
  --output-dir reports/profit_analysis_snapshot
```

输出：

- `skc_recommendations_*.csv`：SKC 最终推荐和核心经济指标；
- `sku_allocations_*.csv`：推荐总量到 SKU 的整数分配；
- `recommendation_details_*.json`：全部候选量、需求场景和评分；
- `quality_report_*.json`：输入覆盖、成本兜底和分配守恒；
- `run_manifest_*.json`：运行参数和产物路径。

质量门禁会拒绝：

- 预测、库存或经济配置重复键；
- 预测行缺库存或经济配置；
- `inventory_snapshot_present = 0`；
- 缺少 `style_id`；
- 非法概率、预测量、库存、成本或售价；
- SKU 分配总量不等于 SKC 推荐总量；
- 可选的成本兜底比例超限。

## 6. 验收结果

```text
单元测试：31/31通过
模板快照：通过
真实快照：通过
真实快照日期：2026-02-15
真实快照SKC：1,349
真实快照SKU：3,950
质量门禁：全部通过
正推荐SKC：2
推荐总量：200
```

真实快照生产入口输出与历史回测一致，推荐 `AK10601651` 和 `AK10801537`。

## 7. 尚需甲方提供

以下缺口不阻塞模块运行，但阻塞“真实财务利润”和正式上线：

1. 实际成交价、成交金额或统一结算折扣；
2. SKC 真实生产周期；
3. 已确认的在途和生产订单；
4. 实际采用推荐后的人工决策、到货和后续销售日志。

当前真实成本优先按 `style_id` 关联；未覆盖的 SKU 使用甲方确认的 `吊牌价 / 7`。实际成交价缺失时必须保留价格敏感性，不应把吊牌价场景解释为真实收入。
