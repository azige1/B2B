# 盈亏分析 V0 规则固化

## 来源

基于甲方对 4.30 会议问题的反馈整理。

## 已确认规则

| 规则 | V0 处理 |
| --- | --- |
| 成本口径 | `生产成本 = 生产件数 * 单件成本` |
| 成本粒度 | 按 SKC 维护成本 |
| 成本缺失兜底 | `单件成本 = 吊牌价 / 7` |
| 首批最小生产件数 | 每个 SKC 默认 `100` 件 |
| 增量粒度 | 默认 `10` 件 |
| 生产周期 | 每个 SKC 不同，后续需要表格提供 |
| 生命周期 | 当前统一按 `45` 天 |
| 生命周期后残值 | `0` |
| 售罄目标 | 45 天内售罄率 `85%` |
| 资金/仓储成本 | V0 暂时忽略 |
| 下单策略 | 越早越好以降低库存为 0 的情况；预测过大时允许分批下单 |

## 代码侧变化

原型模块原来只有 `min_batch_qty`，不能同时表达“100 件起做”和“10 件递增”。

现在已新增：

- `min_batch_qty`：最小生产/补货批量，默认 `100`
- `increment_batch_qty`：增量粒度，默认 `10`

因此候选生产量可以表达为：

`0, 100, 110, 120, ...`

## 配置文件

规则配置已写入：

`modules/profit_analysis/config/profit_analysis_business_defaults_client_feedback_20260515.csv`

其中：

- `min_batch_qty = 100`
- `increment_batch_qty = 10`
- `salvage_ratio_to_unit_cost = 0`
- `holding_cost_ratio_per_day_to_unit_cost = 0`
- `target_sell_through_rate = 0.85`
- `lifecycle_days_assumption = 45`
- `cost_fallback_rule = price_tag_div_7_when_cost_missing`

## 仍缺的数据

这些字段甲方后续仍需提供或确认：

- SKC 单件成本表
- SKC 生产周期表
- 当前库存
- 在途/未到货
- 确认订单或生产订单日志

## 注意

这个 V0 规则框架可以先跑原型，但不是最终经营决策规则。

尤其是：

- 单件成本缺失时用 `吊牌价/7` 只是兜底；
- 生产周期如果缺失，只能用默认值；
- 45 天生命周期是当前统一规则，后续如果有品类/款式差异需要替换。
