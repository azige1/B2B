# 盈亏分析模块当前状态与使用说明

更新时间：2026-06-16

## 1. 当前定位

盈亏分析模块现在是一个独立的 V1 决策模块，目录为 `modules/profit_analysis/`。它不替代预测模型，而是接在预测模型后面，把“未来可能补多少”转换成“现在是否值得生产、生产多少、分到哪些 SKU”。

可以这样理解：

```text
预测模块：未来 30/45 天大概有多少需求？
盈亏模块：考虑库存、成本、售价、批量和风险后，这个需求值值得转成生产计划吗？
```

所以预测量和推荐生产量不是一个概念。预测量是需求信号，推荐生产量是经营决策。

## 2. 当前已经完成什么

模块已经具备以下能力：

- SKU 级预测输入标准化；
- 库存快照、在途量、生产批量、生命周期窗口输入；
- 成本、售价、残值、持有成本、缺货成本等经济参数输入；
- 历史概率校准和 45 天需求场景校准；
- SKU 级 hurdle 输出聚合到 SKC/style 级；
- SKC 级候选生产量生成；
- 多需求场景下的逐日库存流与利润模拟；
- 保守、均衡、激进三种策略；
- SKC 推荐量分配回 SKU；
- 输入质量门禁、运行 manifest、推荐明细、质量报告；
- 单元测试覆盖核心公式、校准、分组、分配、生命周期和质量门禁。

当前不是“正式上线系统”，但已经不是文档原型。它是可以运行、可以回测、可以演示的决策原型。

## 3. 模块输入

生产式入口需要三张标准 CSV。

### 3.1 预测快照 prediction

核心字段：

- `sku_id`：SKU；
- `style_id`：SKC 主键；
- `snapshot_date`：预测日期；
- `pred_prob_positive`：未来窗口发生正向补货需求的概率；
- `pred_qty_30d`：发生正需求条件下的 30 天预测量；
- `prediction_version`：可选，模型版本。

### 3.2 库存快照 inventory

核心字段：

- `sku_id`；
- `style_id`；
- `snapshot_date`；
- `current_inventory`：当前库存；
- `inbound_within_30d`：30 天内已确认在途；
- `inventory_snapshot_present`：是否有真实库存快照；
- `lead_time_days`：生产或到货周期；
- `min_batch_qty`：最小生产批量；
- `increment_batch_qty`：递增粒度。

### 3.3 经济参数 economics

核心字段：

- `sku_id`；
- `style_id`；
- `unit_cost`：单件成本；
- `unit_price`：单件售价或收入代理；
- `holding_cost_per_unit_per_day`：单件每日持有成本；
- `salvage_value_per_unit`：窗口末残值；
- `stockout_penalty_per_unit`：缺货惩罚；
- `target_sell_through_rate`：目标售罄率；
- `lifecycle_days`：分析窗口或生命周期假设；
- `cost_source`：成本来源。

## 4. 模块输出

生产式入口会输出五类文件：

- `skc_recommendations_*.csv`：每个 SKC 的最终推荐量和核心指标；
- `sku_allocations_*.csv`：SKC 推荐总量如何分配到 SKU；
- `recommendation_details_*.json`：每个 SKC 的候选计划、需求场景、评分明细；
- `quality_report_*.json`：输入覆盖、成本兜底、重复键、分配守恒等质量检查；
- `run_manifest_*.json`：本次运行参数和产物路径。

最重要的业务输出是 `skc_recommendations_*.csv`。其中：

- `recommended_plan_qty > 0` 表示建议生产；
- `recommended_plan_qty = 0` 表示即使有预测需求，也不建议生产；
- `expected_profit` 是期望利润；
- `recommendation_score` 是扣除风险、滞销、缺货和售罄目标偏差后的决策评分；
- `expected_leftover_qty` 表示预计剩余库存；
- `expected_lost_sales_qty` 表示预计丢失需求；
- `profit_positive_probability` 表示利润为正的场景概率。

## 5. 最小示例

示例目录：

`modules/profit_analysis/examples/minimal_snapshot/`

运行命令：

```powershell
python modules/profit_analysis/scripts/run_skc_profit_snapshot.py `
  --prediction-csv modules/profit_analysis/examples/minimal_snapshot/prediction.csv `
  --inventory-csv modules/profit_analysis/examples/minimal_snapshot/inventory.csv `
  --economics-csv modules/profit_analysis/examples/minimal_snapshot/economics.csv `
  --policy balanced `
  --horizon-days 45 `
  --run-id minimal_demo `
  --output-dir modules/profit_analysis/examples/minimal_snapshot/output
```

这个例子有两个 SKC：

| style_id | 情况 | 期望结果 |
| --- | --- | --- |
| AK1001 | 两个 SKU，需求概率高，库存为 0，售价明显高于成本 | 推荐生产 |
| AK1002 | 一个 SKU，需求概率较低，已有 30 件库存，生产 100 件起步不划算 | 推荐 0 |

使用当前内置 45 天校准文件时，实际运行结果是：

| style_id | SKC正需求概率 | 30天条件需求 | 当前库存 | 推荐生产量 | 期望利润 |
| --- | ---: | ---: | ---: | ---: | ---: |
| AK1001 | 0.9367 | 111.8911 | 0 | 140 | 7,826.40 |
| AK1002 | 0.1153 | 20.0000 | 30 | 0 | 144.82 |

SKU 分配结果：

| style_id | sku_id | 校准后概率 | 需求缺口分数 | 分配生产量 |
| --- | --- | ---: | ---: | ---: |
| AK1001 | AK1001-36 | 0.8101 | 70.9692 | 87 |
| AK1001 | AK1001-38 | 0.6667 | 43.8023 | 53 |
| AK1002 | AK1002-36 | 0.1153 | 0.0000 | 0 |

这个例子说明：模型预测有需求，不等于一定生产。盈亏模块会进一步看库存、成本、售价和最小批量。如果生产 100 件起步会造成明显过补或收益不够，模块会输出 0。

## 6. 真实数据回测例子

当前最重要的真实回测目录：

`reports/profit_analysis_skc_real_cost_h45_20260612/`

回测口径：

- 锚点：`2026-02-15`、`2026-02-24`；
- 分析窗口：45 天；
- 决策粒度：`style_id`，即甲方确认的 SKC 主键；
- 生产约束：每个 SKC 最少 100 件，10 件递增；
- 主样本：2,682 个完整 SKC-anchor，7,853 个 SKU-anchor；
- 真实未来 45 天需求：15,406 件；
- 锚点库存：169,376 件。

核心对比：

| 策略 | 正计划数 | 总生产量 | 有利计划 | 有害计划 | 相对不补货实验利润 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 不补货 | 0 | 0 | 0 | 0 | 0 |
| 直接按预测缺口 | 210 | 21,000 | 6 | 204 | -3,914,626.24 |
| 盈亏模块 | 3 | 300 | 3 | 0 | +130,980.00 |
| 事后真实缺口 | 97 | 9,700 | 15 | 82 | -769,284.18 |

这个结果说明，当前数据下最大风险不是“少补”，而是“把预测缺口直接转成生产计划会严重过补”。盈亏模块的价值是把预测信号进行经济过滤，只留下少量期望收益和风险都可接受的计划。

Phase8 zero split 作为预测输入时，真实回测里也能进入同一套盈亏模块：

| 预测输入 | 正计划率 | 总生产量 | 相对不补货实验利润 |
| --- | ---: | ---: | ---: |
| phase7_base | 0.0011 | 300 | +130,980.00 |
| phase8_zero_split | 0.0011 | 340 | +134,612.00 |

这说明盈亏模块不是绑定 Phase7 的固定写法，后续主预测模型升级后，可以继续作为下游决策层复用。

## 7. 本次工程整理的实跑验证

### 7.1 最小示例复跑

命令：

```powershell
python modules/profit_analysis/scripts/run_skc_profit_snapshot.py `
  --prediction-csv modules/profit_analysis/examples/minimal_snapshot/prediction.csv `
  --inventory-csv modules/profit_analysis/examples/minimal_snapshot/inventory.csv `
  --economics-csv modules/profit_analysis/examples/minimal_snapshot/economics.csv `
  --policy balanced `
  --horizon-days 45 `
  --run-id minimal_demo `
  --output-dir modules/profit_analysis/examples/minimal_snapshot/output
```

结果：

```text
quality gate: passed
prediction_rows: 3
skc_rows: 2
positive_plan_skc_rows: 1
total_recommended_plan_qty: 140
allocation_mismatch_rows: 0
```

其中 `AK1001` 推荐 140 件，分到 `AK1001-36` 87 件、`AK1001-38` 53 件；`AK1002` 推荐 0 件。

### 7.2 真实快照复跑

命令：

```powershell
python modules/profit_analysis/scripts/run_skc_profit_snapshot.py `
  --prediction-csv reports/profit_analysis_snapshot_real_smoke/inputs/prediction.csv `
  --inventory-csv reports/profit_analysis_snapshot_real_smoke/inputs/inventory.csv `
  --economics-csv reports/profit_analysis_snapshot_real_smoke/inputs/economics.csv `
  --policy balanced `
  --horizon-days 45 `
  --run-id rerun_20260616 `
  --output-dir reports/profit_analysis_snapshot_real_smoke/rerun_20260616
```

结果：

```text
quality gate: passed
prediction_rows: 3,950
skc_rows: 1,349
positive_plan_skc_rows: 2
total_recommended_plan_qty: 200
fallback_cost_rows: 1,227
fallback_cost_rate: 31.06%
allocation_mismatch_rows: 0
```

正推荐 SKC：

| snapshot_date | style_id | SKU数 | 当前库存 | 推荐生产量 | 期望利润 |
| --- | --- | ---: | ---: | ---: | ---: |
| 2026-02-15 | AK10601651 | 3 | 8 | 100 | 17,878.77 |
| 2026-02-15 | AK10801537 | 3 | 66 | 100 | 52,410.74 |

SKU 分配：

| style_id | sku_id | 分配量 |
| --- | --- | ---: |
| AK10601651 | AK1060165136560 | 32 |
| AK10601651 | AK1060165138560 | 43 |
| AK10601651 | AK1060165140560 | 25 |
| AK10801537 | AK1080153736043 | 13 |
| AK10801537 | AK1080153738043 | 24 |
| AK10801537 | AK1080153740043 | 63 |

## 8. 当前边界

当前模块仍有几个边界必须对外讲清楚：

- 实际成交价缺失，所以利润金额是同口径策略比较，不是审计财务利润；
- 生产周期目前主要用默认或配置值，尚未拿到每类商品真实生产周期；
- 在途和已确认生产订单还不完整；
- 成本表覆盖率不是 100%，未覆盖 SKC 使用 `吊牌价 / 7` 兜底；
- 当前更适合做“生产/补货建议过滤”，不适合直接自动下单。

## 9. 后续最值得做的工作

优先级建议：

1. 让甲方确认成交价或结算折扣、真实生产周期、已确认在途和已下生产单；
2. 把 Phase8 最终预测输出接入该模块，形成最新预测版本的盈亏结果；
3. 固定每周或每日的输入输出表结构，便于对账；
4. 保存人工修正记录，后续把“模型推荐、人工决策、真实结果”做闭环；
5. 如果生命周期表稳定，再区分短周期商品和长周期商品的决策方式。
