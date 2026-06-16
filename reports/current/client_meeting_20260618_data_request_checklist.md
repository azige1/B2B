# 2026-06-18 甲方数据需求清单

日期：2026-06-16  
用途：6.18 会议后向甲方确认/索取字段

## 1. 预测模块优先字段

### P0：B2B 渠道真实开售/开订

| 字段 | 说明 | 用途 |
| --- | --- | --- |
| `style_id` / `sku_id` | 商品主键 | 关联商品和预测样本 |
| `b2b_launch_date` | B2B 渠道实际开售日期 | 冷启动和起量判断 |
| `order_open_date` | 买手开放订货日期 | 判断需求释放时点 |
| `planned_first_order_qty` | 首批计划投放量 | 新品初始需求先验 |
| `planned_allocation_qty` | 计划分配量 | 区分计划铺货和自然需求 |
| `launch_batch_id` | 上新批次 | 波段/批次特征 |
| `launch_channel` | 上新渠道 | 渠道差异 |

### P1：生命周期终点

| 字段 | 说明 | 用途 |
| --- | --- | --- |
| `planned_off_shelf_date` | 计划下架日期 | 判断剩余销售窗口 |
| `actual_off_shelf_date` | 实际下架日期 | 回测和复盘 |
| `sale_status` | 未开售/在售/清仓/停售 | 区分不同零需求原因 |
| `stop_replenishment_date` | 停止补货日期 | 避免尾期误补 |

### P2：库存和求购持续增量

| 字段 | 说明 | 用途 |
| --- | --- | --- |
| `snapshot_date` | 库存日期 | 时间对齐 |
| `sku_id` | SKU | 库存关联 |
| `available_qty` | 可用库存 | 判断缺货/过量库存 |
| `reserved_qty` | 占用库存 | 净库存计算 |
| `in_transit_qty` | 在途库存 | 净需求计算 |
| `request_time` | 求购时间 | 需求前置信号 |
| `request_qty` | 求购数量 | 意向强度 |
| `request_reason` | 求购原因 | 解释字段 |

## 2. 盈亏分析模块优先字段

### P0：生产约束

| 字段 | 说明 | 用途 |
| --- | --- | --- |
| `style_id` / `sku_id` | 商品主键 | SKC/SKU 决策 |
| `lead_time_days` | 生产/补货周期 | 判断能否赶上生命周期 |
| `minimum_order_qty` | 最小生产批量 | 候选生产量约束 |
| `order_multiple` | 递增粒度 | 100、110、120 等候选量 |
| `production_capacity` | 产能上限 | 可执行约束 |
| `confirmed_purchase_qty` | 已确认生产/采购数量 | 避免重复补 |
| `in_transit_qty` | 在途量 | 净需求计算 |
| `expected_arrival_date` | 预计到货日期 | 盈亏窗口模拟 |

### P1：真实成本和价格

| 字段 | 说明 | 用途 |
| --- | --- | --- |
| `unit_cost` | 单件成本 | 生产成本 |
| `material_cost` | 材料成本 | 成本拆解 |
| `manufacturing_cost` | 加工成本 | 成本拆解 |
| `logistics_cost` | 物流成本 | 成本拆解 |
| `cost_effective_date` | 成本生效日期 | 防止成本错配 |
| `transaction_price` | 实际成交价 | 真实收入 |
| `settlement_price` | 结算价 | 真实收入 |
| `discount_rate` | 折扣率 | 吊牌价转收入 |
| `net_sales_amount` | 实收金额 | 财务口径 |

### P2：人工决策反馈

| 字段 | 说明 | 用途 |
| --- | --- | --- |
| `decision_date` | 决策日期 | 反馈闭环 |
| `model_recommend_qty` | 模型建议量 | 对账 |
| `human_final_qty` | 人工最终量 | 学习业务修正 |
| `adjust_reason` | 修正原因 | 解释和重训练 |
| `approved_flag` | 是否采纳 | 评估模块价值 |
| `actual_arrival_date` | 实际到货日期 | 复盘 |
| `actual_sales_after_decision` | 决策后真实销售/补货 | 策略收益评估 |

## 3. 商品识别量化字段

### P0：商品基因表

| 字段 | 说明 |
| --- | --- |
| `sku_id` / `style_id` | 商品主键 |
| `image_id` / `image_url` | 图片或图片引用 |
| `category` | 品类 |
| `season` | 季节 |
| `color_family` | 色系 |
| `saturation_level` | 饱和度 |
| `silhouette` | 廓形 |
| `fit_level` | 合体/宽松程度 |
| `fabric_handfeel` | 面料手感 |
| `fabric_weight` | 克重/厚薄 |
| `structure_type` | 结构类型 |
| `design_complexity` | 设计复杂度 |
| `style_keywords` | 风格关键词 |
| `occasion` | 使用场景 |
| `tag_source` | 人工/模型/人工复核 |
| `tag_confidence` | 标签可信度 |

### P1：用于评估标签价值的业务结果

| 字段 | 说明 |
| --- | --- |
| `actual_order_qty` | 实际订货量 |
| `replenishment_qty` | 实际补货量 |
| `sell_through_rate` | 售罄率 |
| `gross_margin` | 毛利 |
| `inventory_leftover_qty` | 剩余库存 |
| `return_qty` | 退货量，可选 |
| `return_reason` | 退货原因，可选 |

## 4. 特殊节点字段

### P0：特殊节点日历

| 字段 | 说明 | 是否可用于预测 |
| --- | --- | --- |
| `event_id` | 活动 ID | 是 |
| `event_name` | 活动名称 | 是 |
| `event_type` | 直播/节日/订货会/主推/折扣/上新波段 | 是 |
| `event_start_date` | 开始日期 | 是 |
| `event_end_date` | 结束日期 | 是 |
| `scope_style_id` | 影响款号 | 是 |
| `scope_sku_id` | 影响 SKU，可选 | 是 |
| `scope_category` | 影响类目 | 是 |
| `scope_channel` | 影响渠道 | 是 |
| `event_strength` | 活动强度，人工 1/2/3 也可以 | 是 |
| `planned_exposure` | 计划曝光 | 是，如果预测日前已知 |
| `planned_discount` | 计划折扣 | 是，如果预测日前已知 |
| `host_or_live_room` | 主播/直播间 | 是，如果预测日前已知 |
| `actual_exposure` | 实际曝光 | 否，只能复盘 |
| `actual_order_qty` | 活动带动订单 | 否，只能复盘 |

### P1：特殊节点建模特征

这些字段由模型侧加工：

```text
days_to_event
days_since_event
is_in_event_window
is_pre_event_window_7d
is_post_event_window_7d
event_type_live_flag
event_type_holiday_flag
event_type_key_style_flag
event_strength
event_scope_match
event_count_next_30d
event_count_past_30d
```

## 5. 优先级总结

最建议会议上推动的顺序：

1. `b2b_launch_date / order_open_date`：直接提升预测冷启动。
2. `planned_off_shelf_date / stop_replenishment_date`：解决生命周期末期误补。
3. `lead_time_days / MOQ / order_multiple / confirmed_purchase_qty`：支撑盈亏模块变成可执行建议。
4. `transaction_price / settlement_price / unit_cost`：支撑盈亏模块从代理利润走向真实利润。
5. `event calendar`：解决直播爆款、节日、订货会导致的大误差。
6. `product gene table`：支撑商品识别量化、相似款生命周期和企划复盘。

