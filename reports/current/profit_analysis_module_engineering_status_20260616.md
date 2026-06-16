# 盈亏分析模块工程整理记录

日期：2026-06-16

## 当前结论

盈亏分析模块已经整理为独立 V1 决策模块，主目录为：

`modules/profit_analysis/`

模块当前可运行、可测试、可演示。它的定位不是预测未来补货量，而是接在预测模型后面，把预测需求转换为 SKC 级生产/补货建议，并把 SKC 总量分配回 SKU。

## 本次整理内容

本次工程整理完成：

- 新增当前状态与使用说明：`modules/profit_analysis/docs/profit_analysis_current_status_20260616.md`
- 新增最小可运行示例：`modules/profit_analysis/examples/minimal_snapshot/`
- 更新模块 README，加入当前入口、真实回测证据、最小示例命令和生命周期说明；
- 生产式入口 `run_skc_profit_snapshot.py` 增加 `--run-id`，便于演示和复现固定输出文件名；
- 确认旧的盈亏分析文档应迁移到模块目录，不再散落在项目根目录或 `reports/current/`。

## 最小示例实跑

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

解释：

- `AK1001` 两个 SKU，预测概率高、库存为 0、售价相对成本高，推荐生产 140 件；
- `AK1001-36` 分配 87 件，`AK1001-38` 分配 53 件；
- `AK1002` 虽然有预测需求，但概率较低且已有 30 件库存，生产 100 件起步不划算，推荐 0。

## 真实快照复跑

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
fallback_cost_rate: 31.06%
allocation_mismatch_rows: 0
```

正推荐 SKC：

| snapshot_date | style_id | 推荐生产量 | 期望利润 |
| --- | --- | ---: | ---: |
| 2026-02-15 | AK10601651 | 100 | 17,878.77 |
| 2026-02-15 | AK10801537 | 100 | 52,410.74 |

## 测试

```text
python -m pytest modules/profit_analysis/tests -q
31 passed in 2.48s
```

## 对外解释口径

预测模块回答“未来有多少需求”，盈亏模块回答“这个需求是否值得转成生产计划”。当前回测显示，直接按预测缺口生产会产生大量有害计划；盈亏模块的主要价值是用成本、售价、库存和批量约束过滤风险，只保留少量高确定性的生产建议。

利润金额当前是同口径策略比较，不是审计财务利润。正式上线前仍需甲方补充实际成交价、真实生产周期、已确认在途/生产订单和人工决策日志。
