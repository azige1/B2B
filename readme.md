# B2B Replenishment System

服装 B2B 补货预测与盈亏分析项目工作区。

## Current Status

当前项目已经从早期 LSTM 探索，演进到以树模型为主的补货预测系统，并拆出了独立的盈亏分析模块。

预测主线当前阶段性结论：

- Phase7 是历史冻结基线，旧 `0.6` 评估口径下 `global_wmape=0.6863`。
- Phase8 推荐候选为 `coverage_router + conservative_blockbuster_uplift`。
- Phase8 推荐候选在旧 `0.6` 四锚点口径下 `global_wmape=0.6482`，相对 Phase7 改善约 `5.56%`。
- 爆款低估指标 `blockbuster_under_wape=0.3101`，相对 Phase7 的 `0.4165` 改善约 `25.55%`。

数据工程当前状态：

- 正式订单流水和训练标签源只使用 `V_IRS_ORDERFTP`。
- 新服务器 `121.40.254.36` 已打通甲方 Oracle 白名单。
- 服务器已配置每日自动取数任务。
- 本地最新原始订单快照为 `data_warehouse/fact_orders/V_IRS_ORDERFTP_6_16.csv`。
- 注意：`6_16` 原始数据已同步，但 silver/gold 和模型还未基于该快照重建。

## Start Here

只想快速了解当前状态时，按这个顺序读：

1. `PROJECT_INDEX.md`
2. `DOCS_INDEX.md`
3. `reports/current/phase8_blockbuster_uplift_0p6_20260616.md`
4. `reports/current/server_whitelist_refresh_20260616.md`
5. `reports/current/server_daily_oracle_snapshot_automation_20260616.md`
6. `modules/profit_analysis/README.md`
7. `RUNNERS_INDEX.md`

## Repository Layout

```text
B2B_Replenishment_System/
|-- src/                    Core ETL, feature, training, analysis, inference code
|-- scripts/                Reproducible runners, data scripts, demos
|-- modules/
|   `-- profit_analysis/    Independent profit-analysis module
|-- data/                   Current asset registry, manifests, processed assets
|-- data_warehouse/         Raw and snapshot source tables
|-- reports/
|   |-- current/            Current conclusions and handoff documents
|   `-- phase*/             Historical or experiment-specific outputs
|-- models/                 Historical frozen official model assets
|-- models_phase8_*/        Phase8 experiment model artifacts
|-- config/                 Configuration files
|-- docs/                   Additional documentation
`-- tests/                  Tests and smoke checks
```

## Current Main Artifacts

Prediction:

- `reports/current/phase8_blockbuster_uplift_0p6_20260616.md`
- `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_candidate_summary.csv`
- `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_recommended_context.csv`
- `src/analysis/analyze_phase8u_blockbuster_uplift.py`

Data:

- `data/current_assets.json`
- `data/manifests/phase8_data_snapshot_20260616.json`
- `reports/current/server_whitelist_refresh_20260616.md`
- `reports/current/server_daily_oracle_snapshot_automation_20260616.md`

Profit analysis:

- `modules/profit_analysis/README.md`
- `modules/profit_analysis/docs/profit_analysis_module_v1_detailed_design_20260522.md`
- `modules/profit_analysis/docs/profit_analysis_module_delivery_20260612.md`
- `modules/profit_analysis/scripts/run_skc_profit_snapshot.py`

## Common Commands

Run Phase8 blockbuster uplift analysis from existing prediction contexts:

```bash
python src/analysis/analyze_phase8u_blockbuster_uplift.py
```

Inspect current data assets:

```bash
python -m json.tool data/current_assets.json
python -m json.tool data/manifests/phase8_data_snapshot_20260616.json
```

Run profit-analysis production-style snapshot:

```bash
python modules/profit_analysis/scripts/run_skc_profit_snapshot.py --prediction-csv <prediction.csv> --inventory-csv <inventory.csv> --economics-csv <economics.csv>
```

## Working Rules

- Use `reports/current/` and `PROJECT_INDEX.md` for current conclusions.
- Use `RUNNERS_INDEX.md` for executable entry points.
- Treat `reports/phase*/` and `models_phase8_*/` as historical or experiment-specific unless explicitly referenced by current docs.
- Do not use `V_IRS_ORDER` or server daily `order_history` files as training labels.
- Do not assume the latest raw data has already rebuilt downstream silver/gold assets.
- Do not delete or move historical experiment artifacts without writing a manifest first.
