# Server Whitelist Refresh Audit (2026-06-16)

## 结论

新服务器白名单已经生效。已从甲方 Oracle 源库重新导出 `2026-06-16` 快照，并下载到本地；本地 CSV 的行数、文件大小、SHA-256 均与导出 manifest 一致。

本次只注册新原始数据快照，不重建 silver/gold，不重训模型。

## 连接状态

- 新服务器：`121.40.254.36`
- 可用登录：`root@121.40.254.36`
- `wzy@121.40.254.36` 当前未授权公钥登录。
- Oracle 连通性：通过。
- `V_IRS_ORDERFTP` 实时库测试结果：`337,837` 行，`BILLDATE=20250101` 到 `20260616`。

## 导出快照

服务器路径：

`/root/client_data_snapshots/client_snapshot_20260616`

本地路径：

`data/incoming/server_20260616/client_snapshot_20260616`

| 表 | 行数 | 当前口径 |
| --- | ---: | --- |
| `V_IRS_ORDERFTP` | 337,837 | 正式订单流水和训练标签源 |
| `V_IRS_PRODUCT` | 28,853 | 商品维表，包含 `LISTING_DATE` |
| `V_IRS_STORAGE` | 6,664 | 当前可用库存快照 |
| `V_IRS_B2BSTORAGE` | 3,455 | B2B 总部库存快照 |
| `V_IRS_EVENT` | 567,825 | 用户行为事件流水 |
| `V_IRS_STORE` | 1,591 | 门店/客户映射，仅审计和映射 |
| `V_IRS_CUS_PROFILE` | 1,559 | 客户画像，shadow-only |
| `V_IRS_PREORDER` | 314 | 预售状态，仅诊断 |
| `V_IRS_ORDER` | 461,920 | 备份，不作为训练标签源 |

`V_IRS_PRO_DATA` 仍然不可用，和此前记录一致，暂不纳入 Phase8 主线。

## 关键字段范围

| 文件 | 字段 | 最小值 | 最大值 | 覆盖 |
| --- | --- | --- | --- | --- |
| `V_IRS_ORDERFTP.csv` | `BILLDATE` | `20250101` | `20260616` | 337,837 / 337,837 |
| `V_IRS_EVENT.csv` | `CREATIONDATE` | `2025-09-18 13:18:06` | `2026-06-16 16:28:09` | 567,825 / 567,825 |
| `V_IRS_PRODUCT.csv` | `LISTING_DATE` | `20241212` | `20260605` | 28,853 / 28,853 |

## 相比 2026-06-14 本地快照

| 数据 | 2026-06-14 | 2026-06-16 | 增量 |
| --- | ---: | ---: | ---: |
| `V_IRS_ORDERFTP` | 337,215 | 337,837 | +622 |
| `V_IRS_EVENT` | 562,822 | 567,825 | +5,003 |

## 本地注册位置

已同步到本地仓库但没有覆盖旧文件：

| 角色 | 本地文件 |
| --- | --- |
| 订单流水 | `data_warehouse/fact_orders/V_IRS_ORDERFTP_6_16.csv` |
| 用户事件 | `data_warehouse/fact_events/V_IRS_EVENT_20260616.csv` |
| 商品维表 | `data_warehouse/dim_product/product_info_20260616.csv` |
| 可用库存 | `data_warehouse/snapshot_inventory/20260616_storage_stock.csv` |
| B2B 库存 | `data_warehouse/snapshot_inventory/20260616_b2b_stock.csv` |
| 门店映射 | `data_warehouse/dim_store/store_info_20260616.csv` |
| 客户画像 | `data_warehouse/snapshot_metrics/20260616_customer_profile.csv` |
| 预售状态 | `data_warehouse/fact_orders/20260616_b2b_preorder.csv` |

Manifest：

`data/manifests/phase8_data_snapshot_20260616.json`

资产登记已更新：

`data/current_assets.json`

当前 `raw_order_baseline` 指向：

`data_warehouse/fact_orders/V_IRS_ORDERFTP_6_16.csv`

## 审计说明

这次 Oracle 导出是实时逐表导出，不是数据库级一致性快照。导出期间库仍在写入，因此导出前后直接查库计数会有轻微变化。例如导出完成后复查，实时库中：

- `V_IRS_ORDERFTP` 已变为 `337,838` 行。
- `V_IRS_EVENT` 已变为 `567,883` 行。

这不影响本地快照的文件级可复现性；本地快照以 manifest 中记录的行数和 SHA 为准。

## 后续建议

- 不建议因为这次只多两天数据就重新定义 Phase8 结论；它更适合做数据基线更新和后续 2026 验证准备。
- 如果要正式使用 `6_16` 训练，需要先重建 silver/gold，再重跑 Phase7 旧口径和 Phase8 候选，避免混用旧特征与新原始表。
- `V_IRS_ORDER` 继续只保留备份，不得替代 `V_IRS_ORDERFTP` 做标签。
