# 甲方数据源表名登记

更新时间：2026-06-14

## 核心口径

正式订单流水和模型标签只使用：

```text
V_IRS_ORDERFTP
```

`V_IRS_ORDERFTP` 是文档清单之外额外提供的原始订单流水表，也是当前项目最重要的数据源。

甲方原文档中的：

```text
V_IRS_ORDER
```

是聚合后的订单状态表，不适合作为当前未来30天补货预测的原始流水或训练标签。
后续不得用 `V_IRS_ORDER`、服务器每日 `order_history` 文件替代或拼接
`V_IRS_ORDERFTP`。

## 表名清单

| 序号 | 表名 | 主要内容 | 当前口径 |
| --- | --- | --- | --- |
| 1 | `V_IRS_PRODUCT` | 商品基础信息、`LISTING_DATE` | 使用；`PL_CYCLE`无区分度，不使用 |
| 2 | `V_IRS_STORE` | 客户/门店名称映射 | 仅映射和审计，不作为模型特征 |
| 3 | `V_IRS_STORAGE` | 现货可用库存每日快照 | Phase8主线特征 |
| 4 | `V_IRS_ORDER` | 聚合订单状态 | 不作为订单流水和训练标签 |
| 5 | `V_IRS_B2BSTORAGE` | B2B总部发布库存每日快照 | Phase8主线特征 |
| 6 | `V_IRS_PREORDER` | 当前预售余量状态 | 仅诊断；记录稀疏且不是完整流水 |
| 7 | `V_IRS_PRO_DATA` | 原商品计算视图 | 表已改名，当前暂不处理，不影响Phase8主线 |
| 8 | `V_IRS_EVENT` | B2B用户行为全量流水 | Phase8主线特征 |
| 9 | `wk_crm_customer` / `wk_crm_customer_data` | CRM客户字段 | 未纳入当前白名单 |
| 10 | `V_IRS_CUS_PROFILE` | 商户衍生画像和经营健康度 | 仅shadow；计算窗口和别名规则未确认 |
| 额外 | `V_IRS_ORDERFTP` | 原始订单流水 | 正式订单源，最高优先级 |
| 外部文件 | 甲方求购明细 | SKU/款级求购意向 | Phase8主线候选；截至2026-04-21 |

## 获取方式

服务器手工全量取表入口：

```text
/root/get_store.py
```

使用时修改脚本中的 Oracle 表名和输出文件名。订单刷新必须查询：

```sql
SELECT * FROM BOSNDS3.V_IRS_ORDERFTP
```

当前正式白名单和用途以以下文档为准：

- `reports/current/phase8_data_semantics_20260614.md`
- `reports/phase8_data_audit/phase8_data_audit_20260614.md`
- `data/manifests/phase8_data_snapshot_20260614.json`
