# Data Index

## Source Of Truth

Official order flow and label source:

```text
V_IRS_ORDERFTP
```

Do not use `V_IRS_ORDER` or server daily `order_history` files as training labels.

## Current Raw Snapshots

Current raw-order checkpoint:

```text
../data_warehouse/fact_orders/V_IRS_ORDERFTP_6_16.csv
```

Current event checkpoint:

```text
../data_warehouse/fact_events/V_IRS_EVENT_20260616.csv
```

Current product checkpoint:

```text
../data_warehouse/dim_product/product_info_20260616.csv
```

Snapshot manifest:

```text
manifests/phase8_data_snapshot_20260616.json
```

Asset registry:

```text
current_assets.json
```

Important limitation:

- The `6_16` raw snapshot is synced and audited.
- `silver/` and `gold/` have not yet been rebuilt on top of `6_16`.
- Existing Phase8 metrics should not be described as retrained on `6_16`.

## Layer Map

- `incoming/`
  - Downloaded server snapshots and raw handoff files.
  - Keep as immutable intake evidence when possible.
- `manifests/`
  - Auditable snapshot manifests, source-to-destination mappings, row counts, hashes.
- `silver/`
  - Cleaned operational tables.
  - Current historical silver may still reflect older raw snapshots.
- `gold/`
  - Modeling-facing merged tables.
  - Current official gold table remains `gold/wide_table_sku.csv`.
- `processed_*` and `artifacts_*`
  - Feature assets for specific phase/anchor experiments.
  - Keep in place for compatibility with reports and runners.

## Current Data Reports

- `reports/current/server_whitelist_refresh_20260616.md`
- `reports/current/server_daily_oracle_snapshot_automation_20260616.md`
- `reports/current/phase8_data_semantics_20260614.md`
- `reports/current/client_source_table_registry.md`
- `reports/current/v_irs_orderftp_refresh_audit_20260614.md`

## Server Automation

The replacement server exports a daily Oracle snapshot at Shanghai time `03:30`.

Server output pattern:

```text
/root/client_data_snapshots/client_snapshot_YYYYMMDD/
/root/client_data_snapshots/client_snapshot_YYYYMMDD.tar.gz
/root/client_data_snapshots/client_snapshot_YYYYMMDD.tar.gz.sha256
```

Local backup scripts:

- `scripts/data/server_run_daily_oracle_snapshot.sh`
- `scripts/data/server_daily_oracle_snapshot.cron`

## Rebuild Rule

When switching the active training baseline to a new raw snapshot:

1. Register the raw snapshot in `data/manifests/`.
2. Update `data/current_assets.json`.
3. Rebuild `silver/`.
4. Rebuild `gold/`.
5. Rebuild feature assets for the target anchors.
6. Re-run Phase7 baseline and Phase8 candidate with the same evaluation口径.
7. Write a report under `reports/current/`.

Do not mix new raw order files with old feature assets when claiming model metrics.
