# Server migration checkpoint (2026-06-15)

## Outcome

The client database was still reachable only from the retiring server. A fresh
Oracle snapshot was exported there and migrated to the replacement server.
Archive-level SHA-256 checks and per-table CSV row/hash checks all passed.

## Fresh Oracle snapshot

Replacement-server path:

`/root/client_data_snapshots/client_snapshot_20260615`

| Table | Rows | Role |
|---|---:|---|
| `V_IRS_ORDERFTP` | 337,480 | Canonical order flow and label source |
| `V_IRS_PRODUCT` | 28,853 | Product dimension and `LISTING_DATE` |
| `V_IRS_STORAGE` | 6,518 | Current available-stock snapshot |
| `V_IRS_B2BSTORAGE` | 3,444 | Current B2B-stock snapshot |
| `V_IRS_EVENT` | 566,348 | User-event history |
| `V_IRS_STORE` | 1,591 | Store mapping and audit |
| `V_IRS_CUS_PROFILE` | 1,561 | Shadow-only customer profile |
| `V_IRS_PREORDER` | 282 | Diagnostic-only preorder state |
| `V_IRS_ORDER` | 461,697 | Backup only; not a training-label source |

Freshness:

- `V_IRS_ORDERFTP.BILLDATE`: 2025-01-01 through 2026-06-15
- `V_IRS_EVENT.CREATIONDATE`: 2025-09-18 through 2026-06-15
- `V_IRS_PRODUCT.LISTING_DATE`: present for all 28,853 rows

`V_IRS_PRO_DATA` was not requested because the view had been renamed/broken and
was explicitly excluded from the current Phase8 work.

## Historical assets

Replacement-server path:

`/root/migration_backup_20260615`

Migrated assets include:

- 1,045 files from the old `B2B` and `B2B_Replenishment` trees
- Daily snapshots through 2026-06-15
- Oracle Instant Client 19.8 and extraction scripts
- 15 legacy root-level CSV files
- Old root and system crontab definitions for reference

The old cron jobs were not enabled on the replacement server.

## Runtime state

- Dedicated Python environment: `/root/oracle-env`
- Oracle client path: `/root/instantclient_19_8`
- Oracle client load test: passed
- Current database connection from the replacement server: blocked with
  `ORA-12170`, consistent with the client IP whitelist

After the client whitelists the replacement server IP, perform a one-table
connection test before restoring any scheduled extraction jobs.
