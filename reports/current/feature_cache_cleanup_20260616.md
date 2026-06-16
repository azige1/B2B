# Feature Cache Cleanup Report (2026-06-16)

## Summary

Deleted old generated feature-cache directories under `data/processed_*` and `data/artifacts_*`.

This cleanup did not delete raw data, warehouse snapshots, `silver/`, `gold/`, manifests, Phase7 current `p7b` assets, or current Phase8 `p8eventreqshadow` assets.

## Deleted Scope

Only directories matching these old experiment patterns were deleted:

- `*p8formal0614*`
- `*p8robustpurge*`
- `*p8oot2026*`
- `*p8einvshadow*`
- `*p8ei_*`
- `*p7refresh*`

All deletion targets were constrained to:

- `data/processed_*`
- `data/artifacts_*`

## Deleted Size

| category | directories | size_gb |
| --- | ---: | ---: |
| `p8formal0614` | 56 | 71.62 |
| `p8robustpurge` | 42 | 45.13 |
| `p8oot2026` | 12 | 23.71 |
| `old_event_inventory` | 16 | 23.61 |
| `p7refresh_validation` | 16 | 20.65 |
| **Total** | **142** | **184.72** |

Disk free space on drive `E:` increased from about `269.91GB` to `454.64GB`.

## Remaining Data Size

After cleanup:

- `data/`: `63.13GB`
- Files under `data/`: `1,152`

Largest remaining feature-cache groups are now:

- Phase7 `p7b` baseline assets.
- Current Phase8 `p8eventreqshadow` assets.
- Listing/lifecycle/peer/request-only experiments that were intentionally left for a later, more selective cleanup.

## Retained Assets Verified

These were verified to still exist after cleanup:

- `data/processed_v5_lite_p7b_20250901_v5_lite`
- `data/processed_v5_lite_p7b_20251001_v5_lite`
- `data/processed_v5_lite_p7b_20251101_v5_lite`
- `data/processed_v5_lite_p7b_20251201_v5_lite`
- `data/processed_v6_event_p8eventreqshadow_20250901_v6_event`
- `data/processed_v6_event_p8eventreqshadow_20251001_v6_event`
- `data/processed_v6_event_p8eventreqshadow_20251101_v6_event`
- `data/processed_v6_event_p8eventreqshadow_20251201_v6_event`

## Manifest

Deletion manifest:

`data/manifests/feature_cache_cleanup_deleted_20260616.csv`

The manifest records deleted path, category, file count, size, action, and reason.

## Notes

- Deleted directories are generated caches and can be rebuilt by their corresponding old runners if needed.
- This cleanup reduces local disk pressure but does not change model metrics or current reports.
- Do not claim any model retraining from this cleanup; no training or evaluation was run.
