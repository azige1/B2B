# p521 Business Review

## Summary
- Experiment: `p521_lstm_l3_v3_filtered_s2026`
- Role: current strongest sequence reference observed in `phase5.2`
- Validation universe: `10,240` rows, `1,734` positive SKUs

## What Is Good
- Overall business balance is stable:
  - `global ratio = 1.0138`
  - `WMAPE = 1.3906`
- SKU-level calibration is materially better than the `phase5.1` V5-lite reference:
  - `SKU P50 = 0.8132` vs `0.6999` on `p512`
- Classification is also stronger:
  - `AUC = 0.8225`
  - `POS F1 = 0.5368`
- Cold-start is still bad, but less bad than the prior V5-lite reference:
  - `Ice P50 = 0.2802` vs `0.2055` on `p512`

## What Is Still Wrong
- The main business demand band is still under-predicted.
- Positive-SKU demand buckets:

| bucket | sku_cnt | sku_pct | true_share | ratio | wmape_like |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1-3` | 1205 | 69.49% | 17.71% | 1.1742 | 0.6485 |
| `4-10` | 284 | 16.38% | 16.14% | 0.5002 | 0.5857 |
| `11-25` | 160 | 9.23% | 24.83% | 0.2543 | 0.7748 |
| `>25` | 85 | 4.90% | 41.32% | 0.1045 | 0.8955 |

- Interpretation:
  - `1-3` is already over-predicted.
  - `4-10` is the first major under-predicted band.
  - `11-25` is severely under-predicted and carries meaningful business volume.
  - `>25` is extremely under-predicted, but should remain a monitored risk slice rather than the current primary optimization target.

## Cold-Start View
- Positive SKU temperature split:
  - `Hot`: `35`
  - `Warm`: `211`
  - `Cold`: `816`
  - `Ice`: `672`
- `Cold + Ice = 85.8%` of positive SKUs
- The main target band `4-25` is also weak inside cold-start:

| segment | sku_cnt | ratio | wmape_like | sku_p50 |
| --- | ---: | ---: | ---: | ---: |
| `4-25 overall` | 444 | 0.3512 | 0.7003 | 0.2899 |
| `Cold 4-25` | 150 | 0.4477 | 0.5958 | 0.4446 |
| `Ice 4-25` | 202 | 0.1466 | 0.8534 | 0.1556 |
| `Warm+Hot 4-25` | 92 | 0.7393 | 0.4526 | 0.7014 |

- Interpretation:
  - The main optimization problem is now clear: `4-25` demand inside `Cold/Ice`.
  - This is why pure sequence tuning is unlikely to be enough.

## Practical Takeaway
- `p521` should be treated as the provisional sequence reference unless a later `phase5.2` run clearly beats it.
- Next-stage evaluation should not be driven by total volume alone.
- The right priority order is:
  1. `SKU P50 / bucket ratio`
  2. `Cold/Ice`, `11-25`, `>25`
  3. `global ratio / WMAPE`
  4. `AUC / F1`
