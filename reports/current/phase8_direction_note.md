# Phase8 Direction Note

## Current Project Status

- Official stage remains `phase7 frozen / phase8 pending`.
- Official winner remains:
  - `tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`
- Official frozen metrics remain those in `reports/current/current_mainline.json`.

This note does **not** replace the official phase7 conclusion. It records the current direction decision for later phase8 work.

## Decision

The primary phase8 direction is now:

- `event + inventory`

This direction is supported by:

1. positive `2025-10/11/12` event-only shadow evidence
2. positive `2026-02-15 / 2026-02-24` event+inventory shadow evidence
3. consistent improvements in the exact areas where the official phase7 line is still weakest:
   - weak-signal blockbuster
   - `4_25 / Ice 4_25`
   - zero-true false positives
   - ranking / allocation

## Why Event+Inventory Is The Main Direction

### 1. Two-anchor 2026 shadow result is clearly positive

Source:
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_summary.md`

Two-anchor average (`2026-02-15`, `2026-02-24`):

| metric | base_tail_full | event_inventory_shadow | delta |
| --- | ---: | ---: | ---: |
| global_ratio | 1.0946 | 1.0476 | -0.0471 |
| global_wmape | 0.8614 | 0.6667 | -0.1947 |
| 4_25_under_wape | 0.4384 | 0.3452 | -0.0932 |
| 4_25_sku_p50 | 0.5966 | 0.6818 | +0.0852 |
| ice_4_25_sku_p50 | 0.5234 | 0.6212 | +0.0978 |
| blockbuster_under_wape | 0.4934 | 0.3773 | -0.1161 |
| blockbuster_sku_p50 | 0.4735 | 0.6326 | +0.1591 |
| top20_true_volume_capture | 0.5606 | 0.5766 | +0.0161 |
| rank_corr_positive_skus | 0.6918 | 0.7522 | +0.0604 |
| 1_3_ratio | 1.1421 | 1.0749 | -0.0671 |

This is strong enough to treat `event + inventory` as the current phase8 priority line.

### 2. Weak-signal blockbuster improves where phase7 is still weakest

Source:
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_detail_summary.md`

Key slice results:

- `2026-02-15 repl0_fut0`: `under_wape 0.6156 -> 0.3725`
- `2026-02-15 repl1_fut0`: `0.4678 -> 0.2817`
- `2026-02-24 repl1_fut0`: `0.5806 -> 0.3907`
- `2026-02-24 repl0_fut0`: `0.7688 -> 0.7334`

Interpretation:

- the signal is not uniformly strong across every slice
- but it is directionally correct on both anchors
- the gains are concentrated exactly where phase7 still needs help

### 3. Zero-true false positives are reduced

Source:
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_zero_true_table.csv`

Results:

- `2026-02-15`: false-positive rate `0.1773 -> 0.1438`
- `2026-02-24`: false-positive rate `0.2254 -> 0.1803`
- `pred>=5` zero-true rows:
  - `35 -> 14`
  - `28 -> 9`

Interpretation:

- event+inventory is not only helping recall
- it is also reducing low-quality positive allocations

### 4. Category improvements are broad enough to matter

Source:
- `reports/phase8_event_inventory_shadow_2026/phase8_event_inventory_shadow_category_delta_table.csv`

Consistent improvements appear in:

- `裤类`
- `外套`
- `裙子`
- `T恤`
- `毛衣`

Known regression already observed:

- `2026-02-24 毛衣开衫`

Interpretation:

- this is not a universal win across every category
- but the improvement is broad enough to justify continuing the line

## Why Preorder Is Not The Main Direction

Current preorder source:
- `data_warehouse/fact_orders/V_IRS_PREORDER.csv`

Current assessment:

- the table is sparse
- dates are not continuous
- `QTY_REM` is mostly zero
- it behaves more like a residual/remaining preorder state table than a complete preorder flow table

Decision:

- keep preorder as auxiliary analysis only
- do not treat preorder as a primary phase8 feature source right now

## What Still Blocks Formal Phase8

Two blockers remain:

1. `V_IRS_ORDERFTP` semantics
   - `TYPE` missing meaning
   - `QTY < 0` meaning
   - duplicate-row meaning
2. lifecycle table

Formal phase8 should not start before those two items are clarified.

## What Phase8 Should Mean Now

Before the blockers are cleared, phase8 should mean:

- exploratory shadow validation
- residual error analysis
- schema and feature-line preparation

It should **not** mean:

- official mainline replacement
- label-definition change
- open-ended hyperparameter search

## Next Actions

### Continue now

- keep using `event + inventory` as the working phase8 direction
- continue 2026 residual / case analysis when needed
- keep preorder as secondary evidence only

### Wait for client reply

- `V_IRS_ORDERFTP` semantics
- lifecycle table

### After client reply

- decide whether to revise label cleaning
- attach lifecycle features
- start formal phase8 targeted optimization on top of the current direction

## Bottom Line

The current direction decision is:

- **official winner stays phase7**
- **phase8 main direction becomes event+inventory**
- **preorder stays secondary**
- **formal phase8 still waits for order semantics and lifecycle**
