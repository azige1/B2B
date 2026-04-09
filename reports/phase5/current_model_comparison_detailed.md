# Current Model Comparison Detailed

## Scope and comparison rules

This note compares three models under two different confidence levels:

- `p527_lstm_l3_v5_lite_s2027`
  - role: current sequence baseline
  - source: `phase5.4` three-seed mean
- `p535_tree_hard_cov_activity`
  - role: current best confirmed tree mainline
  - source: `phase5.4` three-seed mean
- `p56_20251201_covact_lr005_l63_s2026_hard_g020`
  - role: current strongest tuned tree candidate
  - source: `phase5.6` single-anchor, single-seed sweep result

Important:

- `p535` vs `p527` is the fairest confirmed comparison, because both are averaged over `2026/2027/2028`.
- `p56` is the current best tuned configuration, but it is not yet multi-seed / multi-anchor confirmed.
- Business priority remains:
  1. `4-25`
  2. `Ice / Ice 4-25`
  3. `global_ratio / global_wmape`
  4. `AUC / F1`

## Model definitions

### Sequence baseline

- exp_id: `p527_lstm_l3_v5_lite_s2027`
- route: `v5_lite`
- model: `LSTM`
- status: baseline / backup / regression check line

### Confirmed tree mainline

- exp_id: `p535_tree_hard_cov_activity`
- route: `v6_event`
- feature_set: `cov_activity`
- backend: `LightGBM hurdle`
- gate: `hard`
- status: best confirmed mainline after multi-seed validation

### Strongest tuned tree candidate

- exp_id: `p56_20251201_covact_lr005_l63_s2026_hard_g020`
- route: `v6_event`
- feature_set: `cov_activity`
- backend: `LightGBM hurdle`
- gate: `hard`
- qty_gate: `0.20`
- learning_rate: `0.05`
- num_leaves: `63`
- status: best current tuned candidate from `phase5.6`

## Direct metrics

### Confirmed comparison: `p535` vs `p527`

| Metric | `p535` | `p527` | Interpretation |
| --- | ---: | ---: | --- |
| `AUC` | `0.9379` | `0.7664` | Tree ranking quality is much stronger |
| `F1` | `0.7054` | `0.5210` | Tree classification quality is materially better |
| `global_ratio` | `1.0645` | `1.0949` | Both are high, but tree is closer to target |
| `global_wmape` | `1.0469` | `1.5173` | Tree total error is much lower |
| `sku_p50` | `1.2531` | `0.7541` | Baseline underpredicts typical SKU; tree is in the healthy zone but slightly high |

### Current strongest tuned candidate: `p56` vs `p527`

| Metric | `p56` | `p527` | Interpretation |
| --- | ---: | ---: | --- |
| `AUC` | `0.9487` | `0.7664` | Large ranking gain |
| `F1` | `0.7261` | `0.5210` | Large classification gain |
| `global_ratio` | `1.0128` | `1.0949` | Tree is nearly neutral on total volume |
| `global_wmape` | `0.9666` | `1.5173` | Large total error reduction |
| `sku_p50` | `1.2089` | `0.7541` | Typical SKU allocation is much healthier |

### Tuned candidate vs confirmed tree: `p56` vs `p535`

| Metric | `p56` | `p535` | Interpretation |
| --- | ---: | ---: | --- |
| `AUC` | `0.9487` | `0.9379` | Further gain |
| `F1` | `0.7261` | `0.7054` | Further gain |
| `global_ratio` | `1.0128` | `1.0645` | Better total calibration |
| `global_wmape` | `0.9666` | `1.0469` | Better total accuracy |
| `4_25_sku_p50` | `0.5898` | `0.5643` | Better in main business slice |
| `ice_4_25_sku_p50` | `0.4586` | `0.4468` | Better in hardest slice |
| `blockbuster_sku_p50` | `0.3073` | `0.2665` | Better on large-demand SKUs |

## Slice-by-slice comparison

### `Ice` and `Ice 4-25`

These are the hardest slices and the most important proof that tree is not just improving totals.

| Metric | `p535` | `p527` | Gain |
| --- | ---: | ---: | ---: |
| `ice_ratio` | `0.4346` | `0.1400` | `+210.4%` |
| `ice_wmape_like` | `0.6973` | `0.8919` | `-21.8%` |
| `ice_sku_p50` | `0.7995` | `0.2136` | `+274.3%` |
| `ice_4_25_ratio` | `0.4559` | `0.1265` | `+260.4%` |
| `ice_4_25_wmape_like` | `0.5673` | `0.8738` | `-35.1%` |
| `ice_4_25_sku_p50` | `0.4468` | `0.1244` | `+259.2%` |

Interpretation:

- The sequence baseline is structurally weak on cold-start.
- Tree does not fully solve cold-start, but it moves it from near-collapse to a usable range.
- `ice_4_25_sku_p50` is still below the desired zone, so cold-start remains a priority.

### Main business slice: `4-25`

| Metric | `p535` | `p527` | Gain |
| --- | ---: | ---: | ---: |
| `4_25_ratio` | `0.5955` | `0.3045` | `+95.6%` |
| `4_25_wmape_like` | `0.5211` | `0.7300` | `-28.6%` |
| `4_25_sku_p50` | `0.5643` | `0.2707` | `+108.5%` |

Interpretation:

- This is the clearest reason to move the mainline to tree.
- The baseline is systematically underpredicting the main demand zone.
- Tree nearly doubles effective allocation quality in this slice.
- The result is still not ideal, but it is materially closer to a usable business state.

### Small-demand slice: `1-3`

| Metric | `p535` | `p527` | Interpretation |
| --- | ---: | ---: | --- |
| `1_3_ratio` | `1.5709` | `1.0600` | Tree overpredicts this slice |
| `1_3_wmape_like` | `0.7805` | `0.5890` | Baseline is better here |
| `1_3_sku_p50` | `1.5383` | `1.0357` | Tree has a new small-qty bias |

Interpretation:

- Tree is not universally better.
- The current tree route introduces a small-demand overprediction problem.
- This is the most obvious next calibration task if tree becomes the mainline.

### `4-10`

| Metric | `p535` | `p527` | Gain |
| --- | ---: | ---: | ---: |
| `4_10_ratio` | `0.7394` | `0.4231` | `+74.8%` |
| `4_10_wmape_like` | `0.4177` | `0.6222` | `-32.9%` |
| `4_10_sku_p50` | `0.6696` | `0.3774` | `+77.4%` |

Interpretation:

- Tree is materially better in this medium-demand segment.
- This is one of the strongest practical improvements because the segment has both business value and sufficient support.

### `11-25`

| Metric | `p535` | `p527` | Gain |
| --- | ---: | ---: | ---: |
| `11_25_ratio` | `0.4998` | `0.2260` | `+121.2%` |
| `11_25_wmape_like` | `0.5909` | `0.8049` | `-26.6%` |
| `11_25_sku_p50` | `0.3856` | `0.1341` | `+187.5%` |

Interpretation:

- The baseline is extremely weak in this segment.
- Tree is still conservative here, but the improvement is large enough to justify the mainline shift.

### `>25` / blockbuster

| Metric | `p535` | `p527` | Gain |
| --- | ---: | ---: | ---: |
| `blockbuster_ratio` | `0.3257` | `0.0989` | `+229.3%` |
| `blockbuster_wmape_like` | `0.6959` | `0.9011` | `-22.8%` |
| `blockbuster_sku_p50` | `0.2665` | `0.0670` | `+297.8%` |

Interpretation:

- Tree is much better than the baseline, but this slice is still not solved.
- This remains a risk-monitoring slice rather than the current primary optimization target.

## Why the tree route is better

The current problem is not a dense-sequence problem. It is a sparse event and allocation problem.

Current data shape:

- `future` is sparse
- `replenish` is also sparse
- `Ice / Cold` share is high
- target is `future 30d total qty`, not next-step sequence prediction

The tree route fits this shape better because:

- it converts sparse daily signals into dense event statistics
- it uses explicit buyer coverage and activity signals
- it separates occurrence and quantity via hurdle modeling
- it does not depend on sequence compression through a final hidden state

## What is still not solved

### Low-demand calibration

Tree is currently too aggressive on `1-3`.

This is not a reason to go back to LSTM.
It is a reason to add:

- low-demand calibration
- segment-aware post-processing
- possibly gate adjustment or bucket-specific scaling

### Cold-start is improved, not solved

`Ice / Ice 4-25` improved a lot, but these slices are still below healthy levels.

The next optimization step after multi-anchor confirmation should be:

- cold-start fallback
- segment-aware calibration
- not a new large model family

### `>25` remains underpredicted

Tree is much better than sequence, but the highest-demand slice still needs monitoring.

This supports the current business stance:

- keep `>25` as a risk slice
- do not let it dominate the whole model selection process

## Current recommendation

### Best confirmed model

- `p535_tree_hard_cov_activity`

Reason:

- already validated as better than sequence
- already validated across multiple seeds
- good enough to be called the current confirmed mainline

### Best current tuned candidate

- `p56_20251201_covact_lr005_l63_s2026_hard_g020`

Reason:

- better than `p535` on nearly every key business metric
- better total calibration
- better `4-25`
- better `Ice 4-25`
- better blockbuster slice

### Best conservative tuned variant

- `p56_20251201_covact_lr005_l63_s2026_hard_g025`

Reason:

- almost identical slice performance to `g020`
- lower `global_wmape`
- more conservative total output

## Final judgement at this stage

- `event/tree` is no longer a speculative challenger.
- It is the strongest direction in both confirmed and tuned results.
- `p527` should remain the sequence baseline and backup line.
- The next gating experiment is still `phase5.5` multi-anchor validation.
- If `phase5.5` confirms the same pattern on at least `3/4` anchors, the project mainline should formally move to tree.
