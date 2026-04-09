# Current Model Recommendation

## Current judgement

- Current best **tuned tree candidate**: `p56_20251201_covact_lr005_l63_s2026_hard_g020`
- Conservative tree backup: `p56_20251201_covact_lr005_l63_s2026_hard_g025`
- Current sequence baseline: `p527_lstm_l3_v5_lite_s2027`

This recommendation is still **pre-delivery**.

- `phase5.3A` proved `event/tree` beats sequence on the single-anchor benchmark.
- `phase5.4` proved the tree advantage is stable across seeds.
- `phase5.6` improved the tree line further and gave a clear parameter direction.
- `phase5.5` is still the remaining gate for multi-anchor stability.

## Recommended mainline

Use this as the current tree mainline candidate:

- feature route: `v6_event`
- feature set: `cov_activity`
- backend: `LightGBM hurdle`
- learning rate: `0.05`
- num_leaves: `63`
- gate mode: `hard`
- default gate threshold: `0.20`

Reason:

- It preserves the same leading `4-25` and `Ice 4-25` slice quality as the top `g015/g025` variants.
- It keeps `global_ratio` closest to `1.0` while still materially outperforming the sequence baseline.
- It is the safest default for continued local and multi-anchor validation.

## Conservative backup

If the next stage prioritizes total calibration over ratio neutrality, keep this backup:

- `p56_20251201_covact_lr005_l63_s2026_hard_g025`

Reason:

- Same `4-25` and `Ice 4-25` slice quality as `g020`
- Lower `global_wmape`
- Slightly more conservative total output (`global_ratio = 0.9636`)

## Best model vs baseline

### Best current tree candidate

- `exp_id`: `p56_20251201_covact_lr005_l63_s2026_hard_g020`
- `global_ratio`: `1.0128`
- `global_wmape`: `0.9666`
- `4_25_ratio`: `0.6013`
- `4_25_wmape_like`: `0.5025`
- `4_25_sku_p50`: `0.5898`
- `ice_4_25_sku_p50`: `0.4586`
- `blockbuster_sku_p50`: `0.3073`
- `auc`: `0.9487`
- `f1`: `0.7261`

### Sequence baseline

- `exp_id`: `p527_lstm_l3_v5_lite_s2027`
- metrics below are `phase5.4` three-seed means
- `global_ratio_mean`: `1.0949`
- `global_wmape_mean`: `1.5173`
- `4_25_ratio_mean`: `0.3045`
- `4_25_wmape_like_mean`: `0.7300`
- `4_25_sku_p50_mean`: `0.2707`
- `ice_4_25_sku_p50_mean`: `0.1244`
- `blockbuster_sku_p50_mean`: `0.0670`
- `auc_mean`: `0.7664`
- `f1_mean`: `0.5210`

## Direct comparison

Relative to the sequence baseline, the current tuned tree candidate:

- improves `4_25_sku_p50` from `0.2707` to `0.5898`
- improves `4_25_wmape_like` from `0.7300` to `0.5025`
- improves `ice_4_25_sku_p50` from `0.1244` to `0.4586`
- improves `blockbuster_sku_p50` from `0.0670` to `0.3073`
- improves `global_wmape` from `1.5173` to `0.9666`
- reduces `global_ratio` drift from `1.0949` to `1.0128`

## Practical interpretation

- The sequence baseline is no longer competitive on the business slices that matter most.
- The tree route is not just better on totals; it is materially better on `4-25`, `Ice 4-25`, and `>25`.
- The remaining work is no longer “prove tree beats LSTM again”.
- The remaining work is:
  - finish multi-anchor validation
  - calibrate low-demand `1-3`
  - decide whether `g020` or `g025` should be the default deployment setting

## Current stance

- `event/tree` should remain the research mainline.
- `p527` should remain the sequence baseline and backup line.
- Do not continue broad sequence hyperparameter sweeps.
