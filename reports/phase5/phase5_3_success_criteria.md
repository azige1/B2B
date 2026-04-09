# Phase5.3 Success Criteria

## Business Direction
- Primary optimization target: `4-25` units
- `>25` units: keep as a risk-monitoring slice, do not let it dominate current model selection
- `global ratio / WMAPE`: use as guardrails, not as the primary winner criterion

## Why This Direction
- In the current real validation universe:
  - `1-3` units = `69.49%` of positive SKUs, but only `17.71%` of true volume
  - `4-10` units = `16.38%` of positive SKUs, `16.14%` of true volume
  - `11-25` units = `9.23%` of positive SKUs, `24.83%` of true volume
  - `>25` units = `4.90%` of positive SKUs, `41.32%` of true volume
- This means:
  - optimizing only for total volume is too blunt
  - optimizing only for the majority of tiny SKUs is also too narrow
  - the most useful working target is `4-25`

## Winner Priority
1. `SKU P50 / bucket ratio`
2. `Cold/Ice`, `11-25`, `>25`
3. `global ratio / WMAPE`
4. `AUC / F1`

## Promotion Rules For Phase5.3A
- A candidate can only be considered a winner if it improves the main business band first:
  - `4-25 overall ratio` must improve versus the frozen sequence reference
  - `4-25 overall SKU P50` must improve versus the frozen sequence reference
- Cold-start cannot get worse in the main business band:
  - `Cold 4-25` and `Ice 4-25` must not regress materially
  - preferred direction is higher ratio and higher SKU P50
- `>25` does not need to become the best slice immediately, but it must remain visible:
  - do not ignore it in reports
  - do not accept a candidate that makes `>25` materially worse while only improving easy low-demand SKUs
- `global ratio` should remain in a healthy zone:
  - avoid candidates with obvious systematic over- or under-prediction
- `WMAPE` should stay competitive:
  - it is a guardrail for overall quantity stability, not the main business winner signal

## Sequence Reference Freeze Rule
- Freeze one `V3-filtered` reference from `phase5.2`
- Freeze one `V5-lite` reference from `phase5.2`
- Prefer the reference that is strongest on:
  - `SKU P50`
  - `4-25`
  - `Cold/Ice`
- Use `AUC / F1` only as tie-breakers, not as the main freeze criterion

## Event/Tree Promotion Rule
- Promote only `2-4` candidates to the server stage
- Server stage order:
  1. `3-5` seeds
  2. rolling anchors
- Do not expand a candidate that only improves `global ratio` while leaving `4-25` and `Cold/Ice` unchanged
