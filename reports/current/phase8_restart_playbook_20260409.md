# Phase8 Restart Playbook (2026-04-09)

## Current Position

- Official mainline remains `phase7 frozen`.
- Current official tree family remains `LightGBM`.
- Current best `phase8` exploratory baseline is `event_inventory_zero_split`.
- The latest `phase8k` asymmetric training round did not beat the promotion gate.
- Formal `phase8` promotion remains blocked by client-side data clarification rather than by model-line uncertainty.

## What Is Fixed Right Now

- Keep `phase7` as the official delivery-ready reference.
- Use `event_inventory_zero_split` as the default `phase8` comparison baseline.
- Do not roll back to `event_inventory_plus`.
- Do not start broader feature expansion, label changes, or new post-processing search before client data arrives.

## Canonical Reports

- Official freeze state: `reports/current/current_freeze_summary.md`
- Official mainline metadata: `reports/current/current_mainline.json`
- Phase8 direction: `reports/current/phase8_direction_note.md`
- Phase8 exploration summary: `reports/current/phase8_exploration_summary_20260409.md`
- Zero-split follow-up: `reports/current/phase8_zero_split_followup_20260409.md`
- Inventory semantics checkpoint: `reports/current/phase8_inventory_semantics_checkpoint_20260409.md`
- Behavior-control follow-up: `reports/current/phase8_behavior_control_followup_20260409.md`

## Replay Entry Points

### Zero-Split Baseline

- Runner: `scripts/runners/phase8/run_phase8h_inventory_zero_split_shadow_2026.py`
- Summary script: `src/analysis/summarize_phase8h_inventory_zero_split_results.py`
- Result summary: `reports/phase8_inventory_zero_split_shadow_2026/phase8_inventory_zero_split_summary.md`
- Result json: `reports/phase8_inventory_zero_split_shadow_2026/phase8_inventory_zero_split_result.json`

### Asymmetric Training Follow-Up

- Runner: `scripts/runners/phase8/run_phase8k_zero_split_asym_train_2026.py`
- Summary script: `src/analysis/summarize_phase8k_zero_split_asym_results.py`
- Result summary: `reports/phase8_zero_split_asym_train_2026/phase8_zero_split_asym_summary.md`
- Result json: `reports/phase8_zero_split_asym_train_2026/phase8_zero_split_asym_result.json`

### Supporting Analyses

- Snapshot zero-stock case pack: `scripts/runners/phase8/run_phase8g_snapshot_zero_stock_case_pack.py`
- Rule search: `scripts/runners/phase8/run_phase8i_zero_split_rule_search.py`
- Hard-gate search: `scripts/runners/phase8/run_phase8j_zero_split_hard_gate_search.py`
- Event+inventory detail pack: `src/analysis/generate_phase8d_event_inventory_shadow_detail_pack.py`

## Current Decision Gate

The current `phase8` baseline stays in place unless a new branch wins all of the following on the two anchors `2026-02-15` and `2026-02-24`:

- `mean global_wmape <= 0.6667`
- `mean blockbuster_under_wape <= 0.3635`
- `long_zero zero_true_fp_rate < 0.7361`
- `long_zero positive_true_under_wape` does not materially exceed `0.1662`

The latest asymmetric training round improved the `long_zero` false-positive behavior, but did not clear the full gate. Therefore it stays as a checkpoint result, not a promoted baseline.

## Wait Conditions

Do not reopen formal `phase8` promotion work until the client answers at least:

- `V_IRS_ORDERFTP` semantics for `TYPE`, negative `QTY`, and duplicate rows
- lifecycle table availability and grain

## When Work Restarts

1. Confirm the client-side answers and update the request notes under `reports/current/`.
2. Re-check whether label semantics or lifecycle fields require feature-line changes.
3. Resume from `event_inventory_zero_split`, not from `event_inventory_plus`.
4. Keep the scope narrow at first: two anchors, explicit promotion gate, no broad experiment sweep.
