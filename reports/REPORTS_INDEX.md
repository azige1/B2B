# Reports Index

## Current Reports

Use `reports/current/` for current conclusions, handoff, and external-facing summaries.

Most important current files:

- `current/phase8_blockbuster_uplift_0p6_20260616.md`
- `current/phase8_event_coverage_router_0p6_20260616.md`
- `current/phase8_no_listing_0p6_series_candidate_20260616.md`
- `current/phase8_final_freeze_report_20260616.md`
- `current/server_whitelist_refresh_20260616.md`
- `current/server_daily_oracle_snapshot_automation_20260616.md`
- `current/client_source_table_registry.md`
- `current/phase8_data_semantics_20260614.md`

## Current Phase8 Output Directories

- `phase8_blockbuster_uplift_0p6/`
  - Current recommended Phase8 uplift candidate summaries and contexts.
- `phase8_event_request_router/`
  - Coverage-router comparison and best context.
- `phase8_event_request_shadow/`
  - Event+request shadow branch outputs.
- `phase8_purchase_request_shadow/`
  - Purchase-request shadow branch outputs.
- `phase8_data_audit/`
  - Server data audit outputs.

## Profit Analysis Reports

Current useful directories:

- `profit_analysis_skc_real_cost_h45_20260612/`
- `profit_analysis_skc_real_data_20260612/`
- `profit_analysis_real_data_20260612/`
- `profit_analysis_snapshot_smoke_final/`

Prefer module docs first:

- `modules/profit_analysis/README.md`
- `modules/profit_analysis/docs/profit_analysis_module_v1_detailed_design_20260522.md`
- `modules/profit_analysis/docs/profit_analysis_module_delivery_20260612.md`

## Historical Phase Results

These are retained for reproduction and audit:

- `phase1/`, `phase2/`, `phase3/`
- `phase5*/`
- `phase6*/`
- `phase7*/`
- older `phase8_*` directories not listed as current
- `rolling_backtest/`
- `eval_history/`

## Reading Rule

- Start from `current/`.
- Only enter a phase directory if a current report references it or you are reproducing that specific phase.
- Do not infer current status from directory freshness alone.
- Generated report directories are not automatically current just because they are newer.
