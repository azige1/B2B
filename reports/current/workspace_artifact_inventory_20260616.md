# Workspace Artifact Inventory

Generated at: `2026-06-16T17:47:12`

## Purpose

This report separates source code and curated handoff documents from generated local artifacts. It is a cleanup aid, not a deletion list.

## Top-Level Directory Inventory

| path | files | size_mb | category | recommendation |
| --- | ---: | ---: | --- | --- |
| `data` | 1881 | 253805.26 | data_assets | Use data/DATA_INDEX.md and manifests; do not bulk commit generated data. |
| `reports` | 1935 | 1484.36 | reports_and_experiments | Commit curated current docs; leave generated exports ignored. |
| `models_phase8_robust_oot` | 270 | 880.69 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_formal_stage2_20260614` | 270 | 880.46 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `data_warehouse` | 763 | 546.71 | raw_snapshot_warehouse | Keep local snapshots with manifests; do not commit CSV extracts. |
| `.vendor_tree_backends` | 263 | 475.04 | tooling_or_cache | Usually local; commit only config files. |
| `models_phase8_formal_stage1_20260614` | 60 | 195.89 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_formal_stage1b_20260614` | 60 | 195.85 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_event_core_robust_oot` | 54 | 176.04 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_2026_out_of_time` | 54 | 136.19 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_zero_split_asym_train_2026` | 36 | 119.91 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_lifecycle_peer_priors` | 30 | 98.03 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models` | 26 | 93.59 | historical_model_artifact | Keep tracked references only; generated binaries stay ignored. |
| `models_phase8_listing_zero_split_shadow_2026` | 28 | 78.74 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_event_inventory_shadow_2026` | 24 | 78.59 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_event_request_shadow` | 24 | 78.37 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_purchase_request_shadow` | 24 | 78.33 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_listing_date_targeted` | 24 | 78.28 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_listing_date_stage_ablation` | 24 | 78.27 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_peer_prior_46_90` | 24 | 78.27 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_lifecycle_shadow` | 24 | 78.23 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase7_refresh_validation_20260416` | 24 | 78.14 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_event_shadow` | 18 | 58.72 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `models_phase8_inventory_zero_split_shadow_2026` | 12 | 39.34 | generated_model_artifact | Keep local or archive by manifest; do not commit binaries. |
| `src` | 208 | 3.22 | source_or_test_code | Review normally; code changes should be committed intentionally. |
| `.git` | 626 | 3.11 | tooling_or_cache | Usually local; commit only config files. |
| `scripts` | 146 | 1.42 | source_or_test_code | Review normally; code changes should be committed intentionally. |
| `modules` | 69 | 0.62 | active_profit_analysis_module | Review and commit as a separate module-focused change. |
| `__pycache__` | 9 | 0.08 | other | Review manually. |
| `tests` | 23 | 0.08 | source_or_test_code | Review normally; code changes should be committed intentionally. |
| `diagnostics` | 5 | 0.02 | other | Review manually. |
| `docs` | 3 | 0.02 | source_or_test_code | Review normally; code changes should be committed intentionally. |
| `.pytest_cache` | 6 | 0.00 | tooling_or_cache | Usually local; commit only config files. |
| `config` | 2 | 0.00 | source_or_test_code | Review normally; code changes should be committed intentionally. |
| `utils` | 5 | 0.00 | source_or_test_code | Review normally; code changes should be committed intentionally. |

## Remaining Git Working-Tree Groups

### curated_current_report_candidate

- Count: `40`

- ` M` `reports/current/phase8_restart_playbook_20260409.md`
- ` D` `reports/current/profit_analysis_module_v1_proposal_20260410.md`
- ` M` `reports/current/v_irs_orderftp_audit.md`
- ` D` `"reports/current/\347\233\210\344\272\217\345\210\206\346\236\220\346\250\241\345\235\227V1\346\212\200\346\234\257\346\226\271\346\241\210_20260410.md"`
- `??` `reports/current/actual_qty_audit_phase7_anchors_20260417.md`
- `??` `reports/current/client_feedback_20250430_meeting_20260515.md`
- `??` `reports/current/daily_work_summary_for_advisor_20260515.md`
- `??` `reports/current/launch_date_intake_spec_20260515.md`
- `??` `reports/current/launch_date_integration_audit_20260612.md`
- `??` `reports/current/phase7_client_actual_qty_audit_20251201_20260430.json`
- `??` `reports/current/phase7_client_actual_qty_audit_20251201_20260430.md`
- `??` `reports/current/phase7_client_export_non_ai_audit_20251201_20260417.md`
- `??` `reports/current/phase7_client_export_refresh_followup_20251201_20260417.md`
- `??` `reports/current/phase7_current_mainline_multidim_metrics_20260430.md`
- `??` `reports/current/phase7_refresh_validation_checkpoint_20260416.md`
- `??` `reports/current/phase7_stage_meeting_brief_20260430.md`
- `??` `reports/current/phase8_2026_out_of_time_validation_20260612.md`
- `??` `reports/current/phase8_2026_phase7_vs_listing_full_metrics_20260612.md`
- `??` `reports/current/phase8_client_data_gap_priority_20260614.md`
- `??` `reports/current/phase8_client_reply_checkpoint_20260416.md`
- `??` `reports/current/phase8_lifecycle_peer_prior_upgrade_20260612.md`
- `??` `reports/current/phase8_listing_date_lifecycle_shadow_20260612.md`
- `??` `reports/current/phase8_listing_date_prediction_upgrade_20260612.md`
- `??` `reports/current/phase8_listing_date_zero_split_checkpoint_20260614.md`
- `??` `reports/current/phase8_official_style_2026_evaluation_20260616.md`
- `??` `reports/current/phase8_pre_launch_date_work_20260515.md`
- `??` `reports/current/phase8_signal_routes_strict_comparison_20260613.md`
- `??` `reports/current/profit_analysis_style_cost_audit_20260612.json`
- `??` `reports/current/profit_analysis_style_cost_audit_20260612.md`
- `??` `reports/current/project_status_profit_analysis_brief_20260605.md`
- `??` `reports/current/project_status_profit_analysis_roadmap_20260605.md`
- `??` `reports/current/purchase_request_deep_dive_20260515.md`
- `??` `reports/current/purchase_request_feature_shadow_20260515.md`
- `??` `reports/current/purchase_request_mapping_audit_20260515.json`
- `??` `reports/current/purchase_request_mapping_audit_20260515.md`
- `??` `reports/current/purchase_request_table_audit_20260515.json`
- `??` `reports/current/purchase_request_table_audit_20260515.md`
- `??` `reports/current/server_data_intake_audit_20260614.md`
- `??` `reports/current/v_irs_orderftp_refresh_audit_20260416.md`
- `??` `reports/current/v_irs_orderftp_refresh_audit_20260614.md`

### data_manifest_or_registry

- Count: `1`

- `??` `data/manifests/phase8_data_snapshot_20260614.json`

### generated_or_historical_report

- Count: `30`

- `??` `reports/phase7_refresh_validation_20260416/`
- `??` `reports/phase8_2026_out_of_time/`
- `??` `reports/phase8_data_audit/`
- `??` `reports/phase8_event_core_robust_oot/`
- `??` `reports/phase8_formal_stage1_20260614/`
- `??` `reports/phase8_formal_stage1b_20260614/`
- `??` `reports/phase8_formal_stage2_20260614/`
- `??` `reports/phase8_lifecycle_peer_priors/`
- `??` `reports/phase8_lifecycle_shadow/`
- `??` `reports/phase8_listing_date_stage_ablation/`
- `??` `reports/phase8_listing_date_targeted/`
- `??` `reports/phase8_listing_zero_split_shadow_2026/`
- `??` `reports/phase8_peer_prior_46_90/`
- `??` `reports/phase8_purchase_request_diagnostics/`
- `??` `reports/phase8_purchase_request_shadow/`
- `??` `reports/phase8_robust_oot/`
- `??` `reports/profit_analysis_launch_date_safe_20260612/`
- `??` `reports/profit_analysis_lifecycle_smoke/`
- `??` `reports/profit_analysis_real_data_20260612/`
- `??` `reports/profit_analysis_real_data_smoke/`
- `??` `reports/profit_analysis_skc_h45_final_smoke/`
- `??` `reports/profit_analysis_skc_h45_smoke/`
- `??` `reports/profit_analysis_skc_real_cost_20260612/`
- `??` `reports/profit_analysis_skc_real_cost_h45_20260612/`
- `??` `reports/profit_analysis_skc_real_data_20260612/`
- `??` `reports/profit_analysis_skc_real_data_smoke/`
- `??` `reports/profit_analysis_smoke/`
- `??` `reports/profit_analysis_snapshot_real_smoke/`
- `??` `reports/profit_analysis_snapshot_smoke/`
- `??` `reports/profit_analysis_snapshot_smoke_final/`

### other_pending

- Count: `9`

- ` M` `.gitignore`
- ` M` `reports/phase8a_prep/phase8_residual_gap_summary.md`
- ` M` `reports/phase8a_prep/phase8a_prep_manifest.json`
- ` D` `"\347\233\210\344\272\217\345\210\206\346\236\220\346\250\241\345\235\227V1\346\212\200\346\234\257\346\226\271\346\241\210.md"`
- `??` `reports/client_ready/`
- `??` `reports/decision_scheduler_demo/`
- `??` `reports/inventory_snapshot_demo/`
- `??` `reports/phase8a_prep/lifecycle_launch_date_coverage_audit.md`
- `??` `reports/phase8a_prep/phase8_purchase_request_data_coverage_audit.md`

### profit_analysis_module_pending

- Count: `28`

- ` M` `modules/profit_analysis/README.md`
- ` M` `modules/profit_analysis/config/profit_analysis_business_defaults_template.csv`
- ` M` `modules/profit_analysis/config/profit_analysis_economics_config_template.csv`
- ` M` `modules/profit_analysis/config/profit_analysis_inventory_snapshot_template.csv`
- ` M` `modules/profit_analysis/config/profit_analysis_prediction_snapshot_template.csv`
- ` M` `modules/profit_analysis/scripts/backtest_profit_analysis.py`
- ` M` `modules/profit_analysis/scripts/build_profit_analysis_inputs.py`
- ` M` `modules/profit_analysis/scripts/run_profit_analysis_snapshot.py`
- ` M` `modules/profit_analysis/src/profit_analysis/__init__.py`
- ` M` `modules/profit_analysis/src/profit_analysis/builders.py`
- ` M` `modules/profit_analysis/src/profit_analysis/core.py`
- ` M` `modules/profit_analysis/src/profit_analysis/io.py`
- `??` `modules/profit_analysis/config/demand_scenario_calibration_h45_20260612.json`
- `??` `modules/profit_analysis/config/profit_analysis_business_defaults_client_feedback_20260515.csv`
- `??` `modules/profit_analysis/docs/profit_analysis_module_delivery_20260612.md`
- `??` `modules/profit_analysis/docs/profit_analysis_module_v1_detailed_design_20260522.md`
- `??` `modules/profit_analysis/docs/profit_analysis_v0_client_feedback_rules_20260515.md`
- `??` `modules/profit_analysis/docs/source_materials/`
- `??` `modules/profit_analysis/scripts/normalize_style_costs.py`
- `??` `modules/profit_analysis/scripts/run_real_data_experiment.py`
- `??` `modules/profit_analysis/scripts/run_skc_profit_snapshot.py`
- `??` `modules/profit_analysis/scripts/run_skc_real_data_experiment.py`
- `??` `modules/profit_analysis/src/profit_analysis/allocation.py`
- `??` `modules/profit_analysis/src/profit_analysis/calibration.py`
- `??` `modules/profit_analysis/src/profit_analysis/decision.py`
- `??` `modules/profit_analysis/src/profit_analysis/grouping.py`
- `??` `modules/profit_analysis/src/profit_analysis/horizon.py`
- `??` `modules/profit_analysis/tests/`

### source_code_pending

- Count: `47`

- ` M` `evaluate_tabular.py`
- ` M` `src/etl/build_wide_table.py`
- ` M` `src/etl/clean_data.py`
- ` M` `src/features/build_features_v5_lite_sku.py`
- ` M` `src/features/build_features_v6_event_inventory_shadow_sku.py`
- ` M` `src/features/build_features_v6_event_sku.py`
- ` M` `src/features/phase53_feature_utils.py`
- ` M` `src/train/train_tabular_v6.py`
- `??` `scripts/analysis/demo_inventory_snapshot_usage.py`
- `??` `scripts/analysis/demo_replenishment_decision_scheduler.py`
- `??` `scripts/analysis/generate_client_mainline_anchor_pack.py`
- `??` `scripts/maintenance/`
- `??` `scripts/runners/phase7/run_phase7_mainline_refresh_validation_20260416.py`
- `??` `scripts/runners/phase8/run_phase8_formal_stage1_20260614.py`
- `??` `scripts/runners/phase8/run_phase8_formal_stage1b_20260614.py`
- `??` `scripts/runners/phase8/run_phase8_formal_stage2_20260614.py`
- `??` `scripts/runners/phase8/run_phase8_listing_zero_split_shadow_2026.py`
- `??` `scripts/runners/phase8/run_phase8l_purchase_request_shadow.py`
- `??` `scripts/runners/phase8/run_phase8m_lifecycle_shadow.py`
- `??` `scripts/runners/phase8/run_phase8n_listing_date_targeted.py`
- `??` `scripts/runners/phase8/run_phase8o_lifecycle_peer_priors.py`
- `??` `scripts/runners/phase8/run_phase8p_peer_prior_46_90.py`
- `??` `scripts/runners/phase8/run_phase8q_2026_out_of_time.py`
- `??` `scripts/runners/phase8/run_phase8r_robust_oot.py`
- `??` `scripts/runners/phase8/run_phase8s_event_core_robust_oot.py`
- `??` `src/analysis/analyze_phase8r_listing_hybrids.py`
- `??` `src/analysis/build_launch_date_lifecycle_v0_features.py`
- `??` `src/analysis/build_phase8_listing_hybrid_candidates.py`
- `??` `src/analysis/generate_phase8_purchase_request_diagnostics.py`
- `??` `src/analysis/generate_phase8_purchase_request_prep_artifacts.py`
- `??` `src/analysis/generate_phase8q_full_phase7_comparison.py`
- `??` `src/analysis/summarize_phase8_listing_zero_split_results.py`
- `??` `src/analysis/summarize_phase8l_purchase_request_shadow_results.py`
- `??` `src/analysis/summarize_phase8m_lifecycle_shadow_results.py`
- `??` `src/analysis/summarize_phase8n_listing_date_ablation.py`
- `??` `src/analysis/summarize_phase8n_listing_date_targeted_results.py`
- `??` `src/analysis/summarize_phase8o_lifecycle_peer_prior_results.py`
- `??` `src/analysis/summarize_phase8p_peer_prior_46_90_results.py`
- `??` `src/analysis/summarize_phase8q_2026_out_of_time.py`
- `??` `src/analysis/summarize_phase8r_robust_oot.py`
- ... 7 more

## Cleanup Recommendations

- Keep `reports/current/`, root indexes, and small manifest files as the canonical handoff layer.
- Keep raw snapshots and generated feature/model artifacts local or in external storage; do not bulk commit them.
- Review `modules/profit_analysis/` separately and commit it as its own module-focused change.
- Review remaining `src/` and `scripts/` changes separately before committing; they may belong to older Phase8 work.
- Do not delete generated artifacts until a path-level archive manifest has been written.
