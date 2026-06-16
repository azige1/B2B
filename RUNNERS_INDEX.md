# Runners Index

This file lists the executable entry points that are still useful. Historical runners remain in place for reproducibility, but only the current sections should be used for new work.

## Current Phase8 Reproduction

Run the Phase8 blockbuster uplift analysis from existing router contexts:

```bash
python src/analysis/analyze_phase8u_blockbuster_uplift.py
```

Relevant outputs:

- `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_candidate_summary.csv`
- `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_anchor_table.csv`
- `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_recommended_context.csv`
- `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_summary.md`

Rebuild the Event+request shadow branch:

```bash
python scripts/runners/phase8/run_phase8t_event_request_shadow.py
python src/analysis/summarize_phase8t_event_request_shadow_results.py
```

Rebuild the coverage router analysis:

```bash
python src/analysis/analyze_phase8t_event_request_router.py
```

## Current Data Operations

Server-side daily Oracle export script backup:

```bash
scripts/data/server_run_daily_oracle_snapshot.sh
```

Server cron template:

```bash
scripts/data/server_daily_oracle_snapshot.cron
```

Oracle table inspection:

```bash
python scripts/data/inspect_oracle_tables.py
```

Oracle snapshot export:

```bash
python scripts/data/export_oracle_snapshot.py --output-dir <output_dir> --legacy-config <get_store.py>
```

Local snapshot sync helper for the older structured server warehouse format:

```bash
python scripts/data/sync_phase8_server_snapshot.py --source <source> --cutoff <YYYYMMDD>
```

Workspace artifact inventory:

```bash
python scripts/maintenance/inventory_workspace_artifacts.py
```

## Profit Analysis

Production-style SKC snapshot:

```bash
python modules/profit_analysis/scripts/run_skc_profit_snapshot.py --prediction-csv <prediction.csv> --inventory-csv <inventory.csv> --economics-csv <economics.csv>
```

Real-data experiment:

```bash
python modules/profit_analysis/scripts/run_skc_real_data_experiment.py
```

Cost normalization:

```bash
python modules/profit_analysis/scripts/normalize_style_costs.py
```

## Historical Phase7 Baseline

These reproduce the historical frozen Phase7 baseline and external-report artifacts:

```bash
python scripts/runners/phase7/run_phase7_freeze.py
python scripts/runners/phase7/run_phase7i_full_model_compare.py
python scripts/runners/phase7/run_phase7_mainline_refresh_validation_20260416.py
```

## Phase8 Historical Experiments

The following are retained for audit/replay, not as default new-work entry points:

- `scripts/runners/phase8/run_phase8a_prep.py`
- `scripts/runners/phase8/run_phase8l_purchase_request_shadow.py`
- `scripts/runners/phase8/run_phase8m_lifecycle_shadow.py`
- `scripts/runners/phase8/run_phase8n_listing_date_targeted.py`
- `scripts/runners/phase8/run_phase8o_lifecycle_peer_priors.py`
- `scripts/runners/phase8/run_phase8p_peer_prior_46_90.py`
- `scripts/runners/phase8/run_phase8q_2026_out_of_time.py`
- `scripts/runners/phase8/run_phase8r_robust_oot.py`
- `scripts/runners/phase8/run_phase8s_event_core_robust_oot.py`
- `scripts/runners/phase8/run_phase8_formal_stage1_20260614.py`
- `scripts/runners/phase8/run_phase8_formal_stage1b_20260614.py`
- `scripts/runners/phase8/run_phase8_formal_stage2_20260614.py`
- `scripts/runners/phase8/run_phase8_listing_zero_split_shadow_2026.py`

## Working Rule

- Use the "Current Phase8 Reproduction" section for current prediction work.
- Use "Current Data Operations" for server/data refresh work.
- Use the profit-analysis section only inside `modules/profit_analysis`.
- Use historical runners only when reproducing a named historical result.
