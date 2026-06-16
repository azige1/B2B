# Artifacts Index

This file defines how to treat large local artifacts, generated reports, and model outputs.

## Current Policy

Commit:

- Source code under `src/`, `scripts/`, and `modules/` after review.
- Root index documents and current handoff reports.
- Small JSON/Markdown manifests that explain data or experiment provenance.
- Small curated CSV metric tables when they are required to audit a reported result.

Do not bulk commit:

- Raw warehouse CSV files.
- `data/incoming/` server snapshots.
- `data/processed_*` and `data/artifacts_*` feature directories.
- `models_phase*/` experiment model directories.
- Large prediction-context CSV files.
- Generated HTML/XLSX/PNG exports unless explicitly curated for handoff.

## Current Artifact Inventory

Latest inventory:

- `reports/current/workspace_artifact_inventory_20260616.md`
- `data/manifests/workspace_artifact_inventory_20260616.csv`
- `data/manifests/workspace_artifact_inventory_20260616.json`

Regenerate with:

```bash
python scripts/maintenance/inventory_workspace_artifacts.py
```

## Large Local Areas

The largest local areas are:

- `data/`: feature assets, incoming snapshots, processed artifacts.
- `reports/`: historical and experiment outputs.
- `models_phase*/`: generated experiment model binaries.
- `data_warehouse/`: raw Oracle and warehouse snapshots.

These are useful for local reproduction, but they should not make the GitHub repository heavy.

## Archive Rule

Before moving or deleting large generated artifacts, create a path-level archive manifest with:

```text
old_path,new_path,artifact_type,phase,keep_reason,regeneration_command
```

No large cleanup should happen without that manifest.

## Git Status Rule

If `git status` is noisy:

1. First update `.gitignore` for generated local artifacts.
2. Then generate the workspace artifact inventory.
3. Then review source/module changes separately.
4. Do not solve noise by committing raw data or model binaries.
