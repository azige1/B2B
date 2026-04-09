# Phase8 Shadow Experiment Policy

## Status

- Current official mainline remains `tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`.
- Client-facing order semantic questions are still unresolved.
- No experiment in this stage is allowed to replace the official phase7 mainline.

## Allowed Before Client Reply

- Event-only shadow experiments on `2025-10/11/12`
- Inventory and preorder exploratory analysis on `2026` windows
- SHAP and residual-gap diagnostics on the current official phase7 mainline

## Not Allowed Before Client Reply

- Any formal replacement compare against the current official four-anchor phase7 winner
- Any change to the official label-cleaning logic for `V_IRS_ORDERFTP`
- Any claim that a new candidate should replace the current official mainline
- Any open-ended hyperparameter search

## Blocking Issues

- `TYPE` missing semantics in `V_IRS_ORDERFTP`
- `QTY < 0` semantics in `V_IRS_ORDERFTP`
- Duplicate-row business meaning in `V_IRS_ORDERFTP`
- Lifecycle table not yet provided
