# Current Model Compare Summary

## Scope

- Official compare date: `2025-12-01`
- Official current winner: `tail_full_lr005_l63_g027_n800_s2028 + sep098_oct093`
- Previous official tree mainline: `p57_covact_lr005_l63_hard_g025 + sep098_oct093`
- Best LSTM reference in the current cleaned-gold scope: `p527_lstm_l3_v5_lite_s2027_s2027`

## Source Files

- HTML: `reports/current/current_model_compare.html`
- CSV: `reports/current/current_model_compare.csv`
- Historical source directory: `reports/phase7i_full_model_compare/`

## Headline Result

The current official phase7 mainline materially outperforms both the previous official tree mainline and the best LSTM reference on the cleaned-gold December compare.

## Single-Anchor Compare Highlights

- `Global WMAPE`: `0.6942` vs `0.9237` vs `1.2401`
- `4-25 Under WAPE`: `0.3535` vs `0.4512` vs `0.6548`
- `4-25 SKU P50`: `0.6683` vs `0.5898` vs `0.3335`
- `Ice 4-25 SKU P50`: `0.5964` vs `0.4586` vs `0.1658`
- `>25 Under WAPE`: `0.4516` vs `0.6642` vs `0.8731`
- `>25 SKU P50`: `0.5292` vs `0.3073` vs `0.1002`
- `Top20 Capture`: `0.7001` vs `0.6755` vs `0.4874`
- `Rank Corr`: `0.7822` vs `0.6955` vs `0.4198`

Values above are ordered as:

1. Current phase7 mainline
2. Previous official phase6 tree mainline
3. Best LSTM reference

## Universe Note

- This compare uses the cleaned gold universe.
- SKUs missing from `data/silver/clean_products.csv` are excluded from the current official compare.
