import json
from pathlib import Path

import numpy as np
import pandas as pd

from phase_eval_utils import evaluate_context_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "reports" / "phase8_blockbuster_uplift_0p6"
BASE_CONTEXT = PROJECT_ROOT / "reports" / "phase8_event_request_router" / "phase8_event_request_router_best_context.csv"
ROUTER_ANCHOR = PROJECT_ROOT / "reports" / "phase8_event_request_router" / "phase8_event_request_router_anchor_table.csv"

ANCHORS = ["2025-09-01", "2025-10-01", "2025-11-01", "2025-12-01"]
PRED_COLS_TO_SCALE = ["ai_pred_qty", "ai_pred_qty_open"]
SUMMARY_METRICS = [
    "global_wmape",
    "global_ratio",
    "4_25_under_wape",
    "4_25_sku_p50",
    "ice_4_25_sku_p50",
    "blockbuster_under_wape",
    "blockbuster_sku_p50",
    "top20_true_volume_capture",
    "rank_corr_positive_skus",
    "1_3_ratio",
    "false_positive_rate_zero_true",
    "zero_true_pred_ge_3_rate",
]


def anchor_tag(anchor):
    return anchor.replace("-", "")


def normalize_context(df):
    out = df.copy()
    out["sku_id"] = out["sku_id"].astype(str)
    out["anchor_date"] = pd.to_datetime(out["anchor_date"]).dt.strftime("%Y-%m-%d")
    for col in [
        "ai_pred_qty",
        "ai_pred_qty_open",
        "ai_pred_prob",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
        "qty_first_order",
        "event_strong_30",
        "event_any_30",
        "event_active_buyers_30",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def evaluate_by_anchor(df, candidate_key, rule_name, uplift_mult, selected_count):
    rows = []
    for anchor in ANCHORS:
        part = df[df["anchor_date"] == anchor].copy()
        row = evaluate_context_frame(part, f"{candidate_key}_{anchor_tag(anchor)}")
        row["anchor_date"] = anchor
        row["candidate_key"] = candidate_key
        row["uplift_rule"] = rule_name
        row["uplift_mult"] = uplift_mult
        row["selected_count"] = int(selected_count.get(anchor, 0))
        row["selected_rate"] = float(selected_count.get(anchor, 0) / len(part)) if len(part) else 0.0
        rows.append(row)
    anchor_df = pd.DataFrame(rows)
    summary = {
        "candidate_key": candidate_key,
        "uplift_rule": rule_name,
        "uplift_mult": uplift_mult,
        "selected_count": int(sum(selected_count.values())),
        "selected_rate": float(sum(selected_count.values()) / len(df)) if len(df) else 0.0,
    }
    for col in SUMMARY_METRICS:
        summary[col] = float(anchor_df[col].astype(float).mean())
    return summary, anchor_df


def load_router_baselines():
    anchor = pd.read_csv(ROUTER_ANCHOR)
    keys = ["phase7_all", "event_request_all", "router_phase7_anchor_event_covered"]
    out = {}
    for key in keys:
        part = anchor[anchor["candidate_key"] == key]
        out[key] = {col: float(part[col].astype(float).mean()) for col in SUMMARY_METRICS}
        out[key]["max_anchor_global_ratio"] = float(part["global_ratio"].astype(float).max())
        out[key]["anchor_ratio_over_110_count"] = int((part["global_ratio"].astype(float) > 1.10).sum())
        out[key]["candidate_key"] = key
    return out


def score_columns(df):
    out = df.copy()
    hist_signal = np.maximum.reduce(
        [
            out["lookback_repl_sum_90"].to_numpy(dtype=float),
            out["lookback_future_sum_90"].to_numpy(dtype=float),
            out["qty_first_order"].to_numpy(dtype=float),
            out.get("event_strong_30", pd.Series(0.0, index=out.index)).to_numpy(dtype=float),
        ]
    )
    out["uplift_hist_signal"] = hist_signal
    out["uplift_demand_score"] = (
        out["ai_pred_qty"].astype(float)
        * (1.0 + 0.25 * out["ai_pred_prob"].astype(float))
        * (1.0 + 0.02 * np.log1p(np.maximum(hist_signal, 0.0)))
    )
    return out


def make_rule_masks(df):
    rules = []
    for pred_min in [8, 10, 12, 15, 20]:
        for prob_min in [0.50, 0.70, 0.85]:
            for hist_min in [20, 50, 80, 120]:
                name = f"pred_ge_{pred_min}_prob_ge_{prob_min:.2f}_signal_ge_{hist_min}"
                mask = (
                    (df["ai_pred_qty"] >= pred_min)
                    & (df["ai_pred_prob"] >= prob_min)
                    & (df["uplift_hist_signal"] >= hist_min)
                )
                rules.append((name, mask))

    for q in [0.90, 0.93, 0.95, 0.97]:
        mask = pd.Series(False, index=df.index)
        for anchor in ANCHORS:
            idx = df["anchor_date"] == anchor
            threshold = float(df.loc[idx, "uplift_demand_score"].quantile(q))
            mask.loc[idx] = df.loc[idx, "uplift_demand_score"] >= threshold
        rules.append((f"top_demand_score_q{int(q * 100)}", mask))

    for q in [0.90, 0.93, 0.95, 0.97]:
        mask = pd.Series(False, index=df.index)
        for anchor in ANCHORS:
            idx = df["anchor_date"] == anchor
            threshold = float(df.loc[idx, "ai_pred_qty"].quantile(q))
            mask.loc[idx] = df.loc[idx, "ai_pred_qty"] >= threshold
        rules.append((f"top_pred_qty_q{int(q * 100)}", mask))

    dedup = {}
    for name, mask in rules:
        key = tuple(np.asarray(mask, dtype=bool).nonzero()[0].tolist())
        if key and key not in dedup:
            dedup[key] = (name, np.asarray(mask, dtype=bool))
    return list(dedup.values())


def apply_uplift(df, mask, mult, candidate_key, rule_name):
    out = df.copy()
    for col in PRED_COLS_TO_SCALE:
        out.loc[mask, col] = out.loc[mask, col].astype(float) * float(mult)
    out["ai_pred_positive_qty"] = (out["ai_pred_qty"].astype(float) > 0).astype(int)
    out["abs_error"] = (out["ai_pred_qty"].astype(float) - out["true_replenish_qty"].astype(float)).abs()
    out["candidate_key"] = candidate_key
    out["uplift_rule"] = rule_name
    out["uplift_mult"] = float(mult)
    out["uplift_selected"] = mask.astype(int)
    return out


def add_deltas(summary_df, baselines):
    out = summary_df.copy()
    router = baselines["router_phase7_anchor_event_covered"]
    phase7 = baselines["phase7_all"]
    for metric in SUMMARY_METRICS:
        out[f"{metric}_delta_vs_router"] = out[metric].astype(float) - float(router[metric])
        out[f"{metric}_delta_vs_phase7"] = out[metric].astype(float) - float(phase7[metric])
    out["wmape_improvement_pct_vs_phase7"] = (
        (float(phase7["global_wmape"]) - out["global_wmape"].astype(float))
        / float(phase7["global_wmape"])
        * 100.0
    )
    out["blockbuster_under_improvement_pct_vs_phase7"] = (
        (float(phase7["blockbuster_under_wape"]) - out["blockbuster_under_wape"].astype(float))
        / float(phase7["blockbuster_under_wape"])
        * 100.0
    )
    return out


def add_anchor_safety(summary_df, anchor_df):
    ratio_stats = (
        anchor_df.assign(global_ratio=pd.to_numeric(anchor_df["global_ratio"], errors="coerce"))
        .groupby("candidate_key")
        .agg(
            max_anchor_global_ratio=("global_ratio", "max"),
            anchor_ratio_over_110_count=("global_ratio", lambda s: int((s > 1.10).sum())),
        )
        .reset_index()
    )
    return summary_df.merge(ratio_stats, on="candidate_key", how="left")


def gate_candidates(summary_df, baselines):
    router = baselines["router_phase7_anchor_event_covered"]
    out = summary_df.copy()
    out["anchor_ratio_safe"] = out["max_anchor_global_ratio"].astype(float) <= 1.10
    out["phase7_style_gate"] = (
        (out["global_ratio"].between(0.90, 1.10))
        & (out["global_wmape"] <= float(router["global_wmape"]) + 0.003)
        & (out["4_25_under_wape"] <= float(router["4_25_under_wape"]) + 0.003)
        & (out["4_25_sku_p50"] >= float(router["4_25_sku_p50"]) - 0.003)
        & (out["ice_4_25_sku_p50"] >= float(router["ice_4_25_sku_p50"]) - 0.006)
        & (out["top20_true_volume_capture"] >= float(router["top20_true_volume_capture"]) - 0.003)
        & (out["rank_corr_positive_skus"] >= float(router["rank_corr_positive_skus"]) - 0.003)
        & (out["1_3_ratio"] <= float(router["1_3_ratio"]) + 0.020)
        & (out["zero_true_pred_ge_3_rate"] <= float(router["zero_true_pred_ge_3_rate"]) + 0.002)
    )
    out["strict_router_win"] = (
        out["phase7_style_gate"]
        & (out["blockbuster_under_wape"] < float(router["blockbuster_under_wape"]) - 0.005)
        & (out["blockbuster_sku_p50"] > float(router["blockbuster_sku_p50"]))
        & (out["global_wmape"] <= float(router["global_wmape"]))
    )
    out["phase8_final_recommended_gate"] = out["strict_router_win"] & out["anchor_ratio_safe"]
    return out


def rank_candidates(df):
    return str(
        df.sort_values(
            [
                "blockbuster_under_wape",
                "global_wmape",
                "blockbuster_sku_p50",
                "4_25_under_wape",
            ],
            ascending=[True, True, False, True],
        ).iloc[0]["candidate_key"]
    )


def select_metric_best(summary_df):
    strict = summary_df[summary_df["strict_router_win"]].copy()
    if strict.empty:
        strict = summary_df[summary_df["phase7_style_gate"]].copy()
    if strict.empty:
        strict = summary_df.copy()
    return rank_candidates(strict)


def select_recommended(summary_df):
    final_safe = summary_df[summary_df["phase8_final_recommended_gate"]].copy()
    if not final_safe.empty:
        return rank_candidates(final_safe)

    anchor_safe = summary_df[summary_df["anchor_ratio_safe"] & summary_df["phase7_style_gate"]].copy()
    if not anchor_safe.empty:
        return rank_candidates(anchor_safe)

    return select_metric_best(summary_df)


def render_candidate_row(candidate_key, row, rule=None, mult=None, selected_rate=None):
    return (
        f"| {candidate_key} | {rule if rule is not None else row['uplift_rule']} | "
        f"{float(mult if mult is not None else row['uplift_mult']):.2f} | "
        f"{float(selected_rate if selected_rate is not None else row['selected_rate']):.4f} | "
        f"{float(row['global_wmape']):.4f} | {float(row['global_ratio']):.4f} | "
        f"{float(row.get('max_anchor_global_ratio', row['global_ratio'])):.4f} | "
        f"{float(row['4_25_under_wape']):.4f} | {float(row['4_25_sku_p50']):.4f} | "
        f"{float(row['ice_4_25_sku_p50']):.4f} | {float(row['blockbuster_under_wape']):.4f} | "
        f"{float(row['blockbuster_sku_p50']):.4f} | {float(row['top20_true_volume_capture']):.4f} | "
        f"{float(row['rank_corr_positive_skus']):.4f} | {float(row['1_3_ratio']):.4f} | "
        f"{float(row['zero_true_pred_ge_3_rate']):.4f} |"
    )


def render_summary(summary_df, baselines, metric_best_key, recommended_key):
    lookup = summary_df.set_index("candidate_key")
    metric_best = lookup.loc[metric_best_key]
    recommended = lookup.loc[recommended_key]
    router = baselines["router_phase7_anchor_event_covered"]
    phase7 = baselines["phase7_all"]
    event_request = baselines["event_request_all"]

    lines = [
        "# Phase8 Blockbuster Uplift Summary",
        "",
        "- Status: `shadow_postprocess_exploration`",
        "- Base branch: `coverage_router` under old Phase7 0.6-series evaluation.",
        "- This run does not retrain the model. It tests business-safe uplift rules using only prediction-time signals.",
        "",
        "## Key Metrics",
        "",
        "| candidate | rule | mult | selected_rate | global_wmape | global_ratio | max_anchor_ratio | 4_25_under_wape | 4_25_sku_p50 | ice_4_25_sku_p50 | blockbuster_under_wape | blockbuster_sku_p50 | top20_capture | rank_corr | 1_3_ratio | zero_true_ge3 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    base_rows = [
        ("phase7_all", "baseline", 1.0, 0.0, phase7),
        ("event_request_all", "baseline", 1.0, 1.0, event_request),
        ("coverage_router", "anchor_event_covered", 1.0, 0.75, router),
    ]
    for name, rule, mult, rate, row in base_rows:
        lines.append(render_candidate_row(name, row, rule=rule, mult=mult, selected_rate=rate))

    lines.append(render_candidate_row(metric_best_key, metric_best))
    if recommended_key != metric_best_key:
        lines.append(render_candidate_row(recommended_key, recommended))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Metric-best candidate: `{metric_best_key}`.",
            f"- Recommended final candidate: `{recommended_key}`.",
            f"- Recommended candidate changes blockbuster under-WAPE by `{float(recommended['blockbuster_under_wape_delta_vs_router']):+.4f}` and global WMAPE by `{float(recommended['global_wmape_delta_vs_router']):+.4f}` relative to `coverage_router`.",
            f"- Recommended candidate improves Phase7 WMAPE by `{float(recommended['wmape_improvement_pct_vs_phase7']):.2f}%` and blockbuster under-WAPE by `{float(recommended['blockbuster_under_improvement_pct_vs_phase7']):.2f}%`.",
            f"- Metric-best max anchor ratio is `{float(metric_best['max_anchor_global_ratio']):.4f}`; recommended max anchor ratio is `{float(recommended['max_anchor_global_ratio']):.4f}`.",
            "- Use the recommended candidate when Phase8 needs a defensible final branch. Use metric-best only as an aggressive shadow reference.",
            "",
            "## Outputs",
            "",
            "- Candidate summary: `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_candidate_summary.csv`",
            "- Anchor metrics: `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_anchor_table.csv`",
            "- Metric-best context: `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_metric_best_context.csv`",
            "- Recommended context: `reports/phase8_blockbuster_uplift_0p6/phase8_blockbuster_uplift_recommended_context.csv`",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = score_columns(normalize_context(pd.read_csv(BASE_CONTEXT)))
    baselines = load_router_baselines()

    summaries = []
    anchor_tables = []
    contexts = {}
    rules = make_rule_masks(base)
    multipliers = [1.03, 1.05, 1.08, 1.10, 1.12, 1.15, 1.20]

    base_summary, base_anchor = evaluate_by_anchor(
        base,
        "coverage_router_base_recomputed",
        "no_uplift",
        1.0,
        {anchor: 0 for anchor in ANCHORS},
    )
    summaries.append(base_summary)
    anchor_tables.append(base_anchor)
    contexts[base_summary["candidate_key"]] = base

    for rule_name, mask in rules:
        selected_count = {
            anchor: int(mask[(base["anchor_date"] == anchor).to_numpy()].sum())
            for anchor in ANCHORS
        }
        if sum(selected_count.values()) < 20:
            continue
        for mult in multipliers:
            candidate_key = f"uplift_{rule_name}_x{str(mult).replace('.', '')}"
            context = apply_uplift(base, mask, mult, candidate_key, rule_name)
            summary, anchor_df = evaluate_by_anchor(
                context,
                candidate_key,
                rule_name,
                mult,
                selected_count,
            )
            summaries.append(summary)
            anchor_tables.append(anchor_df)
            contexts[candidate_key] = context

    summary_df = pd.DataFrame(summaries)
    anchor_df = pd.concat(anchor_tables, ignore_index=True, sort=False)
    summary_df = add_anchor_safety(summary_df, anchor_df)
    summary_df = add_deltas(summary_df, baselines)
    summary_df = gate_candidates(summary_df, baselines)
    metric_best_key = select_metric_best(summary_df)
    recommended_key = select_recommended(summary_df)
    summary_df = summary_df.sort_values(
        [
            "phase8_final_recommended_gate",
            "strict_router_win",
            "phase7_style_gate",
            "blockbuster_under_wape",
            "global_wmape",
        ],
        ascending=[False, False, False, True, True],
    )

    summary_df.to_csv(OUT_DIR / "phase8_blockbuster_uplift_candidate_summary.csv", index=False, encoding="utf-8-sig")
    anchor_df.to_csv(OUT_DIR / "phase8_blockbuster_uplift_anchor_table.csv", index=False, encoding="utf-8-sig")
    contexts[metric_best_key].to_csv(OUT_DIR / "phase8_blockbuster_uplift_metric_best_context.csv", index=False, encoding="utf-8-sig")
    contexts[recommended_key].to_csv(OUT_DIR / "phase8_blockbuster_uplift_recommended_context.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "phase8_blockbuster_uplift_summary.md").write_text(
        render_summary(summary_df, baselines, metric_best_key, recommended_key),
        encoding="utf-8-sig",
    )
    payload = {
        "status": "shadow_postprocess_exploration",
        "base_branch": "coverage_router",
        "metric_best_candidate_key": metric_best_key,
        "recommended_candidate_key": recommended_key,
        "metric_best_summary": summary_df[summary_df["candidate_key"] == metric_best_key].iloc[0].to_dict(),
        "recommended_summary": summary_df[summary_df["candidate_key"] == recommended_key].iloc[0].to_dict(),
        "outputs": {
            "candidate_summary": str((OUT_DIR / "phase8_blockbuster_uplift_candidate_summary.csv").relative_to(PROJECT_ROOT)),
            "anchor_table": str((OUT_DIR / "phase8_blockbuster_uplift_anchor_table.csv").relative_to(PROJECT_ROOT)),
            "summary_md": str((OUT_DIR / "phase8_blockbuster_uplift_summary.md").relative_to(PROJECT_ROOT)),
            "metric_best_context": str((OUT_DIR / "phase8_blockbuster_uplift_metric_best_context.csv").relative_to(PROJECT_ROOT)),
            "recommended_context": str((OUT_DIR / "phase8_blockbuster_uplift_recommended_context.csv").relative_to(PROJECT_ROOT)),
        },
    }
    (OUT_DIR / "phase8_blockbuster_uplift_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] candidates -> {OUT_DIR / 'phase8_blockbuster_uplift_candidate_summary.csv'}")
    print(f"[OK] anchor table -> {OUT_DIR / 'phase8_blockbuster_uplift_anchor_table.csv'}")
    print(f"[OK] summary -> {OUT_DIR / 'phase8_blockbuster_uplift_summary.md'}")
    print(f"[OK] metric_best={metric_best_key}")
    print(f"[OK] recommended={recommended_key}")


if __name__ == "__main__":
    main()
