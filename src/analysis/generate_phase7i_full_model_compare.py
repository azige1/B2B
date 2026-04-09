import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from phase_eval_utils import evaluate_context_frame


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "reports" / "phase7i_full_model_compare"
OUT_CSV = OUT_DIR / "dec_20251201_full_model_compare.csv"
OUT_HTML = OUT_DIR / "dec_20251201_full_model_compare.html"
OUT_SUMMARY = OUT_DIR / "dec_20251201_full_model_compare_summary.md"

TARGET_ANCHOR = "2025-12-01"
STATIC_PATH = PROJECT_ROOT / "data" / "gold" / "wide_table_sku.csv"
PHASE7_FROZEN = PROJECT_ROOT / "reports" / "phase7" / "phase7_frozen_mainline.json"
PHASE7_STAGE3_DIR = PROJECT_ROOT / "reports" / "phase7_tail_allocation_optimization" / "overnight_20260402_overnight" / "stage_s3" / "20251201" / "phase5"
PHASE6_PATH = PROJECT_ROOT / "reports" / "phase5_7" / "20251201" / "phase5" / "eval_context_p57a_20251201_covact_lr005_l63_s2026_hard_g025.csv"
P527_DIR = PROJECT_ROOT / "reports" / "phase5_4" / "phase5"

STATIC_COLS = [
    "sku_id",
    "style_id",
    "product_name",
    "category",
    "sub_category",
    "season",
    "series",
    "band",
    "size_id",
    "color_id",
    "qty_first_order",
    "price_tag",
]

MODEL_ORDER = ["phase7", "phase6", "lstm"]
MODEL_LABELS = {
    "phase7": "Phase7 当前最强",
    "phase6": "Phase6 冻结主线",
    "lstm": "最佳 LSTM",
}
MODEL_COLORS = {
    "true": "#111827",
    "phase7": "#0f766e",
    "phase6": "#2563eb",
    "lstm": "#dc2626",
    "good": "#0f766e",
    "bad": "#dc2626",
}


def ensure_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def fmt(value, digits=4):
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def esc(value):
    return html.escape("" if value is None else str(value))


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize(v) for v in value]
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if pd.isna(value):
        return None
    return value


def safe_ratio(num, den):
    num = float(num)
    den = float(den)
    if den <= 0:
        return np.nan
    return num / den


def short_label(row):
    product = str(row.get("product_name") or "")[:12]
    if product:
        return f"{row['sku_id']} | {product}"
    return str(row["sku_id"])


def load_static():
    raw = pd.read_csv(STATIC_PATH, usecols=STATIC_COLS)
    raw["sku_id"] = raw["sku_id"].astype(str)
    dedup = raw.drop_duplicates("sku_id").copy()
    return raw, dedup


def load_context(path, valid_skus):
    df = pd.read_csv(path)
    df["sku_id"] = df["sku_id"].astype(str)
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.strftime("%Y-%m-%d")
    return df[(df["anchor_date"] == TARGET_ANCHOR) & (df["sku_id"].isin(valid_skus))].copy().reset_index(drop=True)


def validate_truth(base, other, name):
    merged = base.merge(other, on=["sku_id", "anchor_date"], how="inner", suffixes=("_base", "_other"))
    if len(merged) != len(base):
        raise RuntimeError(f"{name} row mismatch: {len(merged)} vs {len(base)}")
    mismatch = ~np.isclose(
        merged["true_replenish_qty_base"].astype(float),
        merged["true_replenish_qty_other"].astype(float),
        atol=1e-6,
        rtol=0.0,
    )
    if mismatch.any():
        raise RuntimeError(f"{name} true mismatch rows={int(mismatch.sum())}")


def choose_phase7_path():
    frozen = json.loads(PHASE7_FROZEN.read_text(encoding="utf-8"))
    raw_model = frozen["raw_model"]
    matches = [p for p in PHASE7_STAGE3_DIR.glob("eval_context_*.csv") if raw_model in p.name]
    if len(matches) != 1:
        raise RuntimeError(f"phase7 path match != 1: {matches}")
    return matches[0], frozen


def choose_best_lstm(valid_skus):
    rows = []
    contexts = {}
    for seed in (2026, 2027, 2028):
        path = P527_DIR / f"eval_context_p54_p527_lstm_l3_v5_lite_s2027_s{seed}.csv"
        df = load_context(path, valid_skus)
        metrics = evaluate_context_frame(df, f"p527_s{seed}")
        rows.append(
            {
                "seed": seed,
                "path": str(path),
                "global_wmape": metrics["global_wmape"],
                "4_25_under_wape": metrics["4_25_under_wape"],
                "4_25_sku_p50": metrics["4_25_sku_p50"],
                "ice_4_25_sku_p50": metrics["ice_4_25_sku_p50"],
                "blockbuster_under_wape": metrics["blockbuster_under_wape"],
                "blockbuster_sku_p50": metrics["blockbuster_sku_p50"],
                "top20_true_volume_capture": metrics["top20_true_volume_capture"],
                "rank_corr_positive_skus": metrics["rank_corr_positive_skus"],
            }
        )
        contexts[seed] = (df, path)
    seed_df = pd.DataFrame(rows).sort_values(
        [
            "global_wmape",
            "4_25_under_wape",
            "blockbuster_under_wape",
            "4_25_sku_p50",
            "ice_4_25_sku_p50",
            "blockbuster_sku_p50",
            "top20_true_volume_capture",
            "rank_corr_positive_skus",
        ],
        ascending=[True, True, True, False, False, False, False, False],
    )
    best_seed = int(seed_df.iloc[0]["seed"])
    best_df, best_path = contexts[best_seed]
    return best_seed, best_df, best_path, seed_df.reset_index(drop=True)


def rename_model(df, prefix):
    cols = ["sku_id", "anchor_date", "true_replenish_qty", "ai_pred_prob", "ai_pred_qty", "abs_error"]
    return df[cols].rename(
        columns={
            "ai_pred_prob": f"prob_{prefix}",
            "ai_pred_qty": f"pred_{prefix}",
            "abs_error": f"abs_error_{prefix}",
        }
    )


def qfo_bucket(series):
    return pd.cut(
        series.fillna(0).astype(float),
        bins=[-0.01, 0.0, 1.0, 10.0, 30.0, 100.0, np.inf],
        labels=["0", "1", "2-10", "11-30", "31-100", "100+"],
    ).astype(str)


def true_bucket(series):
    return pd.cut(
        series.astype(float),
        bins=[-0.01, 0.0, 3.0, 10.0, 25.0, 50.0, np.inf],
        labels=["0", "1-3", "4-10", "11-25", "26-50", "50+"],
    ).astype(str)


def group_compare(df, group_col, order=None, filter_mask=None):
    work = df.copy()
    if filter_mask is not None:
        work = work.loc[filter_mask].copy()
    rows = []
    groups = order or sorted(work[group_col].dropna().astype(str).unique().tolist())
    for group in groups:
        sub = work[work[group_col].astype(str) == str(group)].copy()
        if sub.empty:
            continue
        true = sub["true_replenish_qty"].astype(float)
        true_sum = float(true.sum())
        row = {
            group_col: str(group),
            "rows": int(len(sub)),
            "true_sum": true_sum,
            "true_mean": float(true.mean()),
        }
        for prefix in MODEL_ORDER:
            pred = sub[f"pred_{prefix}"].astype(float)
            row[f"{prefix}_pred_sum"] = float(pred.sum())
            row[f"{prefix}_pred_mean"] = float(pred.mean())
            row[f"{prefix}_ratio"] = safe_ratio(pred.sum(), true_sum)
            row[f"{prefix}_under_wape"] = safe_ratio(np.clip(true - pred, a_min=0, a_max=None).sum(), true_sum)
            row[f"{prefix}_over_wape"] = safe_ratio(np.clip(pred - true, a_min=0, a_max=None).sum(), true_sum)
        rows.append(row)
    return pd.DataFrame(rows)


def category_compare(df, top_n=12):
    rows = []
    for category, sub in df.groupby("category", dropna=False):
        true = sub["true_replenish_qty"].astype(float)
        true_sum = float(true.sum())
        if true_sum <= 0:
            continue
        row = {"category": str(category), "rows": int(len(sub)), "true_sum": true_sum}
        for prefix in MODEL_ORDER:
            pred = sub[f"pred_{prefix}"].astype(float)
            row[f"{prefix}_ratio"] = safe_ratio(pred.sum(), true_sum)
            row[f"{prefix}_under_wape"] = safe_ratio(np.clip(true - pred, a_min=0, a_max=None).sum(), true_sum)
        rows.append(row)
    cat = pd.DataFrame(rows)
    cat["phase7_ratio_score"] = np.abs(np.log(np.clip(cat["phase7_ratio"].astype(float), 1e-9, None)))
    return (
        cat.sort_values(["phase7_under_wape", "true_sum"], ascending=[False, False]).head(top_n).reset_index(drop=True),
        cat.sort_values(["phase7_ratio_score", "true_sum"], ascending=[False, False]).head(top_n).reset_index(drop=True),
    )


def improvement_tables(df):
    return (
        df.sort_values(["delta_abserr_phase7_vs_phase6", "true_replenish_qty"], ascending=[False, False]).head(20),
        df.sort_values(["delta_abserr_phase7_vs_lstm", "true_replenish_qty"], ascending=[False, False]).head(20),
        df.sort_values(["delta_abserr_phase7_vs_phase6", "true_replenish_qty"], ascending=[True, False]).head(20),
        df.sort_values(["abs_error_phase7", "true_replenish_qty"], ascending=[False, False]).head(20),
    )


def density_bins(df, pred_col, bins=34):
    true_vals = np.log1p(df["true_replenish_qty"].astype(float).clip(lower=0))
    pred_vals = np.log1p(df[pred_col].astype(float).clip(lower=0))
    max_val = max(float(true_vals.max()), float(pred_vals.max()), 1.0)
    hist, xedges, yedges = np.histogram2d(true_vals, pred_vals, bins=bins, range=[[0, max_val], [0, max_val]])
    return hist, xedges, yedges, max_val


def svg_heatmap(df, pred_col, title, color):
    hist, xedges, yedges, max_val = density_bins(df, pred_col)
    width, height = 520, 420
    left, bottom = 72, 44
    top = 26
    plot_w = width - left - 20
    plot_h = height - top - bottom
    max_count = hist.max() if hist.max() > 0 else 1.0
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="svg-plot" role="img" aria-label="{esc(title)}">',
        f'<text x="{width/2:.1f}" y="18" text-anchor="middle" class="svg-title">{esc(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#dbe2ea"/>',
    ]
    for i in range(hist.shape[0]):
        for j in range(hist.shape[1]):
            count = hist[i, j]
            if count <= 0:
                continue
            opacity = 0.12 + 0.88 * (count / max_count)
            x = left + (xedges[i] / max_val) * plot_w
            y = top + plot_h - (yedges[j + 1] / max_val) * plot_h
            w = (xedges[i + 1] - xedges[i]) / max_val * plot_w
            h = (yedges[j + 1] - yedges[j]) / max_val * plot_h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w,0.5):.2f}" height="{max(h,0.5):.2f}" fill="{color}" fill-opacity="{opacity:.3f}"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top}" stroke="#111827" stroke-dasharray="4 4" stroke-width="1.2"/>')
    for t in range(6):
        frac = t / 5
        x = left + plot_w * frac
        y = top + plot_h - plot_h * frac
        label = math.expm1(max_val * frac)
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 4}" stroke="#94a3b8"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - 10}" text-anchor="middle" class="svg-axis">{label:.0f}</text>')
        parts.append(f'<line x1="{left - 4}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#94a3b8"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" class="svg-axis">{label:.0f}</text>')
    parts.append(f'<text x="{left + plot_w/2:.1f}" y="{height - 2}" text-anchor="middle" class="svg-axis-label">真实 30 天补货量</text>')
    parts.append(f'<text x="16" y="{top + plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 16 {top + plot_h/2:.1f})" class="svg-axis-label">预测 30 天补货量</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_grouped_hbar(rows, series_keys, series_labels, series_colors, title, value_key="label", width=1360, label_width=280):
    rows = list(rows)
    if not rows:
        return "<div class='empty'>无数据</div>"
    series_count = len(series_keys)
    row_slot = 28
    top, bottom = 28, 32
    plot_h = len(rows) * row_slot
    height = top + plot_h + bottom
    plot_w = width - label_width - 30
    max_val = max(max(float(row.get(key, 0) or 0) for key in series_keys) for row in rows)
    max_val = max(max_val, 1.0)
    bar_h = max(3.0, (row_slot - 6) / series_count)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="svg-plot" role="img" aria-label="{esc(title)}">',
        f'<text x="{width/2:.1f}" y="18" text-anchor="middle" class="svg-title">{esc(title)}</text>',
        f'<rect x="{label_width}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#dbe2ea"/>',
    ]
    for tick in range(6):
        frac = tick / 5
        x = label_width + plot_w * frac
        val = max_val * frac
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - 8}" text-anchor="middle" class="svg-axis">{val:.0f}</text>')
    for idx, row in enumerate(rows):
        y0 = top + idx * row_slot + 3
        parts.append(f'<text x="{label_width - 8}" y="{y0 + row_slot / 2 + 4:.2f}" text-anchor="end" class="svg-label">{esc(row.get(value_key, ""))}</text>')
        for s_idx, key in enumerate(series_keys):
            val = float(row.get(key, 0) or 0)
            x = label_width
            y = y0 + s_idx * bar_h
            w = plot_w * (val / max_val)
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w,0.5):.2f}" height="{bar_h - 1:.2f}" fill="{series_colors[s_idx]}" rx="1.5"/>')
    parts.append("</svg>")
    legend = "<div class='svg-legend'>" + "".join(
        f"<span><i style='background:{series_colors[i]}'></i>{esc(series_labels[i])}</span>" for i in range(series_count)
    ) + "</div>"
    return "".join(parts) + legend


def svg_grouped_vbar(rows, series_keys, series_labels, series_colors, title, label_key="label", width=760, height=360):
    rows = list(rows)
    if not rows:
        return "<div class='empty'>无数据</div>"
    left, top, bottom = 52, 28, 84
    plot_w = width - left - 16
    plot_h = height - top - bottom
    max_val = max(float(row.get(key, 0) or 0) for row in rows for key in series_keys)
    max_val = max(max_val, 1.0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="svg-plot" role="img" aria-label="{esc(title)}">',
        f'<text x="{width/2:.1f}" y="18" text-anchor="middle" class="svg-title">{esc(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#dbe2ea"/>',
    ]
    for tick in range(6):
        frac = tick / 5
        y = top + plot_h - plot_h * frac
        val = max_val * frac
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4:.2f}" text-anchor="end" class="svg-axis">{val:.2f}</text>')
    slot_w = plot_w / len(rows)
    inner_gap = 4
    bar_w = max(4.0, (slot_w - inner_gap * (len(series_keys) + 1)) / len(series_keys))
    for idx, row in enumerate(rows):
        x0 = left + idx * slot_w
        parts.append(f'<text x="{x0 + slot_w/2:.2f}" y="{top + plot_h + 16}" text-anchor="middle" transform="rotate(35 {x0 + slot_w/2:.2f} {top + plot_h + 16})" class="svg-label">{esc(row.get(label_key, ""))}</text>')
        for s_idx, key in enumerate(series_keys):
            val = float(row.get(key, 0) or 0)
            h = plot_h * (val / max_val)
            x = x0 + inner_gap + s_idx * (bar_w + inner_gap)
            y = top + plot_h - h
            parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{max(h,0.5):.2f}" fill="{series_colors[s_idx]}" rx="1.5"/>')
    parts.append("</svg>")
    legend = "<div class='svg-legend'>" + "".join(
        f"<span><i style='background:{series_colors[i]}'></i>{esc(series_labels[i])}</span>" for i in range(len(series_keys))
    ) + "</div>"
    return "".join(parts) + legend


def svg_single_hbar(rows, label_key, value_key, title, width=760, color="#0f766e", signed=False):
    rows = list(rows)
    if not rows:
        return "<div class='empty'>无数据</div>"
    top, bottom = 28, 26
    label_width, row_h = 230, 24
    plot_h = len(rows) * row_h
    height = top + plot_h + bottom
    plot_w = width - label_width - 20
    values = [float(row.get(value_key, 0) or 0) for row in rows]
    min_val, max_val = (min(values), max(values)) if signed else (0.0, max(values))
    if signed:
        bound = max(abs(min_val), abs(max_val), 1.0)
        min_val, max_val = -bound, bound
    else:
        max_val = max(max_val, 1.0)

    def x_map(v):
        return label_width + plot_w * ((v - min_val) / (max_val - min_val))

    zero_x = x_map(0.0)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="svg-plot" role="img" aria-label="{esc(title)}">',
        f'<text x="{width/2:.1f}" y="18" text-anchor="middle" class="svg-title">{esc(title)}</text>',
        f'<rect x="{label_width}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#dbe2ea"/>',
    ]
    for tick in range(6):
        frac = tick / 5
        val = min_val + (max_val - min_val) * frac
        x = x_map(val)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - 8}" text-anchor="middle" class="svg-axis">{val:.2f}</text>')
    if signed:
        parts.append(f'<line x1="{zero_x:.2f}" y1="{top}" x2="{zero_x:.2f}" y2="{top + plot_h}" stroke="#111827" stroke-width="1.1"/>')
    for idx, row in enumerate(rows):
        y = top + idx * row_h + 5
        value = float(row.get(value_key, 0) or 0)
        x = min(zero_x, x_map(value)) if signed else label_width
        w = abs(x_map(value) - zero_x) if signed else (x_map(value) - label_width)
        fill = MODEL_COLORS["good"] if signed and value >= 0 else MODEL_COLORS["bad"] if signed else color
        parts.append(f'<text x="{label_width - 8}" y="{y + 10:.2f}" text-anchor="end" class="svg-label">{esc(row.get(label_key, ""))}</text>')
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(w,0.5):.2f}" height="14" fill="{fill}" rx="2"/>')
    parts.append("</svg>")
    return "".join(parts)


def svg_ratio_hist(df, title):
    bins = np.linspace(0, 3, 31)
    centers = (bins[:-1] + bins[1:]) / 2
    pos = df[df["true_replenish_qty"].astype(float) > 0].copy()
    series = {}
    for prefix in MODEL_ORDER:
        ratio = pos[f"ratio_{prefix}"].astype(float).clip(lower=0, upper=3)
        counts, _ = np.histogram(ratio, bins=bins, density=False)
        counts = counts.astype(float)
        counts = counts / counts.sum() if counts.sum() > 0 else counts
        series[prefix] = counts
    width, height = 760, 320
    left, top, bottom = 48, 28, 44
    plot_w = width - left - 16
    plot_h = height - top - bottom
    max_val = max(series[p].max() for p in MODEL_ORDER)
    max_val = max(max_val, 1e-6)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" class="svg-plot" role="img" aria-label="{esc(title)}">',
        f'<text x="{width/2:.1f}" y="18" text-anchor="middle" class="svg-title">{esc(title)}</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#ffffff" stroke="#dbe2ea"/>',
    ]
    for tick in range(6):
        frac = tick / 5
        y = top + plot_h - plot_h * frac
        val = max_val * frac
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#eef2f7"/>')
        parts.append(f'<text x="{left - 6}" y="{y + 4:.2f}" text-anchor="end" class="svg-axis">{val:.3f}</text>')
    for tick in range(7):
        x = left + plot_w * (tick / 6)
        val = 3 * tick / 6
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 4}" stroke="#94a3b8"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - 8}" text-anchor="middle" class="svg-axis">{val:.1f}</text>')
    for prefix in MODEL_ORDER:
        points = []
        for i, center in enumerate(centers):
            x = left + plot_w * (center / 3.0)
            y = top + plot_h - plot_h * (series[prefix][i] / max_val)
            points.append(f"{x:.2f},{y:.2f}")
        parts.append(f'<polyline fill="none" stroke="{MODEL_COLORS[prefix]}" stroke-width="2" points="{" ".join(points)}"/>')
    parts.append("</svg>")
    legend = "<div class='svg-legend'>" + "".join(
        f"<span><i style='background:{MODEL_COLORS[p]}'></i>{MODEL_LABELS[p]}</span>" for p in MODEL_ORDER
    ) + "</div>"
    return "".join(parts) + legend


def html_table(df, columns=None, digits=4):
    columns = columns or list(df.columns)
    parts = ["<table class='static-table'><thead><tr>"]
    for col in columns:
        parts.append(f"<th>{esc(col)}</th>")
    parts.append("</tr></thead><tbody>")
    for _, row in df.iterrows():
        parts.append("<tr>")
        for col in columns:
            val = row[col]
            cls = "num" if isinstance(val, (int, float, np.integer, np.floating)) and not isinstance(val, bool) else ""
            text = fmt(val, digits) if cls == "num" else esc(val)
            parts.append(f"<td class='{cls}'>{text}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def render_html(payload):
    detail_json = json.dumps({"columns": payload["detail_columns"], "rows": payload["detail_rows"]}, ensure_ascii=False)
    summary_items = "".join(f"<li>{esc(line)}</li>" for line in payload["summary_lines"])
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{esc(payload['report_title'])}</title>
  <style>
    :root {{ --bg:#f5f7fb; --panel:#fff; --ink:#0f172a; --muted:#475569; --line:#d8e0eb; --shadow:0 10px 28px rgba(15,23,42,.06); }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:var(--bg); color:var(--ink); font-family:"Segoe UI","PingFang SC",sans-serif; }}
    .wrap {{ max-width:1800px; margin:0 auto; padding:24px; }}
    h1,h2,h3 {{ margin:0 0 10px; }}
    .sub {{ color:var(--muted); line-height:1.7; }}
    .grid-hero {{ display:grid; grid-template-columns:1.4fr 1fr; gap:18px; }}
    .grid-2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    .grid-3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; box-shadow:var(--shadow); padding:18px; overflow:hidden; }}
    .chips {{ display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }}
    .chip {{ font-size:12px; padding:6px 10px; border-radius:999px; background:#eef6ff; color:#1d4ed8; border:1px solid #c9dbff; }}
    .section-title {{ margin-top:22px; margin-bottom:12px; font-size:22px; }}
    .svg-plot {{ width:100%; height:auto; display:block; }}
    .svg-title {{ font-size:13px; font-weight:700; fill:#0f172a; }}
    .svg-axis {{ font-size:10px; fill:#64748b; }}
    .svg-axis-label {{ font-size:11px; fill:#475569; }}
    .svg-label {{ font-size:10px; fill:#334155; }}
    .svg-legend {{ display:flex; gap:12px; flex-wrap:wrap; font-size:12px; color:var(--muted); margin-top:8px; }}
    .svg-legend i {{ display:inline-block; width:10px; height:10px; border-radius:999px; margin-right:6px; vertical-align:middle; }}
    .static-table,.detail-table {{ width:100%; border-collapse:collapse; }}
    .static-table th,.static-table td,.detail-table th,.detail-table td {{ padding:8px 10px; border-bottom:1px solid var(--line); font-size:13px; vertical-align:top; }}
    .static-table th,.detail-table th {{ position:sticky; top:0; background:#f8fafc; z-index:1; text-align:left; }}
    .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .table-wrap {{ max-height:420px; overflow:auto; border:1px solid var(--line); border-radius:12px; }}
    .controls {{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:12px; }}
    .controls input,.controls select {{ padding:8px 10px; border-radius:10px; border:1px solid var(--line); background:#fff; }}
    .pager {{ display:flex; align-items:center; gap:10px; margin-top:12px; }}
    .pager button {{ border:1px solid var(--line); background:#fff; border-radius:8px; padding:6px 10px; cursor:pointer; }}
    ul.tight {{ margin:0; padding-left:20px; line-height:1.7; }}
    .muted {{ color:var(--muted); }}
    @media (max-width:1280px) {{ .grid-hero,.grid-2,.grid-3 {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="grid-hero">
      <div class="card">
        <h1>{esc(payload['report_title'])}</h1>
        <div class="sub">这份页面把当前最强版本、Phase6 冻结主线、最佳 LSTM 放到同一口径下直接对照。重点不只看总体 KPI，还看结构性切片、重点 SKU、类别、冷启动、残余误差和逐 SKU 明细。</div>
        <div class="chips">
          <span class="chip">Anchor = {TARGET_ANCHOR}</span>
          <span class="chip">Universe rows = {payload['row_count']}</span>
          <span class="chip">Best LSTM seed = s{payload['best_lstm_seed']}</span>
          <span class="chip">Removed missing-product rows = {payload['removed_missing_product_rows']}</span>
        </div>
      </div>
      <div class="card"><h3>读图顺序</h3><ul class="tight">{summary_items}</ul></div>
    </div>
    <h2 class="section-title">1. 模型来源与总体指标</h2>
    <div class="grid-2">
      <div class="card"><h3>模型来源</h3><div class="table-wrap">{payload['model_meta_html']}</div></div>
      <div class="card"><h3>LSTM 三个 seed 的当前口径重选结果</h3><div class="table-wrap">{payload['lstm_seed_html']}</div></div>
    </div>
    <div class="card"><h3>关键指标总表</h3><div class="table-wrap">{payload['metrics_table_html']}</div></div>
    <h2 class="section-title">2. 重点 SKU 对照</h2>
    <div class="grid-2"><div class="card">{payload['top_true_svg']}</div><div class="card">{payload['top_pred_svg']}</div></div>
    <div class="grid-2"><div class="card">{payload['blockbuster_svg']}</div><div class="card">{payload['ratio_hist_svg']}</div></div>
    <h2 class="section-title">3. 结构性切片</h2>
    <div class="grid-2"><div class="card">{payload['quadrant_svg']}</div><div class="card">{payload['qfo_svg']}</div></div>
    <div class="grid-2"><div class="card"><h3>按真实需求桶汇总</h3><div class="table-wrap">{payload['true_bucket_html']}</div></div><div class="card"><h3>按信号象限汇总</h3><div class="table-wrap">{payload['quadrant_html']}</div></div></div>
    <div class="card"><h3>冷启动按首单量分桶汇总</h3><div class="table-wrap">{payload['qfo_html']}</div></div>
    <h2 class="section-title">4. 类别与个案</h2>
    <div class="grid-2"><div class="card">{payload['cat_under_svg']}</div><div class="card">{payload['cat_ratio_svg']}</div></div>
    <div class="grid-2"><div class="card"><h3>最差类别：Under 侧</h3><div class="table-wrap">{payload['cat_under_html']}</div></div><div class="card"><h3>最差类别：Ratio 偏离</h3><div class="table-wrap">{payload['cat_ratio_html']}</div></div></div>
    <h2 class="section-title">5. 提升与遗留问题</h2>
    <div class="grid-2"><div class="card">{payload['improve_phase6_svg']}</div><div class="card">{payload['improve_lstm_svg']}</div></div>
    <div class="grid-2"><div class="card"><h3>Phase7 仍然最难的 SKU</h3><div class="table-wrap">{payload['hardest_html']}</div></div><div class="card"><h3>相对 Phase6 的主要回退个案</h3><div class="table-wrap">{payload['regress_html']}</div></div></div>
    <h2 class="section-title">6. 真实 vs 预测密度图</h2>
    <div class="grid-3"><div class="card">{payload['phase7_svg']}</div><div class="card">{payload['phase6_svg']}</div><div class="card">{payload['lstm_svg']}</div></div>
    <h2 class="section-title">7. 完整逐 SKU 明细</h2>
    <div class="card">
      <div class="sub">这里展示的是适合交互浏览的核心列。完整数据请直接打开同目录下 CSV。</div>
      <div class="controls"><input id="search" placeholder="搜索 sku / category / style / product_name ..." /><select id="pageSize"><option value="20">20 行</option><option value="50" selected>50 行</option><option value="100">100 行</option></select></div>
      <div class="table-wrap"><table id="detail" class="detail-table"></table></div>
      <div class="pager"><button id="prev">上一页</button><span id="pageInfo" class="muted"></span><button id="next">下一页</button></div>
    </div>
  </div>
  <script>
    const DETAIL = {detail_json};
    const STATE = {{ page: 1, pageSize: 50, sortCol: 'rank_true_desc', sortAsc: true, search: '' }};
    const INDEX = Object.fromEntries(DETAIL.columns.map((c, i) => [c, i]));
    function fmtValue(v) {{ if (v === null || v === undefined) return ''; if (typeof v === 'number') return Number(v).toFixed(4); return String(v); }}
    function filteredRows() {{
      let rows = DETAIL.rows.slice();
      if (STATE.search) {{
        const q = STATE.search.toLowerCase();
        rows = rows.filter(row => row.some(v => String(v ?? '').toLowerCase().includes(q)));
      }}
      const sortIdx = INDEX[STATE.sortCol];
      rows.sort((a, b) => {{
        const av = a[sortIdx], bv = b[sortIdx];
        const aNum = typeof av === 'number', bNum = typeof bv === 'number';
        const cmp = (aNum && bNum) ? ((av ?? -Infinity) - (bv ?? -Infinity)) : String(av ?? '').localeCompare(String(bv ?? ''));
        return STATE.sortAsc ? cmp : -cmp;
      }});
      return rows;
    }}
    function renderDetail() {{
      const rows = filteredRows();
      const totalPages = Math.max(1, Math.ceil(rows.length / STATE.pageSize));
      if (STATE.page > totalPages) STATE.page = totalPages;
      const pageRows = rows.slice((STATE.page - 1) * STATE.pageSize, (STATE.page - 1) * STATE.pageSize + STATE.pageSize);
      const table = document.getElementById('detail');
      let html = '<thead><tr>';
      DETAIL.columns.forEach(col => html += `<th data-col="${{col}}">${{col}}</th>`);
      html += '</tr></thead><tbody>';
      pageRows.forEach(row => {{
        html += '<tr>';
        row.forEach(v => {{
          const cls = typeof v === 'number' ? 'num' : '';
          html += `<td class="${{cls}}">${{fmtValue(v)}}</td>`;
        }});
        html += '</tr>';
      }});
      html += '</tbody>';
      table.innerHTML = html;
      table.querySelectorAll('th').forEach(th => {{
        th.style.cursor = 'pointer';
        th.onclick = () => {{
          const col = th.dataset.col;
          if (STATE.sortCol === col) STATE.sortAsc = !STATE.sortAsc;
          else {{ STATE.sortCol = col; STATE.sortAsc = true; }}
          renderDetail();
        }};
      }});
      document.getElementById('pageInfo').textContent = `第 ${{STATE.page}} / ${{totalPages}} 页，共 ${{rows.length}} 行`;
    }}
    document.getElementById('search').addEventListener('input', e => {{ STATE.search = e.target.value.trim(); STATE.page = 1; renderDetail(); }});
    document.getElementById('pageSize').addEventListener('change', e => {{ STATE.pageSize = Number(e.target.value); STATE.page = 1; renderDetail(); }});
    document.getElementById('prev').onclick = () => {{ STATE.page = Math.max(1, STATE.page - 1); renderDetail(); }};
    document.getElementById('next').onclick = () => {{ const totalPages = Math.max(1, Math.ceil(filteredRows().length / STATE.pageSize)); STATE.page = Math.min(totalPages, STATE.page + 1); renderDetail(); }};
    renderDetail();
  </script>
</body>
</html>"""


def build_summary(metrics_df, model_meta_df, best_lstm_seed, phase7_mainline):
    lines = [
        "# 2025-12 详细模型对照页摘要",
        "",
        f"- 输出 CSV: `{OUT_CSV}`",
        f"- 输出 HTML: `{OUT_HTML}`",
        f"- 当前最强版本: `{phase7_mainline}`",
        "- 中间版本: `p57_covact_lr005_l63_hard_g025 + sep098_oct093`",
        f"- 最佳 LSTM: `p527_lstm_l3_v5_lite_s2027_s{best_lstm_seed}`",
        "",
        "## 模型来源",
        "",
        model_meta_df.to_markdown(index=False),
        "",
        "## 核心指标",
        "",
        metrics_df[["metric", "phase7", "phase6", "lstm", "delta_vs_phase6", "delta_vs_lstm"]].to_markdown(index=False),
        "",
    ]
    return "\n".join(lines)


def main():
    ensure_dir()
    raw_static, static_df = load_static()
    valid_skus = set(static_df["sku_id"])

    phase7_path, phase7_frozen = choose_phase7_path()
    phase7 = load_context(phase7_path, valid_skus)
    phase6 = load_context(PHASE6_PATH, valid_skus)
    best_lstm_seed, lstm, lstm_path, lstm_seed_df = choose_best_lstm(valid_skus)

    validate_truth(phase7, phase6, "phase6")
    validate_truth(phase7, lstm, "lstm")

    shared = [
        "sku_id", "anchor_date", "true_replenish_qty", "signal_quadrant", "activity_bucket",
        "lookback_repl_days_90", "lookback_future_days_90", "lookback_repl_sum_90", "lookback_future_sum_90",
        "category", "style_id", "season", "series", "band",
    ]
    readable = phase7[shared].copy()
    readable = readable.merge(rename_model(phase7, "phase7"), on=["sku_id", "anchor_date", "true_replenish_qty"], how="left")
    readable = readable.merge(rename_model(phase6, "phase6"), on=["sku_id", "anchor_date", "true_replenish_qty"], how="left")
    readable = readable.merge(rename_model(lstm, "lstm"), on=["sku_id", "anchor_date", "true_replenish_qty"], how="left")
    readable = readable.merge(static_df, on="sku_id", how="left", suffixes=("_ctx", ""))

    for col in ("category", "style_id", "season", "series", "band"):
        ctx_col = f"{col}_ctx"
        if ctx_col in readable.columns:
            readable[col] = readable[col].combine_first(readable[ctx_col])
            readable = readable.drop(columns=[ctx_col])

    for prefix in MODEL_ORDER:
        readable[f"ratio_{prefix}"] = np.where(
            readable["true_replenish_qty"].astype(float) > 0,
            readable[f"pred_{prefix}"].astype(float) / readable["true_replenish_qty"].astype(float),
            np.nan,
        )

    true_qty = readable["true_replenish_qty"].astype(float)
    pred_phase7 = readable["pred_phase7"].astype(float)
    readable["delta_abserr_phase7_vs_phase6"] = readable["abs_error_phase6"].astype(float) - readable["abs_error_phase7"].astype(float)
    readable["delta_abserr_phase7_vs_lstm"] = readable["abs_error_lstm"].astype(float) - readable["abs_error_phase7"].astype(float)
    readable["error_direction_phase7"] = np.select(
        [(true_qty <= 0) & (pred_phase7 <= 0), pred_phase7 < true_qty, pred_phase7 > true_qty],
        ["exact_zero", "under", "over"],
        default="exact",
    )
    readable["is_true_blockbuster"] = (true_qty > 25).astype(int)
    readable["is_pred_blockbuster_phase7"] = (pred_phase7 > 25).astype(int)
    readable["is_cold_start"] = (readable["lookback_repl_days_90"].astype(float) <= 0).astype(int)
    readable["is_repl0_fut0"] = (readable["signal_quadrant"].astype(str) == "repl0_fut0").astype(int)
    readable["true_bucket"] = true_bucket(true_qty)
    readable["qfo_bucket"] = qfo_bucket(readable["qty_first_order"])
    readable["rank_true_desc"] = true_qty.rank(method="dense", ascending=False).astype(int)
    readable["rank_pred_phase7_desc"] = pred_phase7.rank(method="dense", ascending=False).astype(int)
    readable["rank_abs_error_phase7_desc"] = readable["abs_error_phase7"].astype(float).rank(method="dense", ascending=False).astype(int)

    csv_columns = [
        "sku_id", "product_name", "category", "sub_category", "style_id", "season", "series", "band", "size_id", "color_id",
        "qty_first_order", "price_tag", "anchor_date", "signal_quadrant", "activity_bucket", "true_bucket", "qfo_bucket",
        "true_replenish_qty", "pred_phase7", "pred_phase6", "pred_lstm", "prob_phase7", "prob_phase6", "prob_lstm",
        "ratio_phase7", "ratio_phase6", "ratio_lstm", "abs_error_phase7", "abs_error_phase6", "abs_error_lstm",
        "delta_abserr_phase7_vs_phase6", "delta_abserr_phase7_vs_lstm", "error_direction_phase7",
        "lookback_repl_days_90", "lookback_future_days_90", "lookback_repl_sum_90", "lookback_future_sum_90",
        "rank_true_desc", "rank_pred_phase7_desc", "rank_abs_error_phase7_desc", "is_true_blockbuster",
        "is_pred_blockbuster_phase7", "is_cold_start", "is_repl0_fut0",
    ]
    readable = readable[csv_columns].sort_values(["rank_true_desc", "rank_abs_error_phase7_desc", "sku_id"]).reset_index(drop=True)

    metrics = {
        "phase7": evaluate_context_frame(phase7, "phase7"),
        "phase6": evaluate_context_frame(phase6, "phase6"),
        "lstm": evaluate_context_frame(lstm, "lstm"),
    }

    true_top = readable.sort_values(["true_replenish_qty", "abs_error_phase7"], ascending=[False, False]).head(30)
    pred_top = readable.sort_values(["pred_phase7", "true_replenish_qty"], ascending=[False, False]).head(30)
    blockbuster_top = readable[readable["true_replenish_qty"].astype(float) > 25].sort_values(["true_replenish_qty", "abs_error_phase7"], ascending=[False, False]).head(20)
    best_vs_phase6, best_vs_lstm, regress_vs_phase6, hardest = improvement_tables(readable)
    quadrant_df = group_compare(readable, "signal_quadrant", order=["repl0_fut0", "repl0_fut1", "repl1_fut0", "repl1_fut1"])
    qfo_df = group_compare(readable, "qfo_bucket", order=["0", "1", "2-10", "11-30", "31-100", "100+"], filter_mask=(readable["is_cold_start"] == 1) & (readable["true_replenish_qty"].astype(float) > 0))
    true_bucket_df = group_compare(readable, "true_bucket", order=["0", "1-3", "4-10", "11-25", "26-50", "50+"])
    worst_under_cat, worst_ratio_cat = category_compare(readable)

    metrics_df = pd.DataFrame([
        {"metric": "Global WMAPE", "rule": "lower", "phase7": metrics["phase7"]["global_wmape"], "phase6": metrics["phase6"]["global_wmape"], "lstm": metrics["lstm"]["global_wmape"]},
        {"metric": "Global Ratio", "rule": "target=1", "phase7": metrics["phase7"]["global_ratio"], "phase6": metrics["phase6"]["global_ratio"], "lstm": metrics["lstm"]["global_ratio"]},
        {"metric": "4-25 Under WAPE", "rule": "lower", "phase7": metrics["phase7"]["4_25_under_wape"], "phase6": metrics["phase6"]["4_25_under_wape"], "lstm": metrics["lstm"]["4_25_under_wape"]},
        {"metric": "4-25 SKU P50", "rule": "higher", "phase7": metrics["phase7"]["4_25_sku_p50"], "phase6": metrics["phase6"]["4_25_sku_p50"], "lstm": metrics["lstm"]["4_25_sku_p50"]},
        {"metric": "Ice 4-25 SKU P50", "rule": "higher", "phase7": metrics["phase7"]["ice_4_25_sku_p50"], "phase6": metrics["phase6"]["ice_4_25_sku_p50"], "lstm": metrics["lstm"]["ice_4_25_sku_p50"]},
        {"metric": ">25 Under WAPE", "rule": "lower", "phase7": metrics["phase7"]["blockbuster_under_wape"], "phase6": metrics["phase6"]["blockbuster_under_wape"], "lstm": metrics["lstm"]["blockbuster_under_wape"]},
        {"metric": ">25 SKU P50", "rule": "higher", "phase7": metrics["phase7"]["blockbuster_sku_p50"], "phase6": metrics["phase6"]["blockbuster_sku_p50"], "lstm": metrics["lstm"]["blockbuster_sku_p50"]},
        {"metric": "Top20 Capture", "rule": "higher", "phase7": metrics["phase7"]["top20_true_volume_capture"], "phase6": metrics["phase6"]["top20_true_volume_capture"], "lstm": metrics["lstm"]["top20_true_volume_capture"]},
        {"metric": "Rank Corr", "rule": "higher", "phase7": metrics["phase7"]["rank_corr_positive_skus"], "phase6": metrics["phase6"]["rank_corr_positive_skus"], "lstm": metrics["lstm"]["rank_corr_positive_skus"]},
        {"metric": "1-3 Ratio", "rule": "guardrail", "phase7": metrics["phase7"]["1_3_ratio"], "phase6": metrics["phase6"]["1_3_ratio"], "lstm": metrics["lstm"]["1_3_ratio"]},
    ])
    metrics_df["delta_vs_phase6"] = metrics_df["phase7"] - metrics_df["phase6"]
    metrics_df["delta_vs_lstm"] = metrics_df["phase7"] - metrics_df["lstm"]

    model_meta_df = pd.DataFrame([
        {"版本": "Phase7 当前最强", "模型": phase7_frozen["mainline_candidate"], "口径": "2025-12 raw 视图", "来源": str(phase7_path)},
        {"版本": "Phase6 冻结主线", "模型": "p57_covact_lr005_l63_hard_g025 + sep098_oct093", "口径": "2025-12 raw 视图", "来源": str(PHASE6_PATH)},
        {"版本": "最佳 LSTM", "模型": f"p527_lstm_l3_v5_lite_s2027_s{best_lstm_seed}", "口径": "2025-12 raw 视图", "来源": str(lstm_path)},
    ])

    top_true_rows = [{"label": short_label(r), "true": r["true_replenish_qty"], "phase7": r["pred_phase7"], "phase6": r["pred_phase6"], "lstm": r["pred_lstm"]} for _, r in true_top.iterrows()]
    top_pred_rows = [{"label": short_label(r), "true": r["true_replenish_qty"], "phase7": r["pred_phase7"], "phase6": r["pred_phase6"], "lstm": r["pred_lstm"]} for _, r in pred_top.iterrows()]
    blockbuster_rows = [{"label": short_label(r), "true": r["true_replenish_qty"], "phase7": r["pred_phase7"], "phase6": r["pred_phase6"], "lstm": r["pred_lstm"]} for _, r in blockbuster_top.iterrows()]
    detail_columns = ["sku_id", "product_name", "category", "style_id", "qty_first_order", "price_tag", "signal_quadrant", "activity_bucket", "true_bucket", "true_replenish_qty", "pred_phase7", "pred_phase6", "pred_lstm", "ratio_phase7", "ratio_phase6", "ratio_lstm", "abs_error_phase7", "abs_error_phase6", "abs_error_lstm", "delta_abserr_phase7_vs_phase6", "delta_abserr_phase7_vs_lstm", "lookback_repl_sum_90", "lookback_future_sum_90", "rank_true_desc", "rank_pred_phase7_desc", "rank_abs_error_phase7_desc", "is_true_blockbuster", "is_cold_start"]
    detail_rows = [[sanitize(row[col]) for col in detail_columns] for _, row in readable[detail_columns].iterrows()]

    payload = sanitize({
        "report_title": "2025-12 详细模型对照看板：Phase7 vs Phase6 vs 最佳 LSTM",
        "row_count": int(len(readable)),
        "best_lstm_seed": best_lstm_seed,
        "removed_missing_product_rows": 100,
        "phase7_svg": svg_heatmap(readable, "pred_phase7", "Phase7：真实 vs 预测密度图", MODEL_COLORS["phase7"]),
        "phase6_svg": svg_heatmap(readable, "pred_phase6", "Phase6：真实 vs 预测密度图", MODEL_COLORS["phase6"]),
        "lstm_svg": svg_heatmap(readable, "pred_lstm", "LSTM：真实 vs 预测密度图", MODEL_COLORS["lstm"]),
        "top_true_svg": svg_grouped_hbar(top_true_rows, ["true", "phase7", "phase6", "lstm"], ["真实", "Phase7", "Phase6", "LSTM"], [MODEL_COLORS["true"], MODEL_COLORS["phase7"], MODEL_COLORS["phase6"], MODEL_COLORS["lstm"]], "真实补货量 Top30 SKU"),
        "top_pred_svg": svg_grouped_hbar(top_pred_rows, ["true", "phase7", "phase6", "lstm"], ["真实", "Phase7", "Phase6", "LSTM"], [MODEL_COLORS["true"], MODEL_COLORS["phase7"], MODEL_COLORS["phase6"], MODEL_COLORS["lstm"]], "Phase7 预测量 Top30 SKU"),
        "blockbuster_svg": svg_grouped_hbar(blockbuster_rows, ["true", "phase7", "phase6", "lstm"], ["真实", "Phase7", "Phase6", "LSTM"], [MODEL_COLORS["true"], MODEL_COLORS["phase7"], MODEL_COLORS["phase6"], MODEL_COLORS["lstm"]], "真实 Blockbuster Top20 SKU"),
        "quadrant_svg": svg_grouped_vbar([{"label": r["signal_quadrant"], "phase7": r["phase7_under_wape"], "phase6": r["phase6_under_wape"], "lstm": r["lstm_under_wape"]} for _, r in quadrant_df.iterrows()], ["phase7", "phase6", "lstm"], ["Phase7", "Phase6", "LSTM"], [MODEL_COLORS["phase7"], MODEL_COLORS["phase6"], MODEL_COLORS["lstm"]], "四象限 Under WAPE 对比"),
        "qfo_svg": svg_grouped_vbar([{"label": r["qfo_bucket"], "true": r["true_mean"], "phase7": r["phase7_pred_mean"], "phase6": r["phase6_pred_mean"], "lstm": r["lstm_pred_mean"]} for _, r in qfo_df.iterrows()], ["true", "phase7", "phase6", "lstm"], ["真实均值", "Phase7", "Phase6", "LSTM"], [MODEL_COLORS["true"], MODEL_COLORS["phase7"], MODEL_COLORS["phase6"], MODEL_COLORS["lstm"]], "冷启动样本按首单量分桶"),
        "cat_under_svg": svg_single_hbar([{"category": r["category"], "value": r["phase7_under_wape"]} for _, r in worst_under_cat.iterrows()], "category", "value", "Phase7 最差类别：Under WAPE", color="#b91c1c"),
        "cat_ratio_svg": svg_single_hbar([{"category": r["category"], "value": abs(math.log(max(float(r["phase7_ratio"]), 1e-9)))} for _, r in worst_ratio_cat.iterrows()], "category", "value", "Phase7 最差类别：Ratio 偏离强度", color="#7c3aed"),
        "improve_phase6_svg": svg_single_hbar([{"label": short_label(r), "delta": r["delta_abserr_phase7_vs_phase6"]} for _, r in best_vs_phase6.iterrows()], "label", "delta", "Phase7 相比 Phase6：绝对误差改善 Top20", signed=True),
        "improve_lstm_svg": svg_single_hbar([{"label": short_label(r), "delta": r["delta_abserr_phase7_vs_lstm"]} for _, r in best_vs_lstm.iterrows()], "label", "delta", "Phase7 相比 LSTM：绝对误差改善 Top20", signed=True),
        "ratio_hist_svg": svg_ratio_hist(readable, "正样本 ratio 分布（截断到 0~3）"),
        "metrics_table_html": html_table(metrics_df[["metric", "rule", "phase7", "phase6", "lstm", "delta_vs_phase6", "delta_vs_lstm"]]),
        "model_meta_html": html_table(model_meta_df),
        "lstm_seed_html": html_table(lstm_seed_df[["seed", "global_wmape", "4_25_under_wape", "4_25_sku_p50", "ice_4_25_sku_p50", "blockbuster_under_wape", "blockbuster_sku_p50", "top20_true_volume_capture", "rank_corr_positive_skus"]]),
        "true_bucket_html": html_table(true_bucket_df),
        "quadrant_html": html_table(quadrant_df),
        "qfo_html": html_table(qfo_df),
        "cat_under_html": html_table(worst_under_cat[["category", "rows", "true_sum", "phase7_under_wape", "phase6_under_wape", "lstm_under_wape"]]),
        "cat_ratio_html": html_table(worst_ratio_cat[["category", "rows", "true_sum", "phase7_ratio", "phase6_ratio", "lstm_ratio"]]),
        "hardest_html": html_table(hardest[["sku_id", "product_name", "category", "true_replenish_qty", "pred_phase7", "pred_phase6", "pred_lstm", "abs_error_phase7", "signal_quadrant", "qty_first_order"]]),
        "regress_html": html_table(regress_vs_phase6[["sku_id", "product_name", "category", "true_replenish_qty", "abs_error_phase7", "abs_error_phase6", "delta_abserr_phase7_vs_phase6"]]),
        "detail_columns": detail_columns,
        "detail_rows": detail_rows,
        "summary_lines": [
            f"当前三线对比已经统一到 cleaned gold，12 月有效行数为 {len(readable)}。",
            f"最佳 LSTM 不是沿用旧 summary，而是在 cleaned gold 口径下重新比较 p527 三个 seed 后选中 s{best_lstm_seed}。",
            f"当前最强版本为 {phase7_frozen['mainline_candidate']}。",
            "这份页面同时展示总体指标、结构性切片、个案改进和完整逐 SKU 明细。",
        ],
    })

    readable.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    OUT_HTML.write_text(render_html(payload), encoding="utf-8")
    OUT_SUMMARY.write_text(build_summary(metrics_df, model_meta_df, best_lstm_seed, phase7_frozen["mainline_candidate"]), encoding="utf-8")
    print(json.dumps({"csv": str(OUT_CSV), "html": str(OUT_HTML), "summary": str(OUT_SUMMARY), "rows": int(len(readable)), "best_lstm_seed": best_lstm_seed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
