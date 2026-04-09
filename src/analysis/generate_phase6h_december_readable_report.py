import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from phase_eval_utils import evaluate_context_frame


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase6h_december_readable_report")
OUT_CSV = os.path.join(OUT_DIR, "dec_20251201_sku_compare_readable.csv")
OUT_HTML = os.path.join(OUT_DIR, "dec_20251201_dashboard.html")
OUT_AUDIT = os.path.join(OUT_DIR, "dec_20251201_static_feature_audit.md")

STATIC_PATH = os.path.join(PROJECT_ROOT, "data", "gold", "wide_table_sku.csv")
MAINLINE_PATH = os.path.join(
    PROJECT_ROOT,
    "reports",
    "phase5_7",
    "20251201",
    "phase5",
    "eval_context_p57a_20251201_covact_lr005_l63_s2026_hard_g025.csv",
)
P535_PATH = os.path.join(
    PROJECT_ROOT,
    "reports",
    "phase5_5",
    "20251201",
    "phase5",
    "eval_context_p55_20251201_p535_tree_hard_cov_activity_s2026.csv",
)
P527_PATH = os.path.join(
    PROJECT_ROOT,
    "reports",
    "phase5_4",
    "phase5",
    "eval_context_p54_p527_lstm_l3_v5_lite_s2027_s2026.csv",
)

MODEL_SPECS = {
    "mainline": {
        "label": "Current Mainline",
        "track": "p57_covact_lr005_l63_hard_g025",
        "path": MAINLINE_PATH,
        "color": "#0f766e",
    },
    "p535": {
        "label": "Confirmed Tree",
        "track": "p535_tree_hard_cov_activity",
        "path": P535_PATH,
        "color": "#2563eb",
    },
    "p527": {
        "label": "Sequence Baseline",
        "track": "p527_lstm_l3_v5_lite_s2027",
        "path": P527_PATH,
        "color": "#dc2626",
    },
}

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

CSV_COL_ORDER = [
    "sku_id",
    "product_name",
    "category",
    "sub_category",
    "style_id",
    "season",
    "series",
    "band",
    "size_id",
    "color_id",
    "qty_first_order",
    "price_tag",
    "anchor_date",
    "true_replenish_qty",
    "pred_mainline",
    "pred_p535",
    "pred_p527",
    "prob_mainline",
    "prob_p535",
    "prob_p527",
    "ratio_mainline",
    "ratio_p535",
    "ratio_p527",
    "abs_error_mainline",
    "abs_error_p535",
    "abs_error_p527",
    "error_direction_mainline",
    "signal_quadrant",
    "activity_bucket",
    "lookback_repl_days_90",
    "lookback_future_days_90",
    "lookback_repl_sum_90",
    "lookback_future_sum_90",
    "rank_true_desc",
    "rank_pred_mainline_desc",
    "rank_abs_error_mainline_desc",
    "is_true_blockbuster",
    "is_pred_blockbuster_mainline",
    "is_cold_start",
    "is_repl0_fut0",
]

AUDIT_SEMANTICS = {
    "sku_id": ("LabelEncoder", "SKU 唯一键。当前更像记忆型 ID，不保留语义距离。"),
    "style_id": ("LabelEncoder", "同款 ID。存在分组语义，但当前编码不保留款式间距离。"),
    "product_name": ("LabelEncoder", "商品名称文本，当前仅作为离散标签使用。"),
    "category": ("LabelEncoder", "一级类目。存在分组语义，但当前编码不保留类目间距离。"),
    "sub_category": ("LabelEncoder", "二级类目。存在分组语义，但当前编码不保留距离。"),
    "season": ("LabelEncoder", "季节标签。分组语义明确。"),
    "series": ("LabelEncoder", "系列标签。分组语义明确。"),
    "band": ("LabelEncoder", "带宽/波段标签。分组语义明确。"),
    "size_id": ("LabelEncoder", "尺码字段原始上有顺序语义，但当前标签编码丢失顺序。"),
    "color_id": ("LabelEncoder", "颜色字段，当前作为纯类别标签使用。"),
    "qty_first_order": ("log1p(max(0,x))", "首单量。是当前最重要的冷启动静态数值特征之一。"),
    "price_tag": ("log1p(max(0,x))", "吊牌价。保留数值语义。"),
    "month": ("anchor_date.month", "运行时写入的锚点月份，不来自原始静态表。"),
}


def ensure_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def load_eval_context(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["sku_id"] = df["sku_id"].astype(str)
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.strftime("%Y-%m-%d")
    return df


def load_static_table() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(STATIC_PATH):
        raise FileNotFoundError(STATIC_PATH)
    raw = pd.read_csv(STATIC_PATH, usecols=STATIC_COLS)
    raw["sku_id"] = raw["sku_id"].astype(str)
    return raw, raw.drop_duplicates("sku_id").copy()


def rename_model_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keep = [
        "sku_id",
        "anchor_date",
        "true_replenish_qty",
        "ai_pred_prob",
        "ai_pred_qty",
        "abs_error",
    ]
    out = df[keep].copy()
    return out.rename(
        columns={
            "ai_pred_prob": f"prob_{prefix}",
            "ai_pred_qty": f"pred_{prefix}",
            "abs_error": f"abs_error_{prefix}",
        }
    )


def validate_truth(base: pd.DataFrame, other: pd.DataFrame, other_name: str) -> None:
    merged = base.merge(other, on=["sku_id", "anchor_date"], how="inner", suffixes=("_base", "_other"))
    if len(merged) != len(base):
        raise ValueError(f"{other_name} row count mismatch after merge: {len(merged)} vs {len(base)}")
    mismatch = ~np.isclose(
        merged["true_replenish_qty_base"].astype(float),
        merged["true_replenish_qty_other"].astype(float),
        atol=1e-6,
        rtol=0.0,
    )
    if mismatch.any():
        raise ValueError(f"{other_name} true_replenish_qty mismatch on {int(mismatch.sum())} rows")


def bucket_qty_first_order(df: pd.DataFrame) -> pd.Series:
    values = df["qty_first_order"].fillna(0).astype(float)
    return pd.cut(
        values,
        bins=[-0.01, 0.0, 1.0, 10.0, 30.0, 100.0, np.inf],
        labels=["0", "1", "2-10", "11-30", "31-100", "100+"],
    ).astype(str)


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(k): sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(v) for v in value]
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


def short_label(row: pd.Series) -> str:
    product = str(row.get("product_name") or "")[:8]
    return f"{row['sku_id']} | {product}" if product else str(row["sku_id"])


def build_readable_frame() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, Dict]]:
    raw_static, static_df = load_static_table()
    model_frames = {key: load_eval_context(spec["path"]) for key, spec in MODEL_SPECS.items()}

    base = model_frames["mainline"].copy()
    for other_key in ("p535", "p527"):
        validate_truth(base, model_frames[other_key], other_key)

    shared_cols = [
        "sku_id",
        "anchor_date",
        "true_replenish_qty",
        "signal_quadrant",
        "activity_bucket",
        "lookback_repl_days_90",
        "lookback_future_days_90",
        "lookback_repl_sum_90",
        "lookback_future_sum_90",
        "category",
        "style_id",
        "season",
        "series",
        "band",
    ]
    readable = base[shared_cols].copy()
    readable = readable.merge(
        rename_model_cols(model_frames["mainline"], "mainline"),
        on=["sku_id", "anchor_date", "true_replenish_qty"],
        how="left",
    )
    readable = readable.merge(
        rename_model_cols(model_frames["p535"], "p535"),
        on=["sku_id", "anchor_date", "true_replenish_qty"],
        how="left",
    )
    readable = readable.merge(
        rename_model_cols(model_frames["p527"], "p527"),
        on=["sku_id", "anchor_date", "true_replenish_qty"],
        how="left",
    )
    readable = readable.merge(static_df, on="sku_id", how="left", suffixes=("_ctx", ""))

    for col in ("category", "style_id", "season", "series", "band"):
        ctx_col = f"{col}_ctx"
        if ctx_col in readable.columns:
            readable[col] = readable[col].combine_first(readable[ctx_col])
            readable = readable.drop(columns=[ctx_col])

    for prefix in ("mainline", "p535", "p527"):
        ratio_col = f"ratio_{prefix}"
        pred_col = f"pred_{prefix}"
        readable[ratio_col] = np.where(
            readable["true_replenish_qty"].astype(float) > 0,
            readable[pred_col].astype(float) / readable["true_replenish_qty"].astype(float),
            np.nan,
        )

    true_qty = readable["true_replenish_qty"].astype(float)
    pred_main = readable["pred_mainline"].astype(float)
    readable["error_direction_mainline"] = np.select(
        [
            (true_qty <= 0) & (pred_main <= 0),
            pred_main < true_qty,
            pred_main > true_qty,
        ],
        ["exact_zero", "under", "over"],
        default="exact",
    )
    readable["is_true_blockbuster"] = (true_qty > 25).astype(int)
    readable["is_pred_blockbuster_mainline"] = (pred_main > 25).astype(int)
    readable["is_cold_start"] = (readable["lookback_repl_days_90"].astype(float) <= 0).astype(int)
    readable["is_repl0_fut0"] = (readable["signal_quadrant"].astype(str) == "repl0_fut0").astype(int)

    readable["rank_true_desc"] = true_qty.rank(method="dense", ascending=False).astype(int)
    readable["rank_pred_mainline_desc"] = pred_main.rank(method="dense", ascending=False).astype(int)
    readable["rank_abs_error_mainline_desc"] = readable["abs_error_mainline"].astype(float).rank(method="dense", ascending=False).astype(int)

    readable = readable.sort_values(
        ["rank_true_desc", "rank_abs_error_mainline_desc", "sku_id"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    for col in CSV_COL_ORDER:
        if col not in readable.columns:
            readable[col] = np.nan
    readable = readable[CSV_COL_ORDER]

    metrics = {}
    for key, frame in model_frames.items():
        metrics[key] = evaluate_context_frame(frame, key)
    return readable, {"raw_static": raw_static, "dedup_static": static_df, **model_frames}, metrics


def build_dashboard_payload(readable: pd.DataFrame, model_metrics: Dict[str, Dict]) -> Dict:
    top_true = readable.sort_values(["true_replenish_qty", "abs_error_mainline"], ascending=[False, False]).head(30).copy()
    top_pred = readable.sort_values(["pred_mainline", "true_replenish_qty"], ascending=[False, False]).head(30).copy()
    top_error = readable.sort_values(["abs_error_mainline", "true_replenish_qty"], ascending=[False, False]).head(30).copy()
    top_blockbuster = readable[readable["true_replenish_qty"].astype(float) > 25].sort_values(
        ["true_replenish_qty", "abs_error_mainline"], ascending=[False, False]
    ).head(20).copy()

    worst_repl0_ice = readable[
        (readable["is_repl0_fut0"] == 1)
        & (readable["is_cold_start"] == 1)
        & (readable["true_replenish_qty"].astype(float) > 10)
    ].sort_values(["abs_error_mainline", "true_replenish_qty"], ascending=[False, False]).head(12)

    over_small = readable[
        (readable["true_replenish_qty"].astype(float) <= 3)
        & (readable["pred_mainline"].astype(float) > readable["true_replenish_qty"].astype(float))
    ].sort_values(["ratio_mainline", "abs_error_mainline"], ascending=[False, False]).head(12)

    quadrant = (
        readable.groupby("signal_quadrant", dropna=False)
        .agg(
            rows=("sku_id", "size"),
            true_mean=("true_replenish_qty", "mean"),
            pred_mean=("pred_mainline", "mean"),
            pred_p535_mean=("pred_p535", "mean"),
            pred_p527_mean=("pred_p527", "mean"),
        )
        .reset_index()
    )
    quadrant_order = ["repl0_fut0", "repl0_fut1", "repl1_fut0", "repl1_fut1"]
    quadrant["ord"] = quadrant["signal_quadrant"].map({q: i for i, q in enumerate(quadrant_order)}).fillna(99)
    quadrant = quadrant.sort_values("ord").drop(columns=["ord"])

    cold_pos = readable[(readable["is_cold_start"] == 1) & (readable["true_replenish_qty"].astype(float) > 0)].copy()
    cold_pos["qfo_bucket"] = bucket_qty_first_order(cold_pos)
    qfo_bucket = (
        cold_pos.groupby("qfo_bucket", dropna=False)
        .agg(
            rows=("sku_id", "size"),
            true_mean=("true_replenish_qty", "mean"),
            pred_mainline_mean=("pred_mainline", "mean"),
            pred_p535_mean=("pred_p535", "mean"),
            pred_p527_mean=("pred_p527", "mean"),
            ratio_mainline_p50=("ratio_mainline", "median"),
        )
        .reset_index()
    )
    bucket_order = ["0", "1", "2-10", "11-30", "31-100", "100+"]
    qfo_bucket["ord"] = qfo_bucket["qfo_bucket"].map({b: i for i, b in enumerate(bucket_order)}).fillna(99)
    qfo_bucket = qfo_bucket.sort_values("ord").drop(columns=["ord"])

    category_ratio = (
        readable.groupby("category", dropna=False)
        .agg(
            total_true=("true_replenish_qty", "sum"),
            total_pred_mainline=("pred_mainline", "sum"),
        )
        .reset_index()
    )
    category_ratio = category_ratio[category_ratio["total_true"] > 0].copy()
    category_ratio["ratio_mainline"] = category_ratio["total_pred_mainline"] / category_ratio["total_true"]
    category_ratio["ratio_score"] = np.abs(np.log(np.clip(category_ratio["ratio_mainline"], 1e-9, None)))
    category_ratio = category_ratio.sort_values(["ratio_score", "total_true"], ascending=[False, False]).head(10)

    category_under = readable.copy()
    category_under["under_mainline"] = np.clip(
        category_under["true_replenish_qty"].astype(float) - category_under["pred_mainline"].astype(float),
        a_min=0,
        a_max=None,
    )
    category_under = (
        category_under.groupby("category", dropna=False)
        .agg(total_true=("true_replenish_qty", "sum"), under_sum=("under_mainline", "sum"))
        .reset_index()
    )
    category_under = category_under[category_under["total_true"] > 0].copy()
    category_under["under_wape"] = category_under["under_sum"] / category_under["total_true"]
    category_under = category_under.sort_values(["under_wape", "total_true"], ascending=[False, False]).head(10)

    summary_cards = [
        {"label": "4-25 Under WAPE", "mainline": model_metrics["mainline"]["4_25_under_wape"], "p535": model_metrics["p535"]["4_25_under_wape"], "p527": model_metrics["p527"]["4_25_under_wape"]},
        {"label": "4-25 SKU P50", "mainline": model_metrics["mainline"]["4_25_sku_p50"], "p535": model_metrics["p535"]["4_25_sku_p50"], "p527": model_metrics["p527"]["4_25_sku_p50"]},
        {"label": "Ice 4-25 SKU P50", "mainline": model_metrics["mainline"]["ice_4_25_sku_p50"], "p535": model_metrics["p535"]["ice_4_25_sku_p50"], "p527": model_metrics["p527"]["ice_4_25_sku_p50"]},
        {"label": "Blockbuster Under WAPE", "mainline": model_metrics["mainline"]["blockbuster_under_wape"], "p535": model_metrics["p535"]["blockbuster_under_wape"], "p527": model_metrics["p527"]["blockbuster_under_wape"]},
        {"label": "Blockbuster SKU P50", "mainline": model_metrics["mainline"]["blockbuster_sku_p50"], "p535": model_metrics["p535"]["blockbuster_sku_p50"], "p527": model_metrics["p527"]["blockbuster_sku_p50"]},
        {"label": "Top20 Capture", "mainline": model_metrics["mainline"]["top20_true_volume_capture"], "p535": model_metrics["p535"]["top20_true_volume_capture"], "p527": model_metrics["p527"]["top20_true_volume_capture"]},
        {"label": "Global Ratio", "mainline": model_metrics["mainline"]["global_ratio"], "p535": model_metrics["p535"]["global_ratio"], "p527": model_metrics["p527"]["global_ratio"]},
    ]

    callout_lines = [
        f"主线 vs p527: 4-25 Under WAPE 从 {model_metrics['p527']['4_25_under_wape']:.4f} 降到 {model_metrics['mainline']['4_25_under_wape']:.4f}",
        f"主线 vs p527: Ice 4-25 SKU P50 从 {model_metrics['p527']['ice_4_25_sku_p50']:.4f} 提到 {model_metrics['mainline']['ice_4_25_sku_p50']:.4f}",
        f"主线当前短板: Blockbuster Under WAPE = {model_metrics['mainline']['blockbuster_under_wape']:.4f}, Blockbuster SKU P50 = {model_metrics['mainline']['blockbuster_sku_p50']:.4f}",
        "当前 12 月主线最典型漏补集中在 repl0_fut0 + ice 的高首单 SKU",
    ]

    payload = {
        "report_title": "2025-12-01 三线对比业务可读看板",
        "summary_cards": summary_cards,
        "models": {k: {"label": v["label"], "track": v["track"], "color": v["color"]} for k, v in MODEL_SPECS.items()},
        "top_true": [{"label": short_label(row), "true": row["true_replenish_qty"], "mainline": row["pred_mainline"], "p535": row["pred_p535"], "p527": row["pred_p527"]} for _, row in top_true.iterrows()],
        "top_pred": [{"label": short_label(row), "true": row["true_replenish_qty"], "mainline": row["pred_mainline"], "p535": row["pred_p535"], "p527": row["pred_p527"]} for _, row in top_pred.iterrows()],
        "top_error": [{"label": short_label(row), "true": row["true_replenish_qty"], "mainline": row["pred_mainline"], "abs_error_mainline": row["abs_error_mainline"], "quadrant": row["signal_quadrant"]} for _, row in top_error.iterrows()],
        "blockbuster_top": [{"label": short_label(row), "true": row["true_replenish_qty"], "mainline": row["pred_mainline"], "p535": row["pred_p535"], "p527": row["pred_p527"], "qfo": row["qty_first_order"]} for _, row in top_blockbuster.iterrows()],
        "quadrant_summary": quadrant.to_dict(orient="records"),
        "qfo_bucket": qfo_bucket.to_dict(orient="records"),
        "category_ratio": category_ratio[["category", "ratio_mainline", "total_true"]].to_dict(orient="records"),
        "category_under": category_under[["category", "under_wape", "total_true"]].to_dict(orient="records"),
        "worst_repl0_ice": worst_repl0_ice[["sku_id", "product_name", "category", "qty_first_order", "true_replenish_qty", "pred_mainline", "ratio_mainline", "abs_error_mainline"]].to_dict(orient="records"),
        "over_small": over_small[["sku_id", "product_name", "category", "true_replenish_qty", "pred_mainline", "ratio_mainline", "abs_error_mainline"]].to_dict(orient="records"),
        "callout_lines": callout_lines,
        "table_columns": list(readable.columns),
        "table_rows": readable.to_dict(orient="records"),
    }
    return sanitize_for_json(payload)


def render_html(payload: Dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{payload['report_title']}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #f6f7fb; --panel: #ffffff; --ink: #0f172a; --sub: #475569; --line: #dbe2ea;
      --main: #0f766e; --p535: #2563eb; --p527: #dc2626; --true: #111827; --accent: #ecfeff;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", "PingFang SC", sans-serif; background: var(--bg); color: var(--ink); }}
    .wrap {{ max-width: 1600px; margin: 0 auto; padding: 24px; }}
    .hero {{ display: grid; grid-template-columns: 1.3fr 1fr; gap: 20px; margin-bottom: 20px; }}
    .hero-main, .hero-side, .panel, .kpi {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; box-shadow: 0 8px 24px rgba(15,23,42,.05); }}
    .hero-main, .hero-side, .panel {{ padding: 20px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .subtitle {{ color: var(--sub); line-height: 1.6; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }}
    .chip {{ background: var(--accent); color: var(--main); border: 1px solid #bfe8ea; border-radius: 999px; padding: 6px 10px; font-size: 12px; }}
    .callouts {{ margin: 0; padding-left: 18px; color: var(--sub); line-height: 1.7; }}
    .grid-kpi {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }}
    .kpi {{ padding: 16px; }}
    .kpi h3 {{ margin: 0 0 10px; font-size: 13px; color: var(--sub); }}
    .kpi .row {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
    .pill {{ background: #f8fafc; border-radius: 12px; padding: 10px; text-align: center; }}
    .pill .name {{ font-size: 11px; color: var(--sub); }}
    .pill .val {{ font-size: 20px; font-weight: 700; }}
    .charts-2, .tables-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
    .panel canvas {{ width: 100% !important; height: 420px !important; }}
    .small canvas {{ height: 300px !important; }}
    .section-title {{ margin: 0 0 12px; font-size: 18px; }}
    .legend {{ display: flex; gap: 14px; flex-wrap: wrap; color: var(--sub); font-size: 12px; margin-top: 6px; }}
    .legend span::before {{ content: ""; display: inline-block; width: 10px; height: 10px; border-radius: 999px; margin-right: 6px; vertical-align: middle; }}
    .lg-true::before {{ background: var(--true); }} .lg-main::before {{ background: var(--main); }} .lg-p535::before {{ background: var(--p535); }} .lg-p527::before {{ background: var(--p527); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #f8fafc; position: sticky; top: 0; }}
    .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .table-wrap {{ max-height: 360px; overflow: auto; }}
    .controls {{ display: flex; gap: 10px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }}
    .controls input, .controls select {{ padding: 8px 10px; border: 1px solid var(--line); border-radius: 10px; background: #fff; }}
    .pager {{ display: flex; align-items: center; gap: 8px; margin-top: 12px; }}
    .pager button {{ border: 1px solid var(--line); background: white; padding: 6px 10px; border-radius: 8px; cursor: pointer; }}
    .tag {{ padding: 2px 8px; border-radius: 999px; background: #eef2ff; color: #3730a3; font-size: 11px; }}
    @media (max-width: 1200px) {{ .hero, .charts-2, .tables-2, .grid-kpi {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <div class="hero-main">
        <h1>{payload['report_title']}</h1>
        <div class="subtitle">当前 12 月业务可读版固定为三线对比：主线 `p57_covact_lr005_l63_hard_g025`、已确认 tree `p535`、baseline `p527`。这份报表只复用现有落盘结果，不新增训练。</div>
        <div class="chips">
          <span class="chip">Anchor: 2025-12-01</span>
          <span class="chip">Validation Rows: 10240</span>
          <span class="chip">Mainline = raw g025</span>
          <span class="chip">三线同口径对比</span>
        </div>
      </div>
      <div class="hero-side"><h3 class="section-title">本页要点</h3><ul class="callouts" id="callout-list"></ul></div>
    </div>
    <div class="grid-kpi" id="kpi-grid"></div>
    <div class="charts-2">
      <div class="panel"><h3 class="section-title">真实 Top30 SKU</h3><canvas id="topTrueChart"></canvas><div class="legend"><span class="lg-true">真实</span><span class="lg-main">主线</span><span class="lg-p535">p535</span><span class="lg-p527">p527</span></div></div>
      <div class="panel"><h3 class="section-title">主线预测 Top30 SKU</h3><canvas id="topPredChart"></canvas><div class="legend"><span class="lg-true">真实</span><span class="lg-main">主线</span><span class="lg-p535">p535</span><span class="lg-p527">p527</span></div></div>
    </div>
    <div class="charts-2">
      <div class="panel"><h3 class="section-title">主线误差最大 Top30</h3><canvas id="topErrorChart"></canvas></div>
      <div class="panel small"><h3 class="section-title">信号象限对比</h3><canvas id="quadrantChart"></canvas></div>
    </div>
    <div class="charts-2">
      <div class="panel"><h3 class="section-title">真实 Blockbuster Top20</h3><canvas id="blockbusterChart"></canvas></div>
      <div class="panel small"><h3 class="section-title">冷启动样本按首单量分桶</h3><canvas id="qfoChart"></canvas></div>
    </div>
    <div class="charts-2">
      <div class="panel small"><h3 class="section-title">最差类别：Under WAPE</h3><canvas id="catUnderChart"></canvas></div>
      <div class="panel small"><h3 class="section-title">最差类别：Ratio 偏离</h3><canvas id="catRatioChart"></canvas></div>
    </div>
    <div class="tables-2">
      <div class="panel"><h3 class="section-title">高价值漏补：repl0_fut0 + ice</h3><div class="table-wrap"><table id="worst-repl0-table"></table></div></div>
      <div class="panel"><h3 class="section-title">小需求高估样本</h3><div class="table-wrap"><table id="over-small-table"></table></div></div>
    </div>
    <div class="panel">
      <h3 class="section-title">完整 12 月逐 SKU 对照表</h3>
      <div class="controls"><input id="table-search" placeholder="搜索 sku/category/style/product_name ..." /><select id="table-page-size"><option value="20">20 行</option><option value="50" selected>50 行</option><option value="100">100 行</option></select><span class="tag">点击表头排序</span></div>
      <div class="table-wrap"><table id="detail-table"></table></div>
      <div class="pager"><button id="prev-page">上一页</button><span id="page-info"></span><button id="next-page">下一页</button></div>
    </div>
  </div>
  <script>
    const DATA = {data_json};
    const COLORS = {{ true: '#111827', mainline: '#0f766e', p535: '#2563eb', p527: '#dc2626' }};
    function fmt(v, digits=4) {{ if (v === null || v === undefined || Number.isNaN(Number(v))) return '-'; return Number(v).toFixed(digits); }}
    function buildKpis() {{
      const grid = document.getElementById('kpi-grid');
      DATA.summary_cards.forEach(card => {{
        const div = document.createElement('div');
        div.className = 'kpi';
        div.innerHTML = `<h3>${{card.label}}</h3><div class="row"><div class="pill"><div class="name">Mainline</div><div class="val">${{fmt(card.mainline)}}</div></div><div class="pill"><div class="name">p535</div><div class="val">${{fmt(card.p535)}}</div></div><div class="pill"><div class="name">p527</div><div class="val">${{fmt(card.p527)}}</div></div></div>`;
        grid.appendChild(div);
      }});
      const list = document.getElementById('callout-list');
      DATA.callout_lines.forEach(line => {{ const li = document.createElement('li'); li.textContent = line; list.appendChild(li); }});
    }}
    function makeBarChart(id, rows) {{
      new Chart(document.getElementById(id), {{
        type: 'bar',
        data: {{
          labels: rows.map(r => r.label),
          datasets: [
            {{ label: '真实', data: rows.map(r => r.true), backgroundColor: COLORS.true }},
            {{ label: 'Mainline', data: rows.map(r => r.mainline), backgroundColor: COLORS.mainline }},
            {{ label: 'p535', data: rows.map(r => r.p535), backgroundColor: COLORS.p535 }},
            {{ label: 'p527', data: rows.map(r => r.p527), backgroundColor: COLORS.p527 }},
          ]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, scales: {{ x: {{ ticks: {{ autoSkip: false, maxRotation: 70, minRotation: 70 }} }}, y: {{ beginAtZero: true }} }} }}
      }});
    }}
    function makeErrorChart() {{
      const rows = DATA.top_error;
      new Chart(document.getElementById('topErrorChart'), {{
        type: 'bar',
        data: {{
          labels: rows.map(r => `${{r.label}} | ${{r.quadrant}}`),
          datasets: [
            {{ label: '真实', data: rows.map(r => r.true), backgroundColor: COLORS.true }},
            {{ label: '主线预测', data: rows.map(r => r.mainline), backgroundColor: COLORS.mainline }},
            {{ label: '绝对误差', data: rows.map(r => r.abs_error_mainline), backgroundColor: '#f59e0b' }},
          ]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, scales: {{ x: {{ ticks: {{ autoSkip: false, maxRotation: 70, minRotation: 70 }} }}, y: {{ beginAtZero: true }} }} }}
      }});
    }}
    function makeQuadrantChart() {{
      const rows = DATA.quadrant_summary;
      new Chart(document.getElementById('quadrantChart'), {{
        type: 'bar',
        data: {{
          labels: rows.map(r => r.signal_quadrant),
          datasets: [
            {{ label: '样本数', data: rows.map(r => r.rows), backgroundColor: '#94a3b8', yAxisID: 'y' }},
            {{ label: '真实均值', data: rows.map(r => r.true_mean), backgroundColor: COLORS.true, yAxisID: 'y1' }},
            {{ label: '主线均值', data: rows.map(r => r.pred_mean), backgroundColor: COLORS.mainline, yAxisID: 'y1' }},
            {{ label: 'p535 均值', data: rows.map(r => r.pred_p535_mean), backgroundColor: COLORS.p535, yAxisID: 'y1' }},
            {{ label: 'p527 均值', data: rows.map(r => r.pred_p527_mean), backgroundColor: COLORS.p527, yAxisID: 'y1' }},
          ]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ type: 'linear', position: 'left', beginAtZero: true }}, y1: {{ type: 'linear', position: 'right', beginAtZero: true, grid: {{ drawOnChartArea: false }} }} }} }}
      }});
    }}
    function makeQfoChart() {{
      const rows = DATA.qfo_bucket;
      new Chart(document.getElementById('qfoChart'), {{
        type: 'bar',
        data: {{
          labels: rows.map(r => `${{r.qfo_bucket}} (n=${{r.rows}})`),
          datasets: [
            {{ label: '真实均值', data: rows.map(r => r.true_mean), backgroundColor: COLORS.true }},
            {{ label: 'Mainline', data: rows.map(r => r.pred_mainline_mean), backgroundColor: COLORS.mainline }},
            {{ label: 'p535', data: rows.map(r => r.pred_p535_mean), backgroundColor: COLORS.p535 }},
            {{ label: 'p527', data: rows.map(r => r.pred_p527_mean), backgroundColor: COLORS.p527 }},
          ]
        }},
        options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ beginAtZero: true }} }} }}
      }});
    }}
    function makeCategoryCharts() {{
      new Chart(document.getElementById('catUnderChart'), {{
        type: 'bar', data: {{ labels: DATA.category_under.map(r => r.category), datasets: [{{ label: 'Under WAPE', data: DATA.category_under.map(r => r.under_wape), backgroundColor: '#b91c1c' }}] }},
        options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false, scales: {{ x: {{ beginAtZero: true }} }} }}
      }});
      new Chart(document.getElementById('catRatioChart'), {{
        type: 'bar', data: {{ labels: DATA.category_ratio.map(r => r.category), datasets: [{{ label: 'Ratio', data: DATA.category_ratio.map(r => r.ratio_mainline), backgroundColor: '#7c3aed' }}] }},
        options: {{ indexAxis: 'y', responsive: true, maintainAspectRatio: false, scales: {{ x: {{ beginAtZero: true }} }} }}
      }});
    }}
    function renderSimpleTable(targetId, rows, cols) {{
      const table = document.getElementById(targetId);
      let html = '<thead><tr>' + cols.map(c => `<th>${{c}}</th>`).join('') + '</tr></thead><tbody>';
      rows.forEach(r => {{ html += '<tr>' + cols.map(c => {{ const v = r[c]; const isNum = typeof v === 'number'; return `<td class="${{isNum ? 'num' : ''}}">${{v === null || v === undefined ? '' : isNum ? fmt(v, 3) : String(v)}}</td>`; }}).join('') + '</tr>'; }});
      table.innerHTML = html + '</tbody>';
    }}
    let state = {{ rows: DATA.table_rows, page: 1, pageSize: 50, sortCol: 'rank_true_desc', sortAsc: true, search: '' }};
    function filteredRows() {{
      let rows = [...DATA.table_rows];
      if (state.search) {{ const q = state.search.toLowerCase(); rows = rows.filter(r => Object.values(r).some(v => String(v ?? '').toLowerCase().includes(q))); }}
      rows.sort((a, b) => {{ const av = a[state.sortCol], bv = b[state.sortCol]; const aNum = typeof av === 'number', bNum = typeof bv === 'number'; let cmp = 0; if (aNum && bNum) cmp = (av ?? -Infinity) - (bv ?? -Infinity); else cmp = String(av ?? '').localeCompare(String(bv ?? '')); return state.sortAsc ? cmp : -cmp; }});
      return rows;
    }}
    function renderDetailTable() {{
      const rows = filteredRows(), totalPages = Math.max(1, Math.ceil(rows.length / state.pageSize));
      if (state.page > totalPages) state.page = totalPages;
      const start = (state.page - 1) * state.pageSize, pageRows = rows.slice(start, start + state.pageSize), cols = DATA.table_columns;
      let html = '<thead><tr>' + cols.map(c => `<th data-col="${{c}}">${{c}}</th>`).join('') + '</tr></thead><tbody>';
      pageRows.forEach(r => {{ html += '<tr>' + cols.map(c => {{ const v = r[c]; const isNum = typeof v === 'number'; return `<td class="${{isNum ? 'num' : ''}}">${{v === null || v === undefined ? '' : isNum ? fmt(v, 4) : String(v)}}</td>`; }}).join('') + '</tr>'; }});
      const table = document.getElementById('detail-table');
      table.innerHTML = html + '</tbody>';
      table.querySelectorAll('th').forEach(th => {{ th.style.cursor = 'pointer'; th.onclick = () => {{ const col = th.dataset.col; if (state.sortCol === col) state.sortAsc = !state.sortAsc; else {{ state.sortCol = col; state.sortAsc = true; }} renderDetailTable(); }}; }});
      document.getElementById('page-info').textContent = `第 ${{state.page}} / ${{totalPages}} 页，共 ${{rows.length}} 行`;
    }}
    function wireControls() {{
      document.getElementById('table-search').addEventListener('input', e => {{ state.search = e.target.value.trim(); state.page = 1; renderDetailTable(); }});
      document.getElementById('table-page-size').addEventListener('change', e => {{ state.pageSize = Number(e.target.value); state.page = 1; renderDetailTable(); }});
      document.getElementById('prev-page').onclick = () => {{ state.page = Math.max(1, state.page - 1); renderDetailTable(); }};
      document.getElementById('next-page').onclick = () => {{ const totalPages = Math.max(1, Math.ceil(filteredRows().length / state.pageSize)); state.page = Math.min(totalPages, state.page + 1); renderDetailTable(); }};
    }}
    buildKpis();
    makeBarChart('topTrueChart', DATA.top_true);
    makeBarChart('topPredChart', DATA.top_pred);
    makeErrorChart();
    makeQuadrantChart();
    makeBarChart('blockbusterChart', DATA.blockbuster_top);
    makeQfoChart();
    makeCategoryCharts();
    renderSimpleTable('worst-repl0-table', DATA.worst_repl0_ice, ['sku_id', 'product_name', 'category', 'qty_first_order', 'true_replenish_qty', 'pred_mainline', 'ratio_mainline', 'abs_error_mainline']);
    renderSimpleTable('over-small-table', DATA.over_small, ['sku_id', 'product_name', 'category', 'true_replenish_qty', 'pred_mainline', 'ratio_mainline', 'abs_error_mainline']);
    wireControls();
    renderDetailTable();
  </script>
</body>
</html>
"""


def format_sample_values(values: List) -> str:
    return json.dumps([sanitize_for_json(v) for v in values], ensure_ascii=False)


def static_feature_audit_markdown(raw_static: pd.DataFrame, dedup_static: pd.DataFrame, readable: pd.DataFrame) -> str:
    lines = [
        "# 2025-12-01 静态特征审计",
        "",
        "这份文档基于 `wide_table_sku.csv` 的原始字段和当前 `v6_event` 特征代码，说明每个静态字段的原始形态、当前变换方式和语义价值。",
        "",
        "## 审计口径",
        "",
        f"- 原始表: `{STATIC_PATH}`",
        "- 去重口径: 按 `sku_id` 去重后审计静态字段，避免同 SKU 被日期/买手重复计数",
        "- 当前代码来源:",
        "  - `E:/LSTM/B2B/B2B_Replenishment_System/src/features/phase53_feature_utils.py`",
        "  - `E:/LSTM/B2B/B2B_Replenishment_System/src/features/build_features_v6_event_sku.py`",
        "",
        "## 静态字段总表",
        "",
        "| field | raw_dtype | distinct_count | missing_rate | transform | semantic_note | sample_values |",
        "| --- | --- | ---: | ---: | --- | --- | --- |",
    ]

    for col in STATIC_COLS:
        series = dedup_static[col]
        raw_dtype = str(raw_static[col].dtype)
        distinct_count = int(series.nunique(dropna=True))
        missing_rate = float(series.isna().mean())
        samples = series.dropna().astype(str).drop_duplicates().head(5).tolist()
        transform, note = AUDIT_SEMANTICS[col]
        lines.append(
            f"| `{col}` | `{raw_dtype}` | {distinct_count} | {missing_rate:.4f} | `{transform}` | {note} | `{format_sample_values(samples)}` |"
        )

    lines.extend(
        [
            "",
            "## 额外运行时静态特征",
            "",
            "| field | transform | semantic_note |",
            "| --- | --- | --- |",
            f"| `month` | `{AUDIT_SEMANTICS['month'][0]}` | {AUDIT_SEMANTICS['month'][1]} |",
            "",
            "## 原始静态样例",
            "",
            dedup_static[STATIC_COLS].head(10).to_markdown(index=False),
            "",
        ]
    )

    pos = readable[readable["true_replenish_qty"].astype(float) > 0].copy()
    ice_pos = readable[(readable["is_cold_start"] == 1) & (readable["true_replenish_qty"].astype(float) > 0)].copy()
    high_true = readable[readable["true_replenish_qty"].astype(float) > 25].copy()
    ice_high_true = readable[(readable["is_cold_start"] == 1) & (readable["true_replenish_qty"].astype(float) > 25)].copy()
    corr_rows = []
    for name, sub in [("all_pos", pos), ("ice_pos", ice_pos), ("high_true", high_true), ("ice_high_true", ice_high_true)]:
        if len(sub) >= 2:
            corr_true = float(sub["qty_first_order"].astype(float).corr(sub["true_replenish_qty"].astype(float)))
            corr_pred = float(sub["qty_first_order"].astype(float).corr(sub["pred_mainline"].astype(float)))
        else:
            corr_true = np.nan
            corr_pred = np.nan
        corr_rows.append(
            {
                "segment": name,
                "rows": int(len(sub)),
                "corr(qty_first_order, true_qty)": corr_true,
                "corr(qty_first_order, pred_mainline)": corr_pred,
            }
        )

    qfo_bucket_df = ice_pos.copy()
    qfo_bucket_df["qfo_bucket"] = bucket_qty_first_order(qfo_bucket_df)
    bucket_summary = (
        qfo_bucket_df.groupby("qfo_bucket", dropna=False)
        .agg(
            rows=("sku_id", "size"),
            true_mean=("true_replenish_qty", "mean"),
            pred_mean=("pred_mainline", "mean"),
            ratio_mainline_p50=("ratio_mainline", "median"),
        )
        .reset_index()
    )
    bucket_order = ["0", "1", "2-10", "11-30", "31-100", "100+"]
    bucket_summary["ord"] = bucket_summary["qfo_bucket"].map({b: i for i, b in enumerate(bucket_order)}).fillna(99)
    bucket_summary = bucket_summary.sort_values("ord").drop(columns=["ord"])

    lines.extend(
        [
            "## `qty_first_order` 专项分析",
            "",
            "`qty_first_order` 已经是当前主线的一部分，而且对冷启动和大需求有显著信息量。下面的表直接基于 2025-12-01 主线结果计算。",
            "",
            "### 相关性",
            "",
            pd.DataFrame(corr_rows).to_markdown(index=False, floatfmt=".4f"),
            "",
            "### 冷启动正样本按首单量分桶",
            "",
            bucket_summary.to_markdown(index=False, floatfmt=".4f"),
            "",
            "## 结论",
            "",
            "- `sku_id/style_id/category/...` 当前都是 `LabelEncoder` 后的离散标签，不保留距离语义",
            "- `qty_first_order`、`price_tag`、`month` 保留了数值语义，其中 `qty_first_order` 对冷启动尤其重要",
            "- 如果后续继续优化静态特征，优先级应是 `qty_first_order` 强化表达、style/category 历史先验、tail 聚合特征，而不是先做 `sku_id` embedding",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(readable: pd.DataFrame, dashboard_payload: Dict, audit_md: str) -> None:
    readable.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    with open(OUT_HTML, "w", encoding="utf-8-sig") as fh:
        fh.write(render_html(dashboard_payload))
    with open(OUT_AUDIT, "w", encoding="utf-8-sig") as fh:
        fh.write(audit_md)


def main() -> None:
    ensure_dir()
    readable, frames, metrics = build_readable_frame()
    dashboard_payload = build_dashboard_payload(readable, metrics)
    audit_md = static_feature_audit_markdown(frames["raw_static"], frames["dedup_static"], readable)
    write_outputs(readable, dashboard_payload, audit_md)
    print("=" * 72)
    print("[done] Phase6h December readable report generated")
    print(f"csv:   {OUT_CSV}")
    print(f"html:  {OUT_HTML}")
    print(f"audit: {OUT_AUDIT}")
    print(f"rows:  {len(readable):,}")
    print("=" * 72)


if __name__ == "__main__":
    main()
