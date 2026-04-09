import json
import math
import os

import numpy as np
import pandas as pd

from phase_eval_utils import evaluate_context_frame


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUN_DIR = os.path.join(PROJECT_ROOT, "reports", "phase7_tail_allocation_optimization", "overnight_20260402_overnight")
OUT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase7h_december_readable_report")
OUT_CSV = os.path.join(OUT_DIR, "dec_20251201_sku_compare_readable.csv")
OUT_HTML = os.path.join(OUT_DIR, "dec_20251201_dashboard.html")
OUT_AUDIT = os.path.join(OUT_DIR, "dec_20251201_static_feature_audit.md")

WINNER_JSON = os.path.join(RUN_DIR, "overnight_winner.json")
STATIC_PATH = os.path.join(PROJECT_ROOT, "data", "gold", "wide_table_sku.csv")
PHASE6_PATH = os.path.join(PROJECT_ROOT, "reports", "phase5_7", "20251201", "phase5", "eval_context_p57a_20251201_covact_lr005_l63_s2026_hard_g025.csv")
P527_PATH = os.path.join(PROJECT_ROOT, "reports", "phase5_4", "phase5", "eval_context_p54_p527_lstm_l3_v5_lite_s2027_s2026.csv")
TARGET_ANCHOR = "2025-12-01"
STATIC_COLS = ["sku_id", "style_id", "product_name", "category", "sub_category", "season", "series", "band", "size_id", "color_id", "qty_first_order", "price_tag"]


def ensure_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


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


def winner_december_path():
    with open(WINNER_JSON, "r", encoding="utf-8") as f:
        payload = json.load(f)
    selected = payload["reason_or_selected"]
    target_dir = os.path.join(RUN_DIR, "stage_s3", "20251201", "phase5")
    matches = [os.path.join(target_dir, name) for name in os.listdir(target_dir) if name.startswith("eval_context_") and selected in name and name.endswith(".csv")]
    if len(matches) != 1:
        raise RuntimeError(f"winner eval_context match count != 1: {matches}")
    return matches[0], selected, payload


def load_static():
    raw = pd.read_csv(STATIC_PATH, usecols=STATIC_COLS)
    raw["sku_id"] = raw["sku_id"].astype(str)
    dedup = raw.drop_duplicates("sku_id").copy()
    return raw, dedup


def load_context(path, valid_skus):
    df = pd.read_csv(path)
    df["sku_id"] = df["sku_id"].astype(str)
    df["anchor_date"] = pd.to_datetime(df["anchor_date"]).dt.strftime("%Y-%m-%d")
    df = df[(df["anchor_date"] == TARGET_ANCHOR) & (df["sku_id"].isin(valid_skus))].copy()
    return df.reset_index(drop=True)


def rename_cols(df, prefix):
    cols = ["sku_id", "anchor_date", "true_replenish_qty", "ai_pred_prob", "ai_pred_qty", "abs_error"]
    out = df[cols].copy()
    return out.rename(columns={"ai_pred_prob": f"prob_{prefix}", "ai_pred_qty": f"pred_{prefix}", "abs_error": f"abs_error_{prefix}"})


def bucket_qfo(series):
    return pd.cut(series.fillna(0).astype(float), bins=[-0.01, 0.0, 1.0, 10.0, 30.0, 100.0, np.inf], labels=["0", "1", "2-10", "11-30", "31-100", "100+"]).astype(str)


def short_label(row):
    product = str(row.get("product_name") or "")[:10]
    return f"{row['sku_id']} | {product}" if product else str(row["sku_id"])


def validate_truth(base, other, name):
    merged = base.merge(other, on=["sku_id", "anchor_date"], how="inner", suffixes=("_base", "_other"))
    if len(merged) != len(base):
        raise ValueError(f"{name} row mismatch: {len(merged)} vs {len(base)}")
    mismatch = ~np.isclose(merged["true_replenish_qty_base"].astype(float), merged["true_replenish_qty_other"].astype(float), atol=1e-6, rtol=0.0)
    if mismatch.any():
        raise ValueError(f"{name} true mismatch rows={int(mismatch.sum())}")


def build_frames():
    raw_static, static_df = load_static()
    valid_skus = set(static_df["sku_id"])
    phase7_path, selected, winner_payload = winner_december_path()
    phase7 = load_context(phase7_path, valid_skus)
    phase6 = load_context(PHASE6_PATH, valid_skus)
    p527 = load_context(P527_PATH, valid_skus)
    validate_truth(phase7, phase6, "phase6")
    validate_truth(phase7, p527, "p527")

    shared = [
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
    readable = phase7[shared].copy()
    readable = readable.merge(rename_cols(phase7, "phase7"), on=["sku_id", "anchor_date", "true_replenish_qty"], how="left")
    readable = readable.merge(rename_cols(phase6, "phase6"), on=["sku_id", "anchor_date", "true_replenish_qty"], how="left")
    readable = readable.merge(rename_cols(p527, "p527"), on=["sku_id", "anchor_date", "true_replenish_qty"], how="left")
    readable = readable.merge(static_df, on="sku_id", how="left", suffixes=("_ctx", ""))
    for col in ("category", "style_id", "season", "series", "band"):
        ctx_col = f"{col}_ctx"
        if ctx_col in readable.columns:
            readable[col] = readable[col].combine_first(readable[ctx_col])
            readable = readable.drop(columns=[ctx_col])

    for prefix in ("phase7", "phase6", "p527"):
        readable[f"ratio_{prefix}"] = np.where(
            readable["true_replenish_qty"].astype(float) > 0,
            readable[f"pred_{prefix}"].astype(float) / readable["true_replenish_qty"].astype(float),
            np.nan,
        )

    true_qty = readable["true_replenish_qty"].astype(float)
    pred = readable["pred_phase7"].astype(float)
    readable["error_direction_phase7"] = np.select(
        [(true_qty <= 0) & (pred <= 0), pred < true_qty, pred > true_qty],
        ["exact_zero", "under", "over"],
        default="exact",
    )
    readable["is_true_blockbuster"] = (true_qty > 25).astype(int)
    readable["is_pred_blockbuster_phase7"] = (pred > 25).astype(int)
    readable["is_cold_start"] = (readable["lookback_repl_days_90"].astype(float) <= 0).astype(int)
    readable["is_repl0_fut0"] = (readable["signal_quadrant"].astype(str) == "repl0_fut0").astype(int)
    readable["rank_true_desc"] = true_qty.rank(method="dense", ascending=False).astype(int)
    readable["rank_pred_phase7_desc"] = pred.rank(method="dense", ascending=False).astype(int)
    readable["rank_abs_error_phase7_desc"] = readable["abs_error_phase7"].astype(float).rank(method="dense", ascending=False).astype(int)

    columns = [
        "sku_id", "product_name", "category", "sub_category", "style_id", "season", "series", "band", "size_id", "color_id",
        "qty_first_order", "price_tag", "anchor_date", "true_replenish_qty",
        "pred_phase7", "pred_phase6", "pred_p527",
        "prob_phase7", "prob_phase6", "prob_p527",
        "ratio_phase7", "ratio_phase6", "ratio_p527",
        "abs_error_phase7", "abs_error_phase6", "abs_error_p527",
        "error_direction_phase7", "signal_quadrant", "activity_bucket",
        "lookback_repl_days_90", "lookback_future_days_90", "lookback_repl_sum_90", "lookback_future_sum_90",
        "rank_true_desc", "rank_pred_phase7_desc", "rank_abs_error_phase7_desc",
        "is_true_blockbuster", "is_pred_blockbuster_phase7", "is_cold_start", "is_repl0_fut0",
    ]
    readable = readable[columns].sort_values(["rank_true_desc", "rank_abs_error_phase7_desc", "sku_id"]).reset_index(drop=True)
    metrics = {
        "phase7": evaluate_context_frame(phase7, "phase7"),
        "phase6": evaluate_context_frame(phase6, "phase6"),
        "p527": evaluate_context_frame(p527, "p527"),
    }
    frames = {"raw_static": raw_static, "dedup_static": static_df, "phase7": phase7, "phase6": phase6, "p527": p527}
    return readable, frames, metrics, phase7_path, selected, winner_payload


def build_payload(readable, metrics):
    top_true = readable.sort_values(["true_replenish_qty", "abs_error_phase7"], ascending=[False, False]).head(30)
    top_pred = readable.sort_values(["pred_phase7", "true_replenish_qty"], ascending=[False, False]).head(30)
    top_error = readable.sort_values(["abs_error_phase7", "true_replenish_qty"], ascending=[False, False]).head(30)
    top_blockbuster = readable[readable["true_replenish_qty"].astype(float) > 25].sort_values(["true_replenish_qty", "abs_error_phase7"], ascending=[False, False]).head(20)
    worst_repl0_ice = readable[(readable["is_repl0_fut0"] == 1) & (readable["is_cold_start"] == 1) & (readable["true_replenish_qty"].astype(float) > 10)].sort_values(["abs_error_phase7", "true_replenish_qty"], ascending=[False, False]).head(12)
    over_small = readable[(readable["true_replenish_qty"].astype(float) <= 3) & (readable["pred_phase7"].astype(float) > readable["true_replenish_qty"].astype(float))].sort_values(["ratio_phase7", "abs_error_phase7"], ascending=[False, False]).head(12)
    quadrant = readable.groupby("signal_quadrant", dropna=False).agg(rows=("sku_id", "size"), true_mean=("true_replenish_qty", "mean"), pred_phase7_mean=("pred_phase7", "mean"), pred_phase6_mean=("pred_phase6", "mean"), pred_p527_mean=("pred_p527", "mean")).reset_index()
    cold_pos = readable[(readable["is_cold_start"] == 1) & (readable["true_replenish_qty"].astype(float) > 0)].copy()
    cold_pos["qfo_bucket"] = bucket_qfo(cold_pos["qty_first_order"])
    qfo = cold_pos.groupby("qfo_bucket", dropna=False).agg(rows=("sku_id", "size"), true_mean=("true_replenish_qty", "mean"), pred_phase7_mean=("pred_phase7", "mean"), pred_phase6_mean=("pred_phase6", "mean"), pred_p527_mean=("pred_p527", "mean"), ratio_phase7_p50=("ratio_phase7", "median")).reset_index()
    qfo["ord"] = qfo["qfo_bucket"].map({"0": 0, "1": 1, "2-10": 2, "11-30": 3, "31-100": 4, "100+": 5}).fillna(99)
    qfo = qfo.sort_values("ord").drop(columns=["ord"])
    cat_ratio = readable.groupby("category", dropna=False).agg(total_true=("true_replenish_qty", "sum"), total_pred=("pred_phase7", "sum")).reset_index()
    cat_ratio = cat_ratio[cat_ratio["total_true"] > 0].copy()
    cat_ratio["ratio_phase7"] = cat_ratio["total_pred"] / cat_ratio["total_true"]
    cat_ratio["ratio_score"] = np.abs(np.log(np.clip(cat_ratio["ratio_phase7"], 1e-9, None)))
    cat_ratio = cat_ratio.sort_values(["ratio_score", "total_true"], ascending=[False, False]).head(10)
    cat_under = readable.copy()
    cat_under["under_phase7"] = np.clip(cat_under["true_replenish_qty"].astype(float) - cat_under["pred_phase7"].astype(float), a_min=0, a_max=None)
    cat_under = cat_under.groupby("category", dropna=False).agg(total_true=("true_replenish_qty", "sum"), under_sum=("under_phase7", "sum")).reset_index()
    cat_under = cat_under[cat_under["total_true"] > 0].copy()
    cat_under["under_wape"] = cat_under["under_sum"] / cat_under["total_true"]
    cat_under = cat_under.sort_values(["under_wape", "total_true"], ascending=[False, False]).head(10)

    cards = [
        {"label": "4-25 Under WAPE", "phase7": metrics["phase7"]["4_25_under_wape"], "phase6": metrics["phase6"]["4_25_under_wape"], "p527": metrics["p527"]["4_25_under_wape"]},
        {"label": "4-25 SKU P50", "phase7": metrics["phase7"]["4_25_sku_p50"], "phase6": metrics["phase6"]["4_25_sku_p50"], "p527": metrics["p527"]["4_25_sku_p50"]},
        {"label": "Ice 4-25 SKU P50", "phase7": metrics["phase7"]["ice_4_25_sku_p50"], "phase6": metrics["phase6"]["ice_4_25_sku_p50"], "p527": metrics["p527"]["ice_4_25_sku_p50"]},
        {"label": "Blockbuster Under WAPE", "phase7": metrics["phase7"]["blockbuster_under_wape"], "phase6": metrics["phase6"]["blockbuster_under_wape"], "p527": metrics["p527"]["blockbuster_under_wape"]},
        {"label": "Blockbuster SKU P50", "phase7": metrics["phase7"]["blockbuster_sku_p50"], "phase6": metrics["phase6"]["blockbuster_sku_p50"], "p527": metrics["p527"]["blockbuster_sku_p50"]},
        {"label": "Top20 Capture", "phase7": metrics["phase7"]["top20_true_volume_capture"], "phase6": metrics["phase6"]["top20_true_volume_capture"], "p527": metrics["p527"]["top20_true_volume_capture"]},
        {"label": "Rank Corr", "phase7": metrics["phase7"]["rank_corr_positive_skus"], "phase6": metrics["phase6"]["rank_corr_positive_skus"], "p527": metrics["p527"]["rank_corr_positive_skus"]},
        {"label": "Global Ratio", "phase7": metrics["phase7"]["global_ratio"], "phase6": metrics["phase6"]["global_ratio"], "p527": metrics["p527"]["global_ratio"]},
    ]
    return sanitize({
        "report_title": "2025-12-01 新主线三线业务对照看板",
        "summary_cards": cards,
        "callout_lines": [
            f"新主线 4-25 Under WAPE 从 {metrics['phase6']['4_25_under_wape']:.4f} 降到 {metrics['phase7']['4_25_under_wape']:.4f}",
            f"新主线 Blockbuster SKU P50 从 {metrics['phase6']['blockbuster_sku_p50']:.4f} 提升到 {metrics['phase7']['blockbuster_sku_p50']:.4f}",
            f"新主线 Top20 Capture 从 {metrics['phase6']['top20_true_volume_capture']:.4f} 提升到 {metrics['phase7']['top20_true_volume_capture']:.4f}",
            "当前最大漏补仍集中在 repl0_fut0 + ice，但新主线已经明显缩小缺口。",
        ],
        "top_true": [{"label": short_label(r), "true": r["true_replenish_qty"], "phase7": r["pred_phase7"], "phase6": r["pred_phase6"], "p527": r["pred_p527"]} for _, r in top_true.iterrows()],
        "top_pred": [{"label": short_label(r), "true": r["true_replenish_qty"], "phase7": r["pred_phase7"], "phase6": r["pred_phase6"], "p527": r["pred_p527"]} for _, r in top_pred.iterrows()],
        "top_error": [{"label": short_label(r), "true": r["true_replenish_qty"], "phase7": r["pred_phase7"], "abs_error_phase7": r["abs_error_phase7"], "quadrant": r["signal_quadrant"]} for _, r in top_error.iterrows()],
        "blockbuster_top": [{"label": short_label(r), "true": r["true_replenish_qty"], "phase7": r["pred_phase7"], "phase6": r["pred_phase6"], "p527": r["pred_p527"]} for _, r in top_blockbuster.iterrows()],
        "quadrant_summary": quadrant.to_dict(orient="records"),
        "qfo_bucket": qfo.to_dict(orient="records"),
        "category_ratio": cat_ratio[["category", "ratio_phase7", "total_true"]].to_dict(orient="records"),
        "category_under": cat_under[["category", "under_wape", "total_true"]].to_dict(orient="records"),
        "worst_repl0_ice": worst_repl0_ice[["sku_id", "product_name", "category", "qty_first_order", "true_replenish_qty", "pred_phase7", "ratio_phase7", "abs_error_phase7"]].to_dict(orient="records"),
        "over_small": over_small[["sku_id", "product_name", "category", "true_replenish_qty", "pred_phase7", "ratio_phase7", "abs_error_phase7"]].to_dict(orient="records"),
        "table_columns": list(readable.columns),
        "table_rows": readable.to_dict(orient="records"),
    })


def render_html(payload):
    data_json = json.dumps(payload, ensure_ascii=False)
    return f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0"/><title>{payload['report_title']}</title><script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script><style>:root{{--bg:#f6f7fb;--panel:#fff;--ink:#0f172a;--sub:#475569;--line:#dbe2ea;--phase7:#0f766e;--phase6:#2563eb;--p527:#dc2626;--true:#111827}}*{{box-sizing:border-box}}body{{margin:0;font-family:"Segoe UI","PingFang SC",sans-serif;background:var(--bg);color:var(--ink)}}.wrap{{max-width:1600px;margin:0 auto;padding:24px}}.hero,.charts,.tables,.kpis{{display:grid;gap:20px}}.hero{{grid-template-columns:1.3fr 1fr}}.charts,.tables{{grid-template-columns:1fr 1fr}}.kpis{{grid-template-columns:repeat(4,1fr)}}.card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:18px;box-shadow:0 8px 24px rgba(15,23,42,.05);padding:18px}}h1{{margin:0 0 8px;font-size:30px}}.sub{{color:var(--sub);line-height:1.7}}.chips{{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}}.chip{{background:#ecfeff;color:var(--phase7);border:1px solid #bfe8ea;border-radius:999px;padding:5px 10px;font-size:12px}}.kpi-title{{margin:0 0 8px;font-size:13px;color:var(--sub)}}.kpi-row{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}.pill{{background:#f8fafc;border-radius:12px;padding:10px;text-align:center}}.pill .name{{font-size:11px;color:var(--sub)}}.pill .val{{font-size:19px;font-weight:700}}canvas{{width:100%!important;height:380px!important}}.small canvas{{height:300px!important}}table{{width:100%;border-collapse:collapse}}th,td{{border-bottom:1px solid var(--line);padding:8px 10px;text-align:left;font-size:13px}}th{{background:#f8fafc;position:sticky;top:0}}.num{{text-align:right;font-variant-numeric:tabular-nums}}.table-wrap{{max-height:360px;overflow:auto}}.legend{{display:flex;gap:14px;flex-wrap:wrap;color:var(--sub);font-size:12px;margin-top:8px}}.legend span::before{{content:"";display:inline-block;width:10px;height:10px;border-radius:999px;margin-right:6px;vertical-align:middle}}.lg-true::before{{background:var(--true)}}.lg-phase7::before{{background:var(--phase7)}}.lg-phase6::before{{background:var(--phase6)}}.lg-p527::before{{background:var(--p527)}}.controls{{display:flex;gap:10px;align-items:center;margin-bottom:12px;flex-wrap:wrap}}.controls input,.controls select{{padding:8px 10px;border:1px solid var(--line);border-radius:10px;background:#fff}}.pager{{display:flex;align-items:center;gap:8px;margin-top:12px}}.pager button{{border:1px solid var(--line);background:#fff;padding:6px 10px;border-radius:8px;cursor:pointer}}@media(max-width:1200px){{.hero,.charts,.tables,.kpis{{grid-template-columns:1fr}}}}</style></head><body><div class="wrap"><div class="hero"><div class="card"><h1>{payload['report_title']}</h1><div class="sub">当前 12 月报表已经统一到 cleaned gold 口径。三线对比固定为：Phase7 新主线、Phase6 冻结主线、p527 baseline。由于 `sep098_oct093` 不作用于 12 月，这里展示的是 12 月 raw 预测对照。</div><div class="chips"><span class="chip">Anchor: 2025-12-01</span><span class="chip">Universe: cleaned gold</span><span class="chip">Rows: {len(payload['table_rows'])}</span><span class="chip">Current winner: tail_full_lr005_l63_g027_n800_s2028</span></div></div><div class="card"><h3>核心结论</h3><ul id="callouts"></ul></div></div><div class="kpis" id="kpis"></div><div class="charts"><div class="panel"><h3>真实 Top30 SKU</h3><canvas id="topTrue"></canvas><div class="legend"><span class="lg-true">真实</span><span class="lg-phase7">Phase7</span><span class="lg-phase6">Phase6</span><span class="lg-p527">p527</span></div></div><div class="panel"><h3>新主线预测 Top30 SKU</h3><canvas id="topPred"></canvas><div class="legend"><span class="lg-true">真实</span><span class="lg-phase7">Phase7</span><span class="lg-phase6">Phase6</span><span class="lg-p527">p527</span></div></div></div><div class="charts"><div class="panel"><h3>新主线误差 Top30</h3><canvas id="topError"></canvas></div><div class="panel small"><h3>信号象限均值对比</h3><canvas id="quadrant"></canvas></div></div><div class="charts"><div class="panel"><h3>真实 Blockbuster Top20</h3><canvas id="blockbuster"></canvas></div><div class="panel small"><h3>冷启动按首单量分桶</h3><canvas id="qfo"></canvas></div></div><div class="charts"><div class="panel small"><h3>最差类别：Under WAPE</h3><canvas id="catUnder"></canvas></div><div class="panel small"><h3>最差类别：Ratio 偏离</h3><canvas id="catRatio"></canvas></div></div><div class="tables"><div class="panel"><h3>高价值漏补：repl0_fut0 + ice</h3><div class="table-wrap"><table id="worstTable"></table></div></div><div class="panel"><h3>小需求但高估</h3><div class="table-wrap"><table id="overTable"></table></div></div></div><div class="panel"><h3>完整 12 月逐 SKU 对照表</h3><div class="controls"><input id="search" placeholder="搜索 sku/category/style/product_name ..."/><select id="pageSize"><option value="20">20 行</option><option value="50" selected>50 行</option><option value="100">100 行</option></select></div><div class="table-wrap"><table id="detail"></table></div><div class="pager"><button id="prev">上一页</button><span id="pageInfo"></span><button id="next">下一页</button></div></div></div><script>const DATA={data_json};const COLORS={{true:'#111827',phase7:'#0f766e',phase6:'#2563eb',p527:'#dc2626'}};function fmt(v,d=4){{if(v===null||v===undefined||Number.isNaN(Number(v)))return '-';return Number(v).toFixed(d)}}function kpis(){{const root=document.getElementById('kpis');DATA.summary_cards.forEach(c=>{{const el=document.createElement('div');el.className='card';el.innerHTML=`<div class="kpi-title">${{c.label}}</div><div class="kpi-row"><div class="pill"><div class="name">Phase7</div><div class="val">${{fmt(c.phase7)}}</div></div><div class="pill"><div class="name">Phase6</div><div class="val">${{fmt(c.phase6)}}</div></div><div class="pill"><div class="name">p527</div><div class="val">${{fmt(c.p527)}}</div></div></div>`;root.appendChild(el)}});const ul=document.getElementById('callouts');DATA.callout_lines.forEach(t=>{{const li=document.createElement('li');li.textContent=t;ul.appendChild(li)}})}}function bar(id,rows){{new Chart(document.getElementById(id),{{type:'bar',data:{{labels:rows.map(r=>r.label),datasets:[{{label:'真实',data:rows.map(r=>r.true),backgroundColor:COLORS.true}},{{label:'Phase7',data:rows.map(r=>r.phase7),backgroundColor:COLORS.phase7}},{{label:'Phase6',data:rows.map(r=>r.phase6),backgroundColor:COLORS.phase6}},{{label:'p527',data:rows.map(r=>r.p527),backgroundColor:COLORS.p527}}]}},options:{{responsive:true,maintainAspectRatio:false,scales:{{x:{{ticks:{{autoSkip:false,maxRotation:70,minRotation:70}}}},y:{{beginAtZero:true}}}}}})}}function topError(){{const rows=DATA.top_error;new Chart(document.getElementById('topError'),{{type:'bar',data:{{labels:rows.map(r=>`${{r.label}} | ${{r.quadrant}}`),datasets:[{{label:'真实',data:rows.map(r=>r.true),backgroundColor:COLORS.true}},{{label:'Phase7 预测',data:rows.map(r=>r.phase7),backgroundColor:COLORS.phase7}},{{label:'绝对误差',data:rows.map(r=>r.abs_error_phase7),backgroundColor:'#f59e0b'}}]}},options:{{responsive:true,maintainAspectRatio:false,scales:{{x:{{ticks:{{autoSkip:false,maxRotation:70,minRotation:70}}}},y:{{beginAtZero:true}}}}}})}}function quadrant(){{const rows=DATA.quadrant_summary;new Chart(document.getElementById('quadrant'),{{type:'bar',data:{{labels:rows.map(r=>r.signal_quadrant),datasets:[{{label:'样本数',data:rows.map(r=>r.rows),backgroundColor:'#94a3b8',yAxisID:'y'}},{{label:'真实均值',data:rows.map(r=>r.true_mean),backgroundColor:COLORS.true,yAxisID:'y1'}},{{label:'Phase7 均值',data:rows.map(r=>r.pred_phase7_mean),backgroundColor:COLORS.phase7,yAxisID:'y1'}},{{label:'Phase6 均值',data:rows.map(r=>r.pred_phase6_mean),backgroundColor:COLORS.phase6,yAxisID:'y1'}},{{label:'p527 均值',data:rows.map(r=>r.pred_p527_mean),backgroundColor:COLORS.p527,yAxisID:'y1'}}]}},options:{{responsive:true,maintainAspectRatio:false,scales:{{y:{{type:'linear',position:'left',beginAtZero:true}},y1:{{type:'linear',position:'right',beginAtZero:true,grid:{{drawOnChartArea:false}}}}}}}})}}function qfo(){{const rows=DATA.qfo_bucket;new Chart(document.getElementById('qfo'),{{type:'bar',data:{{labels:rows.map(r=>`${{r.qfo_bucket}} (n=${{r.rows}})`),datasets:[{{label:'真实均值',data:rows.map(r=>r.true_mean),backgroundColor:COLORS.true}},{{label:'Phase7',data:rows.map(r=>r.pred_phase7_mean),backgroundColor:COLORS.phase7}},{{label:'Phase6',data:rows.map(r=>r.pred_phase6_mean),backgroundColor:COLORS.phase6}},{{label:'p527',data:rows.map(r=>r.pred_p527_mean),backgroundColor:COLORS.p527}}]}},options:{{responsive:true,maintainAspectRatio:false,scales:{{y:{{beginAtZero:true}}}}}})}}function cats(){{new Chart(document.getElementById('catUnder'),{{type:'bar',data:{{labels:DATA.category_under.map(r=>r.category),datasets:[{{label:'Under WAPE',data:DATA.category_under.map(r=>r.under_wape),backgroundColor:'#b91c1c'}}]}},options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,scales:{{x:{{beginAtZero:true}}}}}}}});new Chart(document.getElementById('catRatio'),{{type:'bar',data:{{labels:DATA.category_ratio.map(r=>r.category),datasets:[{{label:'Ratio',data:DATA.category_ratio.map(r=>r.ratio_phase7),backgroundColor:'#7c3aed'}}]}},options:{{indexAxis:'y',responsive:true,maintainAspectRatio:false,scales:{{x:{{beginAtZero:true}}}}}}}})}}function simpleTable(id,rows,cols){{const t=document.getElementById(id);let html='<thead><tr>'+cols.map(c=>`<th>${{c}}</th>`).join('')+'</tr></thead><tbody>';rows.forEach(r=>{{html+='<tr>'+cols.map(c=>{{const v=r[c];const isNum=typeof v==='number';return `<td class="${{isNum?'num':''}}">${{v===null||v===undefined?'':isNum?fmt(v,3):String(v)}}</td>`}}).join('')+'</tr>'}});t.innerHTML=html+'</tbody>'}}let state={{page:1,pageSize:50,sortCol:'rank_true_desc',sortAsc:true,search:''}};function filtered(){{let rows=[...DATA.table_rows];if(state.search){{const q=state.search.toLowerCase();rows=rows.filter(r=>Object.values(r).some(v=>String(v??'').toLowerCase().includes(q)))}}rows.sort((a,b)=>{{const av=a[state.sortCol],bv=b[state.sortCol];const aNum=typeof av==='number',bNum=typeof bv==='number';let cmp=0;if(aNum&&bNum)cmp=(av??-Infinity)-(bv??-Infinity);else cmp=String(av??'').localeCompare(String(bv??''));return state.sortAsc?cmp:-cmp}});return rows}}function detail(){{const rows=filtered();const totalPages=Math.max(1,Math.ceil(rows.length/state.pageSize));if(state.page>totalPages)state.page=totalPages;const pageRows=rows.slice((state.page-1)*state.pageSize,(state.page-1)*state.pageSize+state.pageSize);const cols=DATA.table_columns;let html='<thead><tr>'+cols.map(c=>`<th data-col="${{c}}">${{c}}</th>`).join('')+'</tr></thead><tbody>';pageRows.forEach(r=>{{html+='<tr>'+cols.map(c=>{{const v=r[c];const isNum=typeof v==='number';return `<td class="${{isNum?'num':''}}">${{v===null||v===undefined?'':isNum?fmt(v,4):String(v)}}</td>`}}).join('')+'</tr>'}});const t=document.getElementById('detail');t.innerHTML=html+'</tbody>';t.querySelectorAll('th').forEach(th=>{{th.style.cursor='pointer';th.onclick=()=>{{const c=th.dataset.col;if(state.sortCol===c)state.sortAsc=!state.sortAsc;else{{state.sortCol=c;state.sortAsc=true}}detail()}}}});document.getElementById('pageInfo').textContent=`第 ${{state.page}} / ${{totalPages}} 页，共 ${{rows.length}} 行`}}document.getElementById('search').addEventListener('input',e=>{{state.search=e.target.value.trim();state.page=1;detail()}});document.getElementById('pageSize').addEventListener('change',e=>{{state.pageSize=Number(e.target.value);state.page=1;detail()}});document.getElementById('prev').onclick=()=>{{state.page=Math.max(1,state.page-1);detail()}};document.getElementById('next').onclick=()=>{{const totalPages=Math.max(1,Math.ceil(filtered().length/state.pageSize));state.page=Math.min(totalPages,state.page+1);detail()}};kpis();bar('topTrue',DATA.top_true);bar('topPred',DATA.top_pred);topError();quadrant();bar('blockbuster',DATA.blockbuster_top);qfo();cats();simpleTable('worstTable',DATA.worst_repl0_ice,['sku_id','product_name','category','qty_first_order','true_replenish_qty','pred_phase7','ratio_phase7','abs_error_phase7']);simpleTable('overTable',DATA.over_small,['sku_id','product_name','category','true_replenish_qty','pred_phase7','ratio_phase7','abs_error_phase7']);detail();</script></body></html>"""


def build_audit(raw_static, dedup_static, readable):
    rows = ["# 2025-12-01 静态特征审计", "", "当前报表已经统一到 cleaned gold 口径。缺失商品主数据的 SKU 已被过滤，因此不再出现整行静态字段 `Unknown` 的样本。", "", "| field | raw_dtype | distinct_count | missing_rate | transform | semantic_note | sample_values |", "| --- | --- | ---: | ---: | --- | --- | --- |"]
    semantics = {"sku_id": ("LabelEncoder", "SKU 唯一键。当前更像记忆型 ID，不保留距离语义。"), "style_id": ("LabelEncoder", "同款 ID。有分组语义，但不保留距离语义。"), "product_name": ("LabelEncoder", "商品名称文本，当前只作为离散标签使用。"), "category": ("LabelEncoder", "一级类目，有分组语义。"), "sub_category": ("LabelEncoder", "二级类目，有分组语义。"), "season": ("LabelEncoder", "季节标签。"), "series": ("LabelEncoder", "系列标签。"), "band": ("LabelEncoder", "波段标签。"), "size_id": ("LabelEncoder", "尺码原始上有顺序，但当前编码会丢掉顺序。"), "color_id": ("LabelEncoder", "颜色标签。"), "qty_first_order": ("log1p(max(0,x))", "首单量。冷启动和 tail 的关键静态数值特征。"), "price_tag": ("log1p(max(0,x))", "吊牌价。")}
    for col in STATIC_COLS:
        samples = dedup_static[col].dropna().astype(str).drop_duplicates().head(5).tolist()
        rows.append(f"| `{col}` | `{raw_static[col].dtype}` | {int(dedup_static[col].nunique(dropna=True))} | {float(dedup_static[col].isna().mean()):.4f} | `{semantics[col][0]}` | {semantics[col][1]} | `{json.dumps(samples, ensure_ascii=False)}` |")
    pos = readable[readable['true_replenish_qty'].astype(float) > 0].copy()
    ice_pos = readable[(readable['is_cold_start'] == 1) & (readable['true_replenish_qty'].astype(float) > 0)].copy()
    high_true = readable[readable['true_replenish_qty'].astype(float) > 25].copy()
    ice_high_true = readable[(readable['is_cold_start'] == 1) & (readable['true_replenish_qty'].astype(float) > 25)].copy()
    corr_rows = []
    for name, sub in [('all_pos', pos), ('ice_pos', ice_pos), ('high_true', high_true), ('ice_high_true', ice_high_true)]:
        corr_rows.append({'segment': name, 'rows': int(len(sub)), 'corr(qty_first_order, true_qty)': float(sub['qty_first_order'].astype(float).corr(sub['true_replenish_qty'].astype(float))) if len(sub) >= 2 else np.nan, 'corr(qty_first_order, pred_phase7)': float(sub['qty_first_order'].astype(float).corr(sub['pred_phase7'].astype(float))) if len(sub) >= 2 else np.nan})
    qfo = ice_pos.copy()
    qfo['qfo_bucket'] = bucket_qfo(qfo['qty_first_order'])
    qfo_summary = qfo.groupby('qfo_bucket', dropna=False).agg(rows=('sku_id', 'size'), true_mean=('true_replenish_qty', 'mean'), pred_mean=('pred_phase7', 'mean'), ratio_phase7_p50=('ratio_phase7', 'median')).reset_index()
    rows.extend(["", "## 运行时额外静态特征", "", "| field | transform | semantic_note |", "| --- | --- | --- |", "| `month` | `anchor_date.month` | 运行时写入的锚点月份，不来自原始静态表。 |", "", "## `qty_first_order` 专项分析", "", "### 相关性", "", pd.DataFrame(corr_rows).to_markdown(index=False, floatfmt='.4f'), "", "### 冷启动正样本按首单量分桶", "", qfo_summary.to_markdown(index=False, floatfmt='.4f'), "", "## 结论", "", "- `sku_id/style_id/category/...` 当前仍然是 `LabelEncoder` 后的离散标签，有分组语义，但没有距离语义。", "- `qty_first_order`、`price_tag`、`month` 保留数值语义，其中 `qty_first_order` 对冷启动和 tail 最重要。", "- 如果继续优化静态特征，优先级仍然是 `qty_first_order` 强化表达、style/category 历史先验、tail 聚合特征，而不是先上 `sku_id` embedding。"])
    return "\\n".join(rows) + "\\n"


def main():
    ensure_dir()
    readable, frames, metrics, phase7_path, selected, winner_payload = build_frames()
    readable.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(render_html(build_payload(readable, metrics)))
    with open(OUT_AUDIT, "w", encoding="utf-8") as f:
        f.write(build_audit(frames["raw_static"], frames["dedup_static"], readable))
    old_phase6 = pd.read_csv(PHASE6_PATH)
    old_phase6["sku_id"] = old_phase6["sku_id"].astype(str)
    removed_rows = int((~old_phase6["sku_id"].isin(set(frames["dedup_static"]["sku_id"]))).sum())
    print(json.dumps(sanitize({"out_csv": OUT_CSV, "out_html": OUT_HTML, "out_audit": OUT_AUDIT, "row_count": int(len(readable)), "selected_candidate": selected, "winner_json_action": winner_payload["winner_action"], "winner_december_eval_context": phase7_path, "removed_missing_product_rows_vs_old_phase6_context": removed_rows}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
