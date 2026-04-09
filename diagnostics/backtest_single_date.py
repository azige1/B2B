"""
单日回测对比脚本 — 站在指定日期 T，预测未来 30 天，并与真实值对比
放置位置: diagnostics/backtest_single_date.py
运行方式:
  python diagnostics/backtest_single_date.py              # 默认锚点 2025-10-30
  python diagnostics/backtest_single_date.py 2025-10-15  # 自定义锚点
"""
import os, sys, pickle, argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.models.enhanced_model import EnhancedTwoTowerLSTM
from utils.common import load_yaml
from src.train.run_training import DummyLE  # 用于正确反序列化必须在 pickle.load 之前

import time

# ─── 参数解析 ─────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("anchor_date", nargs="?", default="2025-10-30",
                    help="推断锚点日期，格式 YYYY-MM-DD（默认 2025-10-30）")
args = parser.parse_args()
ANCHOR = pd.Timestamp(args.anchor_date)
WIN_START = ANCHOR + pd.Timedelta(days=1)
WIN_END   = ANCHOR + pd.Timedelta(days=30)

SEP = "=" * 66
print(f"\n{SEP}")
print(f"  单日回测对比报告")
print(f"  推断锚点: {ANCHOR.date()}（站在这一天，向前看 120 天）")
print(f"  预测窗口: {WIN_START.date()} → {WIN_END.date()}（未来 30 天）")
print(SEP)

global_start = time.time()
step_start = time.time()

# ─── 1. 加载数据 ──────────────────────────────────────────────
data_path = os.path.join(PROJECT_ROOT, "data", "gold", "wide_table_sku.csv")
print(f"\n[*] 读取宽表: {data_path}")
df = pd.read_csv(data_path)
df["date"] = pd.to_datetime(df["date"])
print(f"    [耗时] 读取全量数据花去 {time.time() - step_start:.1f} 秒")
step_start = time.time()

# 验证锚点日期是否在数据范围内
data_max = df["date"].max()
if ANCHOR >= data_max:
    print(f"⚠️  锚点 {ANCHOR.date()} ≥ 数据最新日期 {data_max.date()}，无法验证真实值！")
    print(f"   建议锚点 ≤ {(data_max - pd.Timedelta(days=30)).date()}")
    sys.exit(1)
if WIN_END > data_max:
    print(f"⚠️  预测窗口终点 {WIN_END.date()} 超过数据最新日期 {data_max.date()}")
    print(f"   真实值只能覆盖 {WIN_START.date()} ~ {data_max.date()}")
    WIN_END = data_max

# ─── 2. 构建历史切片（120 天） ───────────────────────────────
START = ANCHOR - pd.Timedelta(days=119)
df_hist = df[df["date"].between(START, ANCHOR)].copy()
print(f"[*] 历史切片: {START.date()} ~ {ANCHOR.date()}，共 {df_hist['date'].nunique()} 天，{len(df_hist):,} 行")
print(f"    [耗时] 切片过滤花去 {time.time() - step_start:.1f} 秒")
step_start = time.time()

# ─── 3. 读取真实窗口内数据 ───────────────────────────────────
df_real = df[df["date"].between(WIN_START, WIN_END)].copy()
real_agg = df_real.groupby(["buyer_id", "sku_id"])["qty_replenish"].sum().reset_index()\
           .rename(columns={"qty_replenish": "real_qty_30d"})
print(f"[*] 真实窗口 ({WIN_START.date()}~{WIN_END.date()}): {len(df_real):,} 行，"
      f"有补货记录的组合: {(real_agg['real_qty_30d']>0).sum():,} 个")

# ─── 4. 加载编码器 ───────────────────────────────────────────
encoder_path = os.path.join(PROJECT_ROOT, "data", "artifacts", "label_encoders.pkl")
with open(encoder_path, "rb") as f:
    encoders = pickle.load(f)

static_cat_cols = ["buyer_id", "sku_id", "style_id", "product_name", "category",
                   "sub_category", "season", "series", "band", "size_id", "color_id"]
# ★ 必须与 run_training.py / generate_daily_inference.py 保持完全一致的 7 维
static_num_cols = [
    "qty_first_order",
    "price_tag",
    "cooperation_years",
    "monthly_average_replenishment",
    "avg_discount_rate",
    "replenishment_frequency",
    "item_coverage_rate"
]

# 编码分类特征（极速矢量化）
for c in static_cat_cols:
    if c in df_hist.columns:
        le = encoders[c]
        # 用 np.where 矢量化处理未知类别（极大提速，取代龟速 lambda）
        # 必须显式转为 str 确保与 le.classes_ 类型严格对齐，防止纯数字款式编号失效
        known_mask = df_hist[c].astype(str).isin(le.classes_)
        df_hist.loc[known_mask, c] = le.transform(df_hist.loc[known_mask, c].astype(str))
        df_hist.loc[~known_mask, c] = 0

# ─── 5. 极速提取静态特征字典与动态时序 ───────────────────────
static_dict = {}
daily_sales_dict = {}

print(f"[*] 解析 {len(df_hist):,} 行行为日志，极速构建特征字典...")
for row in tqdm(df_hist.itertuples(index=False), total=len(df_hist), desc="组装特征基座"):
    key = (row.buyer_id, row.sku_id)
    
    if key not in static_dict:
        arr = [getattr(row, c) for c in static_cat_cols]
        arr.append(ANCHOR.month)  # month
        for col in static_num_cols:
            val = getattr(row, col, 0.0)
            val = float(val) if pd.notnull(val) else 0.0
            arr.append(np.log1p(max(0.0, val)))
        static_dict[key] = np.array(arr, dtype=np.float32)
        daily_sales_dict[key] = {}
        
    daily_sales_dict[key][row.date] = (
        getattr(row, "qty_replenish", 0),
        getattr(row, "qty_debt", 0),
        getattr(row, "qty_shipped", 0),
        getattr(row, "qty_inbound", 0),
    )
print(f"    [耗时] 日志字典提取聚合花去 {time.time() - step_start:.1f} 秒")
step_start = time.time()

# ─── 6. 构造特征张量列表 ──────────────────────────────────
x_dyn_list, x_static_list, valid_keys = [], [], []
print(f"    [耗时] 日志字典聚合并映射花去 {time.time() - step_start:.1f} 秒")
step_start = time.time()

print(f"[*] 构建 {len(daily_sales_dict)} 个 (buyer, SKU) 的时序特征块...")
for (buyer, sku), daily in tqdm(daily_sales_dict.items(), desc="提取时序特征"):
    hist = np.zeros((120, 4), dtype=np.float32)
    for b in range(120):
        d = START + pd.Timedelta(days=b)
        if d in daily:
            hist[b] = daily[d]

    cum_r = np.cumsum(np.insert(hist[:, 0], 0, 0.0))
    cum_s = np.cumsum(np.insert(hist[:, 2], 0, 0.0))

    dyn = np.zeros((90, 7), dtype=np.float32)
    dyn[:, 0] = np.log1p(np.maximum(hist[30:, 0], 0))
    dyn[:, 1] = np.log1p(np.maximum(cum_r[31:121] - cum_r[24:114], 0))
    dyn[:, 2] = np.log1p(np.maximum(cum_r[31:121] - cum_r[1:91],  0))
    dyn[:, 3] = np.log1p(np.maximum(hist[30:, 1], 0))
    dyn[:, 4] = np.log1p(np.maximum(hist[30:, 2], 0))
    dyn[:, 5] = np.log1p(np.maximum(cum_s[31:121] - cum_s[24:114], 0))
    dyn[:, 6] = np.log1p(np.maximum(hist[30:, 3], 0))

    if (buyer, sku) in static_dict:
        x_dyn_list.append(dyn)
        x_static_list.append(static_dict[(buyer, sku)])
        valid_keys.append((buyer, sku))

print(f"    [耗时] 纯粹遍历提取时序特征花去 {time.time() - step_start:.1f} 秒")
step_start = time.time()

x_dyn_tensor    = torch.tensor(np.stack(x_dyn_list),    dtype=torch.float32)
x_static_tensor = torch.tensor(np.stack(x_static_list), dtype=torch.float32)
print(f"[*] 张量构建完成: {len(valid_keys):,} 个组合")
print(f"    [耗时] 张量化转换存储花去 {time.time() - step_start:.1f} 秒")
step_start = time.time()

# ─── 7. 加载模型 ─────────────────────────────────────────────
config  = load_yaml(os.path.join(PROJECT_ROOT, "config", "model_config.yaml"))
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] 使用设备: {device}")

dynamic_vocab_sizes = {
    "buyer_id": 1000, "sku_id": 15000, "style_id": 3000, "product_name": 2000,
    "category": 50, "sub_category": 100, "season": 10, "series": 50,
    "band": 50, "size_id": 50, "color_id": 100, "month": 13
}
for col, le in encoders.items():
    if col in dynamic_vocab_sizes:
        dynamic_vocab_sizes[col] = len(le.classes_) + 5

model = EnhancedTwoTowerLSTM(
    dyn_feat_dim=7, lstm_hidden=config["model"]["hidden_size"],
    lstm_layers=config["model"]["num_layers"],
    static_vocab_sizes=dynamic_vocab_sizes,
    static_emb_dim=16, num_numeric_feats=len(static_num_cols), dropout=0.0  # 动态计算，防止硬编码偏移
).to(device)

model_path = os.path.join(PROJECT_ROOT, "models", "best_enhanced_model.pth")
state_dict = torch.load(model_path, map_location=device, weights_only=False)
new_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(new_sd)
model.eval()
print(f"[*] 权重加载成功")

# ─── 8. 推断 ─────────────────────────────────────────────────
static_order_keys = static_cat_cols + ["month"] + static_num_cols
dataset = torch.utils.data.TensorDataset(x_dyn_tensor, x_static_tensor)
loader  = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False)

all_probs, all_qtys = [], []
print("[*] 执行推断...")
with torch.no_grad():
    for b_dyn, b_sta in tqdm(loader, desc="推断中"):
        b_dyn = b_dyn.to(device)
        b_sta = b_sta.to(device)
        logits, preds = model(b_dyn, b_sta, static_order_keys)

        prob = torch.sigmoid(logits)
        
        # 【基础死区掩码】整 90 天全空（纯冷启动新品），毫无预测依据
        dead_mask = (b_dyn.abs().sum(dim=(1, 2)) < 1e-4)
        
        # 【🔥 新增：静默期退市掩码】最近 21 天没有任何指标波动（销量、发货等全为0），说明已过季凉透
        # b_dyn shape: (Batch, 90, 7)
        silent_mask = (b_dyn[:, -21:, :].abs().sum(dim=(1, 2)) < 1e-4)
        
        # 双重拦截：没底子的不要，凉透了的也不要
        prob[dead_mask | silent_mask] = 0.0
        
        preds = torch.clamp(preds, max=8.5)
        # 【阈值收紧】将 0.20 提升至 0.45，大幅砍去对死活不知的滞销品发出的垃圾订单
        qty   = torch.expm1(preds) * (prob > 0.45).float() * 0.8

        all_probs.extend(prob.cpu().numpy().flatten())
        pq = np.nan_to_num(qty.cpu().numpy().flatten(), nan=0.0, posinf=0.0, neginf=0.0)
        all_qtys.extend(pq)

# ─── 9. 解码并合并真实值 ─────────────────────────────────────
df_pred = pd.DataFrame(valid_keys, columns=["buyer_idx", "sku_idx"])
df_pred["ai_prob_30d"]   = np.array(all_probs).round(3)
df_pred["ai_budget_30d"] = np.array(all_qtys).round(1)

# 反向解码 buyer_id / sku_id
df_pred["buyer_id"] = encoders["buyer_id"].inverse_transform(df_pred["buyer_idx"])
df_pred["sku_id"]   = encoders["sku_id"].inverse_transform(df_pred["sku_idx"])
df_pred = df_pred.drop(columns=["buyer_idx","sku_idx"])

# 合并真实值（必须使用 outer join 才能暴露出那些因为根本没有历史数据而无法做出预测的纯粹冷启动漏网之鱼）
df_merged = df_pred.merge(real_agg, on=["buyer_id","sku_id"], how="outer")

# 填充：对于在未来产生了真实消费但 AI 历史数据不足（或推断为 0）的，预测量补 0；对于 AI 预测了但不卖的，真实量补 0
df_merged["ai_budget_30d"] = df_merged["ai_budget_30d"].fillna(0.0)
df_merged["real_qty_30d"] = df_merged["real_qty_30d"].fillna(0.0)

# 保留有预测或有真实的行
df_show = df_merged[(df_merged["ai_budget_30d"] > 0) | (df_merged["real_qty_30d"] > 0)].copy()
df_show["diff"]  = (df_show["ai_budget_30d"] - df_show["real_qty_30d"]).round(1)
df_show["ratio"] = (df_show["ai_budget_30d"] / df_show["real_qty_30d"].replace(0, np.nan)).round(3)
df_show = df_show.sort_values("real_qty_30d", ascending=False)
df_show.insert(0, "anchor_date", ANCHOR.date())

# ─── 10. 输出 ────────────────────────────────────────────────
print(f"\n{SEP}")
print(f"  预测统计汇总")
print(SEP)

ai_pos = df_merged[df_merged["ai_budget_30d"] > 0]
real_pos = df_merged[df_merged["real_qty_30d"] > 0]
both = df_merged[(df_merged["ai_budget_30d"] > 0) & (df_merged["real_qty_30d"] > 0)]
miss = df_merged[(df_merged["ai_budget_30d"] == 0) & (df_merged["real_qty_30d"] > 0)]
fa   = df_merged[(df_merged["ai_budget_30d"] > 0) & (df_merged["real_qty_30d"] == 0)]

print(f"  AI 预测>0:      {len(ai_pos):,} 个组合")
print(f"  真实发生补货:   {len(real_pos):,} 个组合")
print(f"  双阳 (都>0):    {len(both):,} 个")
print(f"  漏报 (AI=0真>0): {len(miss):,} 个")
print(f"  虚报 (AI>0真=0): {len(fa):,} 个")
if len(both) > 0:
    ratio = both["ai_budget_30d"].sum() / both["real_qty_30d"].sum()
    mae   = (both["ai_budget_30d"] - both["real_qty_30d"]).abs().mean()
    print(f"\n  大盘 Ratio:     {ratio:.3f}  （目标 1.20~1.50）")
    print(f"  双阳 MAE:       {mae:.1f} 件/组合")
    if len(ai_pos) > 0:
        print(f"\n  AI件数分布: 中位数={ai_pos['ai_budget_30d'].median():.1f}  "
              f"90%={ai_pos['ai_budget_30d'].quantile(0.9):.1f}  最大={ai_pos['ai_budget_30d'].max():.1f}")

print(f"\n  Top 20（按真实件数排序）:")
print(f"  {'SKU':<24} {'买手':<10} {'AI预测':>8} {'真实':>8} {'差异':>8} {'Ratio':>7}")
print("  " + "-" * 68)
for _, row in df_show.head(20).iterrows():
    ratio_str = f"{row['ratio']:.2f}" if pd.notnull(row["ratio"]) else "∞"
    print(f"  {row['sku_id']:<24} {row['buyer_id']:<10} "
          f"{row['ai_budget_30d']:>8.1f} {row['real_qty_30d']:>8.1f} "
          f"{row['diff']:>8.1f} {ratio_str:>7}")

# ─── 11. 核心业务视角：SKU 全国总量聚合评估 ──────────────────
print(f"\n{SEP}")
print(f"  SKU 总部采购视角 (全国汇总)")
print(SEP)

sku_agg = df_merged.groupby("sku_id")[["ai_budget_30d", "real_qty_30d"]].sum()
sku_agg = sku_agg[(sku_agg["ai_budget_30d"] > 0) | (sku_agg["real_qty_30d"] > 0)].copy()

sku_both = sku_agg[(sku_agg["ai_budget_30d"] > 0) & (sku_agg["real_qty_30d"] > 0)]
if len(sku_both) > 0:
    sku_mae = (sku_both["ai_budget_30d"] - sku_both["real_qty_30d"]).abs().mean()
    print(f"  涉及动销 SKU 数量: {len(sku_agg):,} 款")
    print(f"  预测与真实双阳 SKU: {len(sku_both):,} 款")
    print(f"  SKU 聚合 MAE:      {sku_mae:.1f} 件/款")
    
sku_agg["diff"] = sku_agg["ai_budget_30d"] - sku_agg["real_qty_30d"]
sku_agg["ratio"] = sku_agg["ai_budget_30d"] / sku_agg["real_qty_30d"].replace(0, np.nan)
sku_agg = sku_agg.sort_values("real_qty_30d", ascending=False).reset_index()

print(f"\n  Top 15 款式（全国汇总真实销准对比）:")
print(f"  {'SKU':<24} {'AI全国总推断':>12} {'真实全国满分':>12} {'误差':>10} {'Ratio':>7}")
print("  " + "-" * 68)
for _, row in sku_agg.head(15).iterrows():
    ratio_str = f"{row['ratio']:.2f}" if pd.notnull(row["ratio"]) else "∞"
    print(f"  {row['sku_id']:<24} {row['ai_budget_30d']:>12.1f} {row['real_qty_30d']:>12.1f} "
          f"{row['diff']:>10.1f} {ratio_str:>7}")

# 保存
os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
out = os.path.join(PROJECT_ROOT, "reports",
                   f"backtest_{ANCHOR.strftime('%Y%m%d')}.csv")
df_show.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n{SEP}")
print(f"  完整对比表已保存: {out}")
print(SEP)
