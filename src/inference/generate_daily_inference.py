import os
import sys
import torch
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm

# 使得可以直接从 src/inference/ 目录下或任意位置找到系统包
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.models.enhanced_model import EnhancedTwoTowerLSTM
from utils.common import load_yaml
from src.train.run_training import DummyLE  # 兼容从项目根目录运行

def generate_daily_report():
    print("=" * 80)
    print("🚀 B2B 智能补货系统 - [真实业务时空: 站立于 '今天' 的 30 天预算发单榜]")
    print("=" * 80)

    # 1. 加载配置与模型架构定义
    config_path = os.path.join(PROJECT_ROOT, 'config', 'model_config.yaml')
    config = load_yaml(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 使用计算引擎: {device}")

    # 2. 读取 1.5 版的统一金库表
    data_path = os.path.join(PROJECT_ROOT, 'data', 'gold', 'wide_table_sku.csv')
    print(f"[*] 连通企业宽表中 ({data_path}) ...")
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # ====== 【核心大招】锁定时间锚点（上帝视角的剥离） ======
    latest_date = df['date'].max()
    print(f"[*] 🕒 【时空冻结】检测到您的最新订单存货系统更新至: {latest_date.date()}")
    print(f"[*] AI 将完全站立于此日，放弃对未来的已知干预，利用前置 120 天缓冲，纯预测接下来 30 天的需求！")
    
    lookback = 90 # 历史输入序列长度
    buffer_days = 120 # [30天缓冲池 + 90天可见区]
    
    # 抽取 120 天前到最新最新一天的全表切片
    start_date = latest_date - pd.Timedelta(days=buffer_days - 1)
    df_recent = df[df['date'] >= start_date].copy()
    
    # 3. 准备解码器 (为了获取类的多少、最后需要反查原文字)
    encoder_path = os.path.join(PROJECT_ROOT, 'data', 'artifacts', 'label_encoders.pkl')
    with open(encoder_path, 'rb') as f:
        encoders = pickle.load(f)

    # 提取静态表
    static_cat_cols = ['buyer_id', 'sku_id', 'style_id', 'product_name', 'category', 
                       'sub_category', 'season', 'series', 'band', 'size_id', 'color_id']
    static_num_cols = [
        'qty_first_order', 
        'price_tag',
        'cooperation_years',
        'monthly_average_replenishment',
        'avg_discount_rate',
        'replenishment_frequency',
        'item_coverage_rate'
    ]
    
    # ====== 将源数据的文字映射为字典编号以进入网络 ======
    for c in static_cat_cols:
        if c in df_recent.columns:
            le = encoders[c]
            # 极速矢量化：避免原版全表按行双内层循环造成的近乎死机的卡顿
            known_mask = df_recent[c].astype(str).isin(le.classes_)
            df_recent.loc[known_mask, c] = le.transform(df_recent.loc[known_mask, c].astype(str))
            df_recent.loc[~known_mask, c] = 0
    
    print("[*] 正在穿梭 160万行 记录，极速组装静态档案与 120 天时序...")
    static_dict = {}
    daily_sales_dict = {}
    
    for row in tqdm(df_recent.itertuples(index=False), desc="扫描记录", total=len(df_recent)):
        key = (row.buyer_id, row.sku_id)
        
        if key not in static_dict:
            arr = []
            for col in static_cat_cols: arr.append(getattr(row, col))
            arr.append(latest_date.month)
            for col in static_num_cols:
                val = getattr(row, col)
                val = float(val) if pd.notnull(val) else 0.0
                arr.append(np.log1p(max(0.0, val)))
            static_dict[key] = np.array(arr, dtype=np.float32)
            daily_sales_dict[key] = {}
            
        daily_sales_dict[key][row.date] = (
            getattr(row, 'qty_replenish', 0), getattr(row, 'qty_debt', 0),
            getattr(row, 'qty_shipped', 0), getattr(row, 'qty_inbound', 0)
        )

    x_dyn_list = []
    x_static_list = []
    valid_keys = []
    
    print(f"[*] 解析完毕，准备构建 {len(static_dict)} 组特征张量...")
    for (buyer, sku), daily_sales in tqdm(daily_sales_dict.items(), desc="提取特征张量"):
            
        hist_buffer = np.zeros((120, 4), dtype=np.float32)
        for b in range(120):
            d = start_date + pd.Timedelta(days=b)
            if d in daily_sales: hist_buffer[b] = daily_sales[d]
                
        cum_repl = np.cumsum(np.insert(hist_buffer[:, 0], 0, 0.0))
        cum_ship = np.cumsum(np.insert(hist_buffer[:, 2], 0, 0.0))
        
        window_dyn = np.zeros((90, 7), dtype=np.float32)
        window_dyn[:, 0] = np.log1p(np.maximum(hist_buffer[30:, 0], 0))
        window_dyn[:, 1] = np.log1p(np.maximum(cum_repl[31:121] - cum_repl[24:114], 0))
        window_dyn[:, 2] = np.log1p(np.maximum(cum_repl[31:121] - cum_repl[1:91], 0))
        window_dyn[:, 3] = np.log1p(np.maximum(hist_buffer[30:, 1], 0))
        window_dyn[:, 4] = np.log1p(np.maximum(hist_buffer[30:, 2], 0))
        window_dyn[:, 5] = np.log1p(np.maximum(cum_ship[31:121] - cum_ship[24:114], 0))
        window_dyn[:, 6] = np.log1p(np.maximum(hist_buffer[30:, 3], 0))
        
        x_dyn_list.append(window_dyn)
        x_static_list.append(static_dict[(buyer, sku)])
        valid_keys.append((buyer, sku))
        
    x_dyn_tensor = torch.tensor(np.stack(x_dyn_list), dtype=torch.float32)
    x_static_tensor = torch.tensor(np.stack(x_static_list), dtype=torch.float32)
    
    print(f"[*] 张量转化成功。总计捕捉到有效门店+SKU对决矩阵: {len(valid_keys):,} 个")

    # 4. 构建模型！
    static_order_keys = static_cat_cols + ['month'] + static_num_cols
    dynamic_vocab_sizes = {
        'buyer_id': 1000, 'sku_id': 15000, 'style_id': 3000, 'product_name': 2000, 
        'category': 50, 'sub_category': 100, 'season': 10, 'series': 50,
        'band': 50, 'size_id': 50, 'color_id': 100, 'month': 13
    }
    for col, le in encoders.items():
        if col in dynamic_vocab_sizes: dynamic_vocab_sizes[col] = len(le.classes_) + 5
            
    model = EnhancedTwoTowerLSTM(
        dyn_feat_dim=7, lstm_hidden=config['model']['hidden_size'],
        lstm_layers=config['model']['num_layers'], static_vocab_sizes=dynamic_vocab_sizes,
        static_emb_dim=16, num_numeric_feats=len(static_num_cols), dropout=0.0
    ).to(device)

    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_enhanced_model.pth')
    print(f"[*] 植入已训练完成的大脑 (1.5 版终极权重): {model_path}")
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    new_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 5. 放行大批量流数据预测
    dataset = torch.utils.data.TensorDataset(x_dyn_tensor, x_static_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=False)
    
    all_probs = []
    all_preds_qty = []
    
    print("[*] 发动机点火：高速计算全国各仓 30 天后库存需求...")
    with torch.no_grad():
        for b_dyn, b_static in tqdm(loader, desc="模型大脑运算中"):
            b_dyn = b_dyn.to(device)
            b_static = b_static.to(device)
            logits, preds = model(b_dyn, b_static, static_order_keys)
            
            prob = torch.sigmoid(logits)
            
            # 【基础死区掩码】这 90 天内纯空白（毫无历史信息）
            dead_mask = (b_dyn.abs().sum(dim=(1,2)) < 1e-4)
            
            # 【🔥 V4.4 改进：静默期退市掩码】最近 21 天没有任何流水动作（销量、发货等全0）
            silent_mask = (b_dyn[:, -21:, :].abs().sum(dim=(1, 2)) < 1e-4)
            
            # 【🔥 V4.4 改进：首发款白名单保护】
            # index 12 代表 qty_first_order 首当配货量。如果有首发动作必定是新品或季度爆款激活，免死放行。
            first_order = b_static[:, 12]
            new_item_mask = (first_order > 1e-4)
            
            # 取并集后排除新品
            final_mask = (dead_mask | silent_mask) & (~new_item_mask)
            prob[final_mask] = 0.0
            
            # [V4.4 最终实战系数] 配合底座 0.35 的深重 FP 惩罚，前端大胆放量 (prob>0.15, scale=1.2)
            preds = torch.clamp(preds, max=8.5)
            pred_qty = torch.expm1(preds) * (prob > 0.15).float() * 1.2
            
            all_probs.extend(prob.cpu().numpy().flatten())
            # inf/nan 安全过滤
            pq = pred_qty.cpu().numpy().flatten()
            pq = np.nan_to_num(pq, nan=0.0, posinf=0.0, neginf=0.0)
            all_preds_qty.extend(pq)

    # 6. 建表逆编码反送回人类社会
    df_output = pd.DataFrame(valid_keys, columns=['buyer_idx', 'sku_idx'])
    df_output['prediction_anchor_date'] = latest_date.strftime('%Y-%m-%d')
    df_output['ai_replenish_prob_30d'] = np.array(all_probs).round(3)
    df_output['ai_budget_qty_30d'] = np.array(all_preds_qty).round(1)

    print("[*] 把密码子翻译为仓库大姐能看懂的文字...")
    df_output['buyer_id'] = encoders['buyer_id'].inverse_transform(df_output['buyer_idx'])
    df_output['sku_id'] = encoders['sku_id'].inverse_transform(df_output['sku_idx'])
    
    # 丢掉那些被拦截至零、本就没发生销售的没价值数据
    mask = df_output['ai_budget_qty_30d'] > 0
    df_focused = df_output[mask].drop(columns=['buyer_idx', 'sku_idx']).copy()
    
    # 将最重要的预测行提前
    df_focused = df_focused[['prediction_anchor_date', 'buyer_id', 'sku_id', 'ai_replenish_prob_30d', 'ai_budget_qty_30d']]
    
    # 打印数值健康摘要
    pos_preds = df_focused[df_focused['ai_budget_qty_30d'] > 0]['ai_budget_qty_30d']
    if len(pos_preds) > 0:
        print(f"[*] 📊 预测件数分布: 中位数={pos_preds.median():.1f}件, 90%={pos_preds.quantile(0.9):.1f}件, 最大={pos_preds.max():.1f}件")
    
    # 保存明细预测表
    os.makedirs(os.path.join(PROJECT_ROOT, 'reports'), exist_ok=True)
    out_file = os.path.join(PROJECT_ROOT, 'reports', f'daily_orders_{latest_date.strftime("%Y%m%d")}.csv')
    df_focused.to_csv(out_file, index=False, encoding='utf-8-sig')

    # 🚀 业务交付核心：聚合出全国总仓 SKU 备货报表 🚀
    sku_agg = df_focused.groupby('sku_id')['ai_budget_qty_30d'].sum().reset_index()
    sku_agg = sku_agg.sort_values(by='ai_budget_qty_30d', ascending=False)
    sku_agg.insert(0, 'prediction_anchor_date', latest_date.strftime('%Y-%m-%d'))
    sku_agg.rename(columns={'ai_budget_qty_30d': 'national_total_qty_30d'}, inplace=True)
    
    sku_out_file = os.path.join(PROJECT_ROOT, 'reports', f'daily_sku_replenish_{latest_date.strftime("%Y%m%d")}.csv')
    sku_agg.to_csv(sku_out_file, index=False, encoding='utf-8-sig')

    # ★ V1.8 Demo 额外存档：同步复制到 reports/v1.8_demo/ 统一演示目录
    demo_dir = os.path.join(PROJECT_ROOT, 'reports', 'v1.8_demo')
    os.makedirs(demo_dir, exist_ok=True)
    demo_buyer_file = os.path.join(demo_dir, f'daily_orders_{latest_date.strftime("%Y%m%d")}.csv')
    df_focused.to_csv(demo_buyer_file, index=False, encoding='utf-8-sig')
    demo_sku_file = os.path.join(demo_dir, f'daily_sku_replenish_{latest_date.strftime("%Y%m%d")}.csv')
    sku_agg.to_csv(demo_sku_file, index=False, encoding='utf-8-sig')

    print("\n" + "=" * 80)
    print(f"🎉 真实排产大表成功铸造！")
    print(f"📌 预测起点时间戳 : {latest_date.strftime('%Y-%m-%d 23:59:59')}")
    print(f"📌 [明细版] 买手级指令 : {len(df_focused):,} 条 👉 {out_file}")
    print(f"📌 [汇总版] 仓具备货单 : {len(sku_agg):,} 款 SKU 👉 {sku_out_file} (甲方核心诉求)")
    print(f"📌 [Demo 存档] 统一演示目录 👉 {demo_dir}")
    print("=" * 80)

if __name__ == "__main__":
    generate_daily_report()
