"""
B2B 补货系统 V2.3 -- 多维度完整评估脚本 (evaluate_v2_full.py)
============================================================
维度1: 分类性能 (Classification)     -- 模型能否正确识别"要不要补货"
维度2: 回归性能 (Regression)         -- 模型能否预测"补多少件"
维度3: 业务充盈率 (Business Ratio)   -- AI 总预算 / 实际总补货的大盘比例
维度4: 分位数误差 (Quantile Errors)  -- 不同补货量级区间的精度分层
维度5: 案例展示 (Case Studies)       -- 直观对比最好/最坏的典型样本
维度6: 超参数摘要 (Config Summary)   -- 记录本次评估使用的超参数配置

使用方法: python evaluate_v2_full.py
输出:     控制台打印 + reports/eval_report_v2_{date}.txt
"""
import sys
import os
import json
import pickle
import time
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
)

# 添加源目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src', 'train'))

from src.models.enhanced_model import EnhancedTwoTowerLSTM
from dataset import create_lazy_dataloaders

# Windows GBK 终端兼容: 强制使用 UTF-8 输出
sys.stdout.reconfigure(encoding='utf-8')

# ============ 全局常量与环境路由 ============
EXP_VERSION = os.environ.get('EXP_VERSION', 'v3').lower()
if EXP_VERSION == 'v5':
    ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v5')
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_v5')
    MODEL_DIR     = os.path.join(PROJECT_ROOT, 'models_v5')
    META_NAME     = 'meta_v5.json'
    LE_NAME       = 'label_encoders_v5.pkl'
    print(f"[*] 评估模式: V5 (10维新特征)")
elif EXP_VERSION == 'v5_lite':
    ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v5_lite')
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_v5_lite')
    MODEL_DIR     = os.path.join(PROJECT_ROOT, 'models_v5_lite')
    META_NAME     = 'meta_v5_lite.json'
    LE_NAME       = 'label_encoders_v5_lite.pkl'
    print(f"[*] 评估模式: V5-lite (6维: 补货3维 + 期货3维)")
elif EXP_VERSION == 'v5_lite_cov':
    ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v5_lite_cov')
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_v5_lite_cov')
    MODEL_DIR     = os.path.join(PROJECT_ROOT, 'models_v5_lite_cov')
    META_NAME     = 'meta_v5_lite_cov.json'
    LE_NAME       = 'label_encoders_v5_lite_cov.pkl'
    print(f"[*] 璇勪及妯″紡: V5-lite-cov (6缁? sparse sequence + buyer coverage)")
elif EXP_VERSION == 'v3_filtered':
    ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v3_filtered')
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_v3_filtered')
    MODEL_DIR     = os.path.join(PROJECT_ROOT, 'models_v3_filtered')
    META_NAME     = 'meta_v3_filtered.json'
    LE_NAME       = 'label_encoders_v3_filtered.pkl'
    print(f"[*] 评估模式: V3-filtered (7维原始特征，同 V5-lite SKU 宇宙)")
else:
    ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'data', 'artifacts_v3')
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_v3')
    MODEL_DIR     = os.path.join(PROJECT_ROOT, 'models_v2')
    META_NAME     = 'meta_v2.json'
    LE_NAME       = 'label_encoders_v2.pkl'
    print(f"[*] 评估模式: V3 (7维基线特征)")

ARTIFACTS_DIR = os.environ.get('EXP_ARTIFACTS_DIR', ARTIFACTS_DIR)
PROCESSED_DIR = os.environ.get('EXP_PROCESSED_DIR', PROCESSED_DIR)
MODEL_DIR = os.environ.get('EXP_MODEL_DIR', MODEL_DIR)
META_NAME = os.environ.get('EXP_META_NAME', META_NAME)
LE_NAME = os.environ.get('EXP_LE_NAME', LE_NAME)
MODEL_FILE    = os.environ.get('EXP_MODEL_FILE', 'best_enhanced_model.pth')
MODEL_PATH    = os.environ.get('EXP_MODEL_PATH', os.path.join(MODEL_DIR, MODEL_FILE))
REPORTS_DIR   = os.environ.get('EXP_REPORT_DIR', os.path.join(PROJECT_ROOT, 'reports'))
os.makedirs(REPORTS_DIR, exist_ok=True)

# 从 YAML 动态加载最新的超参数配置 (并允许环境变量覆盖)
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'model_config.yaml')
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        __config = yaml.safe_load(f)
        
    HYPERPARAMS = {
        'LSTM_HIDDEN'    : int(os.environ.get('EXP_HIDDEN', __config['model'].get('hidden_size', 256))),
        'LSTM_LAYERS'    : int(os.environ.get('EXP_LAYERS', __config['model'].get('num_layers', 2))),
        'MODEL_TYPE'     : os.environ.get('EXP_MODEL_TYPE', 'lstm').lower(),
        'DROPOUT'        : float(os.environ.get('EXP_DROPOUT', __config['model'].get('dropout', 0.3))),
        'BATCH_SIZE'     : int(os.environ.get('EXP_BATCH', __config['train'].get('batch_size', 1024))),
        'LEARNING_RATE'  : float(os.environ.get('EXP_LR', __config['train'].get('learning_rate', 0.0002))),
        'EPOCHS'         : int(os.environ.get('EXP_EPOCHS', __config['train'].get('epochs', 30))),
        'EARLY_STOPPING' : int(os.environ.get('EXP_PATIENCE', __config['train'].get('early_stopping_patience', 5))),
        'EXP_VERSION'    : EXP_VERSION,
        'MONITOR_METRIC' : os.environ.get('EXP_MONITOR_METRIC', 'loss').lower(),
        'SEED'           : os.environ.get('EXP_SEED', 'n/a'),
    }
    # 额外导出到全局变量方便调用
    EXP_ID = os.environ.get('EXP_ID', 'default_v2')
except Exception as e:
    HYPERPARAMS = {"Error": f"配置注入失败: {e}"}
    EXP_ID = "error_v2"

# ============ 工具函数 ============
QTY_GATE_THRESHOLD = float(os.environ.get('EXP_QTY_GATE', '0.15'))
QTY_SCALE = float(os.environ.get('EXP_QTY_SCALE', '1.2'))


class DummyLE:
    classes_ = np.arange(13)

def sep(title='', width=60, char='='):
    if title:
        side = (width - len(title) - 2) // 2
        print(f"\n{char * side} {title} {char * side}\n")
    else:
        print(char * width)

def load_model_and_metadata(device):
    """加载模型及元数据 (支持 V3/V5)"""
    with open(os.path.join(ARTIFACTS_DIR, META_NAME), 'r') as f:
        meta = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, LE_NAME), 'rb') as f:
        encoders = pickle.load(f)

    cat_cols = meta['static_cat_cols']
    num_cols = meta['static_num_cols']

    static_vocab_sizes = {c: len(encoders[c].classes_) + 5 for c in cat_cols if c in encoders}
    static_vocab_sizes['month'] = 18

    # [V3.1 动态工厂] 根据实验类型选择模型类
    from src.train.run_training_v2 import MODEL_REGISTRY
    ModelClass = MODEL_REGISTRY.get(HYPERPARAMS['MODEL_TYPE'], EnhancedTwoTowerLSTM)
    print(f"[*] 动态构建评估模型: {ModelClass.__name__} (hidden={HYPERPARAMS['LSTM_HIDDEN']}, layers={HYPERPARAMS['LSTM_LAYERS']})")

    dyn_feat_dim = meta.get('dyn_feat_dim', 7)
    
    model = ModelClass(
        dyn_feat_dim=dyn_feat_dim,
        lstm_hidden=HYPERPARAMS['LSTM_HIDDEN'],
        lstm_layers=HYPERPARAMS['LSTM_LAYERS'],
        static_vocab_sizes=static_vocab_sizes,
        static_emb_dim=16,
        num_numeric_feats=len(num_cols),
        dropout=HYPERPARAMS['DROPOUT']
    ).to(device)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"未找到模型文件: {MODEL_PATH} | EXP_VERSION={EXP_VERSION}"
        )
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model, meta, cat_cols, num_cols

def collect_predictions(model, val_loader, device, cat_cols, num_cols):
    """批量收集验证集上的所有预测结果"""
    all_actual_reg, all_pred_cls, all_pred_reg = [], [], []
    static_keys = cat_cols + ['month'] + num_cols
    first_order_idx = static_keys.index('qty_first_order') if 'qty_first_order' in static_keys else None

    print("  正在批量推断验证集...")
    use_non_blocking = device.type == 'cuda'
    with torch.no_grad():
        for i, (x_dyn, x_static, y_cls, y_reg) in enumerate(val_loader):
            x_dyn    = x_dyn.to(device, non_blocking=use_non_blocking)
            x_static = x_static.to(device, non_blocking=use_non_blocking)
            o_cls, o_reg = model(x_dyn, x_static, static_keys)
            
            # --- [甲方特供规则] 死库存与新款双轨判断 ---
            # 检查时序特征的最后 60 步 (index=0 是 log1p(qty_replenish))
            hist_sales = x_dyn[:, -60:, 0].sum(dim=1)
            
            # 提取首单初始量特征 (静态特征列第 12 维, 已过 log1p)
            if first_order_idx is not None:
                first_order = x_static[:, first_order_idx]
            else:
                first_order = torch.zeros_like(hist_sales)
            
            # 铁律：只有“不仅过去两个月卖不动”，而且“连首发上新调拨量都没有”的长尾烂款
            # 才会被认定为彻底的死库存予以强制熔断。只要有发货量或者首单量，就放行交给 AI 决策
            dead_mask = (hist_sales <= 1e-4) & (first_order <= 1e-4)
            o_reg[dead_mask] = 0.0
            o_cls[dead_mask] = -15.0  # Sigmoid 后概率极低，绝对拦截
            # ----------------------------------------
            
            all_actual_reg.append(y_reg.cpu().numpy())
            all_pred_cls.append(torch.sigmoid(o_cls).cpu().numpy())
            all_pred_reg.append(o_reg.cpu().numpy())
            if i % 20 == 0:
                print(f"    Batch {i} done...")

    y_true_reg  = np.concatenate(all_actual_reg).flatten()
    y_pred_prob = np.concatenate(all_pred_cls).flatten()
    y_pred_reg  = np.concatenate(all_pred_reg).flatten()

    # 标签解码与 [阈值/调配系数] 释放
    # [V4.3] 因为训练阶段已经上了极度苛刻的 FP 惩罚和死库存熔断，模型本身已经极其保守。
    # 所以在此处彻底摘掉以前旧版本为了防止爆仓而加上的 `* 0.8` 压舱枷锁，并将阈值放到 0.15 鼓励模型大胆放量
    actual_qty   = np.expm1(y_true_reg)
    y_true_cls   = (actual_qty > 0).astype(float)
    preds_clamped = np.clip(y_pred_reg, None, 8.5)
    pred_qty_raw  = np.expm1(preds_clamped) * (y_pred_prob > 0.15).astype(float) * 1.2  # 配合激进补偿机制
    pred_qty      = np.nan_to_num(pred_qty_raw, nan=0.0, posinf=0.0, neginf=0.0)

    return actual_qty, pred_qty, y_true_cls, y_pred_prob

# ============ 评估维度函数 ============

def collect_predictions_v2(model, val_loader, device, cat_cols, num_cols):
    """Collect predictions with explicit policy diagnostics."""
    all_actual_reg, all_pred_cls, all_pred_reg, all_dead_mask = [], [], [], []
    static_keys = cat_cols + ['month'] + num_cols
    first_order_idx = static_keys.index('qty_first_order') if 'qty_first_order' in static_keys else None

    print("  正在批量推断验证集..")
    use_non_blocking = device.type == 'cuda'
    with torch.no_grad():
        for i, (x_dyn, x_static, y_cls, y_reg) in enumerate(val_loader):
            x_dyn = x_dyn.to(device, non_blocking=use_non_blocking)
            x_static = x_static.to(device, non_blocking=use_non_blocking)
            o_cls, o_reg = model(x_dyn, x_static, static_keys)

            hist_sales = x_dyn[:, -60:, 0].sum(dim=1)
            if first_order_idx is not None:
                first_order = x_static[:, first_order_idx]
            else:
                first_order = torch.zeros_like(hist_sales)
            dead_mask = (hist_sales <= 1e-4) & (first_order <= 1e-4)
            o_reg[dead_mask] = 0.0
            o_cls[dead_mask] = -15.0

            all_actual_reg.append(y_reg.cpu().numpy())
            all_pred_cls.append(torch.sigmoid(o_cls).cpu().numpy())
            all_pred_reg.append(o_reg.cpu().numpy())
            all_dead_mask.append(dead_mask.cpu().numpy())
            if i % 20 == 0:
                print(f"    Batch {i} done...")

    y_true_reg = np.concatenate(all_actual_reg).flatten()
    y_pred_prob = np.concatenate(all_pred_cls).flatten()
    y_pred_reg = np.concatenate(all_pred_reg).flatten()
    dead_mask = np.concatenate(all_dead_mask).astype(bool).flatten()

    actual_qty = np.expm1(y_true_reg)
    y_true_cls = (actual_qty > 0).astype(float)
    preds_clamped = np.clip(y_pred_reg, None, 8.5)
    pred_qty_open = np.expm1(preds_clamped) * QTY_SCALE
    qty_gate_mask = (y_pred_prob > QTY_GATE_THRESHOLD)
    pred_qty = np.nan_to_num(pred_qty_open * qty_gate_mask.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    return {
        'actual_qty': actual_qty,
        'pred_qty': pred_qty,
        'pred_qty_open': np.nan_to_num(pred_qty_open, nan=0.0, posinf=0.0, neginf=0.0),
        'y_true_cls': y_true_cls,
        'y_pred_prob': y_pred_prob,
        'dead_mask': dead_mask,
        'qty_gate_mask': qty_gate_mask.astype(bool),
    }


def eval_dim1_classification_v2(y_true_cls, y_pred_prob, report_lines):
    """Classification metrics plus exported threshold metadata."""
    sep("维度1: 分类性能 (Classification Metrics)")

    precision, recall, thresholds = precision_recall_curve(y_true_cls, y_pred_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    if len(thresholds) == 0:
        best_threshold = 0.5
        best_f1 = 0.0
    else:
        best_idx = int(np.argmax(f1_scores[:-1]))
        best_threshold = float(thresholds[best_idx])
        best_f1 = float(f1_scores[best_idx])

    print(f"  [V3.0 定标] 自动寻找最优 F1 阈值..")
    print(f"  最佳 F1-Score: {best_f1:.4f}  (对应阈值: {best_threshold:.4f})")

    final_threshold = max(0.35, best_threshold)
    y_pred_cls = (y_pred_prob > final_threshold).astype(float)

    pos_count = y_true_cls.sum()
    neg_count = (1 - y_true_cls).sum()
    auc_roc = roc_auc_score(y_true_cls, y_pred_prob)
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    report = classification_report(
        y_true_cls, y_pred_cls, target_names=['不补货(NEG)', '补货(POS)'], digits=4
    )

    print(f"  验证集总样本:  {len(y_true_cls):>10,}")
    print(f"  正样本(有补货): {int(pos_count):>10,}  ({pos_count/len(y_true_cls)*100:.1f}%)")
    print(f"  负样本(无补货): {int(neg_count):>10,}  ({neg_count/len(y_true_cls)*100:.1f}%)")
    print()
    print(f"  ROC-AUC 分数:  {auc_roc:.4f}  (≥0.85 为优，≥0.90 为极优)")
    print()
    print("  混淆矩阵:")
    print(f"            预测:不补  预测:补货")
    print(f"  真实:不补  {cm[0,0]:>8,}   {cm[0,1]:>8,}  (FP = 误报补货)")
    print(f"  真实:补货  {cm[1,0]:>8,}   {cm[1,1]:>8,}  (FN = 漏报补货)")
    print()
    print("  详细分类报告:")
    for line in report.split('\n'):
        print(f"  {line}")

    report_lines.extend([
        "=== 维度1: 分类性能 ===",
        f"ROC-AUC: {auc_roc:.4f}",
        f"best_f1_threshold: {best_threshold:.4f}",
        f"final_cls_threshold: {final_threshold:.4f}",
        f"正样本: {int(pos_count)}, 负样本: {int(neg_count)}",
        f"混淆矩阵: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}",
        report
    ])
    return {
        'auc': float(auc_roc),
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'final_threshold': final_threshold,
        'y_pred_cls': y_pred_cls.astype(bool),
        'cm': cm,
    }


def eval_dim1b_policy_alignment(y_pred_prob, pred_qty, class_info, dead_mask, report_lines):
    """Show mismatch between classification thresholding and quantity-positive policy."""
    sep("维度1B: 决策口径一致性 (Policy Alignment)")

    cls_positive = class_info['y_pred_cls']
    qty_positive = pred_qty > 0
    cls_only = cls_positive & (~qty_positive)
    qty_only = qty_positive & (~cls_positive)
    both_positive = cls_positive & qty_positive
    both_negative = (~cls_positive) & (~qty_positive)

    print(f"  分类阈值:        {class_info['final_threshold']:.4f}")
    print(f"  数量门控阈值:    {QTY_GATE_THRESHOLD:.4f}")
    print(f"  数量缩放系数:    {QTY_SCALE:.2f}")
    print()
    print(f"  分类判正样本:    {int(cls_positive.sum()):>10,}")
    print(f"  数量>0样本:      {int(qty_positive.sum()):>10,}")
    print(f"  二者同时为正:    {int(both_positive.sum()):>10,}")
    print(f"  仅分类为正:      {int(cls_only.sum()):>10,}")
    print(f"  仅数量为正:      {int(qty_only.sum()):>10,}")
    print(f"  二者同时为负:    {int(both_negative.sum()):>10,}")
    print(f"  死库存硬拦截:    {int(dead_mask.sum()):>10,}")

    report_lines.extend([
        "\n=== 维度1B: Policy Alignment ===",
        f"final_cls_threshold: {class_info['final_threshold']:.4f}",
        f"qty_gate_threshold: {QTY_GATE_THRESHOLD:.4f}",
        f"qty_scale: {QTY_SCALE:.2f}",
        f"class_positive={int(cls_positive.sum())}",
        f"qty_positive={int(qty_positive.sum())}",
        f"both_positive={int(both_positive.sum())}",
        f"class_only={int(cls_only.sum())}",
        f"qty_only={int(qty_only.sum())}",
        f"dead_blocked={int(dead_mask.sum())}",
    ])


def eval_dim1_classification(y_true_cls, y_pred_prob, report_lines):
    """维度1: 分类性能 (补货意图识别能力)"""
    sep("维度1: 分类性能 (Classification Metrics)")

    # --- [NEW V3.0 自动阈值定标] ---
    precision, recall, thresholds = precision_recall_curve(y_true_cls, y_pred_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"  [V3.0 定标] 自动寻找最优 F1 阈值...")
    print(f"  最佳 F1-Score: {best_f1:.4f}  (对应阈值: {best_threshold:.4f})")
    
    # 使用自动寻优结果（如果太低则保底使用 0.35）
    final_threshold = max(0.35, best_threshold)
    y_pred_cls = (y_pred_prob > final_threshold).astype(float)
    # -----------------------------
    pos_count  = y_true_cls.sum()
    neg_count  = (1 - y_true_cls).sum()

    auc_roc = roc_auc_score(y_true_cls, y_pred_prob)
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    report = classification_report(y_true_cls, y_pred_cls, target_names=['不补货(NEG)', '补货(POS)'], digits=4)

    print(f"  验证集总样本:  {len(y_true_cls):>10,}")
    print(f"  正样本(有补货): {int(pos_count):>10,}  ({pos_count/len(y_true_cls)*100:.1f}%)")
    print(f"  负样本(无补货): {int(neg_count):>10,}  ({neg_count/len(y_true_cls)*100:.1f}%)")
    print()
    print(f"  ROC-AUC 分数:  {auc_roc:.4f}  (≥0.85 为优，≥0.90 为极优)")
    print()
    print("  混淆矩阵:")
    print(f"            预测:不补  预测:补货")
    print(f"  真实:不补  {cm[0,0]:>8,}   {cm[0,1]:>8,}  (FP = 误报补货)")
    print(f"  真实:补货  {cm[1,0]:>8,}   {cm[1,1]:>8,}  (FN = 漏报补货)")
    print()
    print("  详细分类报告:")
    for line in report.split('\n'):
        print(f"  {line}")

    lines = [
        "=== 维度1: 分类性能 ===",
        f"ROC-AUC: {auc_roc:.4f}",
        f"正样本: {int(pos_count)}, 负样本: {int(neg_count)}",
        f"混淆矩阵: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}",
        report
    ]
    report_lines.extend(lines)
    return auc_roc


def eval_dim2_regression(actual_qty, pred_qty, report_lines):
    """维度2: 回归性能 (补货件数预测精度)"""
    sep("维度2: 回归性能 (Regression Metrics)")

    abs_err = np.abs(pred_qty - actual_qty)
    sq_err  = (pred_qty - actual_qty) ** 2

    mae  = abs_err.mean()
    rmse = np.sqrt(sq_err.mean())
    # 仅在正样本上计算 (有真实需求的样本)
    pos_mask  = actual_qty > 0
    mae_pos   = abs_err[pos_mask].mean() if pos_mask.sum() > 0 else 0
    rmse_pos  = np.sqrt(sq_err[pos_mask].mean()) if pos_mask.sum() > 0 else 0
    # MAPE (仅正样本，防止除零)
    mape_vals = abs_err[pos_mask] / (actual_qty[pos_mask] + 1e-5)
    mape      = mape_vals.mean() * 100

    print(f"  [全样本含零值的误差]")
    print(f"  MAE  (平均绝对误差): {mae:.4f}  件/样本")
    print(f"  RMSE (均方根误差):   {rmse:.4f}  件/样本")
    print()
    print(f"  [仅在'有真实补货'的正样本上计算]  (共 {pos_mask.sum():,} 个)")
    print(f"  MAE  (正样本):  {mae_pos:.4f} 件/样本")
    print(f"  RMSE (正样本):  {rmse_pos:.4f} 件/样本")
    print(f"  MAPE (正样本):  {mape:.2f}%    (平均百分比误差)")

    lines = [
        f"\n=== 维度2: 回归性能 ===",
        f"全量 MAE: {mae:.4f}, RMSE: {rmse:.4f}",
        f"正样本 MAE: {mae_pos:.4f}, RMSE: {rmse_pos:.4f}, MAPE: {mape:.2f}%"
    ]
    report_lines.extend(lines)
    return mae


def eval_dim3_business_ratio(actual_qty, pred_qty, report_lines):
    """维度3: 业务充盈率 (大盘尺子)"""
    sep("维度3: 业务充盈率 (Business Ratio)")

    total_actual = actual_qty.sum()
    total_pred   = pred_qty.sum()
    ratio        = total_pred / (total_actual + 1e-5)

    if ratio < 0.8:
        status = "偏保守 (AI 总体低估，可能出现断货风险)"
    elif ratio <= 1.5:
        status = "健康区间 (AI 总量预测精准，供货充盈)"
    else:
        status = "偏激进 (AI 总体高估，可能出现库存积压)"

    print(f"  验证集真实总补货量: {total_actual:>12,.0f} 件")
    print(f"  AI 预测总预算:      {total_pred:>12,.0f} 件")
    print(f"  大盘充盈率 (Ratio): {ratio:12.4f}")
    print()
    print(f"  状态判定: [{status}]")
    print()
    print(f"  [业务解读]")
    status_text = "少" if ratio < 1.0 else "多"
    diff_pct = abs(1 - ratio) * 100
    print(f"  Ratio={ratio:.2f} 表示 AI 建议总采购量比实际真实需求{status_text} {diff_pct:.0f}%，")
    if ratio < 1.0:
        print(f"  意味着模型倾向于'防范死库存'，通过严格的门限压制虚报预算。")
    else:
        print(f"  意味着模型预算充沛，采取更为进取的策略以防爆款断货。")

    lines = [
        f"\n=== 维度3: 业务充盈率 ===",
        f"实际总补货: {total_actual:.0f} 件, AI 总预算: {total_pred:.0f} 件",
        f"Ratio: {ratio:.4f} | 状态: {status}"
    ]
    report_lines.extend(lines)
    return ratio


def eval_dim4_quantile(actual_qty, pred_qty, report_lines):
    """维度4: 分位数精度 (按需求量级分层评估)"""
    sep("维度4: 分位数分层误差 (Quantile Analysis)")

    df = pd.DataFrame({'actual': actual_qty, 'pred': pred_qty})
    df['abs_err'] = np.abs(df['pred'] - df['actual'])

    # 只统计正样本, 按真实量级分箱
    pos_df = df[df['actual'] > 0].copy()
    if len(pos_df) == 0:
        print("  无正样本, 跳过")
        return

    q25 = pos_df['actual'].quantile(0.25)
    q75 = pos_df['actual'].quantile(0.75)
    q95 = pos_df['actual'].quantile(0.95)

    bins = [0, q25, q75, q95, np.inf]
    labels = [
        f'微量款 (0~{q25:.0f}件)',
        f'常量款 ({q25:.0f}~{q75:.0f}件)',
        f'大量款 ({q75:.0f}~{q95:.0f}件)',
        f'爆款   (>{q95:.0f}件)'
    ]
    pos_df['group'] = pd.cut(pos_df['actual'], bins=bins, labels=labels)

    print(f"  {'区间':<30} {'样本量':>8} {'MAE':>10} {'RMSE':>10} {'MAPE':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    qty_lines = []
    for gp, grp_df in pos_df.groupby('group', observed=True):
        n    = len(grp_df)
        mae  = grp_df['abs_err'].mean()
        rmse = np.sqrt((grp_df['abs_err'] ** 2).mean())
        mape = (grp_df['abs_err'] / (grp_df['actual'] + 1e-5)).mean() * 100
        print(f"  {str(gp):<30} {n:>8,} {mae:>10.2f} {rmse:>10.2f} {mape:>9.1f}%")
        qty_lines.append(f"  {gp}: n={n}, MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape:.1f}%")

    lines = ["\n=== 维度4: 分位数分层误差 ==="] + qty_lines
    report_lines.extend(lines)


def eval_dim5_case_studies(actual_qty, pred_qty, y_pred_prob, report_lines):
    """维度5: 典型案例展示 (最好/最坏/最大预测)"""
    sep("维度5: 典型案例展示 (Case Studies)")

    df = pd.DataFrame({'actual': actual_qty, 'pred': pred_qty, 'prob': y_pred_prob})
    df['abs_err']  = np.abs(df['pred'] - df['actual'])
    df['err_rate'] = df['abs_err'] / (df['actual'] + 1)

    # 场景A: 在"有真实需求"里准确预测到的案例
    pos_df = df[df['actual'] >= 5].copy()
    if len(pos_df) > 10:
        best_cases = pos_df.nsmallest(5, 'abs_err')
        print("  [场景A] 真实补货≥5件时，AI 预测最接近的 Top 5 案例:")
        print(f"  {'实际补货':>10} {'AI预算':>10} {'绝对误差':>10} {'补货概率':>10}")
        print(f"  {'-'*44}")
        for _, r in best_cases.iterrows():
            print(f"  {r['actual']:>10.0f} {r['pred']:>10.0f} {r['abs_err']:>10.0f} {r['prob']*100:>9.1f}%")

    # 场景B: 在"无真实需求"里，AI 还是放量的虚报案例
    neg_df = df[df['actual'] == 0].copy()
    neg_high = neg_df[neg_df['pred'] > 0].nlargest(5, 'pred')
    if len(neg_high) > 0:
        print(f"\n  [场景B] 真实补货=0件时，AI 误报最严重的 Top 5 案例 (虚报单):")
        print(f"  {'实际补货':>10} {'AI误报预算':>12} {'补货概率':>10}")
        print(f"  {'-'*35}")
        for _, r in neg_high.iterrows():
            print(f"  {r['actual']:>10.0f} {r['pred']:>12.0f} {r['prob']*100:>9.1f}%")

    # 场景C: AI 最大胆预测的 Top 5，看是否命中
    top_preds = df.nlargest(5, 'pred')
    print(f"\n  [场景C] AI 整体最敢'放量'的 Top 5 预测 (对应真实结果):")
    print(f"  {'实际补货':>10} {'AI最大预算':>12} {'绝对误差':>10} {'补货概率':>10}")
    print(f"  {'-'*46}")
    for _, r in top_preds.iterrows():
        print(f"  {r['actual']:>10.0f} {r['pred']:>12.0f} {r['abs_err']:>10.0f} {r['prob']*100:>9.1f}%")

    report_lines.append("\n=== 维度5: 典型案例 (已打印至控制台) ===")


def eval_dim6_config_summary(report_lines):
    """维度6: 超参数与训练配置摘要"""
    sep("维度6: 超参数配置摘要 (Hyperparameter Summary)")

    print(f"  {'参数':<30} {'当前值'}")
    print(f"  {'-'*50}")
    for k, v in HYPERPARAMS.items():
        print(f"  {k:<30} {v}")

    print()
    print("  [模型架构]")
    print("  双塔 LSTM: 左塔 (LSTM 2层×128) + 右塔 (Embedding×10 + BN Dense)")
    print("  损失函数: TwoStageMaskedLoss (FocalLoss + 掩码 HuberLoss)")
    print()
    print("  [v1.7 说明] qty_first_order, price_tag 已进行 log1p 量纲修正")

    report_lines.append("\n=== 维度6: 超参数摘要 ===")
    for k, v in HYPERPARAMS.items():
        report_lines.append(f"  {k}: {v}")


# ============ 主入口 ============
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    print(f"评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    report_lines = [f"B2B 补货 V2.3 完整评估报告 | 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"]

    # 加载模型
    sep("正在加载模型与验证集...")
    model, meta, cat_cols, num_cols = load_model_and_metadata(device)
    eval_workers = max(0, int(os.environ.get('EXP_EVAL_WORKERS', os.environ.get('EXP_WORKERS', '2'))))
    _, val_loader = create_lazy_dataloaders(
        processed_dir=PROCESSED_DIR,
        batch_size=2048,
        num_workers=eval_workers,
        lookback=meta.get('lookback', 90),
        dyn_feat_dim=meta.get('dyn_feat_dim', 7)
    )

    # 收集预测
    pred_data = collect_predictions_v2(model, val_loader, device, cat_cols, num_cols)
    actual_qty = pred_data['actual_qty']
    pred_qty = pred_data['pred_qty']
    y_true_cls = pred_data['y_true_cls']
    y_pred_prob = pred_data['y_pred_prob']
    print(f"  完成. 共 {len(actual_qty):,} 个验证样本.")
    class_info = eval_dim1_classification_v2(y_true_cls, y_pred_prob, report_lines)
    eval_dim1b_policy_alignment(y_pred_prob, pred_qty, class_info, pred_data['dead_mask'], report_lines)

    # [V2.6] 导出明细 CSV 供用户分析 (12月验证集)
    import pandas as pd
    val_keys_path = os.path.join(ARTIFACTS_DIR, 'val_keys.csv')
    if os.path.exists(val_keys_path):
        val_keys_df = pd.read_csv(val_keys_path)
        detail_df = pd.DataFrame({
            'sku_id': val_keys_df['sku_id'],
            'anchor_date': val_keys_df['date'],
            'true_replenish_qty': actual_qty,
            'ai_pred_prob': y_pred_prob,
            'cls_pred_best_f1': class_info['y_pred_cls'].astype(int),
            'ai_pred_qty_open': pred_data['pred_qty_open'],
            'ai_pred_qty': pred_qty,
            'ai_pred_positive_qty': (pred_qty > 0).astype(int),
            'qty_gate_mask': pred_data['qty_gate_mask'].astype(int),
            'dead_blocked': pred_data['dead_mask'].astype(int),
            'abs_error': np.abs(pred_qty - actual_qty)
        })
        # [V3.1] 锁定 EXP_ID 避免多个实验共用一个最新的 CSV
        report_date = time.strftime('%Y%m%d_%H%M')
        detail_path = os.path.join(REPORTS_DIR, f"val_set_detailed_compare_{EXP_ID}_{report_date}.csv")
        detail_df.to_csv(detail_path, index=False, encoding='utf-8-sig')
        eval_meta = {
        'exp_id': EXP_ID,
        'exp_version': EXP_VERSION,
        'model_file': MODEL_FILE,
        'best_f1_threshold': class_info['best_threshold'],
        'final_cls_threshold': class_info['final_threshold'],
        'qty_gate_threshold': QTY_GATE_THRESHOLD,
        'qty_scale': QTY_SCALE,
        'dead_blocked_count': int(pred_data['dead_mask'].sum()),
            'hyperparams': HYPERPARAMS,
        }
        meta_path = os.path.join(REPORTS_DIR, f"eval_meta_{EXP_ID}_{report_date}.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(eval_meta, f, ensure_ascii=False, indent=2)
        print(f"  [V3.1] 实验 {EXP_ID} 预测明细已导出: {detail_path}")
    else:
        print(f"  [Warning] 未找到 {val_keys_path}，无法导出明细对齐表")

    # 执行每个评估维度
    auc  = class_info['auc']
    mae  = eval_dim2_regression(actual_qty, pred_qty, report_lines)
    ratio = eval_dim3_business_ratio(actual_qty, pred_qty, report_lines)
    eval_dim4_quantile(actual_qty, pred_qty, report_lines)
    eval_dim5_case_studies(actual_qty, pred_qty, y_pred_prob, report_lines)
    eval_dim6_config_summary(report_lines)

    # 终局评级
    sep("终局评级 (Final Grade)")
    print(f"  ROC-AUC: {auc:.4f}  |  MAE: {mae:.2f} 件  |  Ratio: {ratio:.2f}")
    # 针对 1.2 万真 SKU 极度稀疏噪音样本重新制定的打分制
    if auc >= 0.70 and 0.3 <= ratio <= 1.3:
        grade = "A  -- [真SKU界] 模型状态极度健康，具备极其纯净的实战防沉降排产能力"
    elif auc >= 0.65 and 0.2 <= ratio <= 2.0:
        grade = "B  -- [真SKU界] 模型可用，整体防守框架确立，但请留意长尾品类方差"
    else:
        grade = "C  -- [真SKU界] 未过槛，建议排查假阳性/数据穿透遗漏"
    print(f"  综合评级: {grade}")
    report_lines.extend([f"\n=== 终局评级 ===", f"ROC-AUC={auc:.4f}, MAE={mae:.2f}, Ratio={ratio:.2f}", grade])

    # 保存报告
    report_date = time.strftime('%Y%m%d_%H%M')
    out_path = os.path.join(REPORTS_DIR, f"eval_report_v2_{report_date}.txt")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  评估报告已保存至: {out_path}")
    sep()

if __name__ == '__main__':
    main()
