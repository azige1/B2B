"""
B2B 时序补货模型的中央训练器 (Trainer Registry)
"""
import torch
import os
import time
import csv
import json
from tqdm import tqdm
from src.models.loss import TwoStageMaskedLoss

def _move_tensor(tensor, device):
    return tensor.to(device, non_blocking=(device.type == 'cuda'))


def _to_jsonable(value):
    """Recursively coerce numpy / torch scalar-like values into JSON-safe natives."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    return value

def train_one_epoch(model, dataloader, optimizer, criterion, device, static_order_keys, amp_scaler):
    model.train()
    total_loss, total_cls, total_reg = 0.0, 0.0, 0.0
    use_amp = device.type == 'cuda'
    
    pbar = tqdm(dataloader, desc="Training", unit="batch")
    for x_dyn, x_static, y_cls, y_reg in pbar:
        x_dyn    = _move_tensor(x_dyn, device)
        x_static = _move_tensor(x_static, device)
        y_cls    = _move_tensor(y_cls, device)
        y_reg    = _move_tensor(y_reg, device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits, preds = model(x_dyn, x_static, static_order_keys)
            loss, loss_cls, loss_reg = criterion(logits, preds, y_cls, y_reg)
        
        # amp_scaler 跨 epoch 复用，保留缩放因子状态
        amp_scaler.scale(loss).backward()
        amp_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        amp_scaler.step(optimizer)
        amp_scaler.update()
        
        total_loss += loss.item()
        total_cls  += loss_cls.item()
        total_reg  += loss_reg.item()
        
        pbar.set_postfix({
            'L_All': f"{loss.item():.7f}",
            'L_Cls': f"{loss_cls.item():.7f}",
            'L_Reg': f"{loss_reg.item():.7f}"
        })
        
    num_batches = len(dataloader)
    return total_loss / num_batches, total_cls / num_batches, total_reg / num_batches


@torch.no_grad()
def validate(model, dataloader, criterion, device, static_order_keys):
    model.eval()
    total_loss, total_cls, total_reg = 0.0, 0.0, 0.0
    
    abs_errors = []
    all_pred_qty_list = []  # ★ 方案A: 收集全量预测值
    all_true_qty_list = []  # ★ 方案A: 收集全量真实值
    global_true_qty = 0.0
    global_pred_qty = 0.0
    global_pos_orders = 0

    # ★ 同步收集 logits/labels，避免为 F1 计算再次遍历 val_loader
    all_probs  = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Validating", unit="batch")
    for x_dyn, x_static, y_cls, y_reg in pbar:
        x_dyn = _move_tensor(x_dyn, device)
        x_static = _move_tensor(x_static, device)
        y_cls = _move_tensor(y_cls, device)
        y_reg = _move_tensor(y_reg, device)
        
        logits, preds = model(x_dyn, x_static, static_order_keys)
        loss, loss_cls, loss_reg = criterion(logits, preds, y_cls, y_reg)
        
        total_loss += loss.item()
        total_cls += loss_cls.item()
        total_reg += loss_reg.item()
        
        prob = torch.sigmoid(logits)
        prob_threshold = 0.45
        pred_qty = torch.expm1(preds) * (prob > prob_threshold).float() * 1.0  
        true_qty = torch.expm1(y_reg)
        
        abs_err = torch.abs(pred_qty - true_qty)
        abs_errors.append(abs_err.cpu().numpy())
        
        global_true_qty += true_qty.sum().item()
        global_pred_qty += pred_qty.sum().item()
        global_pos_orders += (true_qty > 0).sum().item()
        
        # ★ 方案A: 收集样本级 (pred, true) 供 Ratio 聚合
        pred_np = pred_qty.cpu().numpy().flatten()
        true_np = true_qty.cpu().numpy().flatten()
        all_pred_qty_list.append(pred_np)
        all_true_qty_list.append(true_np)

        # ★ 顺便收集 prob/label，供后续 F1 计算（无额外 forward 开销）
        all_probs.extend(prob.cpu().numpy().flatten())
        all_labels.extend(y_cls.cpu().numpy().flatten())
            
        pbar.set_postfix({
            'V_All': f"{loss.item():.7f}", 
            'V_Cls': f"{loss_cls.item():.7f}", 
            'V_Reg': f"{loss_reg.item():.7f}"
        })

    import numpy as np
    num_batches = len(dataloader)
    mean_mae = np.mean(np.concatenate(abs_errors)) if abs_errors else 0.0

    # ★ 方案A: SKU 级中位 Ratio
    # 把所有样本的 (pred, true) 都拿到，按正样本筛选后取中位数
    all_pred_arr = np.concatenate(all_pred_qty_list) if all_pred_qty_list else np.array([])
    all_true_arr = np.concatenate(all_true_qty_list) if all_true_qty_list else np.array([])
    pos_mask = all_true_arr > 0
    if pos_mask.sum() > 0:
        # 方案A: 直接用样本级 ratio 的中位数（训练时无 SKU ID 可用）
        # 完整的 SKU 聚合在 evaluate.py 中实现
        sample_ratios = all_pred_arr[pos_mask] / all_true_arr[pos_mask]
        median_ratio = float(np.median(sample_ratios))
        # WMAPE: sum(|pred-true|) / sum(true)
        wmape = float(np.sum(np.abs(all_pred_arr - all_true_arr)) / (np.sum(all_true_arr) + 1e-9))
    else:
        median_ratio = 0.0
        wmape = 0.0

    return (
        total_loss / num_batches,
        total_cls / num_batches,
        total_reg / num_batches,
        mean_mae, median_ratio, global_true_qty, global_pred_qty, global_pos_orders,
        np.array(all_probs), np.array(all_labels), wmape
    )


def fit_model(model, train_loader, val_loader, config_epochs, learning_rate, device, static_order_keys, save_dir, val_every=1, patience_limit=10):
    """实施完整的训练管线"""
    import numpy as np
    
    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    
    print("=" * 60)
    print("  B2B Replenishment - Two-Stage Training Pipeline")
    print("=" * 60)
    print(f"  Device        : {device}")
    print(f"  Train samples : {n_train:,}")
    print(f"  Val   samples : {n_val:,}")
    print(f"  Batch size    : {train_loader.batch_size}")
    print(f"  Learning rate : {learning_rate}")
    print(f"  Max epochs    : {config_epochs}")
    print(f"  Patience      : {patience_limit}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU           : {gpu_name} ({total_mem:.1f} GB)")
    print("=" * 60)
    
    model = model.to(device)
    
    # AMP GradScaler 在这里创建一次，跨 epoch 复用（保留缩放因子学习状态）
    use_amp = device.type == 'cuda'
    amp_scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # LR Warmup (前 3 轮预热)
    warmup_epochs = 3
    base_lr = learning_rate
    warmup_start_lr = max(base_lr * 0.1, 1e-5)  # 下限保护：防止 lr=1e-5 时 warmup 起始太小
    # 注意: optimizer 在下方 env_weight_decay 分支中创建（Adam / AdamW 二选一）
    
    # [V2.1 优化] 遵照用户嘱咐不修改 Loss 底层逻辑：
    # 我们废除了采样器，数据回档到 85.4% 负样本 : 14.6% 正样本。
    # 根据严格分布：pos_weight = 85.4 / 14.6 ≈ 5.85，天然补偿稀疏正类的被忽视问题，而不制造幻觉。
    # [V2.1 优化] 核心分布矫正 (Calibration)：
    # 训练集正负比为 62.1%    # [V3.0 彻夜挂机自动化流水线改造]
    env_pos_weight = float(os.environ.get('EXP_POS_WEIGHT', '5.85'))
    env_fp_penalty = float(os.environ.get('EXP_FP_PENALTY', '0.15'))
    
    # [V3.0 新增] 第四期新增 Loss 超参
    env_reg_weight   = float(os.environ.get('EXP_REG_WEIGHT',   '0.5'))
    env_soft_f1      = float(os.environ.get('EXP_SOFT_F1',      '0.5'))
    env_huber        = float(os.environ.get('EXP_HUBER',        '1.5'))
    # ★ Phase 5 新增: Label Smoothing + Weight Decay
    env_label_smooth = float(os.environ.get('EXP_LABEL_SMOOTH', '0.0'))
    env_weight_decay = float(os.environ.get('EXP_WEIGHT_DECAY', '0.0'))

    print("=" * 60)
    print(f"  [实验热注入] pos_weight={env_pos_weight} | reg_fp={env_fp_penalty}")
    print(f"  [Loss热注入] reg_weight={env_reg_weight} | soft_f1={env_soft_f1} | huber={env_huber}")
    if env_label_smooth > 0:
        print(f"  [★ Phase5]  label_smooth={env_label_smooth} | weight_decay={env_weight_decay}")
    print("=" * 60)

    criterion = TwoStageMaskedLoss(
        cls_weight=1.0,  reg_weight=env_reg_weight, huber_delta=env_huber,
        pos_weight=env_pos_weight, fp_penalty=env_fp_penalty,
        soft_f1_coeff=env_soft_f1, label_smoothing=env_label_smooth  # ★
    ).to(device)

    # 优化器: 有 weight_decay 时用 AdamW，否则用 Adam
    if env_weight_decay > 0:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=warmup_start_lr,
                                      weight_decay=env_weight_decay)
        print(f"  [Optimizer] AdamW | weight_decay={env_weight_decay}")
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=warmup_start_lr,
                                     weight_decay=1e-5)
    
    # [v2.3 深度调优] 因子调小，过程更平滑
    monitor_metric = os.environ.get('EXP_MONITOR_METRIC', 'loss').lower()
    if monitor_metric not in {'loss', 'f1', 'wmape'}:
        print(f"  [Monitor] unsupported={monitor_metric}, fallback -> loss")
        monitor_metric = 'loss'
    scheduler_mode = 'max' if monitor_metric == 'f1' else 'min'
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=scheduler_mode, factor=0.3, patience=3, min_lr=1e-6
    )

    best_val_loss = float('inf')
    best_mae = float('inf')
    best_f1 = -1.0
    best_wmape = float('inf')
    best_monitor = float('-inf') if monitor_metric == 'f1' else float('inf')
    best_epochs = {'loss': None, 'f1': None, 'wmape': None, 'monitor': None}
    patience_counter = 0

    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, 'best_enhanced_model.pth')
    best_loss_model_path = os.path.join(save_dir, 'best_loss_enhanced_model.pth')
    best_f1_model_path = os.path.join(save_dir, 'best_f1_enhanced_model.pth')
    best_wmape_model_path = os.path.join(save_dir, 'best_wmape_enhanced_model.pth')
    last_model_path = os.path.join(save_dir, 'last_enhanced_model.pth')

    exp_id = os.environ.get('EXP_ID', 'default_exp')
    exp_seed = os.environ.get('EXP_SEED')
    reports_dir = os.path.join(os.path.dirname(save_dir), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    history_csv = os.path.join(reports_dir, f'history_{exp_id}.csv')
    selection_meta_path = os.path.join(reports_dir, f'model_selection_{exp_id}.json')
    with open(history_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'lr', 'train_loss', 'val_loss', 'val_f1', 'val_precision', 'val_recall', 'val_ratio', 'val_wmape', 'duration_min'])
    
    for epoch in range(1, config_epochs + 1):
        try:
            # LR Warmup 手动调整
            if epoch <= warmup_epochs:
                # 线性预热
                current_lr = warmup_start_lr + (base_lr - warmup_start_lr) * ((epoch - 1) / max(1, warmup_epochs - 1))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                print(f"  🔥 [Warmup] Epoch {epoch}/{warmup_epochs}: 调整 LR 到 {current_lr:.6f}")
            else:
                current_lr = optimizer.param_groups[0]['lr']
                
            # V3.5 加速补丁：前 2 轮 Warmup 期跳过全量验证
            do_val = (epoch >= 3) and ((epoch % val_every == 0) or (epoch == config_epochs))
            print(f"\n[Epoch {epoch}/{config_epochs}]  LR: {current_lr:.6f}  {'(+Val)' if do_val else '(skip val)'}")
            epoch_start = time.time()
            
            # 1. 训练阶段（每轮必跑）
            t0 = time.time()
            t_loss, t_cls, t_reg = train_one_epoch(
                model, train_loader, optimizer, criterion, device, static_order_keys, amp_scaler)
            train_secs = time.time() - t0
            
            if not do_val:
                # 跳过验证的轮次：只打印训练信息
                print(f"  Train Loss : {t_loss:.7f} (Cls={t_cls:.7f}, Reg={t_reg:.7f})")
                print(f"  Training   : {train_secs:.0f}s  |  下次验证: Epoch {epoch + (val_every - epoch % val_every)}")
                continue
            
            # 2. 评估阶段（validate 同时返回 logits/labels，避免二次遍历）
            t1 = time.time()
            v_loss, v_cls, v_reg, v_mae, v_ratio, sum_true, sum_pred, pos_count, all_probs, all_labels, v_wmape = \
                validate(model, val_loader, criterion, device, static_order_keys)
            val_secs = time.time() - t1
            epoch_secs = time.time() - epoch_start
            
            # 3. 详细日志输出
            print("=" * 60)
            print(f"  EPOCH {epoch}/{config_epochs}  |  LR: {current_lr:.6f}")
            print(f"  Training   : {train_secs/60:.1f} min ({train_secs:.0f}s)")
            print(f"  Validating : {val_secs/60:.1f} min ({val_secs:.0f}s)")
            print(f"  Epoch Total: {epoch_secs/60:.1f} min")
            remaining_epochs = config_epochs - epoch
            eta_secs = remaining_epochs * (train_secs + val_secs / val_every)
            eta_str = f"{eta_secs/3600:.1f}h" if eta_secs > 3600 else f"{eta_secs/60:.0f}min"
            print(f"  ETA (est.) : ~{eta_str} ({remaining_epochs} epochs left)")
            print("-" * 60)
            print(f"  Loss       : Train={t_loss:.7f}  Val={v_loss:.7f}")
            print(f"  Cls Loss   : Train={t_cls:.7f}  Val={v_cls:.7f}")
            print(f"  Reg Loss   : Train={t_reg:.7f}  Val={v_reg:.7f}")
            print("-" * 60)
            print(f"  MAE (件)   : {v_mae:.2f}")
            print(f"  中位 Ratio  : {v_ratio:.4f}  (正样本预测/真实的中位数)")
            print(f"  WMAPE      : {v_wmape:.4f}  (加权绝对百分比误差)")
            print(f"  True  qty  : {int(sum_true):>12,} 件")
            print(f"  Pred  qty  : {int(sum_pred):>12,} 件")
            pos_rate = int(pos_count) / max(n_val, 1) * 100
            print(f"  Pos orders : {int(pos_count):>12,} 笔  ({pos_rate:.3f}% of val)")
            if device.type == 'cuda':
                peak_mem  = torch.cuda.max_memory_allocated(0) / 1024**2
                total_mem_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
                print(f"  GPU VRAM   : peak {peak_mem:.0f} MB / {total_mem_mb:.0f} MB  ({peak_mem/total_mem_mb*100:.1f}%)")
                torch.cuda.reset_peak_memory_stats(0)  # 每轮重置，下轮重新从0计峰值
            print("-" * 60)

            # [V4.1] 计算 F1/Precision/Recall（直接使用 validate() 返回的数据，无二次遍历）
            try:
                from sklearn.metrics import f1_score, precision_score, recall_score
                # ★ 直接使用 validate 已经收集的 all_probs / all_labels
                best_thr, best_ep_f1 = 0.5, 0.0
                for thr in np.arange(0.3, 0.8, 0.02):
                    preds_t = (all_probs >= thr).astype(int)
                    f1_t = f1_score(all_labels, preds_t, zero_division=0)
                    if f1_t > best_ep_f1:
                        best_ep_f1 = f1_t
                        best_thr = thr

                preds_best = (all_probs >= best_thr).astype(int)
                ep_prec = precision_score(all_labels, preds_best, zero_division=0)
                ep_rec  = recall_score(all_labels, preds_best, zero_division=0)

                print(f"  F1 (best阈值={best_thr:.2f}): {best_ep_f1:.4f}  "
                      f"Precision={ep_prec:.4f}  Recall={ep_rec:.4f}")
                print("-" * 60)

            except Exception as _e:
                print(f"  [F1 计算失败] {_e}")
                best_ep_f1 = -v_loss
                ep_prec = ep_rec = 0.0
            
            # 记录历史轨迹 (CSV)
            with open(history_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, current_lr, t_loss, v_loss, best_ep_f1, ep_prec, ep_rec, v_ratio, v_wmape, epoch_secs/60])
            
            # Early Stopping + LR 衰减（只在验证轮、且超过 warmup 后才触发）
            # ★ 修复: >= 而非 >，确保第一个真实验证轮（epoch 3）的结果被正确追踪
            if epoch >= warmup_epochs:
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_epochs['loss'] = epoch
                    torch.save(model.state_dict(), best_loss_model_path)
                    print(f"  [Best-Loss] epoch={epoch} | val_loss={v_loss:.7f}")

                if best_ep_f1 > best_f1:
                    best_f1 = best_ep_f1
                    best_epochs['f1'] = epoch
                    torch.save(model.state_dict(), best_f1_model_path)
                    print(f"  [Best-F1]   epoch={epoch} | f1={best_ep_f1:.4f}")

                if v_wmape < best_wmape:
                    best_wmape = v_wmape
                    best_epochs['wmape'] = epoch
                    torch.save(model.state_dict(), best_wmape_model_path)
                    print(f"  [Best-WMAPE] epoch={epoch} | wmape={v_wmape:.4f}")

                monitor_value_map = {
                    'loss': v_loss,
                    'f1': best_ep_f1,
                    'wmape': v_wmape,
                }
                current_monitor = monitor_value_map[monitor_metric]
                improved = current_monitor > best_monitor if monitor_metric == 'f1' else current_monitor < best_monitor

                if improved:
                    best_monitor = current_monitor
                    best_mae = v_mae
                    best_epochs['monitor'] = epoch
                    torch.save(model.state_dict(), best_model_path)
                    print(
                        f"  💡 New Primary Best! monitor={monitor_metric} "
                        f"value={current_monitor:.7f} | epoch={epoch} | "
                        f"F1={best_ep_f1:.4f} | WMAPE={v_wmape:.4f}"
                    )
                    patience_counter = 0
                else:
                    patience_counter += 1
                    print(
                        f"  ⏳ No {monitor_metric} improvement. "
                        f"Patience: {patience_counter}/{patience_limit}"
                    )
                    if patience_counter >= patience_limit:
                        print(
                            f"\n  ⏹️  Early Stopping! 连续 {patience_limit} 次 "
                            f"{monitor_metric} 无改善，终止。"
                        )
                        torch.save(model.state_dict(), last_model_path)
                        break

                prev_lr = optimizer.param_groups[0]['lr']
                scheduler.step(current_monitor)
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr < prev_lr:
                    print(f"  📉 LR 衰减: {prev_lr:.6f} → {new_lr:.6f}")
                    
            torch.save(model.state_dict(), last_model_path)  # 覆盖保存本轮末尾权重
                    
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"⛔ 显存溢出！请降低 batch_size（当前 {train_loader.batch_size}）。")
                torch.cuda.empty_cache()
                break
            else:
                raise e
                
    selection_meta = {
        'exp_id': exp_id,
        'seed': int(exp_seed) if exp_seed is not None else None,
        'monitor_metric': monitor_metric,
        'best_epochs': best_epochs,
        'best_values': {
            'loss': best_val_loss,
            'f1': best_f1,
            'wmape': best_wmape,
            'monitor': best_monitor,
            'mae_at_primary_best': best_mae,
        },
        'paths': {
            'primary': best_model_path,
            'loss': best_loss_model_path,
            'f1': best_f1_model_path,
            'wmape': best_wmape_model_path,
            'last': last_model_path,
        },
    }
    with open(selection_meta_path, 'w', encoding='utf-8') as f:
        json.dump(_to_jsonable(selection_meta), f, ensure_ascii=False, indent=2)

    print(f"\n🎉 训练终止，最佳记录 MAE = {best_mae:.2f}  |  Best F1 = {best_f1:.4f}")
    print(f"  📌 Primary Best ({monitor_metric}) : {best_model_path}")
    print(f"  📌 Best by loss                    : {best_loss_model_path}")
    print(f"  📌 Best by f1                      : {best_f1_model_path}")
    print(f"  📌 Best by wmape                   : {best_wmape_model_path}")
    print(f"  📌 Last Model (最后一轮权重)      : {last_model_path}")
    print(f"  📌 Selection meta                  : {selection_meta_path}")
    return best_model_path

