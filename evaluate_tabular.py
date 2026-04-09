import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score

from src.models.tabular_hurdle import TabularHurdleModel


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.environ.get("EXP_REPORT_DIR", os.path.join(PROJECT_ROOT, "reports"))
os.makedirs(REPORTS_DIR, exist_ok=True)


def main():
    exp_id = os.environ.get("EXP_ID", "phase53_tabular")
    exp_version = os.environ.get("EXP_VERSION", "v6_event").lower()
    artifacts_dir = os.environ.get("EXP_ARTIFACTS_DIR", os.path.join(PROJECT_ROOT, "data", "artifacts_v6_event"))
    processed_dir = os.environ.get("EXP_PROCESSED_DIR", os.path.join(PROJECT_ROOT, "data", "processed_v6_event"))
    model_dir = os.environ.get("EXP_MODEL_DIR", os.path.join(PROJECT_ROOT, "models_v6_event"))

    cls_path = os.path.join(model_dir, f"{exp_id}_cls.pkl")
    reg_path = os.path.join(model_dir, f"{exp_id}_reg.pkl")
    meta_path = os.path.join(model_dir, f"{exp_id}_meta.json")
    if not (os.path.exists(cls_path) and os.path.exists(reg_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f"Missing trained tabular hurdle artifacts for {exp_id}")

    model, train_meta = TabularHurdleModel.load(cls_path, reg_path, meta_path)
    data_meta_path = os.path.join(artifacts_dir, "meta_v6_event.json")
    with open(data_meta_path, "r", encoding="utf-8") as fh:
        data_meta = json.load(fh)

    selected_idx = train_meta["selected_feature_indices"]
    x_val = np.load(os.path.join(processed_dir, "X_val.npy"), mmap_mode="r")
    y_val_cls = np.load(os.path.join(processed_dir, "y_val_cls.npy"), mmap_mode="r")
    y_val_reg = np.load(os.path.join(processed_dir, "y_val_reg.npy"), mmap_mode="r")
    x_val_sel = np.nan_to_num(np.asarray(x_val[:, selected_idx], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    qty_gate = float(os.environ.get("EXP_QTY_GATE", "0.20"))
    gate_mode = os.environ.get("EXP_GATE_MODE", train_meta.get("gate_mode", "hard")).lower()
    pred = model.predict_quantity(x_val_sel, gate_mode=gate_mode, gate_threshold=qty_gate)

    actual_qty = np.expm1(y_val_reg).astype(np.float32)
    precision, recall, thresholds = precision_recall_curve(y_val_cls, pred["prob"])
    f1_scores = 2 * (precision * recall) / np.maximum(precision + recall, 1e-8)
    if len(thresholds) == 0:
        best_threshold = 0.5
    else:
        best_threshold = float(thresholds[int(np.argmax(f1_scores[:-1]))])
    cls_pred_best_f1 = (pred["prob"] >= best_threshold).astype(int)

    ratio = float(pred["qty"].sum() / max(actual_qty.sum(), 1e-9))
    wmape = float(np.abs(pred["qty"] - actual_qty).sum() / max(actual_qty.sum(), 1e-9))
    auc = float(roc_auc_score(y_val_cls, pred["prob"])) if np.unique(y_val_cls).size > 1 else float("nan")
    f1 = float(f1_score(y_val_cls.astype(int), cls_pred_best_f1, zero_division=0))

    val_keys_path = os.path.join(artifacts_dir, "val_keys.csv")
    val_keys = pd.read_csv(val_keys_path)
    detail_df = pd.DataFrame(
        {
            "sku_id": val_keys["sku_id"],
            "anchor_date": val_keys["date"],
            "true_replenish_qty": actual_qty,
            "ai_pred_prob": pred["prob"],
            "cls_pred_best_f1": cls_pred_best_f1,
            "ai_pred_qty_open": pred["qty_open"],
            "ai_pred_qty": pred["qty"],
            "ai_pred_positive_qty": (pred["qty"] > 0).astype(int),
            "qty_gate_mask": pred["gate_mask"].astype(int),
            "dead_blocked": np.zeros(len(actual_qty), dtype=int),
            "abs_error": np.abs(pred["qty"] - actual_qty),
        }
    )

    report_date = time.strftime("%Y%m%d_%H%M")
    detail_path = os.path.join(REPORTS_DIR, f"val_set_detailed_compare_{exp_id}_{report_date}.csv")
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")

    meta_out = {
        "exp_id": exp_id,
        "exp_version": exp_version,
        "model_kind": "tabular_hurdle",
        "backend": train_meta.get("backend", "unknown"),
        "feature_set": train_meta.get("feature_set"),
        "gate_mode": gate_mode,
        "best_f1_threshold": best_threshold,
        "qty_gate_threshold": qty_gate,
        "auc": auc,
        "f1": f1,
        "ratio": ratio,
        "wmape": wmape,
        "selected_feature_cols": train_meta.get("selected_feature_cols", []),
        "data_feature_groups": data_meta.get("feature_groups", {}),
    }
    meta_out_path = os.path.join(REPORTS_DIR, f"eval_meta_{exp_id}_{report_date}.json")
    with open(meta_out_path, "w", encoding="utf-8") as fh:
        json.dump(meta_out, fh, ensure_ascii=False, indent=2)

    text_lines = [
        "=== Phase 5.3 Tabular Evaluation ===",
        f"exp_id={exp_id}",
        f"backend={train_meta.get('backend', 'unknown')}",
        f"feature_set={train_meta.get('feature_set')}",
        f"gate_mode={gate_mode}",
        f"AUC={auc:.4f}",
        f"F1(best)={f1:.4f}",
        f"Ratio={ratio:.4f}",
        f"WMAPE={wmape:.4f}",
        f"best_f1_threshold={best_threshold:.4f}",
        f"qty_gate_threshold={qty_gate:.4f}",
        f"detail_csv={os.path.relpath(detail_path, PROJECT_ROOT)}",
    ]
    print("\n".join(text_lines))


if __name__ == "__main__":
    main()
