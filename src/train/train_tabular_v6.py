import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.models.tabular_hurdle import TabularHurdleModel


def parse_json_env(name):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must decode to a JSON object.")
    return payload


def load_paths():
    exp_version = os.environ.get("EXP_VERSION", "v6_event").lower()
    if exp_version != "v6_event":
        raise ValueError(f"Unsupported tabular version: {exp_version}")

    artifacts_dir = os.environ.get(
        "EXP_ARTIFACTS_DIR", os.path.join(PROJECT_ROOT, "data", "artifacts_v6_event")
    )
    processed_dir = os.environ.get(
        "EXP_PROCESSED_DIR", os.path.join(PROJECT_ROOT, "data", "processed_v6_event")
    )
    model_dir = os.environ.get(
        "EXP_MODEL_DIR", os.path.join(PROJECT_ROOT, "models_v6_event")
    )
    meta_path = os.path.join(artifacts_dir, "meta_v6_event.json")
    return exp_version, artifacts_dir, processed_dir, model_dir, meta_path


def build_feature_indices(meta, feature_set):
    feature_cols = meta["feature_cols"]
    feature_groups = meta["feature_groups"]
    selected = []
    selected.extend(feature_groups["static"])
    selected.extend(feature_groups["core"])
    if feature_set in {
        "cov",
        "cov_activity",
        "cov_activity_qfo",
        "cov_activity_tail",
        "cov_activity_priors",
        "cov_activity_tail_full",
        "cov_activity_tail_full_event",
    }:
        selected.extend(feature_groups["buyer"])
    if feature_set in {
        "cov_activity",
        "cov_activity_qfo",
        "cov_activity_tail",
        "cov_activity_priors",
        "cov_activity_tail_full",
        "cov_activity_tail_full_event",
    }:
        selected.extend(feature_groups["activity"])
    if feature_set in {"cov_activity_qfo", "cov_activity_tail_full", "cov_activity_tail_full_event"}:
        selected.extend(feature_groups.get("qfo", []))
    if feature_set in {"cov_activity_tail", "cov_activity_tail_full", "cov_activity_tail_full_event"}:
        selected.extend(feature_groups.get("tail", []))
    if feature_set in {"cov_activity_priors", "cov_activity_tail_full", "cov_activity_tail_full_event"}:
        selected.extend(feature_groups.get("priors", []))
    if feature_set in {"cov_activity_tail_full_event"}:
        selected.extend(feature_groups.get("event", []))

    index_map = {name: idx for idx, name in enumerate(feature_cols)}
    return [index_map[name] for name in selected], selected


def build_asymmetric_state_weights(x_train_sel, y_train_cls, y_train_reg, selected_cols):
    config = {
        "short_zero_neg_mult": float(os.environ.get("EXP_SHORT_ZERO_NEG_MULT", "1.0")),
        "short_zero_pos_mult": float(os.environ.get("EXP_SHORT_ZERO_POS_MULT", "1.0")),
        "short_zero_reg_mult": float(os.environ.get("EXP_SHORT_ZERO_REG_MULT", "1.0")),
        "long_zero_neg_mult": float(os.environ.get("EXP_LONG_ZERO_NEG_MULT", "1.0")),
        "long_zero_pos_mult": float(os.environ.get("EXP_LONG_ZERO_POS_MULT", "1.0")),
        "long_zero_reg_mult": float(os.environ.get("EXP_LONG_ZERO_REG_MULT", "1.0")),
    }
    enabled = any(abs(value - 1.0) > 1e-6 for value in config.values())

    pos = max(float(y_train_cls.sum()), 1.0)
    neg = max(float(len(y_train_cls) - y_train_cls.sum()), 1.0)
    pos_weight = neg / pos
    cls_weights = np.where(y_train_cls > 0, pos_weight, 1.0).astype(np.float32)
    reg_weights = np.ones_like(y_train_reg, dtype=np.float32)

    info = {
        "enabled": enabled,
        "config": config,
        "short_zero_rows": 0,
        "long_zero_rows": 0,
        "short_zero_positive_rows": 0,
        "long_zero_positive_rows": 0,
    }
    if not enabled:
        return cls_weights, reg_weights, info

    feature_to_idx = {name: idx for idx, name in enumerate(selected_cols)}
    short_idx = feature_to_idx.get("inv_short_zero")
    long_idx = feature_to_idx.get("inv_long_zero")
    if short_idx is None or long_idx is None:
        info["enabled"] = False
        info["note"] = "missing_zero_split_features"
        return cls_weights, reg_weights, info

    short_mask = np.asarray(x_train_sel[:, short_idx] > 0.5, dtype=bool)
    long_mask = np.asarray(x_train_sel[:, long_idx] > 0.5, dtype=bool)
    pos_mask = np.asarray(y_train_cls > 0, dtype=bool)
    neg_mask = ~pos_mask

    cls_weights[short_mask & neg_mask] *= np.float32(config["short_zero_neg_mult"])
    cls_weights[short_mask & pos_mask] *= np.float32(config["short_zero_pos_mult"])
    cls_weights[long_mask & neg_mask] *= np.float32(config["long_zero_neg_mult"])
    cls_weights[long_mask & pos_mask] *= np.float32(config["long_zero_pos_mult"])

    reg_weights[short_mask & pos_mask] *= np.float32(config["short_zero_reg_mult"])
    reg_weights[long_mask & pos_mask] *= np.float32(config["long_zero_reg_mult"])

    info.update(
        {
            "short_zero_rows": int(short_mask.sum()),
            "long_zero_rows": int(long_mask.sum()),
            "short_zero_positive_rows": int((short_mask & pos_mask).sum()),
            "long_zero_positive_rows": int((long_mask & pos_mask).sum()),
        }
    )
    return cls_weights, reg_weights, info


def main():
    exp_id = os.environ.get("EXP_ID", "phase53_tabular")
    feature_set = os.environ.get("EXP_FEATURE_SET", "core").lower()
    gate_mode = os.environ.get("EXP_GATE_MODE", "hard").lower()
    seed = int(os.environ.get("EXP_SEED", "2026"))
    backend = os.environ.get("EXP_TREE_BACKEND", "lightgbm").lower()

    classifier_params = parse_json_env("EXP_TREE_CLS_PARAMS_JSON")
    regressor_params = parse_json_env("EXP_TREE_REG_PARAMS_JSON")
    if backend == "lightgbm":
        if os.environ.get("EXP_LGBM_N_ESTIMATORS"):
            value = int(os.environ["EXP_LGBM_N_ESTIMATORS"])
            classifier_params["n_estimators"] = value
            regressor_params["n_estimators"] = value
        if os.environ.get("EXP_LGBM_LR"):
            value = float(os.environ["EXP_LGBM_LR"])
            classifier_params["learning_rate"] = value
            regressor_params["learning_rate"] = value
        if os.environ.get("EXP_LGBM_NUM_LEAVES"):
            value = int(os.environ["EXP_LGBM_NUM_LEAVES"])
            classifier_params["num_leaves"] = value
            regressor_params["num_leaves"] = value
        if os.environ.get("EXP_LGBM_SUBSAMPLE"):
            value = float(os.environ["EXP_LGBM_SUBSAMPLE"])
            classifier_params["subsample"] = value
            regressor_params["subsample"] = value
        if os.environ.get("EXP_LGBM_COLSAMPLE"):
            value = float(os.environ["EXP_LGBM_COLSAMPLE"])
            classifier_params["colsample_bytree"] = value
            regressor_params["colsample_bytree"] = value
        if os.environ.get("EXP_LGBM_CLS_CHILD"):
            classifier_params["min_child_samples"] = int(os.environ["EXP_LGBM_CLS_CHILD"])
        if os.environ.get("EXP_LGBM_REG_CHILD"):
            regressor_params["min_child_samples"] = int(os.environ["EXP_LGBM_REG_CHILD"])

    exp_version, artifacts_dir, processed_dir, model_dir, meta_path = load_paths()
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta: {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    x_train = np.load(os.path.join(processed_dir, "X_train.npy"), mmap_mode="r")
    x_val = np.load(os.path.join(processed_dir, "X_val.npy"), mmap_mode="r")
    y_train_cls = np.load(os.path.join(processed_dir, "y_train_cls.npy"), mmap_mode="r")
    y_train_reg = np.load(os.path.join(processed_dir, "y_train_reg.npy"), mmap_mode="r")
    y_val_cls = np.load(os.path.join(processed_dir, "y_val_cls.npy"), mmap_mode="r")
    y_val_reg = np.load(os.path.join(processed_dir, "y_val_reg.npy"), mmap_mode="r")

    selected_idx, selected_cols = build_feature_indices(meta, feature_set)
    x_train_sel = np.nan_to_num(np.asarray(x_train[:, selected_idx], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    x_val_sel = np.nan_to_num(np.asarray(x_val[:, selected_idx], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    cls_weights, reg_weights, asym_info = build_asymmetric_state_weights(
        x_train_sel,
        y_train_cls,
        y_train_reg,
        selected_cols,
    )
    pos = max(float(y_train_cls.sum()), 1.0)
    neg = max(float(len(y_train_cls) - y_train_cls.sum()), 1.0)
    pos_weight = neg / pos

    print("=" * 72)
    print("Phase 5.3 tabular hurdle training")
    print(
        f"exp_id={exp_id} | version={exp_version} | feature_set={feature_set} | "
        f"gate_mode={gate_mode} | seed={seed} | backend={backend}"
    )
    print(f"train={len(y_train_cls):,} | val={len(y_val_cls):,} | pos_weight={pos_weight:.3f}")
    if asym_info["enabled"]:
        print(
            "[asym] short_zero rows="
            f"{asym_info['short_zero_rows']:,} "
            f"(pos={asym_info['short_zero_positive_rows']:,}) | "
            "long_zero rows="
            f"{asym_info['long_zero_rows']:,} "
            f"(pos={asym_info['long_zero_positive_rows']:,})"
        )
        print(
            "[asym] "
            f"short_neg={asym_info['config']['short_zero_neg_mult']:.2f} "
            f"short_pos={asym_info['config']['short_zero_pos_mult']:.2f} "
            f"short_reg={asym_info['config']['short_zero_reg_mult']:.2f} | "
            f"long_neg={asym_info['config']['long_zero_neg_mult']:.2f} "
            f"long_pos={asym_info['config']['long_zero_pos_mult']:.2f} "
            f"long_reg={asym_info['config']['long_zero_reg_mult']:.2f}"
        )
    print("=" * 72)

    t0 = time.time()
    model = TabularHurdleModel(
        random_state=seed,
        backend=backend,
        classifier_params=classifier_params,
        regressor_params=regressor_params,
    )
    model.fit(
        x_train_sel,
        y_train_cls,
        y_train_reg,
        cls_sample_weight=cls_weights,
        reg_sample_weight=reg_weights,
        x_val=x_val_sel,
        y_val_cls=y_val_cls,
        y_val_reg=y_val_reg,
    )
    elapsed_min = (time.time() - t0) / 60.0

    pred = model.predict_quantity(
        x_val_sel,
        gate_mode=gate_mode,
        gate_threshold=float(os.environ.get("EXP_QTY_GATE", "0.20")),
    )
    actual_qty = np.expm1(y_val_reg)
    ratio = float(pred["qty"].sum() / max(actual_qty.sum(), 1e-9))
    wmape = float(np.abs(pred["qty"] - actual_qty).sum() / max(actual_qty.sum(), 1e-9))
    y_pred_cls = (pred["prob"] >= 0.50).astype(int)
    auc = float(roc_auc_score(y_val_cls, pred["prob"])) if np.unique(y_val_cls).size > 1 else float("nan")
    f1 = float(f1_score(y_val_cls.astype(int), y_pred_cls, zero_division=0))

    os.makedirs(model_dir, exist_ok=True)
    cls_path = os.path.join(model_dir, f"{exp_id}_cls.pkl")
    reg_path = os.path.join(model_dir, f"{exp_id}_reg.pkl")
    meta_out = os.path.join(model_dir, f"{exp_id}_meta.json")
    extra_meta = {
        "exp_id": exp_id,
        "exp_version": exp_version,
        "feature_set": feature_set,
        "gate_mode": gate_mode,
        "backend": backend,
        "selected_feature_cols": selected_cols,
        "selected_feature_indices": selected_idx,
        "classifier_params": classifier_params,
        "regressor_params": regressor_params,
        "asymmetric_state_weighting": asym_info,
        "elapsed_min": elapsed_min,
        "val_auc_at_050": auc,
        "val_f1_at_050": f1,
        "val_ratio": ratio,
        "val_wmape": wmape,
    }
    model.save(cls_path, reg_path, meta_out, extra_meta=extra_meta)

    print(f"[OK] cls  -> {cls_path}")
    print(f"[OK] reg  -> {reg_path}")
    print(f"[OK] meta -> {meta_out}")
    print(f"[VAL] auc@0.50={auc:.4f} | f1@0.50={f1:.4f} | ratio={ratio:.4f} | wmape={wmape:.4f} | elapsed={elapsed_min:.1f} min")


if __name__ == "__main__":
    main()
