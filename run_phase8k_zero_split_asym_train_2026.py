import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from run_phase8d_event_inventory_shadow_2026 import ANCHORS, BASE_CFG, FORCE_REBUILD, SEED, TREE_BACKEND


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "phase8_zero_split_asym_train_2026")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models_phase8_zero_split_asym_train_2026")
TIMING_LOG = os.path.join(REPORTS_DIR, "phase8_zero_split_asym_train_timing_log.csv")
VAL_MODE = "single_anchor"

ZERO_SPLIT_VARIANT = {
    "candidate_key": "event_inventory_zero_split",
    "feature_set": "cov_activity_tail_full_event",
}

ASYM_CANDIDATES = [
    {
        "candidate_key": "zero_split_asym_mild",
        "weights": {
            "EXP_SHORT_ZERO_NEG_MULT": "1.05",
            "EXP_SHORT_ZERO_POS_MULT": "1.00",
            "EXP_SHORT_ZERO_REG_MULT": "1.00",
            "EXP_LONG_ZERO_NEG_MULT": "1.20",
            "EXP_LONG_ZERO_POS_MULT": "1.05",
            "EXP_LONG_ZERO_REG_MULT": "1.15",
        },
    },
    {
        "candidate_key": "zero_split_asym_balanced",
        "weights": {
            "EXP_SHORT_ZERO_NEG_MULT": "1.10",
            "EXP_SHORT_ZERO_POS_MULT": "1.00",
            "EXP_SHORT_ZERO_REG_MULT": "1.05",
            "EXP_LONG_ZERO_NEG_MULT": "1.35",
            "EXP_LONG_ZERO_POS_MULT": "1.10",
            "EXP_LONG_ZERO_REG_MULT": "1.25",
        },
    },
    {
        "candidate_key": "zero_split_asym_strong",
        "weights": {
            "EXP_SHORT_ZERO_NEG_MULT": "1.10",
            "EXP_SHORT_ZERO_POS_MULT": "1.00",
            "EXP_SHORT_ZERO_REG_MULT": "1.05",
            "EXP_LONG_ZERO_NEG_MULT": "1.50",
            "EXP_LONG_ZERO_POS_MULT": "1.10",
            "EXP_LONG_ZERO_REG_MULT": "1.35",
        },
    },
]


def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def init_timing_log():
    if os.path.exists(TIMING_LOG):
        return
    with open(TIMING_LOG, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "anchor_date",
                "candidate_key",
                "stage",
                "exp_id",
                "start_time",
                "end_time",
                "elapsed_min",
                "status",
                "note",
            ]
        )


def append_timing(anchor_date, candidate_key, stage, exp_id, t0, t1, status, note=""):
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                anchor_date,
                candidate_key,
                stage,
                exp_id,
                datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S"),
                datetime.fromtimestamp(t1).strftime("%Y-%m-%d %H:%M:%S"),
                round((t1 - t0) / 60.0, 2),
                status,
                note,
            ]
        )


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def report_dir(anchor_date):
    path = os.path.join(REPORTS_DIR, anchor_tag(anchor_date))
    os.makedirs(path, exist_ok=True)
    return path


def model_paths(exp_id):
    return (
        os.path.join(MODEL_DIR, f"{exp_id}_cls.pkl"),
        os.path.join(MODEL_DIR, f"{exp_id}_reg.pkl"),
        os.path.join(MODEL_DIR, f"{exp_id}_meta.json"),
    )


def copy_model_artifacts(src_exp, dst_exp):
    src_cls, src_reg, src_meta = model_paths(src_exp)
    dst_cls, dst_reg, dst_meta = model_paths(dst_exp)
    shutil.copy2(src_cls, dst_cls)
    shutil.copy2(src_reg, dst_reg)
    shutil.copy2(src_meta, dst_meta)


def ensure_zero_split_assets(anchor_date):
    tag = anchor_tag(anchor_date)
    output_tag = f"p8einvshadow_zero_split_{tag}_v6_event"
    processed_dir = os.path.join(PROJECT_ROOT, "data", f"processed_v6_event_{output_tag}")
    artifacts_dir = os.path.join(PROJECT_ROOT, "data", f"artifacts_v6_event_{output_tag}")
    meta_path = os.path.join(artifacts_dir, "meta_v6_event.json")
    train_path = os.path.join(processed_dir, "X_train.npy")
    if (not FORCE_REBUILD) and os.path.exists(meta_path) and os.path.exists(train_path):
        return processed_dir, artifacts_dir

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["FEATURE_SPLIT_DATE"] = anchor_date
    env["FEATURE_VAL_MODE"] = VAL_MODE
    env["FEATURE_SHADOW_SUFFIX"] = "zero_split"
    env["FEATURE_INVENTORY_ZERO_SPLIT_MODE"] = "split"
    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "features", "build_features_v6_event_inventory_shadow_sku.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    t1 = time.time()
    append_timing(anchor_date, ZERO_SPLIT_VARIANT["candidate_key"], "build_zero_split_assets", output_tag, t0, t1, "success" if rc == 0 else "failed")
    if rc != 0:
        raise RuntimeError(f"Failed to build zero_split assets for {anchor_date}")
    return processed_dir, artifacts_dir


def run_training(anchor_date, variant, processed_dir, artifacts_dir):
    exp_base = f"p8k_{anchor_tag(anchor_date)}_{variant['candidate_key']}_s{SEED}"
    cls_path, reg_path, meta_path = model_paths(exp_base)
    if (not FORCE_REBUILD) and os.path.exists(cls_path) and os.path.exists(reg_path) and os.path.exists(meta_path):
        return exp_base

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXP_ID"] = exp_base
    env["EXP_VERSION"] = "v6_event"
    env["EXP_FEATURE_SET"] = ZERO_SPLIT_VARIANT["feature_set"]
    env["EXP_GATE_MODE"] = BASE_CFG["gate_mode"]
    env["EXP_QTY_GATE"] = BASE_CFG["qty_gate"]
    env["EXP_SEED"] = SEED
    env["EXP_TREE_BACKEND"] = TREE_BACKEND
    env["EXP_PROCESSED_DIR"] = processed_dir
    env["EXP_ARTIFACTS_DIR"] = artifacts_dir
    env["EXP_MODEL_DIR"] = MODEL_DIR
    env["EXP_REPORT_DIR"] = report_dir(anchor_date)
    env["EXP_LGBM_LR"] = BASE_CFG["lr"]
    env["EXP_LGBM_NUM_LEAVES"] = BASE_CFG["num_leaves"]
    env["EXP_LGBM_CLS_CHILD"] = BASE_CFG["cls_child"]
    env["EXP_LGBM_REG_CHILD"] = BASE_CFG["reg_child"]
    env["EXP_LGBM_N_ESTIMATORS"] = BASE_CFG["n_estimators"]
    env["EXP_LGBM_SUBSAMPLE"] = BASE_CFG["subsample"]
    env["EXP_LGBM_COLSAMPLE"] = BASE_CFG["colsample"]
    env.update(variant["weights"])

    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "train", "train_tabular_v6.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    t1 = time.time()
    append_timing(anchor_date, variant["candidate_key"], "train", exp_base, t0, t1, "success" if rc == 0 else "failed")
    if rc != 0:
        raise RuntimeError(f"Training failed for {exp_base}")
    return exp_base


def run_eval(anchor_date, variant, base_exp, processed_dir, artifacts_dir):
    eval_exp = f"{base_exp}_{BASE_CFG['gate_mode']}_g{BASE_CFG['qty_gate'].replace('.', '')}"
    base_report_dir = report_dir(anchor_date)
    eval_txt = os.path.join(base_report_dir, f"eval_{eval_exp}.txt")
    agg_txt = os.path.join(base_report_dir, f"agg_{eval_exp}.txt")
    context_csv = os.path.join(base_report_dir, "phase5", f"eval_context_{eval_exp}.csv")
    if (not FORCE_REBUILD) and os.path.exists(eval_txt) and os.path.exists(agg_txt) and os.path.exists(context_csv):
        return eval_exp

    copy_model_artifacts(base_exp, eval_exp)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXP_ID"] = eval_exp
    env["EXP_VERSION"] = "v6_event"
    env["EXP_FEATURE_SET"] = ZERO_SPLIT_VARIANT["feature_set"]
    env["EXP_GATE_MODE"] = BASE_CFG["gate_mode"]
    env["EXP_QTY_GATE"] = BASE_CFG["qty_gate"]
    env["EXP_SEED"] = SEED
    env["EXP_TREE_BACKEND"] = TREE_BACKEND
    env["EXP_PROCESSED_DIR"] = processed_dir
    env["EXP_ARTIFACTS_DIR"] = artifacts_dir
    env["EXP_MODEL_DIR"] = MODEL_DIR
    env["EXP_REPORT_DIR"] = base_report_dir

    t0 = time.time()
    rc = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "evaluate_tabular.py")], cwd=PROJECT_ROOT, env=env).returncode
    if rc == 0:
        rc = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "evaluate_agg.py")], cwd=PROJECT_ROOT, env=env).returncode
    t1 = time.time()
    append_timing(anchor_date, variant["candidate_key"], "eval", eval_exp, t0, t1, "success" if rc == 0 else "failed")
    if rc != 0:
        raise RuntimeError(f"Evaluation failed for {eval_exp}")
    return eval_exp


def main():
    ensure_dirs()
    init_timing_log()
    print(f"[Phase8 zero_split asym train 2026] anchors={ANCHORS} seed={SEED}")
    for anchor_date in ANCHORS:
        zero_split_processed, zero_split_artifacts = ensure_zero_split_assets(anchor_date)
        for variant in ASYM_CANDIDATES:
            base_exp = run_training(anchor_date, variant, zero_split_processed, zero_split_artifacts)
            run_eval(anchor_date, variant, base_exp, zero_split_processed, zero_split_artifacts)

    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "analysis", "summarize_phase8k_zero_split_asym_results.py")],
        cwd=PROJECT_ROOT,
        check=False,
    )
    print("[DONE] phase8 zero_split asym train 2026 completed")


if __name__ == "__main__":
    main()
