import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_7")
TIMING_LOG = os.path.join(REPORTS_DIR, "phase5_7a_timing_log.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models_phase5_7")

ANCHORS = [x.strip() for x in os.environ.get(
    "PHASE57A_ANCHORS",
    "2025-09-01,2025-10-01,2025-11-01,2025-12-01",
).split(",") if x.strip()]
SEED = os.environ.get("PHASE57A_SEED", "2026")
TREE_BACKEND = os.environ.get("PHASE57A_TREE_BACKEND", "lightgbm").lower()
VAL_MODE = os.environ.get("PHASE57A_VAL_MODE", "single_anchor").strip().lower()
RUN_SUMMARY = os.environ.get("PHASE57A_RUN_SUMMARY", "1") == "1"

BASE_CFG = {
    "tag": "covact_lr005_l63",
    "feature_set": "cov_activity",
    "lr": "0.05",
    "num_leaves": "63",
    "cls_child": "20",
    "reg_child": "20",
}

EVAL_VARIANTS = [
    {"gate_mode": "hard", "qty_gate": "0.20"},
    {"gate_mode": "hard", "qty_gate": "0.25"},
]


def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)


def init_timing_log():
    if os.path.exists(TIMING_LOG):
        return
    with open(TIMING_LOG, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "anchor_date", "base_exp", "eval_exp", "feature_set", "gate_mode",
            "qty_gate", "lr", "num_leaves", "seed", "stage", "start_time",
            "end_time", "elapsed_min", "status", "note",
        ])


def append_timing(anchor_date, base_exp, eval_exp, gate_mode, qty_gate, stage, t_start, t_end, status, note=""):
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            anchor_date,
            base_exp,
            eval_exp,
            BASE_CFG["feature_set"],
            gate_mode,
            qty_gate,
            BASE_CFG["lr"],
            BASE_CFG["num_leaves"],
            SEED,
            stage,
            datetime.fromtimestamp(t_start).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(t_end).strftime("%Y-%m-%d %H:%M:%S"),
            round((t_end - t_start) / 60.0, 2),
            status,
            note,
        ])


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
    shutil.copy(src_cls, dst_cls)
    shutil.copy(src_reg, dst_reg)
    shutil.copy(src_meta, dst_meta)


def ensure_assets(anchor_date):
    anchor_suffix = anchor_tag(anchor_date)

    def candidate_dirs(version, tags, meta_name, train_name):
        for tag in tags:
            if tag is None:
                processed_dir = os.path.join(PROJECT_ROOT, "data", f"processed_{version}")
                artifacts_dir = os.path.join(PROJECT_ROOT, "data", f"artifacts_{version}")
            else:
                processed_dir = os.path.join(PROJECT_ROOT, "data", f"processed_{version}_{tag}")
                artifacts_dir = os.path.join(PROJECT_ROOT, "data", f"artifacts_{version}_{tag}")
            meta_path = os.path.join(artifacts_dir, meta_name)
            train_path = os.path.join(processed_dir, train_name)
            if os.path.exists(meta_path) and os.path.exists(train_path):
                return processed_dir, artifacts_dir
        return None, None

    v5_tags = [
        f"p57a_{anchor_suffix}_v5_lite",
        f"p56_{anchor_suffix}_v5_lite",
        f"p55_{anchor_suffix}_v5_lite",
        None,
    ]
    v5_processed, v5_artifacts = candidate_dirs("v5_lite", v5_tags, "meta_v5_lite.json", "X_train_dyn.bin")
    if not (v5_processed and v5_artifacts):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["FEATURE_SPLIT_DATE"] = anchor_date
        env["FEATURE_OUTPUT_TAG"] = f"p57a_{anchor_suffix}_v5_lite"
        env["FEATURE_VAL_MODE"] = VAL_MODE
        rc = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "src", "features", "build_features_v5_lite_sku.py")],
            cwd=PROJECT_ROOT,
            env=env,
        ).returncode
        if rc != 0:
            raise RuntimeError(f"Failed to build v5_lite assets for {anchor_date}")

    v6_tags = [
        f"p57a_{anchor_suffix}_v6_event",
        f"p56_{anchor_suffix}_v6_event",
        f"p55_{anchor_suffix}_v6_event",
        None,
    ]
    processed_dir, artifacts_dir = candidate_dirs("v6_event", v6_tags, "meta_v6_event.json", "X_train.npy")
    if processed_dir and artifacts_dir:
        return processed_dir, artifacts_dir

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["FEATURE_SPLIT_DATE"] = anchor_date
    env["FEATURE_OUTPUT_TAG"] = f"p57a_{anchor_suffix}_v6_event"
    env["FEATURE_VAL_MODE"] = VAL_MODE
    rc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "features", "build_features_v6_event_sku.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    if rc != 0:
        raise RuntimeError(f"Failed to build v6_event assets for {anchor_date}")

    return (
        os.path.join(PROJECT_ROOT, "data", f"processed_v6_event_p57a_{anchor_suffix}_v6_event"),
        os.path.join(PROJECT_ROOT, "data", f"artifacts_v6_event_p57a_{anchor_suffix}_v6_event"),
    )


def preflight():
    ensure_dirs()
    required = [
        os.path.join(PROJECT_ROOT, "src", "train", "train_tabular_v6.py"),
        os.path.join(PROJECT_ROOT, "evaluate_tabular.py"),
        os.path.join(PROJECT_ROOT, "evaluate_agg.py"),
        os.path.join(PROJECT_ROOT, "src", "analysis", "summarize_phase5_7_results.py"),
    ]
    for path in required:
        if not os.path.exists(path):
            return False
    if TREE_BACKEND == "lightgbm":
        check = subprocess.run(
            [sys.executable, "-c", "import lightgbm"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if check.returncode != 0:
            return False
    return True


def run_training(anchor_date, processed_dir, artifacts_dir):
    base_exp = f"p57a_{anchor_tag(anchor_date)}_{BASE_CFG['tag']}_s{SEED}"
    cls_path, reg_path, meta_path = model_paths(base_exp)
    if os.path.exists(cls_path) and os.path.exists(reg_path) and os.path.exists(meta_path):
        return base_exp

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXP_ID"] = base_exp
    env["EXP_VERSION"] = "v6_event"
    env["EXP_FEATURE_SET"] = BASE_CFG["feature_set"]
    env["EXP_GATE_MODE"] = "hard"
    env["EXP_QTY_GATE"] = "0.20"
    env["EXP_SEED"] = str(SEED)
    env["EXP_TREE_BACKEND"] = TREE_BACKEND
    env["EXP_PROCESSED_DIR"] = processed_dir
    env["EXP_ARTIFACTS_DIR"] = artifacts_dir
    env["EXP_MODEL_DIR"] = MODEL_DIR
    env["EXP_REPORT_DIR"] = report_dir(anchor_date)
    env["EXP_LGBM_LR"] = BASE_CFG["lr"]
    env["EXP_LGBM_NUM_LEAVES"] = BASE_CFG["num_leaves"]
    env["EXP_LGBM_CLS_CHILD"] = BASE_CFG["cls_child"]
    env["EXP_LGBM_REG_CHILD"] = BASE_CFG["reg_child"]

    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "train", "train_tabular_v6.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    t1 = time.time()
    append_timing(anchor_date, base_exp, "", "hard", "0.20", "train", t0, t1, "success" if rc == 0 else "failed")
    if rc != 0:
        raise RuntimeError(f"Training failed for {base_exp}")
    return base_exp


def run_eval(anchor_date, base_exp, processed_dir, artifacts_dir, variant):
    eval_exp = f"{base_exp}_{variant['gate_mode']}_g{variant['qty_gate'].replace('.', '')}"
    base_report_dir = report_dir(anchor_date)
    eval_txt = os.path.join(base_report_dir, f"eval_{eval_exp}.txt")
    agg_txt = os.path.join(base_report_dir, f"agg_{eval_exp}.txt")
    if os.path.exists(eval_txt) and os.path.exists(agg_txt):
        return

    copy_model_artifacts(base_exp, eval_exp)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXP_ID"] = eval_exp
    env["EXP_VERSION"] = "v6_event"
    env["EXP_FEATURE_SET"] = BASE_CFG["feature_set"]
    env["EXP_GATE_MODE"] = variant["gate_mode"]
    env["EXP_QTY_GATE"] = variant["qty_gate"]
    env["EXP_SEED"] = str(SEED)
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
    append_timing(anchor_date, base_exp, eval_exp, variant["gate_mode"], variant["qty_gate"], "eval", t0, t1, "success" if rc == 0 else "failed")
    if rc != 0:
        raise RuntimeError(f"Evaluation failed for {eval_exp}")


def main():
    if not preflight():
        print("[BLOCKED] phase5.7A preflight failed")
        raise SystemExit(1)
    init_timing_log()
    print(f"[Phase 5.7A] anchors={ANCHORS} seed={SEED} variants={len(EVAL_VARIANTS)}")
    for anchor_date in ANCHORS:
        processed_dir, artifacts_dir = ensure_assets(anchor_date)
        base_exp = run_training(anchor_date, processed_dir, artifacts_dir)
        for variant in EVAL_VARIANTS:
            run_eval(anchor_date, base_exp, processed_dir, artifacts_dir, variant)

    if RUN_SUMMARY:
        subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "src", "analysis", "summarize_phase5_7_results.py")],
            cwd=PROJECT_ROOT,
            check=False,
        )
    print("[DONE] phase5.7A tuned tree anchors completed")


if __name__ == "__main__":
    main()
