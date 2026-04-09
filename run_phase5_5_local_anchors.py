import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_ROOT = os.path.join(PROJECT_ROOT, "reports", "phase5_5")
TIMING_LOG = os.path.join(REPORTS_ROOT, "phase5_5_timing_log.csv")
WINNERS_JSON = os.path.join(PROJECT_ROOT, "reports", "phase5_3", "phase5_3_winners.json")
PHASE52_REFS = os.path.join(PROJECT_ROOT, "reports", "phase5_2", "phase5_2_sequence_refs.json")

ANCHORS = [x.strip() for x in os.environ.get(
    "PHASE55_ANCHORS",
    "2025-09-01,2025-10-01,2025-11-01,2025-12-01",
).split(",") if x.strip()]
SEEDS = [int(x.strip()) for x in os.environ.get("PHASE55_SEEDS", "2026").split(",") if x.strip()]
INCLUDE_RUNNER_UP = os.environ.get("PHASE55_INCLUDE_RUNNER_UP", "0") == "1"
INCLUDE_SEQ_CHALLENGER = os.environ.get("PHASE55_INCLUDE_SEQ_CHALLENGER", "0") == "1"
VAL_MODE = os.environ.get("PHASE55_VAL_MODE", "single_anchor").strip().lower()

SEQ_EPOCHS = os.environ.get("PHASE55_SEQ_EPOCHS", "10")
SEQ_PATIENCE = os.environ.get("PHASE55_SEQ_PATIENCE", "3")
SEQ_WORKERS = os.environ.get("PHASE55_SEQ_WORKERS", "0")
SEQ_MONITOR = os.environ.get("PHASE55_SEQ_MONITOR", "wmape").lower()
QTY_GATE = os.environ.get("PHASE55_QTY_GATE", "0.20")
TREE_BACKEND = os.environ.get("PHASE55_TREE_BACKEND", "lightgbm").lower()
RUN_TREE_SWEEP = os.environ.get("PHASE55_RUN_TREE_SWEEP", "1") == "1"

DEFAULT_SEQ_LOSS = {
    "pos_w": "5.85",
    "fp": "0.15",
    "gamma": "1.5",
    "reg_w": "0.5",
    "sf1": "0.5",
    "hub": "1.5",
}

EVENT_BASE = {
    "p531_tree_hard_core": {"feature_set": "core", "gate_mode": "hard"},
    "p532_tree_soft_core": {"feature_set": "core", "gate_mode": "soft"},
    "p533_tree_hard_cov": {"feature_set": "cov", "gate_mode": "hard"},
    "p534_tree_soft_cov": {"feature_set": "cov", "gate_mode": "soft"},
    "p535_tree_hard_cov_activity": {"feature_set": "cov_activity", "gate_mode": "hard"},
    "p536_tree_soft_cov_activity": {"feature_set": "cov_activity", "gate_mode": "soft"},
}

SEQ_BASE = {
    "p527_lstm_l3_v5_lite_s2027": {
        "version": "v5_lite",
        "builder": os.path.join(PROJECT_ROOT, "src", "features", "build_features_v5_lite_sku.py"),
        "meta_name": "meta_v5_lite.json",
        "processed_name": "X_train_dyn.bin",
        "model_type": "lstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
    },
    "p537_lstm_pool_v5_lite_cov": {
        "version": "v5_lite_cov",
        "builder": os.path.join(PROJECT_ROOT, "src", "features", "build_features_v5_lite_cov_sku.py"),
        "meta_name": "meta_v5_lite_cov.json",
        "processed_name": "X_train_dyn.bin",
        "model_type": "lstm_pool",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
    },
    "p539_attn_v5_lite_cov": {
        "version": "v5_lite_cov",
        "builder": os.path.join(PROJECT_ROOT, "src", "features", "build_features_v5_lite_cov_sku.py"),
        "meta_name": "meta_v5_lite_cov.json",
        "processed_name": "X_train_dyn.bin",
        "model_type": "attn",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
    },
}


def ensure_dirs():
    os.makedirs(REPORTS_ROOT, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "models_phase5_5"), exist_ok=True)


def init_timing_log():
    ensure_dirs()
    if os.path.exists(TIMING_LOG):
        return
    with open(TIMING_LOG, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "exp_id", "anchor_date", "base_exp", "track", "seed",
            "start_time", "end_time", "elapsed_min", "status", "note",
        ])


def append_timing(exp_id, anchor_date, base_exp, track, seed, t_start, t_end, status, note=""):
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            exp_id,
            anchor_date,
            base_exp,
            track,
            seed,
            datetime.fromtimestamp(t_start).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(t_end).strftime("%Y-%m-%d %H:%M:%S"),
            round((t_end - t_start) / 60.0, 2),
            status,
            note,
        ])


def load_selection():
    with open(WINNERS_JSON, "r", encoding="utf-8") as fh:
        winners = json.load(fh)
    with open(PHASE52_REFS, "r", encoding="utf-8") as fh:
        refs = json.load(fh)

    event_candidates = winners.get("event_tree_candidates", [])
    if winners.get("main_line") and winners["main_line"] not in event_candidates:
        event_candidates.insert(0, winners["main_line"])
    event_candidates = [exp for exp in event_candidates if exp in EVENT_BASE]
    include_runner_up = INCLUDE_RUNNER_UP or VAL_MODE == "single_anchor"
    if include_runner_up:
        event_candidates = event_candidates[:2]
    else:
        event_candidates = event_candidates[:1]

    sequence_keep = [refs["v5_lite"]["exp_id"]]
    if INCLUDE_SEQ_CHALLENGER:
        for exp in winners.get("sequence_keep", []):
            if exp != refs["v5_lite"]["exp_id"] and exp in SEQ_BASE:
                sequence_keep.append(exp)
        sequence_keep = sequence_keep[:2]
    else:
        sequence_keep = sequence_keep[:1]
    return winners, refs, event_candidates, sequence_keep


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def feature_tag(version, anchor_date):
    return f"p55_{anchor_tag(anchor_date)}_{version}"


def build_asset(version, anchor_date):
    def _run_builder(builder_path, tag_value):
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["FEATURE_SPLIT_DATE"] = anchor_date
        env["FEATURE_OUTPUT_TAG"] = tag_value
        env["FEATURE_VAL_MODE"] = VAL_MODE
        return subprocess.run([sys.executable, builder_path], cwd=PROJECT_ROOT, env=env).returncode

    if version in {"v6_event", "v5_lite_cov", "v3_filtered"}:
        base_tag = feature_tag("v5_lite", anchor_date)
        base_processed = os.path.join(PROJECT_ROOT, "data", f"processed_v5_lite_{base_tag}")
        base_artifacts = os.path.join(PROJECT_ROOT, "data", f"artifacts_v5_lite_{base_tag}")
        base_meta = os.path.join(base_artifacts, "meta_v5_lite.json")
        base_dyn = os.path.join(base_processed, "X_train_dyn.bin")
        if not (os.path.exists(base_meta) and os.path.exists(base_dyn)):
            base_builder = os.path.join(PROJECT_ROOT, "src", "features", "build_features_v5_lite_sku.py")
            rc = _run_builder(base_builder, base_tag)
            if rc != 0:
                raise RuntimeError(f"Failed to build v5_lite assets for {anchor_date}")

    if version == "v6_event":
        builder = os.path.join(PROJECT_ROOT, "src", "features", "build_features_v6_event_sku.py")
        meta_name = "meta_v6_event.json"
        processed_name = "X_train.npy"
    else:
        cfg = SEQ_BASE[next(base for base in SEQ_BASE if SEQ_BASE[base]["version"] == version)]
        builder = cfg["builder"]
        meta_name = cfg["meta_name"]
        processed_name = cfg["processed_name"]

    tag = feature_tag(version, anchor_date)
    processed_dir = os.path.join(PROJECT_ROOT, "data", f"processed_{version}_{tag}")
    artifacts_dir = os.path.join(PROJECT_ROOT, "data", f"artifacts_{version}_{tag}")
    meta_path = os.path.join(artifacts_dir, meta_name)
    processed_path = os.path.join(processed_dir, processed_name)

    if os.path.exists(meta_path) and os.path.exists(processed_path):
        return processed_dir, artifacts_dir, meta_path

    rc = _run_builder(builder, tag)
    if rc != 0:
        raise RuntimeError(f"Failed to build {version} assets for {anchor_date}")
    return processed_dir, artifacts_dir, meta_path


def report_dir(anchor_date):
    path = os.path.join(REPORTS_ROOT, anchor_tag(anchor_date))
    os.makedirs(path, exist_ok=True)
    return path


def model_dir(anchor_date, base_exp):
    path = os.path.join(PROJECT_ROOT, "models_phase5_5", anchor_tag(anchor_date), base_exp)
    os.makedirs(path, exist_ok=True)
    return path


def is_done(anchor_date, exp_id):
    base = report_dir(anchor_date)
    return (
        os.path.exists(os.path.join(base, f"eval_{exp_id}.txt"))
        and os.path.exists(os.path.join(base, f"agg_{exp_id}.txt"))
    )


def preflight():
    ensure_dirs()
    required = [
        os.path.join(PROJECT_ROOT, "src", "analysis", "summarize_phase5_5_results.py"),
        os.path.join(PROJECT_ROOT, "run_phase5_6_tree_sweep.py"),
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


def run_event_tree(exp_id, anchor_date, base_exp, seed):
    processed_dir, artifacts_dir, _ = build_asset("v6_event", anchor_date)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXP_ID"] = exp_id
    env["EXP_VERSION"] = "v6_event"
    env["EXP_FEATURE_SET"] = EVENT_BASE[base_exp]["feature_set"]
    env["EXP_GATE_MODE"] = EVENT_BASE[base_exp]["gate_mode"]
    env["EXP_SEED"] = str(seed)
    env["EXP_TREE_BACKEND"] = TREE_BACKEND
    env["EXP_QTY_GATE"] = QTY_GATE
    env["EXP_PROCESSED_DIR"] = processed_dir
    env["EXP_ARTIFACTS_DIR"] = artifacts_dir
    env["EXP_MODEL_DIR"] = model_dir(anchor_date, base_exp)
    env["EXP_REPORT_DIR"] = report_dir(anchor_date)

    rc = subprocess.run([sys.executable, os.path.join("src", "train", "train_tabular_v6.py")], cwd=PROJECT_ROOT, env=env).returncode
    if rc != 0:
        return rc
    rc = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "evaluate_tabular.py")], cwd=PROJECT_ROOT, env=env).returncode
    if rc != 0:
        return rc
    rc = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "evaluate_agg.py")], cwd=PROJECT_ROOT, env=env).returncode
    return rc


def backup_sequence_variants(exp_id, exp_model_dir):
    mapping = {
        "best_enhanced_model.pth": f"best_{exp_id}.pth",
        "best_loss_enhanced_model.pth": f"best_loss_{exp_id}.pth",
        "best_f1_enhanced_model.pth": f"best_f1_{exp_id}.pth",
        "best_wmape_enhanced_model.pth": f"best_wmape_{exp_id}.pth",
        "last_enhanced_model.pth": f"last_{exp_id}.pth",
    }
    for src_name, dst_name in mapping.items():
        src = os.path.join(exp_model_dir, src_name)
        if not os.path.exists(src):
            return False
        shutil.copy(src, os.path.join(exp_model_dir, dst_name))
    return True


def run_sequence(exp_id, anchor_date, base_exp, seed):
    cfg = SEQ_BASE[base_exp]
    processed_dir, artifacts_dir, _ = build_asset(cfg["version"], anchor_date)
    exp_model_dir = model_dir(anchor_date, base_exp)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXP_ID"] = exp_id
    env["EXP_VERSION"] = cfg["version"]
    env["EXP_MODEL_TYPE"] = cfg["model_type"]
    env["EXP_HIDDEN"] = cfg["hidden"]
    env["EXP_LAYERS"] = cfg["layers"]
    env["EXP_DROPOUT"] = cfg["dropout"]
    env["EXP_BATCH"] = cfg["batch"]
    env["EXP_LR"] = cfg["lr"]
    env["EXP_EPOCHS"] = SEQ_EPOCHS
    env["EXP_PATIENCE"] = SEQ_PATIENCE
    env["EXP_WORKERS"] = SEQ_WORKERS
    env["EXP_EVAL_WORKERS"] = SEQ_WORKERS
    env["EXP_MONITOR_METRIC"] = SEQ_MONITOR
    env["EXP_SEED"] = str(seed)
    env["EXP_DETERMINISTIC"] = "1"
    env["EXP_QTY_GATE"] = QTY_GATE
    env["EXP_POS_WEIGHT"] = DEFAULT_SEQ_LOSS["pos_w"]
    env["EXP_FP_PENALTY"] = DEFAULT_SEQ_LOSS["fp"]
    env["EXP_GAMMA"] = DEFAULT_SEQ_LOSS["gamma"]
    env["EXP_REG_WEIGHT"] = DEFAULT_SEQ_LOSS["reg_w"]
    env["EXP_SOFT_F1"] = DEFAULT_SEQ_LOSS["sf1"]
    env["EXP_HUBER"] = DEFAULT_SEQ_LOSS["hub"]
    env["EXP_PROCESSED_DIR"] = processed_dir
    env["EXP_ARTIFACTS_DIR"] = artifacts_dir
    env["EXP_MODEL_DIR"] = exp_model_dir
    env["EXP_REPORT_DIR"] = report_dir(anchor_date)

    rc = subprocess.run([sys.executable, os.path.join("src", "train", "run_training_v2.py")], cwd=PROJECT_ROOT, env=env).returncode
    if rc != 0:
        return rc
    if not backup_sequence_variants(exp_id, exp_model_dir):
        return 1
    rc = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "evaluate.py")], cwd=PROJECT_ROOT, env=env).returncode
    if rc != 0:
        return rc
    rc = subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "evaluate_agg.py")], cwd=PROJECT_ROOT, env=env).returncode
    return rc


def main():
    if not preflight():
        print("[BLOCKED] phase5.5 preflight failed")
        raise SystemExit(1)
    init_timing_log()
    winners, refs, event_candidates, sequence_keep = load_selection()

    jobs = []
    for anchor_date in ANCHORS:
        for seed in SEEDS:
            for base_exp in event_candidates:
                jobs.append(("event_tree", anchor_date, base_exp, seed))
            for base_exp in sequence_keep:
                jobs.append(("sequence", anchor_date, base_exp, seed))

    print(json.dumps({
        "anchors": ANCHORS,
        "seeds": SEEDS,
        "val_mode": VAL_MODE,
        "event_candidates": event_candidates,
        "sequence_keep": sequence_keep,
        "winners": winners,
    }, ensure_ascii=False, indent=2))
    print(f"[Phase 5.5] local anchor jobs={len(jobs)}")

    for track, anchor_date, base_exp, seed in jobs:
        exp_id = f"p55_{anchor_tag(anchor_date)}_{base_exp}_s{seed}"
        if is_done(anchor_date, exp_id):
            print(f"[SKIP] already done: {exp_id}")
            continue
        t0 = time.time()
        print("\n" + "=" * 72)
        print(f"[RUN] {exp_id} | anchor={anchor_date} | track={track} | base={base_exp} | seed={seed}")
        print("=" * 72)
        if track == "event_tree":
            rc = run_event_tree(exp_id, anchor_date, base_exp, seed)
        else:
            rc = run_sequence(exp_id, anchor_date, base_exp, seed)
        t1 = time.time()
        append_timing(exp_id, anchor_date, base_exp, track, seed, t0, t1, "success" if rc == 0 else "failed", "")
        if rc != 0:
            print(f"[STOP] anchor job failed: {exp_id}")
            raise SystemExit(rc)

    summary_rc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "analysis", "summarize_phase5_5_results.py")],
        cwd=PROJECT_ROOT,
    ).returncode
    if summary_rc != 0:
        print("[STOP] phase5.5 summary failed")
        raise SystemExit(summary_rc)

    if RUN_TREE_SWEEP:
        print("[Phase 5.5] launching phase5.6 tree sweep ...")
        sweep_env = os.environ.copy()
        sweep_env.setdefault("PHASE56_VAL_MODE", VAL_MODE)
        rc = subprocess.run(
            [sys.executable, os.path.join(PROJECT_ROOT, "run_phase5_6_tree_sweep.py")],
            cwd=PROJECT_ROOT,
            env=sweep_env,
        ).returncode
        if rc != 0:
            print("[STOP] phase5.6 tree sweep failed")
            raise SystemExit(rc)

    print("[DONE] phase5.5 local anchors completed")


if __name__ == "__main__":
    main()
