import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_4")
TIMING_LOG = os.path.join(REPORTS_DIR, "phase5_4_timing_log.csv")
WINNERS_JSON = os.path.join(PROJECT_ROOT, "reports", "phase5_3", "phase5_3_winners.json")
PHASE52_REFS = os.path.join(PROJECT_ROOT, "reports", "phase5_2", "phase5_2_sequence_refs.json")

SEEDS = [int(x.strip()) for x in os.environ.get("PHASE54_SEEDS", "2026,2027,2028").split(",") if x.strip()]
SEQ_EPOCHS = os.environ.get("PHASE54_SEQ_EPOCHS", "10")
SEQ_PATIENCE = os.environ.get("PHASE54_SEQ_PATIENCE", "3")
SEQ_WORKERS = os.environ.get("PHASE54_SEQ_WORKERS", "2")
SEQ_MONITOR = os.environ.get("PHASE54_SEQ_MONITOR", "wmape").lower()
QTY_GATE = os.environ.get("PHASE54_QTY_GATE", "0.20")
TREE_BACKEND = os.environ.get("PHASE54_TREE_BACKEND", "lightgbm").lower()

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
        "model_type": "lstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
    },
    "p537_lstm_pool_v5_lite_cov": {
        "version": "v5_lite_cov",
        "model_type": "lstm_pool",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
    },
    "p539_attn_v5_lite_cov": {
        "version": "v5_lite_cov",
        "model_type": "attn",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
    },
}


def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "models_phase5_4"), exist_ok=True)


def init_timing_log():
    ensure_dirs()
    if os.path.exists(TIMING_LOG):
        return
    with open(TIMING_LOG, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "exp_id", "base_exp", "track", "seed", "start_time", "end_time",
            "elapsed_min", "status", "note",
        ])


def append_timing(exp_id, base_exp, track, seed, t_start, t_end, status, note=""):
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            exp_id,
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
    if not os.path.exists(WINNERS_JSON):
        raise FileNotFoundError(f"Missing phase5.3 winners: {WINNERS_JSON}")
    if not os.path.exists(PHASE52_REFS):
        raise FileNotFoundError(f"Missing phase5.2 refs: {PHASE52_REFS}")

    with open(WINNERS_JSON, "r", encoding="utf-8") as fh:
        winners = json.load(fh)
    with open(PHASE52_REFS, "r", encoding="utf-8") as fh:
        refs = json.load(fh)

    event_candidates = winners.get("event_tree_candidates", [])
    sequence_keep = winners.get("sequence_keep", [])
    if not sequence_keep:
        sequence_keep = [refs["v5_lite"]["exp_id"]]
    return winners, refs, event_candidates, sequence_keep


def get_model_dir(track, base_exp):
    sub = "event_tree" if track == "event_tree" else "sequence"
    path = os.path.join(PROJECT_ROOT, "models_phase5_4", sub, base_exp)
    os.makedirs(path, exist_ok=True)
    return path


def ensure_assets(version):
    if version == "v6_event":
        processed = os.path.join(PROJECT_ROOT, "data", "processed_v6_event", "X_train.npy")
        meta = os.path.join(PROJECT_ROOT, "data", "artifacts_v6_event", "meta_v6_event.json")
        builder = os.path.join(PROJECT_ROOT, "src", "features", "build_features_v6_event_sku.py")
    elif version == "v5_lite":
        processed = os.path.join(PROJECT_ROOT, "data", "processed_v5_lite", "X_train_dyn.bin")
        meta = os.path.join(PROJECT_ROOT, "data", "artifacts_v5_lite", "meta_v5_lite.json")
        builder = os.path.join(PROJECT_ROOT, "src", "features", "build_features_v5_lite_sku.py")
    elif version == "v5_lite_cov":
        processed = os.path.join(PROJECT_ROOT, "data", "processed_v5_lite_cov", "X_train_dyn.bin")
        meta = os.path.join(PROJECT_ROOT, "data", "artifacts_v5_lite_cov", "meta_v5_lite_cov.json")
        builder = os.path.join(PROJECT_ROOT, "src", "features", "build_features_v5_lite_cov_sku.py")
    else:
        raise ValueError(f"Unsupported version: {version}")

    if os.path.exists(processed) and os.path.exists(meta):
        return True
    return subprocess.run([sys.executable, builder], cwd=PROJECT_ROOT).returncode == 0


def preflight():
    ensure_dirs()
    winners, refs, event_candidates, sequence_keep = load_selection()
    required = [
        os.path.join(PROJECT_ROOT, "src", "train", "run_training_v2.py"),
        os.path.join(PROJECT_ROOT, "src", "train", "train_tabular_v6.py"),
        os.path.join(PROJECT_ROOT, "evaluate.py"),
        os.path.join(PROJECT_ROOT, "evaluate_tabular.py"),
        os.path.join(PROJECT_ROOT, "evaluate_agg.py"),
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

    for version in {"v6_event"} | {SEQ_BASE[exp]["version"] for exp in sequence_keep if exp in SEQ_BASE}:
        if not ensure_assets(version):
            return False

    print(json.dumps({
        "phase5_3_winners": winners,
        "phase5_2_refs": refs,
        "event_candidates": event_candidates,
        "sequence_keep": sequence_keep,
        "seeds": SEEDS,
    }, ensure_ascii=False, indent=2))
    return True


def event_env(exp_id, base_exp, seed):
    cfg = EVENT_BASE[base_exp]
    model_dir = get_model_dir("event_tree", base_exp)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXP_ID"] = exp_id
    env["EXP_VERSION"] = "v6_event"
    env["EXP_FEATURE_SET"] = cfg["feature_set"]
    env["EXP_GATE_MODE"] = cfg["gate_mode"]
    env["EXP_SEED"] = str(seed)
    env["EXP_MODEL_DIR"] = model_dir
    env["EXP_ARTIFACTS_DIR"] = os.path.join(PROJECT_ROOT, "data", "artifacts_v6_event")
    env["EXP_PROCESSED_DIR"] = os.path.join(PROJECT_ROOT, "data", "processed_v6_event")
    env["EXP_REPORT_DIR"] = REPORTS_DIR
    env["EXP_QTY_GATE"] = QTY_GATE
    env["EXP_TREE_BACKEND"] = TREE_BACKEND
    return env


def seq_env(exp_id, base_exp, seed):
    cfg = SEQ_BASE[base_exp]
    version = cfg["version"]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXP_ID"] = exp_id
    env["EXP_VERSION"] = version
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
    env["EXP_REPORT_DIR"] = REPORTS_DIR
    env["EXP_MODEL_DIR"] = get_model_dir("sequence", base_exp)
    env["EXP_ARTIFACTS_DIR"] = os.path.join(PROJECT_ROOT, "data", f"artifacts_{version}")
    env["EXP_PROCESSED_DIR"] = os.path.join(PROJECT_ROOT, "data", f"processed_{version}")
    env["EXP_POS_WEIGHT"] = DEFAULT_SEQ_LOSS["pos_w"]
    env["EXP_FP_PENALTY"] = DEFAULT_SEQ_LOSS["fp"]
    env["EXP_GAMMA"] = DEFAULT_SEQ_LOSS["gamma"]
    env["EXP_REG_WEIGHT"] = DEFAULT_SEQ_LOSS["reg_w"]
    env["EXP_SOFT_F1"] = DEFAULT_SEQ_LOSS["sf1"]
    env["EXP_HUBER"] = DEFAULT_SEQ_LOSS["hub"]
    env["EXP_QTY_GATE"] = QTY_GATE
    return env


def run_event_tree(exp_id, env):
    train_rc = subprocess.run(
        [sys.executable, os.path.join("src", "train", "train_tabular_v6.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    if train_rc != 0:
        return train_rc
    eval_rc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "evaluate_tabular.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    if eval_rc != 0:
        return eval_rc
    agg_rc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "evaluate_agg.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    return agg_rc


def backup_sequence_variants(exp_id, model_dir):
    mapping = {
        "best_enhanced_model.pth": f"best_{exp_id}.pth",
        "best_loss_enhanced_model.pth": f"best_loss_{exp_id}.pth",
        "best_f1_enhanced_model.pth": f"best_f1_{exp_id}.pth",
        "best_wmape_enhanced_model.pth": f"best_wmape_{exp_id}.pth",
        "last_enhanced_model.pth": f"last_{exp_id}.pth",
    }
    for src_name, dst_name in mapping.items():
        src = os.path.join(model_dir, src_name)
        if not os.path.exists(src):
            return False
        shutil.copy(src, os.path.join(model_dir, dst_name))
    return True


def run_sequence(exp_id, env):
    train_rc = subprocess.run(
        [sys.executable, os.path.join("src", "train", "run_training_v2.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    if train_rc != 0:
        return train_rc
    if not backup_sequence_variants(exp_id, env["EXP_MODEL_DIR"]):
        return 1
    eval_rc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "evaluate.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    if eval_rc != 0:
        return eval_rc
    agg_rc = subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "evaluate_agg.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    return agg_rc


def is_done(exp_id):
    return (
        os.path.exists(os.path.join(REPORTS_DIR, f"eval_{exp_id}.txt"))
        and os.path.exists(os.path.join(REPORTS_DIR, f"agg_{exp_id}.txt"))
    )


def main():
    if not preflight():
        print("[BLOCKED] phase5.4 preflight failed")
        raise SystemExit(1)
    init_timing_log()

    winners, refs, event_candidates, sequence_keep = load_selection()

    jobs = []
    for base_exp in event_candidates:
        if base_exp not in EVENT_BASE:
            continue
        for seed in SEEDS:
            jobs.append(("event_tree", base_exp, seed))
    for base_exp in sequence_keep:
        if base_exp not in SEQ_BASE:
            continue
        for seed in SEEDS:
            jobs.append(("sequence", base_exp, seed))

    print(f"[Phase 5.4] local confirm jobs={len(jobs)}")
    for track, base_exp, seed in jobs:
        exp_id = f"p54_{base_exp}_s{seed}"
        if is_done(exp_id):
            print(f"[SKIP] already done: {exp_id}")
            continue
        t0 = time.time()
        print("\n" + "=" * 72)
        print(f"[RUN] {exp_id} | track={track} | base={base_exp} | seed={seed}")
        print("=" * 72)
        if track == "event_tree":
            rc = run_event_tree(exp_id, event_env(exp_id, base_exp, seed))
        else:
            rc = run_sequence(exp_id, seq_env(exp_id, base_exp, seed))
        t1 = time.time()
        append_timing(exp_id, base_exp, track, seed, t0, t1, "success" if rc == 0 else "failed", "")
        if rc != 0:
            print(f"[STOP] job failed: {exp_id}")
            raise SystemExit(rc)

    print("[DONE] phase5.4 local confirm completed")


if __name__ == "__main__":
    main()
