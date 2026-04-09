import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_3")
TIMING_LOG = os.path.join(PROJECT_ROOT, "reports", "phase5_3_timing_log.csv")
PHASE52_REFS = os.path.join(PROJECT_ROOT, "reports", "phase5_2", "phase5_2_sequence_refs.json")

SEQ_EPOCHS = os.environ.get("PHASE53_SEQ_EPOCHS", "10")
SEQ_PATIENCE = os.environ.get("PHASE53_SEQ_PATIENCE", "3")
SEQ_WORKERS = os.environ.get("PHASE53_SEQ_WORKERS", "2")
SEQ_MONITOR = os.environ.get("PHASE53_SEQ_MONITOR", "wmape").lower()
PHASE53_SEED = os.environ.get("PHASE53_SEED", "2026")
QTY_GATE = os.environ.get("PHASE53_QTY_GATE", "0.20")
TREE_BACKEND = os.environ.get("PHASE53_TREE_BACKEND", "lightgbm").lower()

DEFAULT_SEQ_LOSS = {
    "pos_w": "5.85",
    "fp": "0.15",
    "gamma": "1.5",
    "reg_w": "0.5",
    "sf1": "0.5",
    "hub": "1.5",
}

EXPERIMENTS = [
    {
        "id": "p531_tree_hard_core",
        "track": "event_tree",
        "version": "v6_event",
        "feature_set": "core",
        "gate_mode": "hard",
        "desc": "Core replenish/future rolling features with hard gate.",
    },
    {
        "id": "p532_tree_soft_core",
        "track": "event_tree",
        "version": "v6_event",
        "feature_set": "core",
        "gate_mode": "soft",
        "desc": "Core replenish/future rolling features with soft gate.",
    },
    {
        "id": "p533_tree_hard_cov",
        "track": "event_tree",
        "version": "v6_event",
        "feature_set": "cov",
        "gate_mode": "hard",
        "desc": "Core event features plus buyer coverage, hard gate.",
    },
    {
        "id": "p534_tree_soft_cov",
        "track": "event_tree",
        "version": "v6_event",
        "feature_set": "cov",
        "gate_mode": "soft",
        "desc": "Core event features plus buyer coverage, soft gate.",
    },
    {
        "id": "p535_tree_hard_cov_activity",
        "track": "event_tree",
        "version": "v6_event",
        "feature_set": "cov_activity",
        "gate_mode": "hard",
        "desc": "Buyer coverage plus activity/cold-start features, hard gate.",
    },
    {
        "id": "p536_tree_soft_cov_activity",
        "track": "event_tree",
        "version": "v6_event",
        "feature_set": "cov_activity",
        "gate_mode": "soft",
        "desc": "Buyer coverage plus activity/cold-start features, soft gate.",
    },
    {
        "id": "p537_lstm_pool_v5_lite_cov",
        "track": "sequence",
        "version": "v5_lite_cov",
        "model_type": "lstm_pool",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
        "desc": "V5-lite sparse sequence with buyer coverage and last+mean+max pooled LSTM.",
    },
    {
        "id": "p539_attn_v5_lite_cov",
        "track": "sequence",
        "version": "v5_lite_cov",
        "model_type": "attn",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
        "desc": "V5-lite sparse sequence with buyer coverage and attention LSTM.",
    },
]


def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "models_v6_event"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "models_v5_lite_cov"), exist_ok=True)


def init_timing_log():
    if os.path.exists(TIMING_LOG):
        return
    with open(TIMING_LOG, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "exp_id", "track", "version", "config", "start_time", "end_time",
            "elapsed_min", "status", "note",
        ])


def append_timing(exp, t_start, t_end, status, note=""):
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            exp["id"],
            exp["track"],
            exp["version"],
            exp.get("feature_set", exp.get("model_type", "")),
            datetime.fromtimestamp(t_start).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(t_end).strftime("%Y-%m-%d %H:%M:%S"),
            round((t_end - t_start) / 60.0, 2),
            status,
            note,
        ])


def load_phase52_refs():
    if not os.path.exists(PHASE52_REFS):
        return None
    with open(PHASE52_REFS, "r", encoding="utf-8") as fh:
        return json.load(fh)


def ensure_assets(version):
    if version == "v6_event":
        processed = os.path.join(PROJECT_ROOT, "data", "processed_v6_event", "X_train.npy")
        meta = os.path.join(PROJECT_ROOT, "data", "artifacts_v6_event", "meta_v6_event.json")
        builder = os.path.join(PROJECT_ROOT, "src", "features", "build_features_v6_event_sku.py")
    elif version == "v5_lite_cov":
        processed = os.path.join(PROJECT_ROOT, "data", "processed_v5_lite_cov", "X_train_dyn.bin")
        meta = os.path.join(PROJECT_ROOT, "data", "artifacts_v5_lite_cov", "meta_v5_lite_cov.json")
        builder = os.path.join(PROJECT_ROOT, "src", "features", "build_features_v5_lite_cov_sku.py")
    else:
        raise ValueError(f"Unsupported version: {version}")

    if os.path.exists(processed) and os.path.exists(meta):
        return True
    print(f"  [AUTO-BUILD] {version} assets missing. Building now...")
    result = subprocess.run([sys.executable, builder], cwd=PROJECT_ROOT)
    return result.returncode == 0 and os.path.exists(processed) and os.path.exists(meta)


def preflight():
    print("\n[Preflight] Checking Phase 5.3 prerequisites...")
    ensure_dirs()
    refs = load_phase52_refs()
    if not refs:
        print(f"  [MISSING] frozen refs: {os.path.relpath(PHASE52_REFS, PROJECT_ROOT)}")
        print("  Run phase5.2 summary first after phase5.2 completes.")
        return False
    print(f"  [OK] frozen refs: {os.path.relpath(PHASE52_REFS, PROJECT_ROOT)}")
    print(json.dumps(refs, ensure_ascii=False, indent=2))

    required = [
        os.path.join(PROJECT_ROOT, "src", "train", "run_training_v2.py"),
        os.path.join(PROJECT_ROOT, "src", "train", "train_tabular_v6.py"),
        os.path.join(PROJECT_ROOT, "evaluate.py"),
        os.path.join(PROJECT_ROOT, "evaluate_tabular.py"),
        os.path.join(PROJECT_ROOT, "evaluate_agg.py"),
    ]
    missing = [path for path in required if not os.path.exists(path)]
    for path in required:
        print(f"  [{'OK' if os.path.exists(path) else 'MISSING'}] {os.path.relpath(path, PROJECT_ROOT)}")
    if missing:
        return False

    if TREE_BACKEND == "lightgbm":
        check = subprocess.run(
            [sys.executable, "-c", "import lightgbm"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  [{'OK' if check.returncode == 0 else 'MISSING'}] python dependency: lightgbm")
        if check.returncode != 0:
            return False
    elif TREE_BACKEND != "sklearn":
        print(f"  [MISSING] unsupported tree backend: {TREE_BACKEND}")
        return False

    for version in ("v6_event", "v5_lite_cov"):
        ok = ensure_assets(version)
        print(f"  [{'OK' if ok else 'MISSING'}] data assets: {version}")
        if not ok:
            return False
    return True


def is_done(exp):
    if exp["track"] == "event_tree":
        cls_path = os.path.join(PROJECT_ROOT, "models_v6_event", f"{exp['id']}_cls.pkl")
        eval_txt = os.path.join(REPORTS_DIR, f"eval_{exp['id']}.txt")
        agg_txt = os.path.join(REPORTS_DIR, f"agg_{exp['id']}.txt")
        return os.path.exists(cls_path) and os.path.exists(eval_txt) and os.path.exists(agg_txt)
    model_path = os.path.join(PROJECT_ROOT, "models_v5_lite_cov", f"best_{exp['id']}.pth")
    eval_txt = os.path.join(REPORTS_DIR, f"eval_{exp['id']}.txt")
    agg_txt = os.path.join(REPORTS_DIR, f"agg_{exp['id']}.txt")
    return os.path.exists(model_path) and os.path.exists(eval_txt) and os.path.exists(agg_txt)


def build_env(exp):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXP_ID"] = exp["id"]
    env["EXP_VERSION"] = exp["version"]
    env["EXP_SEED"] = PHASE53_SEED
    env["EXP_REPORT_DIR"] = REPORTS_DIR
    env["EXP_QTY_GATE"] = QTY_GATE
    if exp["track"] == "event_tree":
        env["EXP_MODEL_DIR"] = os.path.join(PROJECT_ROOT, "models_v6_event")
        env["EXP_ARTIFACTS_DIR"] = os.path.join(PROJECT_ROOT, "data", "artifacts_v6_event")
        env["EXP_PROCESSED_DIR"] = os.path.join(PROJECT_ROOT, "data", "processed_v6_event")
        env["EXP_FEATURE_SET"] = exp["feature_set"]
        env["EXP_GATE_MODE"] = exp["gate_mode"]
        env["EXP_TREE_BACKEND"] = TREE_BACKEND
    else:
        env["EXP_MODEL_DIR"] = os.path.join(PROJECT_ROOT, "models_v5_lite_cov")
        env["EXP_ARTIFACTS_DIR"] = os.path.join(PROJECT_ROOT, "data", "artifacts_v5_lite_cov")
        env["EXP_PROCESSED_DIR"] = os.path.join(PROJECT_ROOT, "data", "processed_v5_lite_cov")
        env["EXP_MODEL_TYPE"] = exp["model_type"]
        env["EXP_HIDDEN"] = exp["hidden"]
        env["EXP_LAYERS"] = exp["layers"]
        env["EXP_DROPOUT"] = exp["dropout"]
        env["EXP_BATCH"] = exp["batch"]
        env["EXP_LR"] = exp["lr"]
        env["EXP_EPOCHS"] = SEQ_EPOCHS
        env["EXP_PATIENCE"] = SEQ_PATIENCE
        env["EXP_WORKERS"] = SEQ_WORKERS
        env["EXP_EVAL_WORKERS"] = SEQ_WORKERS
        env["EXP_MONITOR_METRIC"] = SEQ_MONITOR
        env["EXP_MODEL_FILE"] = "best_enhanced_model.pth"
        env["EXP_POS_WEIGHT"] = DEFAULT_SEQ_LOSS["pos_w"]
        env["EXP_FP_PENALTY"] = DEFAULT_SEQ_LOSS["fp"]
        env["EXP_GAMMA"] = DEFAULT_SEQ_LOSS["gamma"]
        env["EXP_REG_WEIGHT"] = DEFAULT_SEQ_LOSS["reg_w"]
        env["EXP_SOFT_F1"] = DEFAULT_SEQ_LOSS["sf1"]
        env["EXP_HUBER"] = DEFAULT_SEQ_LOSS["hub"]
    return env


def run_with_log(cmd, env, out_path):
    with open(out_path, "w", encoding="utf-8") as fh:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return result.returncode


def run_event_tree(exp, env):
    code = subprocess.run(
        [sys.executable, os.path.join("src", "train", "train_tabular_v6.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    if code != 0:
        return False, "train_error"

    eval_out = os.path.join(REPORTS_DIR, f"eval_{exp['id']}.txt")
    code = run_with_log([sys.executable, "evaluate_tabular.py"], env, eval_out)
    if code != 0:
        return False, "eval_error"

    agg_out = os.path.join(REPORTS_DIR, f"agg_{exp['id']}.txt")
    code = run_with_log([sys.executable, "evaluate_agg.py", "--exp", exp["id"]], env, agg_out)
    if code != 0:
        return False, "agg_error"
    return True, ""


def run_sequence(exp, env):
    code = subprocess.run(
        [sys.executable, os.path.join("src", "train", "run_training_v2.py")],
        cwd=PROJECT_ROOT,
        env=env,
    ).returncode
    if code != 0:
        return False, "train_error"

    model_dir = env["EXP_MODEL_DIR"]
    generic = os.path.join(model_dir, "best_enhanced_model.pth")
    backup = os.path.join(model_dir, f"best_{exp['id']}.pth")
    if not os.path.exists(generic):
        return False, "missing_best_checkpoint"
    shutil.copy(generic, backup)

    eval_out = os.path.join(REPORTS_DIR, f"eval_{exp['id']}.txt")
    code = run_with_log([sys.executable, "evaluate.py"], env, eval_out)
    if code != 0:
        return False, "eval_error"

    agg_out = os.path.join(REPORTS_DIR, f"agg_{exp['id']}.txt")
    code = run_with_log([sys.executable, "evaluate_agg.py", "--exp", exp["id"]], env, agg_out)
    if code != 0:
        return False, "agg_error"
    return True, ""


def main():
    print("=" * 76)
    print("Phase 5.3A dual-track local runner")
    print(f"sequence: epochs={SEQ_EPOCHS} | patience={SEQ_PATIENCE} | workers={SEQ_WORKERS} | monitor={SEQ_MONITOR}")
    print(f"event/tree backend={TREE_BACKEND} | gate threshold={QTY_GATE} | seed={PHASE53_SEED}")
    print("=" * 76)

    if not preflight():
        sys.exit(1)

    init_timing_log()
    total = len(EXPERIMENTS)
    for idx, exp in enumerate(EXPERIMENTS, start=1):
        if is_done(exp):
            print(f"\n[SKIP] [{idx}/{total}] {exp['id']} already complete.")
            continue

        print(f"\n{'=' * 76}")
        print(f"[RUN] [{idx}/{total}] {exp['id']}")
        print(f"track={exp['track']} | version={exp['version']} | note={exp['desc']}")
        env = build_env(exp)
        t_start = time.time()

        if exp["track"] == "event_tree":
            ok, note = run_event_tree(exp, env)
        else:
            ok, note = run_sequence(exp, env)

        t_end = time.time()
        append_timing(exp, t_start, t_end, "success" if ok else note, note="" if ok else note)
        if not ok:
            print(f"[STOP] {exp['id']} failed: {note}")
            sys.exit(1)
        print(f"[OK] {exp['id']} closed | elapsed={(t_end - t_start) / 60:.1f} min")

    print(f"\nDone. Reports: {os.path.relpath(REPORTS_DIR, PROJECT_ROOT)}")
    print(f"Timing: {os.path.relpath(TIMING_LOG, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
