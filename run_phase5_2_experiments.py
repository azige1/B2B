"""
Phase 5.2 fair-universe runner.

Purpose:
- Compare V3-filtered vs V5-lite on the same SKU universe.
- Use multiple seeds to reduce single-run variance.
- Monitor business-aligned WMAPE by default while still saving loss/f1/wmape checkpoints.

Default budget:
- epochs=12
- patience=4
- workers=2
- seeds=2026,2027
- primary monitor=wmape
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_2")
TIMING_LOG = os.path.join(PROJECT_ROOT, "reports", "phase5_2_timing_log.csv")

GLOBAL_EPOCHS = os.environ.get("PHASE52_EPOCHS", "12")
GLOBAL_PATIENCE = os.environ.get("PHASE52_PATIENCE", "4")
GLOBAL_WORKERS = os.environ.get("PHASE52_WORKERS", "2")
GLOBAL_MONITOR = os.environ.get("PHASE52_MONITOR", "wmape").lower()
GLOBAL_SEEDS = [int(x.strip()) for x in os.environ.get("PHASE52_SEEDS", "2026,2027").split(",") if x.strip()]

DEFAULT_LOSS = {
    "pos_w": "5.85",
    "fp": "0.15",
    "gamma": "1.5",
    "reg_w": "0.5",
    "sf1": "0.5",
    "hub": "1.5",
}

BASE_EXPERIMENTS = [
    {
        "tag": "lstm_l3_v3_filtered",
        "version": "v3_filtered",
        "model_type": "lstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
        "desc": "Fair-universe V3 baseline with original 7-dim features.",
    },
    {
        "tag": "bilstm_l3_v3_filtered",
        "version": "v3_filtered",
        "model_type": "bilstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
        "desc": "Fair-universe V3 BiLSTM baseline.",
    },
    {
        "tag": "lstm_l3_v5_lite",
        "version": "v5_lite",
        "model_type": "lstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
        "desc": "Fair-universe V5-lite LSTM candidate.",
    },
    {
        "tag": "bilstm_l3_v5_lite",
        "version": "v5_lite",
        "model_type": "bilstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
        "desc": "Fair-universe V5-lite BiLSTM candidate.",
    },
]


def build_experiments():
    experiments = []
    counter = 521
    for seed in GLOBAL_SEEDS:
        for base in BASE_EXPERIMENTS:
            exp = dict(base)
            exp["seed"] = str(seed)
            exp["id"] = f"p{counter}_{base['tag']}_s{seed}"
            experiments.append(exp)
            counter += 1
    return experiments


EXPERIMENTS = build_experiments()


def get_model_dir(version):
    if version == "v3":
        return os.path.join(PROJECT_ROOT, "models_v2")
    if version == "v5":
        return os.path.join(PROJECT_ROOT, "models_v5")
    if version == "v5_lite":
        return os.path.join(PROJECT_ROOT, "models_v5_lite")
    if version == "v3_filtered":
        return os.path.join(PROJECT_ROOT, "models_v3_filtered")
    return os.path.join(PROJECT_ROOT, f"models_{version}")


def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    for version in ("v3_filtered", "v5_lite"):
        os.makedirs(get_model_dir(version), exist_ok=True)


def init_timing_log():
    os.makedirs(os.path.dirname(TIMING_LOG), exist_ok=True)
    if os.path.exists(TIMING_LOG):
        return
    with open(TIMING_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_id", "version", "model_type", "seed", "monitor",
            "start_time", "end_time", "elapsed_min",
            "actual_epochs", "min_per_epoch", "status", "note"
        ])


def append_timing(exp, t_start, t_end, actual_epochs, status, note=""):
    elapsed_min = round((t_end - t_start) / 60, 2)
    min_per_epoch = round(elapsed_min / max(actual_epochs, 1), 2)
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            exp["id"], exp["version"], exp["model_type"], exp["seed"], GLOBAL_MONITOR,
            datetime.fromtimestamp(t_start).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(t_end).strftime("%Y-%m-%d %H:%M:%S"),
            elapsed_min, actual_epochs, min_per_epoch, status, note
        ])
    return elapsed_min


def get_actual_epochs(exp_id):
    history_file = os.path.join(PROJECT_ROOT, "reports", f"history_{exp_id}.csv")
    if not os.path.exists(history_file):
        return 0
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            return max(0, len(f.readlines()) - 1)
    except Exception:
        return 0


def is_done(exp):
    primary = os.path.join(get_model_dir(exp["version"]), f"best_{exp['id']}.pth")
    return os.path.exists(primary)


def ensure_version_assets(version):
    version_files = {
        "v3_filtered": {
            "processed": os.path.join(PROJECT_ROOT, "data", "processed_v3_filtered", "X_train_dyn.bin"),
            "meta": os.path.join(PROJECT_ROOT, "data", "artifacts_v3_filtered", "meta_v3_filtered.json"),
            "builder": os.path.join(PROJECT_ROOT, "src", "features", "build_features_v3_filtered_sku.py"),
        },
        "v5_lite": {
            "processed": os.path.join(PROJECT_ROOT, "data", "processed_v5_lite", "X_train_dyn.bin"),
            "meta": os.path.join(PROJECT_ROOT, "data", "artifacts_v5_lite", "meta_v5_lite.json"),
            "builder": None,
        },
    }
    info = version_files[version]
    if os.path.exists(info["processed"]) and os.path.exists(info["meta"]):
        return True

    if info["builder"] is None:
        return False

    print(f"  [AUTO-BUILD] {version} assets missing. Building now...")
    result = subprocess.run([sys.executable, info["builder"]], cwd=PROJECT_ROOT)
    return result.returncode == 0 and os.path.exists(info["processed"]) and os.path.exists(info["meta"])


def preflight():
    print("\n[Preflight] Checking Phase 5.2 prerequisites...")
    ensure_dirs()

    required = [
        ("train entry", os.path.join(PROJECT_ROOT, "src", "train", "run_training_v2.py")),
        ("eval entry", os.path.join(PROJECT_ROOT, "evaluate.py")),
        ("agg eval", os.path.join(PROJECT_ROOT, "evaluate_agg.py")),
    ]

    missing = []
    for label, path in required:
        ok = os.path.exists(path)
        print(f"  [{'OK' if ok else 'MISSING'}] {label}: {os.path.relpath(path, PROJECT_ROOT)}")
        if not ok:
            missing.append(path)

    for version in ("v3_filtered", "v5_lite"):
        ok = ensure_version_assets(version)
        print(f"  [{'OK' if ok else 'MISSING'}] data assets: {version}")
        if not ok:
            missing.append(version)

    if missing:
        print("\n[Preflight] Blocked. Missing required files or assets.")
        return False

    print("[Preflight] Passed.")
    return True


def backup_model_family(exp):
    model_dir = get_model_dir(exp["version"])
    mapping = {
        "best_enhanced_model.pth": f"best_{exp['id']}.pth",
        "best_loss_enhanced_model.pth": f"best_loss_{exp['id']}.pth",
        "best_f1_enhanced_model.pth": f"best_f1_{exp['id']}.pth",
        "best_wmape_enhanced_model.pth": f"best_wmape_{exp['id']}.pth",
        "last_enhanced_model.pth": f"last_{exp['id']}.pth",
    }
    for src_name, dst_name in mapping.items():
        src = os.path.join(model_dir, src_name)
        if not os.path.exists(src):
            print(f"  [FAIL] Missing checkpoint variant: {os.path.relpath(src, PROJECT_ROOT)}")
            return False
        dst = os.path.join(model_dir, dst_name)
        shutil.copy(src, dst)
    print(f"  [OK] checkpoint family backed up for {exp['id']}")
    return True


def run_training(exp, env):
    proc = subprocess.Popen(
        [sys.executable, os.path.join("src", "train", "run_training_v2.py")],
        cwd=PROJECT_ROOT,
        env=env,
    )
    proc.wait()
    return proc.returncode, get_actual_epochs(exp["id"])


def run_eval(exp, env):
    eval_out = os.path.join(REPORTS_DIR, f"eval_{exp['id']}.txt")
    with open(eval_out, "w", encoding="utf-8") as f:
        result = subprocess.run(
            [sys.executable, "evaluate.py"],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        print(f"  [FAIL] evaluate.py -> {os.path.relpath(eval_out, PROJECT_ROOT)}")
        return False

    agg_out = os.path.join(REPORTS_DIR, f"agg_{exp['id']}.txt")
    with open(agg_out, "w", encoding="utf-8") as f:
        result = subprocess.run(
            [sys.executable, "evaluate_agg.py", "--exp", exp["id"]],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        print(f"  [FAIL] evaluate_agg.py -> {os.path.relpath(agg_out, PROJECT_ROOT)}")
        return False

    print(f"  [OK] eval -> {os.path.relpath(eval_out, PROJECT_ROOT)}")
    print(f"  [OK] agg  -> {os.path.relpath(agg_out, PROJECT_ROOT)}")
    return True


def build_env(exp):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    env["EXP_ID"] = exp["id"]
    env["EXP_VERSION"] = exp["version"]
    env["EXP_MODEL_TYPE"] = exp["model_type"]
    env["EXP_HIDDEN"] = exp["hidden"]
    env["EXP_LAYERS"] = exp["layers"]
    env["EXP_DROPOUT"] = exp["dropout"]
    env["EXP_BATCH"] = exp["batch"]
    env["EXP_LR"] = exp["lr"]
    env["EXP_EPOCHS"] = GLOBAL_EPOCHS
    env["EXP_PATIENCE"] = GLOBAL_PATIENCE
    env["EXP_WORKERS"] = GLOBAL_WORKERS
    env["EXP_EVAL_WORKERS"] = GLOBAL_WORKERS
    env["EXP_MONITOR_METRIC"] = GLOBAL_MONITOR
    env["EXP_MODEL_FILE"] = "best_enhanced_model.pth"
    env["EXP_SEED"] = exp["seed"]
    env["EXP_DETERMINISTIC"] = "1"
    env["EXP_POS_WEIGHT"] = DEFAULT_LOSS["pos_w"]
    env["EXP_FP_PENALTY"] = DEFAULT_LOSS["fp"]
    env["EXP_GAMMA"] = DEFAULT_LOSS["gamma"]
    env["EXP_REG_WEIGHT"] = DEFAULT_LOSS["reg_w"]
    env["EXP_SOFT_F1"] = DEFAULT_LOSS["sf1"]
    env["EXP_HUBER"] = DEFAULT_LOSS["hub"]
    env["EXP_LABEL_SMOOTH"] = exp.get("label_smooth", "0.0")
    env["EXP_WEIGHT_DECAY"] = exp.get("weight_decay", "0.0")
    return env


def print_eta(completed_minutes, remaining):
    if not completed_minutes:
        return
    avg = sum(completed_minutes) / len(completed_minutes)
    eta_min = avg * remaining
    finish_at = datetime.now() + timedelta(minutes=eta_min)
    print(
        f"  [ETA] avg={avg:.1f} min/exp | remaining={remaining} | "
        f"finish~{finish_at.strftime('%Y-%m-%d %H:%M')}"
    )


def main():
    print("=" * 76)
    print("Phase 5.2 fair-universe comparison")
    print(
        f"epochs={GLOBAL_EPOCHS} | patience={GLOBAL_PATIENCE} | "
        f"workers={GLOBAL_WORKERS} | monitor={GLOBAL_MONITOR} | seeds={GLOBAL_SEEDS}"
    )
    print("Goal: compare V3-filtered vs V5-lite on the same SKU universe with repeatable seeds.")
    print("=" * 76)

    if not preflight():
        sys.exit(1)

    init_timing_log()
    completed_minutes = []
    total = len(EXPERIMENTS)
    pipeline_start = time.time()

    for idx, exp in enumerate(EXPERIMENTS, start=1):
        if is_done(exp):
            print(f"\n[SKIP] [{idx}/{total}] {exp['id']} already has primary backup.")
            continue

        print(f"\n{'=' * 76}")
        print(f"[RUN] [{idx}/{total}] {exp['id']}")
        print(
            f"version={exp['version']} | model={exp['model_type']} | "
            f"layers={exp['layers']} | hidden={exp['hidden']} | dropout={exp['dropout']} | "
            f"batch={exp['batch']} | lr={exp['lr']} | seed={exp['seed']}"
        )
        print(f"note: {exp['desc']}")
        print(f"start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        env = build_env(exp)
        t_start = time.time()
        returncode, actual_epochs = run_training(exp, env)
        t_train_end = time.time()

        if returncode != 0:
            elapsed = append_timing(exp, t_start, t_train_end, actual_epochs, "train_error")
            print(f"[STOP] training failed: {exp['id']} | {elapsed:.1f} min | epochs={actual_epochs}")
            sys.exit(1)

        print(
            f"  [OK] training finished | elapsed={(t_train_end - t_start) / 60:.1f} min | "
            f"epochs={actual_epochs}"
        )

        if not backup_model_family(exp):
            append_timing(exp, t_start, time.time(), actual_epochs, "backup_error")
            sys.exit(1)

        if not run_eval(exp, env):
            append_timing(exp, t_start, time.time(), actual_epochs, "eval_error")
            sys.exit(1)

        t_end = time.time()
        elapsed = append_timing(exp, t_start, t_end, actual_epochs, "success")
        completed_minutes.append(elapsed)
        print(f"  [OK] experiment closed | total={elapsed:.1f} min")
        print_eta(completed_minutes, total - idx)

    total_hours = (time.time() - pipeline_start) / 3600
    print(f"\nDone. Phase 5.2 finished in {total_hours:.2f} hours.")
    print(f"Reports: {os.path.relpath(REPORTS_DIR, PROJECT_ROOT)}")
    print(f"Timing:  {os.path.relpath(TIMING_LOG, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
