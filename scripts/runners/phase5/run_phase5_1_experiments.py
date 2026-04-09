"""
Phase 5.1 diagnostic overnight runner.

Purpose:
- Isolate whether dense derived V5 features are hurting performance.
- Compare V5-lite against the existing V5/V3 references.
- Keep the run small enough for a single overnight session on RTX 3050 Laptop.

Default budget:
- epochs=12
- patience=3
- workers=2

Expected runtime on current machine:
- ~9 to 12 hours for the full set, depending on early stopping.
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


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_1")
TIMING_LOG = os.path.join(PROJECT_ROOT, "reports", "phase5_1_timing_log.csv")

GLOBAL_EPOCHS = os.environ.get("PHASE51_EPOCHS", "12")
GLOBAL_PATIENCE = os.environ.get("PHASE51_PATIENCE", "3")
GLOBAL_WORKERS = os.environ.get("PHASE51_WORKERS", "2")

DEFAULT_LOSS = {
    "pos_w": "5.85",
    "fp": "0.15",
    "gamma": "1.5",
    "reg_w": "0.5",
    "sf1": "0.5",
    "hub": "1.5",
}

EXPERIMENTS = [
    {
        "id": "p511_lstm_l3_v5_lite",
        "version": "v5_lite",
        "model_type": "lstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
        "desc": "Direct counterpart to e53. Test whether removing dense derived features rescues V5 on LSTM.",
    },
    {
        "id": "p512_bilstm_l3_v5_lite",
        "version": "v5_lite",
        "model_type": "bilstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
        "desc": "Main candidate. Direct counterpart to e54 with only replenish+future features.",
    },
    {
        "id": "p513_lstm_l3_v5_lite_lr1e4",
        "version": "v5_lite",
        "model_type": "lstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.0001",
        "desc": "Check whether V5-lite on LSTM mainly needs a faster learning rate.",
    },
    {
        "id": "p514_bilstm_l3_v5_lite_lr1e4",
        "version": "v5_lite",
        "model_type": "bilstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.0001",
        "desc": "Check whether V5-lite benefits from a faster learning rate.",
    },
    {
        "id": "p515_bilstm_l3_v5_lr1e4",
        "version": "v5",
        "model_type": "bilstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.0001",
        "desc": "Control run. Check whether original V5 mainly failed because lr=5e-5 was too conservative.",
    },
    {
        "id": "p516_lstm_l3_v3_ref",
        "version": "v3",
        "model_type": "lstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
        "desc": "Same-night reference. Rebuild the V3 LSTM baseline under patience=3.",
    },
    {
        "id": "p517_bilstm_l3_v3_ref",
        "version": "v3",
        "model_type": "bilstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
        "desc": "Same-night reference. Rebuild the strongest current baseline under patience=3.",
    },
]


def get_model_dir(version):
    if version == "v3":
        return os.path.join(PROJECT_ROOT, "models_v2")
    if version == "v5":
        return os.path.join(PROJECT_ROOT, "models_v5")
    if version == "v5_lite":
        return os.path.join(PROJECT_ROOT, "models_v5_lite")
    return os.path.join(PROJECT_ROOT, f"models_{version}")


def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(get_model_dir("v3"), exist_ok=True)
    os.makedirs(get_model_dir("v5"), exist_ok=True)
    os.makedirs(get_model_dir("v5_lite"), exist_ok=True)


def init_timing_log():
    os.makedirs(os.path.dirname(TIMING_LOG), exist_ok=True)
    if os.path.exists(TIMING_LOG):
        return
    with open(TIMING_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_id", "version", "model_type", "lr", "dropout",
            "start_time", "end_time", "elapsed_min",
            "actual_epochs", "min_per_epoch", "status", "note"
        ])


def append_timing(exp, t_start, t_end, actual_epochs, status, note=""):
    elapsed_min = round((t_end - t_start) / 60, 2)
    min_per_epoch = round(elapsed_min / max(actual_epochs, 1), 2)
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            exp["id"], exp["version"], exp["model_type"], exp["lr"], exp["dropout"],
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
    backup = os.path.join(get_model_dir(exp["version"]), f"best_{exp['id']}.pth")
    return os.path.exists(backup)


def preflight():
    print("\n[Preflight] Checking Phase 5.1 prerequisites...")
    ensure_dirs()

    required = [
        ("train entry", os.path.join(PROJECT_ROOT, "src", "train", "run_training_v2.py")),
        ("eval entry", os.path.join(PROJECT_ROOT, "evaluate.py")),
        ("agg eval", os.path.join(PROJECT_ROOT, "evaluate_agg.py")),
        ("v5 train", os.path.join(PROJECT_ROOT, "data", "processed_v5", "X_train_dyn.bin")),
        ("v5 val", os.path.join(PROJECT_ROOT, "data", "processed_v5", "X_val_dyn.bin")),
        ("v5 meta", os.path.join(PROJECT_ROOT, "data", "artifacts_v5", "meta_v5.json")),
        ("v5-lite train", os.path.join(PROJECT_ROOT, "data", "processed_v5_lite", "X_train_dyn.bin")),
        ("v5-lite val", os.path.join(PROJECT_ROOT, "data", "processed_v5_lite", "X_val_dyn.bin")),
        ("v5-lite meta", os.path.join(PROJECT_ROOT, "data", "artifacts_v5_lite", "meta_v5_lite.json")),
    ]

    missing = []
    for label, path in required:
        ok = os.path.exists(path)
        status = "OK" if ok else "MISSING"
        print(f"  [{status}] {label}: {os.path.relpath(path, PROJECT_ROOT)}")
        if not ok:
            missing.append(path)

    for label, rel_path in (
        ("V5", os.path.join("data", "artifacts_v5", "meta_v5.json")),
        ("V5-lite", os.path.join("data", "artifacts_v5_lite", "meta_v5_lite.json")),
    ):
        meta_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(
            f"  [META] {label}: lookback={meta.get('lookback')} | "
            f"dyn_feat_dim={meta.get('dyn_feat_dim')} | "
            f"train_cnt={meta.get('train_cnt')} | val_cnt={meta.get('val_cnt')}"
        )

    if missing:
        print("\n[Preflight] Blocked. Missing required files.")
        return False

    print("[Preflight] Passed.")
    return True


def backup_model(exp):
    model_src = os.path.join(get_model_dir(exp["version"]), "best_enhanced_model.pth")
    if not os.path.exists(model_src):
        print(f"  [FAIL] Missing model checkpoint: {os.path.relpath(model_src, PROJECT_ROOT)}")
        return False
    model_dst = os.path.join(get_model_dir(exp["version"]), f"best_{exp['id']}.pth")
    shutil.copy(model_src, model_dst)
    print(f"  [OK] backup -> {os.path.relpath(model_dst, PROJECT_ROOT)}")
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
    print(f"  [OK] eval -> {os.path.relpath(eval_out, PROJECT_ROOT)}")

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
    print("=" * 72)
    print("Phase 5.1 overnight diagnostics")
    print(f"epochs={GLOBAL_EPOCHS} | patience={GLOBAL_PATIENCE} | workers={GLOBAL_WORKERS}")
    print("Goal: test V5-lite against original V5 under a controlled overnight budget.")
    print("=" * 72)

    if not preflight():
        sys.exit(1)

    init_timing_log()
    completed_minutes = []
    total = len(EXPERIMENTS)
    pipeline_start = time.time()

    for idx, exp in enumerate(EXPERIMENTS, start=1):
        if is_done(exp):
            print(f"\n[SKIP] [{idx}/{total}] {exp['id']} already has backup checkpoint.")
            continue

        print(f"\n{'=' * 72}")
        print(f"[RUN] [{idx}/{total}] {exp['id']}")
        print(
            f"version={exp['version']} | model={exp['model_type']} | "
            f"layers={exp['layers']} | hidden={exp['hidden']} | "
            f"dropout={exp['dropout']} | batch={exp['batch']} | lr={exp['lr']}"
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

        if not backup_model(exp):
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
    print(f"\nDone. Phase 5.1 finished in {total_hours:.2f} hours.")
    print(f"Reports: {os.path.relpath(REPORTS_DIR, PROJECT_ROOT)}")
    print(f"Timing:  {os.path.relpath(TIMING_LOG, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

