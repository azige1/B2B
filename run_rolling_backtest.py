"""
Rolling backtest runner for fair-universe V3-filtered vs V5-lite.

Goals:
- Use multiple anchor dates instead of a single 2025-12-01 cut.
- Rebuild features per anchor date into isolated directories.
- Compare V3-filtered vs V5-lite on the same SKU universe.
- Keep outputs grouped by anchor for later analysis.

Default budget:
- anchors: 2025-09-01, 2025-10-01, 2025-11-01, 2025-12-01
- seeds: 2026
- models: LSTM + BiLSTM
- monitor: wmape
"""
import csv
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
ROLLING_ROOT = os.path.join(PROJECT_ROOT, "reports", "rolling_backtest")
TIMING_LOG = os.path.join(ROLLING_ROOT, "rolling_backtest_timing.csv")

GLOBAL_EPOCHS = os.environ.get("ROLLING_EPOCHS", "12")
GLOBAL_PATIENCE = os.environ.get("ROLLING_PATIENCE", "4")
GLOBAL_WORKERS = os.environ.get("ROLLING_WORKERS", "2")
GLOBAL_MONITOR = os.environ.get("ROLLING_MONITOR", "wmape").lower()
GLOBAL_SEEDS = [int(x.strip()) for x in os.environ.get("ROLLING_SEEDS", "2026").split(",") if x.strip()]
GLOBAL_ANCHORS = [x.strip() for x in os.environ.get(
    "ROLLING_ANCHORS",
    "2025-09-01,2025-10-01,2025-11-01,2025-12-01"
).split(",") if x.strip()]

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
        "tag": "lstm_v3f",
        "version": "v3_filtered",
        "model_type": "lstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
    },
    {
        "tag": "bilstm_v3f",
        "version": "v3_filtered",
        "model_type": "bilstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
    },
    {
        "tag": "lstm_v5lite",
        "version": "v5_lite",
        "model_type": "lstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
    },
    {
        "tag": "bilstm_v5lite",
        "version": "v5_lite",
        "model_type": "bilstm",
        "hidden": "256",
        "layers": "3",
        "dropout": "0.3",
        "batch": "1024",
        "lr": "0.00005",
    },
]


def anchor_tag(anchor_date):
    return anchor_date.replace("-", "")


def feature_paths(version, anchor_date):
    tag = anchor_tag(anchor_date)
    suffix = f"_bt_{tag}"
    if version == "v3_filtered":
        return {
            "processed_dir": os.path.join(PROJECT_ROOT, "data", f"processed_v3_filtered{suffix}"),
            "artifacts_dir": os.path.join(PROJECT_ROOT, "data", f"artifacts_v3_filtered{suffix}"),
            "meta_path": os.path.join(PROJECT_ROOT, "data", f"artifacts_v3_filtered{suffix}", "meta_v3_filtered.json"),
            "encoder_path": os.path.join(PROJECT_ROOT, "data", f"artifacts_v3_filtered{suffix}", "label_encoders_v3_filtered.pkl"),
        }
    if version == "v5_lite":
        return {
            "processed_dir": os.path.join(PROJECT_ROOT, "data", f"processed_v5_lite{suffix}"),
            "artifacts_dir": os.path.join(PROJECT_ROOT, "data", f"artifacts_v5_lite{suffix}"),
            "meta_path": os.path.join(PROJECT_ROOT, "data", f"artifacts_v5_lite{suffix}", "meta_v5_lite.json"),
            "encoder_path": os.path.join(PROJECT_ROOT, "data", f"artifacts_v5_lite{suffix}", "label_encoders_v5_lite.pkl"),
        }
    raise ValueError(f"unsupported version: {version}")


def feature_builder(version):
    if version == "v3_filtered":
        return os.path.join(PROJECT_ROOT, "src", "features", "build_features_v3_filtered_sku.py")
    if version == "v5_lite":
        return os.path.join(PROJECT_ROOT, "src", "features", "build_features_v5_lite_sku.py")
    raise ValueError(f"unsupported version: {version}")


def build_experiments():
    experiments = []
    for anchor_date in GLOBAL_ANCHORS:
        for seed in GLOBAL_SEEDS:
            for base in BASE_EXPERIMENTS:
                exp = dict(base)
                exp["anchor_date"] = anchor_date
                exp["seed"] = str(seed)
                exp["id"] = f"rb_{anchor_tag(anchor_date)}_{base['tag']}_s{seed}"
                experiments.append(exp)
    return experiments


EXPERIMENTS = build_experiments()


def ensure_dirs():
    os.makedirs(ROLLING_ROOT, exist_ok=True)


def init_timing_log():
    ensure_dirs()
    if os.path.exists(TIMING_LOG):
        return
    with open(TIMING_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "exp_id", "anchor_date", "version", "model_type", "seed", "monitor",
            "start_time", "end_time", "elapsed_min", "actual_epochs",
            "min_per_epoch", "status", "note"
        ])


def append_timing(exp, t_start, t_end, actual_epochs, status, note=""):
    elapsed_min = round((t_end - t_start) / 60, 2)
    min_per_epoch = round(elapsed_min / max(actual_epochs, 1), 2)
    with open(TIMING_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            exp["id"], exp["anchor_date"], exp["version"], exp["model_type"], exp["seed"], GLOBAL_MONITOR,
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


def report_dir(anchor_date):
    path = os.path.join(ROLLING_ROOT, anchor_tag(anchor_date))
    os.makedirs(path, exist_ok=True)
    return path


def model_dir(anchor_date, version):
    path = os.path.join(PROJECT_ROOT, "models_backtest", anchor_tag(anchor_date), version)
    os.makedirs(path, exist_ok=True)
    return path


def is_done(exp):
    return os.path.exists(os.path.join(model_dir(exp["anchor_date"], exp["version"]), f"best_{exp['id']}.pth"))


def ensure_feature_assets(anchor_date):
    tag = f"bt_{anchor_tag(anchor_date)}"
    for version in ("v5_lite", "v3_filtered"):
        info = feature_paths(version, anchor_date)
        if os.path.exists(info["meta_path"]) and os.path.exists(os.path.join(info["processed_dir"], "X_train_dyn.bin")):
            print(f"  [OK] features ready: {version} @ {anchor_date}")
            continue

        print(f"  [BUILD] features: {version} @ {anchor_date}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["FEATURE_SPLIT_DATE"] = anchor_date
        env["FEATURE_OUTPUT_TAG"] = tag
        result = subprocess.run([sys.executable, feature_builder(version)], cwd=PROJECT_ROOT, env=env)
        if result.returncode != 0:
            return False
    return True


def preflight():
    print("\n[Preflight] Checking rolling backtest prerequisites...")
    ensure_dirs()
    required = [
        ("train entry", os.path.join(PROJECT_ROOT, "src", "train", "run_training_v2.py")),
        ("eval entry", os.path.join(PROJECT_ROOT, "evaluate.py")),
        ("agg eval", os.path.join(PROJECT_ROOT, "evaluate_agg.py")),
        ("v5-lite builder", feature_builder("v5_lite")),
        ("v3-filtered builder", feature_builder("v3_filtered")),
    ]
    missing = []
    for label, path in required:
        ok = os.path.exists(path)
        print(f"  [{'OK' if ok else 'MISSING'}] {label}: {os.path.relpath(path, PROJECT_ROOT)}")
        if not ok:
            missing.append(path)
    if missing:
        print("\n[Preflight] Blocked. Missing files.")
        return False
    print("[Preflight] Passed.")
    return True


def backup_model_family(exp):
    exp_model_dir = model_dir(exp["anchor_date"], exp["version"])
    mapping = {
        "best_enhanced_model.pth": f"best_{exp['id']}.pth",
        "best_loss_enhanced_model.pth": f"best_loss_{exp['id']}.pth",
        "best_f1_enhanced_model.pth": f"best_f1_{exp['id']}.pth",
        "best_wmape_enhanced_model.pth": f"best_wmape_{exp['id']}.pth",
        "last_enhanced_model.pth": f"last_{exp['id']}.pth",
    }
    for src_name, dst_name in mapping.items():
        src = os.path.join(exp_model_dir, src_name)
        if not os.path.exists(src):
            print(f"  [FAIL] Missing checkpoint variant: {src}")
            return False
        shutil.copy(src, os.path.join(exp_model_dir, dst_name))
    return True


def build_env(exp):
    info = feature_paths(exp["version"], exp["anchor_date"])
    exp_report_dir = report_dir(exp["anchor_date"])
    exp_model_dir = model_dir(exp["anchor_date"], exp["version"])

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
    env["EXP_REPORT_DIR"] = exp_report_dir
    env["EXP_ARTIFACTS_DIR"] = info["artifacts_dir"]
    env["EXP_PROCESSED_DIR"] = info["processed_dir"]
    env["EXP_META_PATH"] = info["meta_path"]
    env["EXP_ENCODER_PATH"] = info["encoder_path"]
    env["EXP_MODEL_DIR"] = exp_model_dir
    env["EXP_POS_WEIGHT"] = DEFAULT_LOSS["pos_w"]
    env["EXP_FP_PENALTY"] = DEFAULT_LOSS["fp"]
    env["EXP_GAMMA"] = DEFAULT_LOSS["gamma"]
    env["EXP_REG_WEIGHT"] = DEFAULT_LOSS["reg_w"]
    env["EXP_SOFT_F1"] = DEFAULT_LOSS["sf1"]
    env["EXP_HUBER"] = DEFAULT_LOSS["hub"]
    env["EXP_LABEL_SMOOTH"] = exp.get("label_smooth", "0.0")
    env["EXP_WEIGHT_DECAY"] = exp.get("weight_decay", "0.0")
    return env


def run_training(exp, env):
    proc = subprocess.Popen(
        [sys.executable, os.path.join("src", "train", "run_training_v2.py")],
        cwd=PROJECT_ROOT,
        env=env,
    )
    proc.wait()
    return proc.returncode, get_actual_epochs(exp["id"])


def run_eval(exp, env):
    exp_report_dir = report_dir(exp["anchor_date"])
    eval_out = os.path.join(exp_report_dir, f"eval_{exp['id']}.txt")
    with open(eval_out, "w", encoding="utf-8") as f:
        result = subprocess.run(
            [sys.executable, "evaluate.py"],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        print(f"  [FAIL] evaluate.py -> {eval_out}")
        return False

    agg_out = os.path.join(exp_report_dir, f"agg_{exp['id']}.txt")
    with open(agg_out, "w", encoding="utf-8") as f:
        result = subprocess.run(
            [sys.executable, "evaluate_agg.py", "--exp", exp["id"]],
            cwd=PROJECT_ROOT,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        print(f"  [FAIL] evaluate_agg.py -> {agg_out}")
        return False
    return True


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
    print("=" * 80)
    print("Rolling backtest runner")
    print(
        f"anchors={GLOBAL_ANCHORS} | seeds={GLOBAL_SEEDS} | "
        f"epochs={GLOBAL_EPOCHS} | patience={GLOBAL_PATIENCE} | "
        f"workers={GLOBAL_WORKERS} | monitor={GLOBAL_MONITOR}"
    )
    print("=" * 80)

    if not preflight():
        sys.exit(1)

    init_timing_log()
    completed_minutes = []
    total = len(EXPERIMENTS)
    pipeline_start = time.time()
    built_anchors = set()

    for idx, exp in enumerate(EXPERIMENTS, start=1):
        if exp["anchor_date"] not in built_anchors:
            print(f"\n[Anchor] preparing assets for {exp['anchor_date']}")
            if not ensure_feature_assets(exp["anchor_date"]):
                print(f"[STOP] feature build failed for anchor={exp['anchor_date']}")
                sys.exit(1)
            built_anchors.add(exp["anchor_date"])

        if is_done(exp):
            print(f"\n[SKIP] [{idx}/{total}] {exp['id']} already has primary backup.")
            continue

        print(f"\n{'=' * 80}")
        print(f"[RUN] [{idx}/{total}] {exp['id']}")
        print(
            f"anchor={exp['anchor_date']} | version={exp['version']} | model={exp['model_type']} | "
            f"layers={exp['layers']} | hidden={exp['hidden']} | dropout={exp['dropout']} | "
            f"batch={exp['batch']} | lr={exp['lr']} | seed={exp['seed']}"
        )
        print(f"start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        env = build_env(exp)
        t_start = time.time()
        returncode, actual_epochs = run_training(exp, env)
        t_train_end = time.time()

        if returncode != 0:
            elapsed = append_timing(exp, t_start, t_train_end, actual_epochs, "train_error")
            print(f"[STOP] training failed: {exp['id']} | {elapsed:.1f} min | epochs={actual_epochs}")
            sys.exit(1)

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
    print(f"\nDone. Rolling backtest finished in {total_hours:.2f} hours.")
    print(f"Reports: {ROLLING_ROOT}")
    print(f"Timing:  {TIMING_LOG}")


if __name__ == "__main__":
    main()
