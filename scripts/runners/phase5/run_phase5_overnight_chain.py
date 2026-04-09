import os
import subprocess
import sys
import time
from datetime import datetime
import csv


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "overnight_chain")
LOG_PATH = os.path.join(REPORTS_DIR, f"overnight_chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
PHASE53_REPORT_DIR = os.path.join(PROJECT_ROOT, "reports", "phase5_3")
PHASE53_TIMING = os.path.join(PROJECT_ROOT, "reports", "phase5_3_timing_log.csv")
PHASE53_EXPECTED = 8

POLL_SECONDS = int(os.environ.get("PHASE_OVERNIGHT_POLL_SECONDS", "60"))
PHASE54_SEEDS = os.environ.get("PHASE54_SEEDS", "2026,2027,2028")
PHASE55_SEEDS = os.environ.get("PHASE55_SEEDS", "2026")
PHASE55_ANCHORS = os.environ.get("PHASE55_ANCHORS", "2025-09-01,2025-10-01,2025-11-01,2025-12-01")


def ensure_dirs():
    os.makedirs(REPORTS_DIR, exist_ok=True)


def log(message):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def running_phase53_pids():
    cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*run_phase5_3_experiments.py*' } | "
        "Select-Object -ExpandProperty ProcessId"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", cmd],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def wait_for_phase53():
    log("Watcher started. Waiting for active run_phase5_3_experiments.py to finish.")
    saw_process = False
    while True:
        pids = running_phase53_pids()
        if pids:
            saw_process = True
            log(f"phase5.3A still running. pids={','.join(pids)}")
            time.sleep(POLL_SECONDS)
            continue
        if saw_process:
            if phase53_completed_successfully():
                log("phase5.3A finished with all expected outputs. Starting overnight continuation.")
                return
            log("phase5.3A process ended but outputs are incomplete. Overnight chain will stop.")
            raise SystemExit(2)
        log("No active phase5.3A process found yet. Waiting.")
        time.sleep(POLL_SECONDS)


def phase53_completed_successfully():
    if not os.path.exists(PHASE53_TIMING):
        return False
    success_rows = {}
    with open(PHASE53_TIMING, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            success_rows[row["exp_id"]] = row.get("status", "")
    if len(success_rows) < PHASE53_EXPECTED:
        log(f"phase5.3A timing rows insufficient: found={len(success_rows)} expected={PHASE53_EXPECTED}")
        return False
    failed = [exp_id for exp_id, status in success_rows.items() if status != "success"]
    if failed:
        log(f"phase5.3A has non-success statuses: {failed}")
        return False

    eval_count = len([
        name for name in os.listdir(PHASE53_REPORT_DIR)
        if name.startswith("eval_context_") and name.endswith(".csv")
    ]) if os.path.exists(PHASE53_REPORT_DIR) else 0
    if eval_count < PHASE53_EXPECTED:
        log(f"phase5.3A eval_context outputs insufficient: found={eval_count} expected={PHASE53_EXPECTED}")
        return False
    return True


def run_step(label, command, extra_env=None):
    log(f"START {label}: {' '.join(command)}")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)

    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        proc = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    if proc.returncode != 0:
        log(f"FAIL {label}: exit={proc.returncode}")
        raise SystemExit(proc.returncode)
    log(f"OK {label}")


def main():
    ensure_dirs()
    log("Overnight chain bootstrap.")
    wait_for_phase53()

    run_step(
        "phase5.3 summary",
        [sys.executable, os.path.join(PROJECT_ROOT, "src", "analysis", "summarize_phase5_3_results.py")],
    )
    run_step(
        "phase5.4 local confirm",
        [sys.executable, os.path.join(PROJECT_ROOT, "run_phase5_4_local_confirm.py")],
        {"PHASE54_SEEDS": PHASE54_SEEDS},
    )
    run_step(
        "phase5.5 local anchors",
        [sys.executable, os.path.join(PROJECT_ROOT, "run_phase5_5_local_anchors.py")],
        {
            "PHASE55_SEEDS": PHASE55_SEEDS,
            "PHASE55_ANCHORS": PHASE55_ANCHORS,
        },
    )
    log("Overnight chain completed.")


if __name__ == "__main__":
    main()

