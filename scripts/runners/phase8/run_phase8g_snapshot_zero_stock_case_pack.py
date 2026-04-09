import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    script = os.path.join(
        PROJECT_ROOT, "src", "analysis", "generate_phase8g_snapshot_zero_stock_case_pack.py"
    )
    rc = subprocess.run([sys.executable, script], cwd=PROJECT_ROOT, env=env).returncode
    if rc != 0:
        raise SystemExit(rc)
    print("[DONE] phase8 snapshot_zero_stock case pack generated")


if __name__ == "__main__":
    main()

