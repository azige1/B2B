import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    script = os.path.join(PROJECT_ROOT, "src", "analysis", "generate_phase8c_extended_signal_2026_pack.py")
    rc = subprocess.run([sys.executable, script], cwd=PROJECT_ROOT, env=env).returncode
    if rc != 0:
        raise SystemExit(rc)
    print("[DONE] phase8 extended signal 2026 pack generated")


if __name__ == "__main__":
    main()

