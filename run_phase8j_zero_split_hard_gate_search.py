import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    script = os.path.join(PROJECT_ROOT, "src", "analysis", "generate_phase8j_zero_split_hard_gate_search.py")
    rc = subprocess.run([sys.executable, script], cwd=PROJECT_ROOT, env=env).returncode
    if rc != 0:
        raise SystemExit(rc)
    print("[DONE] phase8 zero-split hard-gate search generated")


if __name__ == "__main__":
    main()
