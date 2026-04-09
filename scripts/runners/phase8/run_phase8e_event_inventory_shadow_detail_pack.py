import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    subprocess.run(
        [
            sys.executable,
            os.path.join(PROJECT_ROOT, "src", "analysis", "generate_phase8d_event_inventory_shadow_detail_pack.py"),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()

