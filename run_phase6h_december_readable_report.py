import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "analysis"))

from generate_phase6h_december_readable_report import main


if __name__ == "__main__":
    main()
