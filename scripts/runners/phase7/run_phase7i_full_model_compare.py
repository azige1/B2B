import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC_ANALYSIS = os.path.join(ROOT, "src", "analysis")
if SRC_ANALYSIS not in sys.path:
    sys.path.insert(0, SRC_ANALYSIS)

from generate_phase7i_full_model_compare import main


if __name__ == "__main__":
    main()

