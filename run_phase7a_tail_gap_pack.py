import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "analysis"))

from summarize_phase7a_tail_gap_pack_results import main


if __name__ == "__main__":
    main()
