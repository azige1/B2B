import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(PROJECT_ROOT, "src", "analysis"))

from summarize_phase6e_tree_validation_pack_results import main


if __name__ == "__main__":
    main()
