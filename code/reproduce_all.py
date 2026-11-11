"""依次运行全部复现脚本。用法：python code/reproduce_all.py"""
import runpy, sys
from pathlib import Path
here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
for s in ("a3_readout_panel.py", "a3b_ties.py", "a1_f20_rules.py", "a2_same_threshold.py"):
    print("\n" + "=" * 30, s, "=" * 30)
    runpy.run_path(str(here / s), run_name="__main__")
