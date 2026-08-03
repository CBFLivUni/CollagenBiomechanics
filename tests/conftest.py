"""
Make the pipeline importable as `biomech` regardless of its filename.

The script is called `01_biomechanical_processing.py`. A module name cannot
start with a digit, so `import 01_biomechanical_processing` is a SyntaxError.
We load it by path with importlib and expose it under a clean name.

(The cleaner long-term fix is to rename the module to something importable,
e.g. `biomech_processing.py`, and keep `01_...py` as a one-line CLI shim.
Then this shim goes away and tests just `import biomech_processing`.)
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = ROOT / "01_biomechanical_processing.py"

spec = importlib.util.spec_from_file_location("biomech", _SCRIPT)
biomech = importlib.util.module_from_spec(spec)
sys.modules["biomech"] = biomech
spec.loader.exec_module(biomech)
