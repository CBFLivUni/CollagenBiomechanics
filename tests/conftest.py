"""
Make the pipeline importable as `biomech` regardless of its filename.

Module names cannot start with a digit, so `01_biomechanical_processing.py` cannot be imported directly. 
It is loaded by path with importlib and exposed as `biomech`. 
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
