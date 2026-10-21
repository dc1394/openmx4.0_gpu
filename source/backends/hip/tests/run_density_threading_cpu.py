#!/usr/bin/env python3
"""The HIP and CUDA host grid paths use the same C++ implementation and tests."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[4]
runpy.run_path(str(ROOT / "tests/run_density_threading_cpu.py"), run_name="__main__")
