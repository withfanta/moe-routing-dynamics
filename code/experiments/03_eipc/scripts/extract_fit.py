"""EIPC-P0 GPU worker A: extract all 1024 FIT samples.

Pin with CUDA_VISIBLE_DEVICES to one free Tesla V100 before launching.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from _extract_worker import run_extraction

if __name__ == "__main__":
    raise SystemExit(run_extraction("fit"))
