"""DREV-P0 worker B: horizons Δ = 4 and Δ = 8 (target Layers 8 and 12).

Pin with CUDA_VISIBLE_DEVICES to one free Tesla V100 before launching.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from _horizon_worker import run_worker

if __name__ == "__main__":
    raise SystemExit(run_worker("gpu_b", (4, 8)))
