"""DREV-P0 worker A: horizons Δ = 1 and Δ = 2 (target Layers 5 and 6).

Pin with CUDA_VISIBLE_DEVICES to one free Tesla V100 before launching.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from _horizon_worker import run_worker

if __name__ == "__main__":
    raise SystemExit(run_worker("gpu_a", (1, 2)))
