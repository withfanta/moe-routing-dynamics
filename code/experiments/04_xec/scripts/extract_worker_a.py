"""XEC-P0 GPU worker A: TRAIN extraction + oracle generation.

Pin with CUDA_VISIBLE_DEVICES to one idle Tesla V100 before launching.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from _extract_common import run_worker

if __name__ == "__main__":
    raise SystemExit(run_worker("gpu_a", ("train",)))
