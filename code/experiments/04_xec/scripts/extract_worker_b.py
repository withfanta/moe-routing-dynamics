"""XEC-P0 GPU worker B: VALIDATION + TEST extraction.

VALIDATION gets oracle labels (it drives checkpoint selection). TEST does not: its oracle
is computed only after policy training is frozen, by finalize_xec.py.

Pin with CUDA_VISIBLE_DEVICES to one idle Tesla V100 before launching.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from _extract_common import run_worker

if __name__ == "__main__":
    raise SystemExit(run_worker("gpu_b", ("validation", "test")))
