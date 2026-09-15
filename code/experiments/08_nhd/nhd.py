"""NHD-P0 shared constants, source verification, and frozen model definitions.

Everything analyze.py and test_nhd.py must agree on lives here, so the frozen protocol has
exactly one implementation. Scope is fixed by protocol.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Dict, List, Sequence, Tuple

import numpy as np

EXPERIMENT = "NHD-P0"
TITLE = "Nonlinear History Decoding Pilot"

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

# --------------------------------------------------------------- frozen source artifacts
# RMC-P0 is CLOSED. It is read, never written. Hashes are pinned at this project's first
# commit, so a later change to that tree makes this project fail loudly instead of silently
# analysing different data.

SOURCE_DIR = "/home/h-li/work/routing_memory_crossmodel"
SOURCE_EXPERIMENT = "RMC-P0"
SOURCE_COMMIT = "cc60edd"

SOURCE_HASHES = {
    "artifacts/merged.npz":
        "7dafe8048772937c239bf9ab61577d9cce648847af4a6227e4ca5d86a9758726",
    "artifacts/data_manifest.json":
        "371601f7ef6c491d571b1528fb15ef5dcd8cd361844786aee9fc8c4aef8afe02",
    "artifacts/model_provenance.json":
        "e26d8ec79271f1312baad4c7993b6bd153d69f5426abb104221cae9e25585335",
    "artifacts/results.json":
        "595ee0b98ed70bd5a9758e8ede0042ac1ab7ccaef3192f8c44077584795c0851",
}

NOT_AVAILABLE = "REQUIRED_ARTIFACT_NOT_AVAILABLE"

# ---------------------------------------------------------------------- architecture facts

N_LAYERS = 24
NUM_EXPERTS = 8
TOP_K = 2

N_FIT = 512
N_TEST = 256
N_TOTAL = N_FIT + N_TEST

TARGET_LAYERS = (12, 20)

# ------------------------------------------------------------------------------ variables

RECENT_DIM = NUM_EXPERTS            # R = S_{m-1}, 8
HISTORY_OFFSETS = (4, 3, 2)         # H = [S_{m-4}, S_{m-3}, S_{m-2}]
HISTORY_DIM = len(HISTORY_OFFSETS) * NUM_EXPERTS      # 24
FULL_DIM = HISTORY_DIM + RECENT_DIM                   # 32

ALPHA = 1.0

# ------------------------------------------------------------------------------- training

HIDDEN = 32
FIT_SPLIT_SEED = 20260921
N_TRAIN = 384
N_VAL = 128
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 30
INIT_SEEDS = (42, 123, 2026)

CROSSFIT_SEED = 20260922
N_FOLD = 256
PERMUTE_SEED = 271828

NEAR_ZERO = 0.02                    # frozen; never a significance threshold

NONLINEAR_ACCESSIBILITY = "NONLINEAR-ACCESSIBILITY-COMPATIBLE"
RESIDUAL_HISTORY_VALUE = "RESIDUAL-HISTORY-VALUE"
JOINT_ONLY = "JOINT-ONLY"
NO_CLEAR_PATTERN = "NO-CLEAR-PATTERN"


def recent_layer(target: int) -> int:
    """The immediately preceding layer, m-1."""
    return target - 1


def history_layers(target: int) -> Tuple[int, ...]:
    """The three older layers, m-4 .. m-2, excluding m-1 and m itself."""
    layers = tuple(target - off for off in HISTORY_OFFSETS)
    if min(layers) < 1:
        raise ValueError(f"target {target} cannot support history {layers}")
    if target - 1 in layers or target in layers:
        raise AssertionError("history must exclude the recent layer and the target")
    return layers


def matched_width(hidden: int = HIDDEN, in_dim: int = FULL_DIM,
                  out_dim: int = NUM_EXPERTS, recent_dim: int = RECENT_DIM) -> Tuple[int, int, int]:
    """Width W whose parameter count is closest to MLP_HISTORY's, chosen mechanically.

    MLP_HISTORY is Linear(in_dim, hidden) -> GELU -> Linear(hidden, out_dim) with biases.
    MLP_RECENT_MATCHED is Linear(recent_dim, W) -> GELU -> Linear(W, out_dim). Returns
    (W, matched_params, history_params). No sweep and no dependence on any result.
    """
    history_params = in_dim * hidden + hidden + hidden * out_dim + out_dim
    best_w, best_params, best_diff = None, None, None
    for w in range(1, 4097):
        p = recent_dim * w + w + w * out_dim + out_dim
        diff = abs(p - history_params)
        if best_diff is None or diff < best_diff:
            best_w, best_params, best_diff = w, p, diff
    return best_w, best_params, history_params


MATCHED_WIDTH = matched_width()[0]


# -------------------------------------------------------------------------- verification


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_repo_is_clean() -> Tuple[bool, str]:
    """Check A: the closed source repository has no uncommitted change."""
    try:
        out = subprocess.run(["git", "-C", SOURCE_DIR, "status", "--porcelain"],
                             capture_output=True, text=True, check=True).stdout
        head = subprocess.run(["git", "-C", SOURCE_DIR, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        return (out.strip() == ""), head
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"cannot inspect source repository: {e}")


def verify_source() -> Dict:
    """Checks A and B, plus the presence of every required array.

    Stops with REQUIRED_ARTIFACT_NOT_AVAILABLE if the frozen artifacts lack the raw
    per-layer Top-2 expert IDs, the router logits, or the FIT/TEST membership.
    """
    clean, head = source_repo_is_clean()
    if not clean:
        raise SystemExit(f"{SOURCE_EXPERIMENT} has uncommitted changes; it must stay frozen")

    observed = {}
    for rel, expected in SOURCE_HASHES.items():
        path = os.path.join(SOURCE_DIR, rel)
        if not os.path.exists(path):
            raise SystemExit(f"{NOT_AVAILABLE}: {rel} missing from {SOURCE_EXPERIMENT}")
        got = sha256_file(path)
        if got != expected:
            raise SystemExit(f"{SOURCE_EXPERIMENT} {rel} hash {got} != pinned {expected}")
        observed[rel] = got

    with np.load(os.path.join(SOURCE_DIR, "artifacts/merged.npz")) as d:
        for key in ("ids", "logits", "is_test"):
            if key not in d.files:
                raise SystemExit(f"{NOT_AVAILABLE}: merged.npz has no '{key}'")
        ids, logits, is_test = d["ids"], d["logits"], d["is_test"].astype(bool)
        if ids.shape != (N_TOTAL, N_LAYERS, TOP_K):
            raise SystemExit(f"{NOT_AVAILABLE}: ids shape {ids.shape}")
        if logits.shape != (N_TOTAL, N_LAYERS, NUM_EXPERTS):
            raise SystemExit(f"{NOT_AVAILABLE}: logits shape {logits.shape}")
        if int((~is_test).sum()) != N_FIT or int(is_test.sum()) != N_TEST:
            raise SystemExit(f"{NOT_AVAILABLE}: split is not {N_FIT}/{N_TEST}")

    return {"source_experiment": SOURCE_EXPERIMENT, "source_dir": SOURCE_DIR,
            "source_head": head, "source_clean": True, "hashes": observed,
            "arrays_present": ["ids", "logits", "is_test"]}


def load_source() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read-only load of the frozen RMC-P0 arrays, after verification."""
    verify_source()
    with np.load(os.path.join(SOURCE_DIR, "artifacts/merged.npz")) as d:
        return (d["ids"].astype(np.int64), d["logits"].astype(np.float64),
                d["is_test"].astype(bool))
