"""RMC-P0 shared constants, data manifest, and router identification.

Everything that both extract.py and analyze.py must agree on lives here, so the frozen
protocol has exactly one implementation. Scope is fixed by protocol.md.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

EXPERIMENT = "RMC-P0"
TITLE = "Routing Memory Cross-Model Pilot"

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

# ------------------------------------------------------------------ pinned provenance

MODEL_ID = "jetmoe/jetmoe-8b"
MODEL_REVISION = "d8fd02ccf7911aa8148a63c7984ffd2e465b0352"   # immutable, resolved once
IMPLEMENTATION = "transformers.models.jetmoe (official in-tree)"

DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_SPLIT = "train"

# ------------------------------------------------------------------ architecture facts
# All asserted against the loaded config at extraction time; never assumed silently.

N_LAYERS = 24
NUM_EXPERTS = 8
TOP_K = 2
HIDDEN_SIZE = 2048

# ------------------------------------------------------------------------------- data

CONTEXT_LEN = 128
BLOCK_LEN = 129            # 0..127 context, 128 continuation
EXPERIMENTAL_POS = 127     # only this token's routing is analyzed

N_FIT = 512
N_TEST = 256
N_TOTAL = N_FIT + N_TEST

SAMPLE_SEED = 20260920
PCA_SEED = 20260920
BOOTSTRAP_SEED = 314159
N_BOOTSTRAP = 10000

N_SHARDS = 4
SHARD_SIZE = N_TOTAL // N_SHARDS          # 192 = 128 FIT + 64 TEST
SHARD_FIT = N_FIT // N_SHARDS             # 128
SHARD_TEST = N_TEST // N_SHARDS           # 64

# Enough of the stream to draw 768 disjoint blocks from without touching the tail.
N_BLOCKS_POOL = 20000
MICROBATCH = 8

# --------------------------------------------------------------------------- analysis

TARGET_LAYERS = (12, 20)                  # human-readable block numbers, fixed
K_VALUES = (1, 2, 4, 8)
PRIMARY_K = 4                             # the k=1 vs k=4 comparison
RECENT_DIM = NUM_EXPERTS                  # 8
PCA_COMPONENTS = NUM_EXPERTS              # 8
FEATURE_DIM = RECENT_DIM + PCA_COMPONENTS # 16
ALPHA = 1.0

DELTA4_THRESHOLD = 0.02                   # frozen; never modified after results

REPLICATED = "REPLICATED"
NOT_REPLICATED = "NOT_REPLICATED"
ROUTER_BLOCKER = "ROUTER_IDENTIFICATION_BLOCKER"
MEMORY_BLOCKER = "TECHNICAL_MEMORY_BLOCKER"


def history_layers(target: int, k: int) -> Tuple[int, ...]:
    """Human-readable layers used as history for `target` at depth `k`.

    k=1 -> (m-1,); k=4 -> (m-4, m-3, m-2, m-1). Always contiguous and ending at m-1.
    """
    if k not in K_VALUES:
        raise ValueError(f"k must be one of {K_VALUES}, got {k}")
    if target - k < 1:
        raise ValueError(f"target {target} cannot support k={k} history")
    return tuple(range(target - k, target))


def older_layers(target: int, k: int) -> Tuple[int, ...]:
    """History strictly before S_{m-1}: empty for k=1."""
    return tuple(l for l in history_layers(target, k) if l != target - 1)


# ------------------------------------------------------------------------ manifest


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(blocks: Sequence[Sequence[int]]) -> str:
    """Content hash of the actual token ids, so a manifest cannot silently drift."""
    h = hashlib.sha256()
    for b in blocks:
        h.update(np.asarray(b, dtype=np.int64).tobytes())
    return h.hexdigest()


@dataclass
class Manifest:
    n_fit: int
    n_test: int
    fit_index: List[int]        # indices into the frozen block pool
    test_index: List[int]
    fit_fingerprint: str
    test_fingerprint: str
    all_fingerprint: str
    tokenizer_name: str
    n_pool_blocks: int
    block_len: int
    experimental_pos: int
    sample_seed: int

    def to_json(self) -> Dict:
        return asdict(self)


def shard_of(role_position: int) -> int:
    """Deterministic shard assignment: FIT and TEST are each split evenly in order."""
    return role_position % N_SHARDS
