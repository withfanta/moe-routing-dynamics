"""REDV-V2 untouched-sample construction and the one fixed shuffled control.

Two responsibilities, both frozen by protocols/REDV_V2_PREREGISTRATION.md:

1. Draw 1024 fitting and 1024 final-test contexts from WikiText blocks that
   REDV-V1 never observed. Every REDV-V1 block id is excluded; the two V2 roles
   are disjoint.
2. Build the single deterministic derangement used for the matched control.

REDV-V1 artifacts are read-only here: the V1 manifest is only ever loaded.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import CONTEXT_LEN, TRANSITIONS

V2_SEED = 20260915
SHUFFLE_SEED = 314159
N_FIT = 1024
N_TEST = 1024


@dataclass(frozen=True)
class V2Context:
    """One frozen V2 context, tagged with provenance and role."""

    role: str  # "fit" or "test"
    source_split: str  # "validation" or "test"
    block_id: int  # original block index within its source split
    stream_start: int  # token-stream offset of the block
    tokens: tuple  # length 129

    @property
    def context(self) -> tuple:
        return self.tokens[:CONTEXT_LEN]

    @property
    def target(self) -> int:
        return self.tokens[CONTEXT_LEN]


def load_v1_block_ids(v1_manifest_path: str) -> Dict[str, set]:
    """Read-only load of the REDV-V1 manifest. These blocks are forbidden in V2."""
    with open(v1_manifest_path) as fh:
        m = json.load(fh)
    return {split: set(m[split]["block_ids"]) for split in ("validation", "test")}


def untouched_pool(
    blocks_by_split: Dict[str, Sequence], v1_ids: Dict[str, set]
) -> List[Tuple[str, object]]:
    """Candidate pool: every block whose id REDV-V1 did not use.

    Order is deterministic: validation remainder in ascending block id, then test
    remainder in ascending block id.
    """
    pool: List[Tuple[str, object]] = []
    for split in ("validation", "test"):
        used = v1_ids[split]
        for b in blocks_by_split[split]:
            if b.block_id not in used:
                pool.append((split, b))
    return pool


def draw_v2_sample(
    pool: Sequence[Tuple[str, object]], seed: int = V2_SEED
) -> Tuple[List[V2Context], List[V2Context]]:
    """Draw 2048 distinct candidates; first 1024 -> fit, last 1024 -> test."""
    need = N_FIT + N_TEST
    if len(pool) < need:
        raise RuntimeError(
            f"insufficient untouched blocks: have {len(pool)}, need {need}. "
            "Do not reduce sample size; report the exact count."
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=need, replace=False)

    def make(rows, role):
        out = []
        for i in rows:
            split, b = pool[int(i)]
            out.append(
                V2Context(
                    role=role,
                    source_split=split,
                    block_id=b.block_id,
                    stream_start=b.stream_start,
                    tokens=b.tokens,
                )
            )
        return out

    return make(idx[:N_FIT], "fit"), make(idx[N_FIT:], "test")


def context_key(c: V2Context) -> Tuple[str, int]:
    """Identity of a context for overlap checks: (source split, block id)."""
    return (c.source_split, c.block_id)


def fingerprint(contexts: Sequence[V2Context]) -> str:
    h = hashlib.sha256()
    for c in contexts:
        h.update(c.source_split.encode())
        h.update(np.asarray([c.block_id, c.stream_start], dtype=np.int64).tobytes())
        h.update(np.asarray(c.tokens, dtype=np.int64).tobytes())
    return h.hexdigest()


# --------------------------------------------------------------- derangement


def derange(n: int, rng: np.random.Generator) -> np.ndarray:
    """A permutation of 0..n-1 with no fixed point.

    Draw ``rng.permutation(n)``, then scan i ascending and, whenever ``p[i] == i``,
    swap ``p[i]`` with ``p[(i + 1) % n]``. One forward pass suffices: the swap
    moves the fixed point to position i+1 and brings in a value != i, and the
    final wrap-around at i = n-1 pulls from position 0, which is already settled.
    The seed is never changed to obtain a derangement.
    """
    if n < 2:
        raise ValueError("derangement requires n >= 2")
    p = rng.permutation(n)
    for i in range(n):
        if p[i] == i:
            j = (i + 1) % n
            p[i], p[j] = p[j], p[i]
    if np.any(p == np.arange(n)):  # pragma: no cover - guarded by construction
        raise AssertionError("derangement failed")
    return p


def build_control_permutations(n_fit: int, n_test: int) -> Dict[Tuple[int, int], Dict[str, np.ndarray]]:
    """The one fixed set of control permutations.

    A single generator seeded with 314159 produces, in fixed order per transition
    (3->4, 7->8, 11->12), first the fit-set derangement then the test-set
    derangement. Six permutations, one seed, no averaging, no selection.
    """
    rng = np.random.default_rng(SHUFFLE_SEED)
    out: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    for pair in TRANSITIONS:
        out[pair] = {
            "fit": derange(n_fit, rng),
            "test": derange(n_test, rng),
        }
    return out


# ----------------------------------------------------------------- manifest


def v2_manifest_payload(
    fit: Sequence[V2Context],
    test: Sequence[V2Context],
    pool_size: int,
    v1_ids: Dict[str, set],
    extra: Dict | None = None,
) -> Dict:
    payload: Dict[str, object] = {
        "experiment": "REDV-V2",
        "dataset_id": "Salesforce/wikitext",
        "dataset_config": "wikitext-103-raw-v1",
        "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        "v2_seed": V2_SEED,
        "shuffle_seed": SHUFFLE_SEED,
        "context_len": CONTEXT_LEN,
        "block_len": 129,
        "overlapping_windows": False,
        "train_split_used": False,
        "v1_blocks_excluded": {k: len(v) for k, v in v1_ids.items()},
        "untouched_pool_size": pool_size,
        "n_required": N_FIT + N_TEST,
    }
    if extra:
        payload.update(extra)
    for role, contexts in (("fit", fit), ("test", test)):
        payload[role] = {
            "n": len(contexts),
            "fingerprint": fingerprint(contexts),
            "source_splits": [c.source_split for c in contexts],
            "block_ids": [c.block_id for c in contexts],
            "stream_starts": [c.stream_start for c in contexts],
            "targets": [c.target for c in contexts],
        }
    return payload


def write_v2_manifest(path: str, payload: Dict) -> str:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()
