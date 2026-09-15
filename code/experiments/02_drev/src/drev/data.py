"""DREV-P0 data construction on previously untouched WikiText blocks.

Token-stream and block construction are copied from the validated REDV
implementation (`src/redv/data.py`, REDV commit 6d1c57f): exact OLMoE tokenizer,
EOS inserted between WikiText records, non-overlapping 129-token blocks, tokens
0..127 the context and token 128 the next-token target.

New here: every block ever used by REDV-V1 or REDV-V2 is excluded before
sampling. The REDV repository is read-only and only ever loaded.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import (
    BLOCK_LEN,
    CONTEXT_LEN,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    N_FIT,
    N_TEST,
    REDV_ROOT,
    SAMPLE_SEED,
)


@dataclass(frozen=True)
class Block:
    """One 129-token block within a split's token stream."""

    block_id: int
    stream_start: int
    tokens: tuple


@dataclass(frozen=True)
class DrevContext:
    """One frozen DREV context, tagged with provenance and role."""

    role: str  # "fit" or "test"
    source_split: str  # "validation" or "test"
    block_id: int
    stream_start: int
    tokens: tuple

    @property
    def context(self) -> tuple:
        return self.tokens[:CONTEXT_LEN]

    @property
    def target(self) -> int:
        return self.tokens[CONTEXT_LEN]


# ------------------------------------------------------- stream construction


def build_token_stream(texts: Sequence[str], tokenizer, eos_token_id: int) -> List[int]:
    """Concatenate a split's records into one stream, EOS between records."""
    stream: List[int] = []
    for text in texts:
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        stream.extend(ids)
        stream.append(eos_token_id)
    return stream


def make_blocks(stream: Sequence[int]) -> List[Block]:
    """Cut a stream into non-overlapping 129-token blocks. No sliding windows."""
    n = len(stream) // BLOCK_LEN
    return [
        Block(block_id=i, stream_start=i * BLOCK_LEN,
              tokens=tuple(stream[i * BLOCK_LEN : (i + 1) * BLOCK_LEN]))
        for i in range(n)
    ]


def load_split_texts(split: str, dataset_dir: str | None = None) -> List[str]:
    """Load one WikiText-103-raw split at the pinned revision. Train is refused."""
    if split not in ("validation", "test"):
        raise ValueError(f"DREV-P0 uses validation and test only, got {split!r}")

    if dataset_dir is not None:
        import glob
        import pyarrow.parquet as pq

        pattern = f"{dataset_dir}/{DATASET_CONFIG}/{split}-*.parquet"
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no parquet for {split} at {pattern}")
        texts: List[str] = []
        for f in files:
            texts.extend(pq.read_table(f)["text"].to_pylist())
        return texts

    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=split, revision=DATASET_REVISION)
    return list(ds["text"])


# ------------------------------------------------------- prior-use exclusion


def load_prior_used_keys(redv_root: str = REDV_ROOT) -> Tuple[set, Dict[str, int]]:
    """Every (split, block_id) ever used by REDV-V1 or REDV-V2. Read-only."""
    v1_path = os.path.join(redv_root, "artifacts", "REDV_V1", "data_manifest.json")
    v2_path = os.path.join(redv_root, "artifacts", "REDV_V2", "data_manifest.json")
    with open(v1_path) as fh:
        m1 = json.load(fh)
    with open(v2_path) as fh:
        m2 = json.load(fh)

    v1 = {("validation", b) for b in m1["validation"]["block_ids"]}
    v1 |= {("test", b) for b in m1["test"]["block_ids"]}

    v2 = set(zip(m2["fit"]["source_splits"], m2["fit"]["block_ids"]))
    v2 |= set(zip(m2["test"]["source_splits"], m2["test"]["block_ids"]))

    counts = {"redv_v1": len(v1), "redv_v2": len(v2), "union": len(v1 | v2)}
    return v1 | v2, counts


def untouched_pool(
    blocks_by_split: Dict[str, Sequence[Block]], used: set
) -> List[Tuple[str, Block]]:
    """Blocks never used by REDV, validation first then test, ascending id."""
    pool: List[Tuple[str, Block]] = []
    for split in ("validation", "test"):
        for b in blocks_by_split[split]:
            if (split, b.block_id) not in used:
                pool.append((split, b))
    return pool


def draw_sample(
    pool: Sequence[Tuple[str, Block]], seed: int = SAMPLE_SEED
) -> Tuple[List[DrevContext], List[DrevContext]]:
    """Draw 512 fit + 512 test contexts without replacement."""
    need = N_FIT + N_TEST
    if len(pool) < need:
        raise RuntimeError(
            f"TECHNICAL_DATA_BLOCKER: only {len(pool)} untouched blocks, need {need}"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=need, replace=False)

    def make(rows, role):
        out = []
        for i in rows:
            split, b = pool[int(i)]
            out.append(DrevContext(role=role, source_split=split, block_id=b.block_id,
                                   stream_start=b.stream_start, tokens=b.tokens))
        return out

    return make(idx[:N_FIT], "fit"), make(idx[N_FIT:], "test")


def fingerprint(contexts: Sequence[DrevContext]) -> str:
    h = hashlib.sha256()
    for c in contexts:
        h.update(c.source_split.encode())
        h.update(np.asarray([c.block_id, c.stream_start], dtype=np.int64).tobytes())
        h.update(np.asarray(c.tokens, dtype=np.int64).tobytes())
    return h.hexdigest()


def manifest_payload(
    fit: Sequence[DrevContext],
    test: Sequence[DrevContext],
    pool_size: int,
    prior_counts: Dict[str, int],
    extra: Dict | None = None,
) -> Dict:
    payload: Dict[str, object] = {
        "experiment": "DREV-P0",
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "sample_seed": SAMPLE_SEED,
        "context_len": CONTEXT_LEN,
        "block_len": BLOCK_LEN,
        "overlapping_windows": False,
        "train_split_used": False,
        "prior_blocks_excluded": prior_counts,
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


def write_manifest(path: str, payload: Dict) -> str:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_manifest_contexts(path: str) -> Tuple[List[DrevContext], List[DrevContext]]:
    """Rebuild contexts from a frozen manifest without re-tokenizing.

    Tokens are not stored in the manifest, so callers that need token ids should
    use ``rebuild_from_manifest``. This returns provenance-only contexts.
    """
    with open(path) as fh:
        m = json.load(fh)
    out = {}
    for role in ("fit", "test"):
        d = m[role]
        out[role] = [
            DrevContext(role=role, source_split=s, block_id=b, stream_start=st, tokens=())
            for s, b, st in zip(d["source_splits"], d["block_ids"], d["stream_starts"])
        ]
    return out["fit"], out["test"]


def rebuild_from_manifest(
    path: str, tokenizer, eos_token_id: int, dataset_dir: str | None = None
) -> Tuple[List[DrevContext], List[DrevContext]]:
    """Reconstruct the exact frozen contexts, tokens included."""
    with open(path) as fh:
        m = json.load(fh)

    blocks_by_split = {}
    for split in ("validation", "test"):
        texts = load_split_texts(split, dataset_dir=dataset_dir)
        stream = build_token_stream(texts, tokenizer, eos_token_id)
        blocks_by_split[split] = {b.block_id: b for b in make_blocks(stream)}

    out = {}
    for role in ("fit", "test"):
        d = m[role]
        ctxs = []
        for split, bid, start in zip(d["source_splits"], d["block_ids"], d["stream_starts"]):
            b = blocks_by_split[split][bid]
            if b.stream_start != start:
                raise AssertionError(f"stream offset mismatch for {split}:{bid}")
            ctxs.append(DrevContext(role=role, source_split=split, block_id=bid,
                                    stream_start=start, tokens=b.tokens))
        if fingerprint(ctxs) != d["fingerprint"]:
            raise AssertionError(f"{role} fingerprint mismatch on rebuild")
        out[role] = ctxs
    return out["fit"], out["test"]
