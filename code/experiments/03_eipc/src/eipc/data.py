"""EIPC-P0 data construction on the WikiText-103-raw TRAIN split.

Train is used exclusively: the prior REDV/DREV projects used validation and test and
explicitly never touched train, so this is fresh data and no prior manifest is
needed as input.

Exact OLMoE tokenizer, EOS inserted between original WikiText records,
non-overlapping 129-token blocks; tokens 0..127 are the context and token 128 is the
next-token continuation token. The experimental token is context position 127.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import (
    BLOCK_LEN,
    CONTEXT_LEN,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    N_FIT,
    N_TEST,
    SAMPLE_SEED,
)


@dataclass(frozen=True)
class Block:
    block_id: int
    stream_start: int
    tokens: tuple


@dataclass(frozen=True)
class EipcContext:
    """One frozen EIPC context, tagged with role."""

    role: str  # "fit" or "test"
    block_id: int
    stream_start: int
    tokens: tuple

    @property
    def context(self) -> tuple:
        return self.tokens[:CONTEXT_LEN]

    @property
    def continuation(self) -> int:
        return self.tokens[CONTEXT_LEN]


def load_train_texts(dataset_dir: str | None = None, max_records: int | None = None) -> List[str]:
    """Load the WikiText-103-raw train split at the pinned revision.

    ``max_records`` truncates the record list, which keeps manifest construction
    cheap: the train split is far larger than the 2048 blocks EIPC-P0 needs. The
    truncation is part of the frozen manifest and is recorded there.
    """
    if dataset_dir is not None:
        import glob
        import pyarrow.parquet as pq

        pattern = f"{dataset_dir}/{DATASET_CONFIG}/{DATASET_SPLIT}-*.parquet"
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no parquet for {DATASET_SPLIT} at {pattern}")
        texts: List[str] = []
        for f in files:
            texts.extend(pq.read_table(f)["text"].to_pylist())
            if max_records is not None and len(texts) >= max_records:
                break
        return texts[:max_records] if max_records is not None else texts

    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT,
                      revision=DATASET_REVISION)
    texts = list(ds["text"])
    return texts[:max_records] if max_records is not None else texts


def build_token_stream(texts: Sequence[str], tokenizer, eos_token_id: int) -> List[int]:
    """Concatenate records into one stream, EOS between records."""
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
    """Cut a stream into non-overlapping 129-token blocks."""
    n = len(stream) // BLOCK_LEN
    return [
        Block(block_id=i, stream_start=i * BLOCK_LEN,
              tokens=tuple(stream[i * BLOCK_LEN : (i + 1) * BLOCK_LEN]))
        for i in range(n)
    ]


def draw_sample(
    blocks: Sequence[Block], seed: int = SAMPLE_SEED
) -> Tuple[List[EipcContext], List[EipcContext]]:
    """Draw 1024 FIT + 1024 TEST blocks without replacement, disjoint by construction."""
    need = N_FIT + N_TEST
    if len(blocks) < need:
        raise RuntimeError(f"TECHNICAL_BLOCKER: only {len(blocks)} blocks, need {need}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(blocks), size=need, replace=False)

    def make(rows, role):
        return [
            EipcContext(role=role, block_id=blocks[int(i)].block_id,
                        stream_start=blocks[int(i)].stream_start,
                        tokens=blocks[int(i)].tokens)
            for i in rows
        ]

    return make(idx[:N_FIT], "fit"), make(idx[N_FIT:], "test")


def fingerprint(contexts: Sequence[EipcContext]) -> str:
    h = hashlib.sha256()
    for c in contexts:
        h.update(np.asarray([c.block_id, c.stream_start], dtype=np.int64).tobytes())
        h.update(np.asarray(c.tokens, dtype=np.int64).tobytes())
    return h.hexdigest()


def manifest_payload(
    fit: Sequence[EipcContext],
    test: Sequence[EipcContext],
    n_blocks_available: int,
    n_records: int,
    n_stream_tokens: int,
    max_records: int | None,
    extra: Dict | None = None,
) -> Dict:
    payload: Dict[str, object] = {
        "experiment": "EIPC-P0",
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "split_used": DATASET_SPLIT,
        "validation_or_test_used": False,
        "sample_seed": SAMPLE_SEED,
        "context_len": CONTEXT_LEN,
        "block_len": BLOCK_LEN,
        "overlapping_windows": False,
        "experimental_token_position": CONTEXT_LEN - 1,
        "n_records_used": n_records,
        "max_records_truncation": max_records,
        "n_stream_tokens": n_stream_tokens,
        "n_blocks_available": n_blocks_available,
        "n_required": N_FIT + N_TEST,
    }
    if extra:
        payload.update(extra)
    for role, contexts in (("fit", fit), ("test", test)):
        payload[role] = {
            "n": len(contexts),
            "fingerprint": fingerprint(contexts),
            "block_ids": [c.block_id for c in contexts],
            "stream_starts": [c.stream_start for c in contexts],
            "continuations": [c.continuation for c in contexts],
        }
    return payload


def write_manifest(path: str, payload: Dict) -> str:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def rebuild_from_manifest(
    path: str, tokenizer, eos_token_id: int, dataset_dir: str | None = None
) -> Tuple[List[EipcContext], List[EipcContext]]:
    """Reconstruct the exact frozen contexts and verify their fingerprints."""
    with open(path) as fh:
        m = json.load(fh)

    texts = load_train_texts(dataset_dir=dataset_dir, max_records=m["max_records_truncation"])
    stream = build_token_stream(texts, tokenizer, eos_token_id)
    by_id = {b.block_id: b for b in make_blocks(stream)}

    out = {}
    for role in ("fit", "test"):
        d = m[role]
        ctxs = []
        for bid, start in zip(d["block_ids"], d["stream_starts"]):
            b = by_id[bid]
            if b.stream_start != start:
                raise AssertionError(f"stream offset mismatch for block {bid}")
            ctxs.append(EipcContext(role=role, block_id=bid, stream_start=start,
                                    tokens=b.tokens))
        if fingerprint(ctxs) != d["fingerprint"]:
            raise AssertionError(f"{role} fingerprint mismatch on rebuild")
        out[role] = ctxs
    return out["fit"], out["test"]
