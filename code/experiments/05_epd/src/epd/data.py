"""EPD-P0 data construction on fresh WikiText-103-raw TRAIN blocks.

Excludes every block used by EIPC-P0 and XEC-P0, and verifies REDV/DREV never used train.
Block construction matches the prior projects exactly: EOS between original records,
non-overlapping 129-token blocks, tokens 0..127 the context, token 128 the next token,
experimental token at context position 127.

Samples are split into four deterministic shards that preserve FIT/TEST identity.
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
    EIPC_MANIFEST,
    N_FIT,
    N_SHARDS,
    N_TEST,
    N_TOTAL,
    PRIOR_TRAIN_UNUSED_MANIFESTS,
    SAMPLE_SEED,
    XEC_MANIFEST,
)

# All prior projects built their stream from this record window, so block ids align.
MAX_RECORDS = 40000


@dataclass(frozen=True)
class Block:
    block_id: int
    stream_start: int
    tokens: tuple


@dataclass(frozen=True)
class EpdContext:
    """One frozen EPD context: role, shard assignment, provenance, tokens."""

    role: str  # "fit" or "test"
    shard: int
    block_id: int
    stream_start: int
    tokens: tuple

    @property
    def context(self) -> tuple:
        return self.tokens[:CONTEXT_LEN]

    @property
    def next_token(self) -> int:
        return self.tokens[CONTEXT_LEN]


def load_train_texts(dataset_dir: str | None = None, max_records: int | None = MAX_RECORDS):
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
    n = len(stream) // BLOCK_LEN
    return [
        Block(block_id=i, stream_start=i * BLOCK_LEN,
              tokens=tuple(stream[i * BLOCK_LEN : (i + 1) * BLOCK_LEN]))
        for i in range(n)
    ]


def load_prior_used_blocks() -> Tuple[set, Dict]:
    """Every train block used by EIPC-P0 or XEC-P0. Read-only."""
    with open(EIPC_MANIFEST) as fh:
        e = json.load(fh)
    with open(XEC_MANIFEST) as fh:
        x = json.load(fh)

    if e.get("split_used") != "train" or x.get("split_used") != "train":
        raise AssertionError("expected both prior projects to have used the train split")

    eipc = set(e["fit"]["block_ids"]) | set(e["test"]["block_ids"])
    xec = set(x["train"]["block_ids"]) | set(x["validation"]["block_ids"]) | set(x["test"]["block_ids"])

    info = {
        "eipc_blocks": len(eipc),
        "xec_blocks": len(xec),
        "eipc_xec_overlap": len(eipc & xec),
        "union_excluded": len(eipc | xec),
        "eipc_blocks_available": e["n_blocks_available"],
        "xec_blocks_total": x["n_blocks_total"],
        "shared_max_records": e["max_records_truncation"],
    }
    return eipc | xec, info


def verify_prior_projects_avoided_train() -> Dict[str, bool]:
    """Verify REDV/DREV never used the train split. Read-only."""
    out = {}
    for p in PRIOR_TRAIN_UNUSED_MANIFESTS:
        with open(p) as fh:
            d = json.load(fh)
        name = p.split("/")[-2]
        if d.get("train_split_used") is not False:
            raise AssertionError(f"{name} does not record train_split_used = False")
        out[name] = True
    return out


def fresh_blocks(blocks: Sequence[Block], excluded: set) -> List[Block]:
    return [b for b in blocks if b.block_id not in excluded]


def draw_sample(pool: Sequence[Block], seed: int = SAMPLE_SEED) -> List[EpdContext]:
    """Draw 512 FIT + 256 TEST blocks, disjoint, then assign four deterministic shards.

    Shard assignment is round-robin over the drawn order *within* each role, so every
    shard holds the same FIT/TEST proportion and FIT/TEST identity is preserved.
    """
    if len(pool) < N_TOTAL:
        raise RuntimeError(f"TECHNICAL_BLOCKER: only {len(pool)} fresh blocks, need {N_TOTAL}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=N_TOTAL, replace=False)

    contexts: List[EpdContext] = []
    for role, rows in (("fit", idx[:N_FIT]), ("test", idx[N_FIT:])):
        for position, i in enumerate(rows):
            b = pool[int(i)]
            contexts.append(EpdContext(
                role=role, shard=position % N_SHARDS, block_id=b.block_id,
                stream_start=b.stream_start, tokens=b.tokens))
    return contexts


def shard_contexts(contexts: Sequence[EpdContext], shard: int) -> List[EpdContext]:
    """Contexts for one shard, in stable manifest order."""
    return [c for c in contexts if c.shard == shard]


def fingerprint(contexts: Sequence[EpdContext]) -> str:
    h = hashlib.sha256()
    for c in contexts:
        h.update(c.role.encode())
        h.update(np.asarray([c.shard, c.block_id, c.stream_start], dtype=np.int64).tobytes())
        h.update(np.asarray(c.tokens, dtype=np.int64).tobytes())
    return h.hexdigest()


def manifest_payload(contexts: Sequence[EpdContext], pool_size: int, n_records: int,
                     n_stream_tokens: int, n_blocks_total: int, prior_info: Dict,
                     prior_checked: Dict[str, bool], projection_meta: Dict,
                     extra: Dict | None = None) -> Dict:
    payload: Dict[str, object] = {
        "experiment": "EPD-P0",
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "split_used": DATASET_SPLIT,
        "sample_seed": SAMPLE_SEED,
        "context_len": CONTEXT_LEN,
        "block_len": BLOCK_LEN,
        "overlapping_windows": False,
        "experimental_token_position": CONTEXT_LEN - 1,
        "n_records_used": n_records,
        "max_records_truncation": MAX_RECORDS,
        "n_stream_tokens": n_stream_tokens,
        "n_blocks_total": n_blocks_total,
        "fresh_pool_size": pool_size,
        "n_fit": N_FIT,
        "n_test": N_TEST,
        "n_total": N_TOTAL,
        "n_shards": N_SHARDS,
        "prior_exclusion": prior_info,
        "prior_projects_train_unused_verified": prior_checked,
        "reused_eipc_projection": projection_meta,
        "target_layer_human": 12,
        "target_layer_code": 11,
        "history_layers_human": list(range(1, 12)),
        "fingerprint_all": fingerprint(contexts),
    }
    if extra:
        payload.update(extra)

    for role in ("fit", "test"):
        sel = [c for c in contexts if c.role == role]
        payload[role] = {
            "n": len(sel),
            "fingerprint": fingerprint(sel),
            "block_ids": [c.block_id for c in sel],
            "stream_starts": [c.stream_start for c in sel],
            "shards": [c.shard for c in sel],
            "next_tokens": [c.next_token for c in sel],
        }
    payload["shard_sizes"] = {
        str(s): len([c for c in contexts if c.shard == s]) for s in range(N_SHARDS)}
    payload["shard_role_counts"] = {
        str(s): {r: len([c for c in contexts if c.shard == s and c.role == r])
                 for r in ("fit", "test")}
        for s in range(N_SHARDS)}
    return payload


def write_manifest(path: str, payload: Dict) -> str:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def rebuild_from_manifest(path: str, tokenizer, eos_token_id: int,
                          dataset_dir: str | None = None) -> List[EpdContext]:
    """Reconstruct the exact frozen contexts and verify fingerprints."""
    with open(path) as fh:
        m = json.load(fh)

    texts = load_train_texts(dataset_dir=dataset_dir, max_records=m["max_records_truncation"])
    stream = build_token_stream(texts, tokenizer, eos_token_id)
    by_id = {b.block_id: b for b in make_blocks(stream)}

    contexts: List[EpdContext] = []
    for role in ("fit", "test"):
        d = m[role]
        sel = []
        for bid, start, shard in zip(d["block_ids"], d["stream_starts"], d["shards"]):
            b = by_id[bid]
            if b.stream_start != start:
                raise AssertionError(f"stream offset mismatch for block {bid}")
            sel.append(EpdContext(role=role, shard=shard, block_id=bid,
                                  stream_start=start, tokens=b.tokens))
        if fingerprint(sel) != d["fingerprint"]:
            raise AssertionError(f"{role} fingerprint mismatch on rebuild")
        contexts.extend(sel)

    if fingerprint(contexts) != m["fingerprint_all"]:
        raise AssertionError("combined fingerprint mismatch on rebuild")
    return contexts
