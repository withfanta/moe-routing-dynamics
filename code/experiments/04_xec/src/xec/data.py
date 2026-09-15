"""XEC-P0 data construction on fresh WikiText-103-raw TRAIN blocks.

Every block used by EIPC-P0 is excluded. REDV/DREV recorded that train was unused, and
that claim is verified rather than assumed.

Block construction matches the validated EIPC-P0 implementation: exact OLMoE tokenizer,
EOS between records, non-overlapping 129-token blocks, tokens 0..127 the context and
token 128 the next-token target, experimental token at context position 127.
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
    N_TEST,
    N_TRAIN,
    N_VALIDATION,
    SAMPLE_SEED,
)

PRIOR_MANIFESTS = (
    "/home/h-li/work/rejected_expert_delayed_value/artifacts/REDV_V1/data_manifest.json",
    "/home/h-li/work/rejected_expert_delayed_value/artifacts/REDV_V2/data_manifest.json",
    "/home/h-li/work/delayed_rejected_evidence/artifacts/DREV_P0/data_manifest.json",
)


@dataclass(frozen=True)
class Block:
    block_id: int
    stream_start: int
    tokens: tuple


@dataclass(frozen=True)
class XecContext:
    """One frozen XEC context, tagged with role."""

    role: str  # "train", "validation" or "test"
    block_id: int
    stream_start: int
    tokens: tuple

    @property
    def context(self) -> tuple:
        return self.tokens[:CONTEXT_LEN]

    @property
    def target(self) -> int:
        return self.tokens[CONTEXT_LEN]


def load_train_texts(dataset_dir: str | None = None, max_records: int | None = None) -> List[str]:
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


def load_eipc_used_blocks(path: str = EIPC_MANIFEST) -> Tuple[set, Dict]:
    """Every train block id EIPC-P0 consumed. Read-only."""
    with open(path) as fh:
        m = json.load(fh)
    if m.get("split_used") != "train":
        raise AssertionError(f"expected EIPC-P0 to use train, got {m.get('split_used')!r}")
    used = set(m["fit"]["block_ids"]) | set(m["test"]["block_ids"])
    info = {
        "eipc_split_used": m["split_used"],
        "eipc_fit_n": m["fit"]["n"],
        "eipc_test_n": m["test"]["n"],
        "eipc_blocks_excluded": len(used),
        "eipc_max_records": m["max_records_truncation"],
        "eipc_blocks_available": m["n_blocks_available"],
    }
    return used, info


def verify_prior_projects_avoided_train(paths: Sequence[str] = PRIOR_MANIFESTS) -> Dict[str, bool]:
    """Verify REDV/DREV really never used the train split. Read-only."""
    out = {}
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        name = p.split("/")[-2]
        flag = d.get("train_split_used")
        if flag is not False:
            raise AssertionError(f"{name} does not record train_split_used = False (got {flag!r})")
        out[name] = True
    return out


def untouched_blocks(blocks: Sequence[Block], excluded: set) -> List[Block]:
    return [b for b in blocks if b.block_id not in excluded]


def draw_sample(
    pool: Sequence[Block], seed: int = SAMPLE_SEED
) -> Tuple[List[XecContext], List[XecContext], List[XecContext]]:
    """Draw disjoint TRAIN / VALIDATION / TEST splits without replacement."""
    need = N_TRAIN + N_VALIDATION + N_TEST
    if len(pool) < need:
        raise RuntimeError(
            f"TECHNICAL_BLOCKER: only {len(pool)} untouched blocks, need {need}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=need, replace=False)

    def make(rows, role):
        return [
            XecContext(role=role, block_id=pool[int(i)].block_id,
                       stream_start=pool[int(i)].stream_start, tokens=pool[int(i)].tokens)
            for i in rows
        ]

    a = N_TRAIN
    b = N_TRAIN + N_VALIDATION
    return make(idx[:a], "train"), make(idx[a:b], "validation"), make(idx[b:], "test")


def fingerprint(contexts: Sequence[XecContext]) -> str:
    h = hashlib.sha256()
    for c in contexts:
        h.update(np.asarray([c.block_id, c.stream_start], dtype=np.int64).tobytes())
        h.update(np.asarray(c.tokens, dtype=np.int64).tobytes())
    return h.hexdigest()


def manifest_payload(
    train: Sequence[XecContext],
    validation: Sequence[XecContext],
    test: Sequence[XecContext],
    pool_size: int,
    n_records: int,
    n_stream_tokens: int,
    n_blocks_total: int,
    max_records: int | None,
    eipc_info: Dict,
    prior_checked: Dict[str, bool],
    extra: Dict | None = None,
) -> Dict:
    payload: Dict[str, object] = {
        "experiment": "XEC-P0",
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
        "max_records_truncation": max_records,
        "n_stream_tokens": n_stream_tokens,
        "n_blocks_total": n_blocks_total,
        "untouched_pool_size": pool_size,
        "n_required": N_TRAIN + N_VALIDATION + N_TEST,
        "eipc_exclusion": eipc_info,
        "prior_projects_train_unused_verified": prior_checked,
        "target_layer_human": 12,
        "target_layer_code": 11,
        "history_layers_human": list(range(1, 12)),
    }
    if extra:
        payload.update(extra)
    for role, contexts in (("train", train), ("validation", validation), ("test", test)):
        payload[role] = {
            "n": len(contexts),
            "fingerprint": fingerprint(contexts),
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


def rebuild_from_manifest(
    path: str, tokenizer, eos_token_id: int, dataset_dir: str | None = None
) -> Dict[str, List[XecContext]]:
    """Reconstruct the exact frozen contexts and verify their fingerprints."""
    with open(path) as fh:
        m = json.load(fh)

    texts = load_train_texts(dataset_dir=dataset_dir, max_records=m["max_records_truncation"])
    stream = build_token_stream(texts, tokenizer, eos_token_id)
    by_id = {b.block_id: b for b in make_blocks(stream)}

    out: Dict[str, List[XecContext]] = {}
    for role in ("train", "validation", "test"):
        d = m[role]
        ctxs = []
        for bid, start in zip(d["block_ids"], d["stream_starts"]):
            b = by_id[bid]
            if b.stream_start != start:
                raise AssertionError(f"stream offset mismatch for block {bid}")
            ctxs.append(XecContext(role=role, block_id=bid, stream_start=start, tokens=b.tokens))
        if fingerprint(ctxs) != d["fingerprint"]:
            raise AssertionError(f"{role} fingerprint mismatch on rebuild")
        out[role] = ctxs
    return out
