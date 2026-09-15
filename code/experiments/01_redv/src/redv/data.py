"""Frozen WikiText-103-raw context construction for REDV-V1.

Rules (preregistered, do not change):

* Only ``validation`` (probe fitting) and ``test`` (held-out evaluation) splits.
  ``train`` is never touched.
* One token stream is built independently per split, inserting the model EOS
  token between original WikiText text records.
* The stream is cut into NON-OVERLAPPING blocks of 129 tokens: the first 128 are
  the context, token 129 is the next-token target.
* Exactly 512 blocks are sampled without replacement per split, seed 20260914.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from . import BLOCK_LEN, CONTEXT_LEN, DATASET_CONFIG, DATASET_ID, DATASET_REVISION, SEED


@dataclass(frozen=True)
class Block:
    """One frozen 129-token block.

    ``stream_start`` is the token offset of the block inside the split's token
    stream, which makes the sample reconstructible from the stream alone.
    """

    block_id: int
    stream_start: int
    tokens: tuple  # length 129

    @property
    def context(self) -> tuple:
        return self.tokens[:CONTEXT_LEN]

    @property
    def target(self) -> int:
        return self.tokens[CONTEXT_LEN]


def build_token_stream(texts: Sequence[str], tokenizer, eos_token_id: int) -> List[int]:
    """Concatenate the split's text records into one stream, EOS between records.

    Records are tokenized without special tokens; the EOS token is the only
    separator inserted. Empty WikiText records (blank lines) contribute no
    tokens but still act as record boundaries, so they are skipped rather than
    producing bare EOS runs.
    """
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
    n_blocks = len(stream) // BLOCK_LEN
    return [
        Block(
            block_id=i,
            stream_start=i * BLOCK_LEN,
            tokens=tuple(stream[i * BLOCK_LEN : (i + 1) * BLOCK_LEN]),
        )
        for i in range(n_blocks)
    ]


def sample_blocks(blocks: Sequence[Block], n: int, seed: int, salt: int) -> List[Block]:
    """Sample exactly ``n`` blocks without replacement.

    The split-specific ``salt`` keeps validation and test draws independent while
    remaining fully determined by the frozen seed.
    """
    if len(blocks) < n:
        raise RuntimeError(f"only {len(blocks)} blocks available, need {n}")
    rng = np.random.default_rng([seed, salt])
    idx = rng.choice(len(blocks), size=n, replace=False)
    idx = np.sort(idx)
    return [blocks[int(i)] for i in idx]


def blocks_fingerprint(blocks: Sequence[Block]) -> str:
    """Content hash over (block_id, stream_start, tokens) for determinism tests."""
    h = hashlib.sha256()
    for b in blocks:
        h.update(np.asarray([b.block_id, b.stream_start], dtype=np.int64).tobytes())
        h.update(np.asarray(b.tokens, dtype=np.int64).tobytes())
    return h.hexdigest()


def load_split_texts(split: str, dataset_dir: str | None = None) -> List[str]:
    """Load one WikiText-103-raw split at the pinned revision.

    ``train`` is refused: REDV-V1 uses validation and test only.
    """
    if split not in ("validation", "test"):
        raise ValueError(f"REDV-V1 uses validation and test only, got {split!r}")

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

    ds = load_dataset(
        DATASET_ID, DATASET_CONFIG, split=split, revision=DATASET_REVISION
    )
    return list(ds["text"])


def build_frozen_sample(tokenizer, eos_token_id: int, dataset_dir: str | None = None) -> Dict:
    """Build the frozen validation/test samples and the manifest payload."""
    out: Dict[str, object] = {}
    salts = {"validation": 1, "test": 2}
    for split in ("validation", "test"):
        texts = load_split_texts(split, dataset_dir=dataset_dir)
        stream = build_token_stream(texts, tokenizer, eos_token_id)
        blocks = make_blocks(stream)
        chosen = sample_blocks(blocks, 512, SEED, salts[split])
        out[split] = {
            "blocks": chosen,
            "n_records": len(texts),
            "n_stream_tokens": len(stream),
            "n_blocks_available": len(blocks),
            "n_sampled": len(chosen),
            "sample_salt": salts[split],
            "fingerprint": blocks_fingerprint(chosen),
        }
    return out


def manifest_payload(sample: Dict, extra: Dict | None = None) -> Dict:
    """Serializable manifest: enough to reconstruct the exact sampled blocks."""
    payload: Dict[str, object] = {
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "splits_used": {"validation": "probe fitting", "test": "held-out evaluation"},
        "train_split_used": False,
        "seed": SEED,
        "context_len": CONTEXT_LEN,
        "block_len": BLOCK_LEN,
        "overlapping_windows": False,
    }
    if extra:
        payload.update(extra)
    for split in ("validation", "test"):
        s = sample[split]
        blocks = s["blocks"]
        payload[split] = {
            "n_records": s["n_records"],
            "n_stream_tokens": s["n_stream_tokens"],
            "n_blocks_available": s["n_blocks_available"],
            "n_sampled": s["n_sampled"],
            "sample_salt": s["sample_salt"],
            "fingerprint": s["fingerprint"],
            "block_ids": [b.block_id for b in blocks],
            "stream_starts": [b.stream_start for b in blocks],
            "targets": [b.target for b in blocks],
        }
    return payload


def write_manifest(path: str, payload: Dict) -> str:
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()
