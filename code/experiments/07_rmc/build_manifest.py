"""Freeze the RMC-P0 data manifest.

Runs before any model inference. Tokenizes WikiText-103 raw TRAIN with the pinned JetMoE
tokenizer, cuts non-overlapping 129-token blocks, and draws a disjoint FIT/TEST sample
with seed 20260920. Writes artifacts/data_manifest.json and nothing else.

No OLMoE block indices are read: this tokenization is model-specific.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

from rmc import (
    ART, BLOCK_LEN, DATASET_CONFIG, DATASET_ID, DATASET_REVISION, DATASET_SPLIT,
    EXPERIMENTAL_POS, MODEL_ID, MODEL_REVISION, N_BLOCKS_POOL, N_FIT, N_SHARDS, N_TEST,
    N_TOTAL, SAMPLE_SEED, SHARD_FIT, SHARD_TEST, Manifest, fingerprint,
)


def log(msg: str) -> None:
    print(f"[manifest] {msg}", flush=True)


def build_blocks(tokenizer, target_blocks: int) -> list:
    """Non-overlapping 129-token blocks, EOS inserted between original records."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT,
                      revision=DATASET_REVISION)
    log(f"dataset loaded: {len(ds)} records")

    eos = tokenizer.eos_token_id
    if eos is None:
        raise SystemExit("tokenizer has no eos_token_id")

    blocks, buf, n_records = [], [], 0
    for rec in ds:
        text = rec["text"]
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        buf.extend(ids)
        buf.append(eos)                      # EOS between original WikiText records
        n_records += 1
        while len(buf) >= BLOCK_LEN:
            blocks.append(buf[:BLOCK_LEN])
            del buf[:BLOCK_LEN]              # non-overlapping
            if len(blocks) >= target_blocks:
                log(f"reached {len(blocks)} blocks after {n_records} records")
                return blocks
    raise SystemExit(f"only {len(blocks)} blocks available, needed {target_blocks}")


def main() -> int:
    os.makedirs(ART, exist_ok=True)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    log(f"tokenizer: {type(tok).__name__} vocab {tok.vocab_size} eos {tok.eos_token_id}")

    blocks = build_blocks(tok, N_BLOCKS_POOL)
    log(f"pool: {len(blocks)} non-overlapping blocks of {BLOCK_LEN} tokens")

    # Disjoint FIT/TEST draw from the frozen pool.
    rng = np.random.default_rng(SAMPLE_SEED)
    chosen = rng.choice(len(blocks), size=N_TOTAL, replace=False)
    chosen = np.sort(chosen)
    perm = rng.permutation(N_TOTAL)
    fit_idx = sorted(chosen[perm[:N_FIT]].tolist())
    test_idx = sorted(chosen[perm[N_FIT:]].tolist())

    assert len(fit_idx) == N_FIT and len(test_idx) == N_TEST
    assert not (set(fit_idx) & set(test_idx)), "FIT and TEST overlap"

    fit_blocks = [blocks[i] for i in fit_idx]
    test_blocks = [blocks[i] for i in test_idx]

    man = Manifest(
        n_fit=N_FIT, n_test=N_TEST,
        fit_index=fit_idx, test_index=test_idx,
        fit_fingerprint=fingerprint(fit_blocks),
        test_fingerprint=fingerprint(test_blocks),
        all_fingerprint=fingerprint(fit_blocks + test_blocks),
        tokenizer_name=type(tok).__name__,
        n_pool_blocks=len(blocks),
        block_len=BLOCK_LEN,
        experimental_pos=EXPERIMENTAL_POS,
        sample_seed=SAMPLE_SEED,
    )

    payload = {
        "experiment": "RMC-P0",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": DATASET_SPLIT,
        "reused_prior_block_indices": False,
        "note": ("model-specific tokenization; OLMoE block indices are not reused and no "
                 "prior manifest was read"),
        "shards": {"n_shards": N_SHARDS, "fit_per_shard": SHARD_FIT,
                   "test_per_shard": SHARD_TEST},
        **man.to_json(),
    }
    out = os.path.join(ART, "data_manifest.json")
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)

    log(f"FIT  n={N_FIT}  fingerprint {man.fit_fingerprint[:16]}")
    log(f"TEST n={N_TEST} fingerprint {man.test_fingerprint[:16]}")
    log(f"combined fingerprint {man.all_fingerprint[:16]}")
    log(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
