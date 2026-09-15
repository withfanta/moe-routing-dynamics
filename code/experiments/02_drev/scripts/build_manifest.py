"""Freeze the DREV-P0 sample from blocks never used by REDV-V1 or REDV-V2.

Runs on CPU. Produces no research statistic.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoTokenizer

from drev import MODEL_ID, MODEL_REVISION, N_FIT, N_TEST, SAMPLE_SEED
from drev.data import (
    build_token_stream,
    draw_sample,
    load_prior_used_keys,
    load_split_texts,
    make_blocks,
    manifest_payload,
    untouched_pool,
    write_manifest,
)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "DREV_P0")
DATASET_DIR = os.environ.get("DREV_DATASET_DIR")
MODEL_DIR = os.environ.get("DREV_MODEL_DIR")


def main():
    os.makedirs(ART, exist_ok=True)

    src = MODEL_DIR or MODEL_ID
    kwargs = {} if MODEL_DIR else {"revision": MODEL_REVISION}
    tok = AutoTokenizer.from_pretrained(src, **kwargs)
    eos = 50279

    used, counts = load_prior_used_keys()
    print(f"prior blocks excluded: {counts}")

    blocks_by_split = {}
    for split in ("validation", "test"):
        texts = load_split_texts(split, dataset_dir=DATASET_DIR)
        stream = build_token_stream(texts, tok, eos)
        blocks_by_split[split] = make_blocks(stream)
        print(f"  {split}: {len(texts)} records -> {len(stream)} tokens -> "
              f"{len(blocks_by_split[split])} blocks")

    pool = untouched_pool(blocks_by_split, used)
    print(f"untouched pool: {len(pool)} blocks (need {N_FIT + N_TEST})")
    if len(pool) < N_FIT + N_TEST:
        print(f"TECHNICAL_DATA_BLOCKER: only {len(pool)} untouched blocks available")
        return 1

    fit, test = draw_sample(pool, seed=SAMPLE_SEED)
    payload = manifest_payload(
        fit, test, len(pool), counts,
        extra={"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
               "tokenizer_class": tok.__class__.__name__,
               "source_layer_human": 4, "source_layer_code": 3,
               "horizons": {"1": 5, "2": 6, "4": 8, "8": 12}},
    )
    h = write_manifest(os.path.join(ART, "data_manifest.json"), payload)

    print(f"data_manifest.json written, sha256 {h}")
    print(f"  fit  n={payload['fit']['n']} fingerprint {payload['fit']['fingerprint'][:16]}")
    print(f"  test n={payload['test']['n']} fingerprint {payload['test']['fingerprint'][:16]}")

    fit_keys = set(zip(payload["fit"]["source_splits"], payload["fit"]["block_ids"]))
    test_keys = set(zip(payload["test"]["source_splits"], payload["test"]["block_ids"]))
    print(f"  fit ∩ test = {len(fit_keys & test_keys)}")
    print(f"  DREV ∩ prior = {len((fit_keys | test_keys) & used)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
