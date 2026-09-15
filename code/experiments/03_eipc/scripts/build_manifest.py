"""Freeze the EIPC-P0 FIT/TEST sample from the WikiText-103-raw TRAIN split.

Runs on CPU. Produces no research statistic.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoTokenizer

from eipc import MODEL_ID, MODEL_REVISION, N_FIT, N_TEST, SAMPLE_SEED
from eipc.data import (
    build_token_stream,
    draw_sample,
    load_train_texts,
    make_blocks,
    manifest_payload,
    write_manifest,
)
from eipc.expert_states import build_projection, write_projection

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "EIPC_P0")
DATASET_DIR = os.environ.get("EIPC_DATASET_DIR")
MODEL_DIR = os.environ.get("EIPC_MODEL_DIR")

# The train split is far larger than the 2048 blocks needed; truncating the record
# list keeps manifest construction cheap. Recorded in the manifest as part of the
# frozen sample definition.
MAX_RECORDS = 40000


def main():
    os.makedirs(ART, exist_ok=True)

    src = MODEL_DIR or MODEL_ID
    kwargs = {} if MODEL_DIR else {"revision": MODEL_REVISION}
    tok = AutoTokenizer.from_pretrained(src, **kwargs)
    eos = 50279

    texts = load_train_texts(dataset_dir=DATASET_DIR, max_records=MAX_RECORDS)
    stream = build_token_stream(texts, tok, eos)
    blocks = make_blocks(stream)
    print(f"train: {len(texts)} records -> {len(stream)} tokens -> {len(blocks)} blocks")
    print(f"need {N_FIT + N_TEST}")

    if len(blocks) < N_FIT + N_TEST:
        print(f"TECHNICAL_BLOCKER: only {len(blocks)} blocks available")
        return 1

    fit, test = draw_sample(blocks, seed=SAMPLE_SEED)

    P = build_projection()
    proj_sha = write_projection(os.path.join(ART, "projection.json"), P)
    print(f"projection.json written, sha256 {proj_sha}")

    payload = manifest_payload(
        fit, test, len(blocks), len(texts), len(stream), MAX_RECORDS,
        extra={"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
               "tokenizer_class": tok.__class__.__name__,
               "projection_sha256": proj_sha,
               "targets": {"8": {"history_human": list(range(1, 8))},
                           "12": {"history_human": list(range(1, 12))}}},
    )
    h = write_manifest(os.path.join(ART, "data_manifest.json"), payload)

    print(f"data_manifest.json written, sha256 {h}")
    print(f"  fit  n={payload['fit']['n']} fingerprint {payload['fit']['fingerprint'][:16]}")
    print(f"  test n={payload['test']['n']} fingerprint {payload['test']['fingerprint'][:16]}")

    fit_ids = set(payload["fit"]["block_ids"])
    test_ids = set(payload["test"]["block_ids"])
    print(f"  fit ∩ test = {len(fit_ids & test_ids)} (must be 0)")
    print(f"  unique fit {len(fit_ids)}, unique test {len(test_ids)}")
    print(f"  split used: {payload['split_used']}, validation/test used: {payload['validation_or_test_used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
