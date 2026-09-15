"""Freeze the XEC-P0 TRAIN/VALIDATION/TEST sample from fresh WikiText train blocks.

Excludes every block EIPC-P0 used and verifies REDV/DREV never touched train. Runs on
CPU and produces no research statistic.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from transformers import AutoTokenizer

from xec import (
    MODEL_ID,
    MODEL_REVISION,
    N_TEST,
    N_TRAIN,
    N_VALIDATION,
    SAMPLE_SEED,
)
from xec.cache_features import build_projection, write_projection
from xec.data import (
    build_token_stream,
    draw_sample,
    load_eipc_used_blocks,
    load_train_texts,
    make_blocks,
    manifest_payload,
    untouched_blocks,
    verify_prior_projects_avoided_train,
    write_manifest,
)

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "XEC_P0")
DATASET_DIR = os.environ.get("XEC_DATASET_DIR")
MODEL_DIR = os.environ.get("XEC_MODEL_DIR")

# Same 40000-record window EIPC-P0 used, so its block ids refer to the same stream and
# exclusion is exact.
MAX_RECORDS = 40000


def main():
    os.makedirs(ART, exist_ok=True)

    checked = verify_prior_projects_avoided_train()
    print("prior projects verified to have never used train:", checked)

    excluded, eipc_info = load_eipc_used_blocks()
    print(f"EIPC-P0 exclusion: {eipc_info}")

    src = MODEL_DIR or MODEL_ID
    kwargs = {} if MODEL_DIR else {"revision": MODEL_REVISION}
    tok = AutoTokenizer.from_pretrained(src, **kwargs)
    eos = 50279

    texts = load_train_texts(dataset_dir=DATASET_DIR, max_records=MAX_RECORDS)
    stream = build_token_stream(texts, tok, eos)
    blocks = make_blocks(stream)
    print(f"train: {len(texts)} records -> {len(stream)} tokens -> {len(blocks)} blocks")

    if len(blocks) != eipc_info["eipc_blocks_available"]:
        raise AssertionError(
            f"stream mismatch: {len(blocks)} blocks vs EIPC-P0's "
            f"{eipc_info['eipc_blocks_available']}; block ids would not align")
    print("stream matches EIPC-P0's block indexing, so exclusion is exact")

    pool = untouched_blocks(blocks, excluded)
    need = N_TRAIN + N_VALIDATION + N_TEST
    print(f"untouched pool: {len(pool)} blocks (need {need})")
    if len(pool) < need:
        print(f"TECHNICAL_BLOCKER: only {len(pool)} untouched blocks")
        return 1

    train, validation, test = draw_sample(pool, seed=SAMPLE_SEED)

    P = build_projection()
    proj_sha = write_projection(os.path.join(ART, "projection.json"), P)
    print(f"projection.json written, sha256 {proj_sha}")

    payload = manifest_payload(
        train, validation, test, len(pool), len(texts), len(stream), len(blocks),
        MAX_RECORDS, eipc_info, checked,
        extra={"model_id": MODEL_ID, "model_revision": MODEL_REVISION,
               "tokenizer_class": tok.__class__.__name__,
               "projection_sha256": proj_sha},
    )
    h = write_manifest(os.path.join(ART, "data_manifest.json"), payload)

    print(f"data_manifest.json written, sha256 {h}")
    ids = {}
    for role in ("train", "validation", "test"):
        d = payload[role]
        ids[role] = set(d["block_ids"])
        print(f"  {role:11s} n={d['n']:5d} fingerprint {d['fingerprint'][:16]}")

    print(f"  train ∩ validation = {len(ids['train'] & ids['validation'])}")
    print(f"  train ∩ test       = {len(ids['train'] & ids['test'])}")
    print(f"  validation ∩ test  = {len(ids['validation'] & ids['test'])}")
    allxec = ids["train"] | ids["validation"] | ids["test"]
    print(f"  XEC ∩ EIPC-P0      = {len(allxec & excluded)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
