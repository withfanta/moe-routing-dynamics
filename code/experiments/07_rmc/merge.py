"""Merge the four RMC-P0 shards into artifacts/merged.npz.

Verifies that shard coverage is exact and disjoint and that FIT/TEST counts match the
frozen manifest. Computes no statistic.
"""

from __future__ import annotations

import json
import os

import numpy as np

from rmc import ART, N_FIT, N_LAYERS, NUM_EXPERTS, N_SHARDS, N_TEST, N_TOTAL, TOP_K


def main() -> int:
    parts = []
    for s in range(N_SHARDS):
        p = os.path.join(ART, f"shard_{s}.npz")
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")
        parts.append(np.load(p))
        print(f"[merge] shard {s}: {parts[-1]['ids'].shape[0]} samples, "
              f"final microbatch {int(parts[-1]['microbatch_final'][0])}, "
              f"agreement {float(parts[-1]['agreement'][0]):.6f}")

    ids = np.concatenate([p["ids"] for p in parts], axis=0)
    logits = np.concatenate([p["logits"] for p in parts], axis=0)
    is_test = np.concatenate([p["is_test"] for p in parts], axis=0)
    block_id = np.concatenate([p["block_id"] for p in parts], axis=0)

    assert ids.shape == (N_TOTAL, N_LAYERS, TOP_K), ids.shape
    assert logits.shape == (N_TOTAL, N_LAYERS, NUM_EXPERTS), logits.shape
    assert int((is_test == 0).sum()) == N_FIT
    assert int(is_test.sum()) == N_TEST
    unique_ids = np.unique(block_id)
    assert unique_ids.size == N_TOTAL, "duplicate block ids: shards overlap"

    with open(os.path.join(ART, "data_manifest.json")) as fh:
        man = json.load(fh)
    assert set(block_id[is_test == 0].tolist()) == set(man["fit_index"])
    assert set(block_id[is_test == 1].tolist()) == set(man["test_index"])

    out = os.path.join(ART, "merged.npz")
    np.savez_compressed(out, ids=ids, logits=logits, is_test=is_test, block_id=block_id)
    print(f"[merge] wrote {out}: n={N_TOTAL} "
          f"(FIT {N_FIT}, TEST {N_TEST}), {N_LAYERS} layers, {NUM_EXPERTS} experts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
