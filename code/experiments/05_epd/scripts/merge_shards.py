"""Merge the four EPD-P0 shards once on CPU, restoring frozen manifest order.

Verifies that the union of shards is exactly the manifest sample, with FIT/TEST identity
preserved. Builds no representation and fits no probe.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from epd import N_FIT, N_SHARDS, N_TEST, N_TOTAL, PROJ_DIM, TOP_K

ART = os.path.join(os.path.dirname(__file__), "..", "artifacts", "EPD_P0")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [merge] {msg}"
    print(line, flush=True)
    with open(os.path.join(ART, "run.log"), "a") as fh:
        fh.write(line + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    log("=== EPD-P0 merge shards (CPU) ===")

    with open(os.path.join(ART, "data_manifest.json")) as fh:
        man = json.load(fh)

    parts = {}
    recon_abs, recon_rel = 0.0, 0.0
    for shard in range(N_SHARDS):
        path = os.path.join(ART, f"shard_{shard}.npz")
        if not os.path.exists(path):
            log(f"MISSING {path}; all four shards must finish first")
            return 1
        d = np.load(path)
        parts[shard] = d
        recon_abs = max(recon_abs, float(d["recon_max_abs"][0]))
        recon_rel = max(recon_rel, float(d["recon_max_rel"][0]))
        log(f"  shard {shard}: n={d['s'].shape[0]} sha256 {sha256_file(path)[:16]}")
    log(f"  worst reconstruction across shards: abs {recon_abs:.3e}, relative {recon_rel:.3e}")

    keys = ("s", "hist_ids", "hist_probs", "g12", "top8_12", "next_token", "is_test", "block_id")
    cat = {k: np.concatenate([parts[s][k] for s in range(N_SHARDS)], axis=0) for k in keys}
    if cat["s"].shape[0] != N_TOTAL:
        raise AssertionError(f"merged n={cat['s'].shape[0]}, expected {N_TOTAL}")

    # Restore frozen manifest order per role, keyed by block id.
    index = {int(b): i for i, b in enumerate(cat["block_id"])}
    order, roles = [], []
    for role, flag in (("fit", 0), ("test", 1)):
        for bid in man[role]["block_ids"]:
            i = index[int(bid)]
            if int(cat["is_test"][i]) != flag:
                raise AssertionError(f"block {bid} has wrong role flag")
            order.append(i)
            roles.append(flag)
    order = np.asarray(order)
    roles = np.asarray(roles)

    merged = {k: cat[k][order] for k in keys}
    merged["is_test"] = roles

    n_fit = int((roles == 0).sum())
    n_test = int((roles == 1).sum())
    if (n_fit, n_test) != (N_FIT, N_TEST):
        raise AssertionError(f"role counts ({n_fit}, {n_test}) != ({N_FIT}, {N_TEST})")

    # Block ids must match the manifest exactly, in order.
    expected_ids = np.array(man["fit"]["block_ids"] + man["test"]["block_ids"], dtype=np.int64)
    if not np.array_equal(merged["block_id"], expected_ids):
        raise AssertionError("merged block ids do not match manifest order")
    # Next tokens must match too.
    expected_next = np.array(man["fit"]["next_tokens"] + man["test"]["next_tokens"], dtype=np.int64)
    if not np.array_equal(merged["next_token"], expected_next):
        raise AssertionError("merged next tokens do not match manifest")

    assert merged["s"].shape[1:] == (11, TOP_K, PROJ_DIM), merged["s"].shape
    assert merged["g12"].shape[1] == 64

    out = os.path.join(ART, "merged_raw.npz")
    np.savez_compressed(
        out,
        recon_max_abs=np.array([recon_abs]), recon_max_rel=np.array([recon_rel]),
        **merged)
    log(f"merged_raw.npz written, sha256 {sha256_file(out)}")
    log(f"  n_fit {n_fit}, n_test {n_test}, s{merged['s'].shape}, g12{merged['g12'].shape}")
    log("merge complete; no representation built, no probe fitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
