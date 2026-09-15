"""RMC-P0 shard extraction: one independent single-GPU worker.

Captures, for the experimental token of every block in its shard, the native MLP-MoE
router logits and native Top-2 MLP expert identities at all 24 JetMoE blocks. Nothing
else is captured.

The MLP router is identified structurally (identity against `layers[i].mlp.router`), never
by name matching, because JetMoE's attention mixture carries a router of the identical
shape. The library's `output_router_logits` path is not used: it interleaves attention and
MLP logits in one flat tuple.

Usage:  CUDA_VISIBLE_DEVICES=<physical> python extract.py --shard <0..3>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from rmc import (
    ART, BLOCK_LEN, DATASET_CONFIG, DATASET_ID, DATASET_REVISION, DATASET_SPLIT,
    EXPERIMENTAL_POS, MEMORY_BLOCKER, MICROBATCH, MODEL_ID, MODEL_REVISION, N_LAYERS,
    NUM_EXPERTS, N_SHARDS, ROUTER_BLOCKER, SHARD_FIT, SHARD_TEST, TOP_K, fingerprint,
)


def log(shard: int, msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [shard {shard}] {msg}"
    print(line, flush=True)
    with open(os.path.join(ART, f"extract_{shard}.log"), "a") as fh:
        fh.write(line + "\n")


# ------------------------------------------------------------------ router identification


def mlp_routers(model) -> List[torch.nn.Module]:
    """The 24 MLP-MoE routers, verified to be the MLP mixture and not the MoA.

    Raises with ROUTER_IDENTIFICATION_BLOCKER if the structure is not exactly as expected.
    """
    from transformers.models.jetmoe.modeling_jetmoe import JetMoeMoA, JetMoeMoE

    layers = model.model.layers
    if len(layers) != N_LAYERS:
        raise SystemExit(f"{ROUTER_BLOCKER}: {len(layers)} blocks, expected {N_LAYERS}")

    routers = []
    for i, block in enumerate(layers):
        mlp = getattr(block, "mlp", None)
        if not isinstance(mlp, JetMoeMoE):
            raise SystemExit(f"{ROUTER_BLOCKER}: block {i} .mlp is {type(mlp).__name__}, "
                             "not JetMoeMoE")
        moa = getattr(getattr(block, "self_attention", None), "experts", None)
        if not isinstance(moa, JetMoeMoA):
            raise SystemExit(f"{ROUTER_BLOCKER}: block {i} attention experts is "
                             f"{type(moa).__name__}, not JetMoeMoA")
        r = mlp.router
        if r is moa.router:
            raise SystemExit(f"{ROUTER_BLOCKER}: block {i} MLP and MoA share a router")
        if r.num_experts != NUM_EXPERTS or r.top_k != TOP_K:
            raise SystemExit(f"{ROUTER_BLOCKER}: block {i} router is "
                             f"{r.num_experts} experts top-{r.top_k}")
        if r.layer.out_features != NUM_EXPERTS:
            raise SystemExit(f"{ROUTER_BLOCKER}: block {i} router width "
                             f"{r.layer.out_features}")
        routers.append(r)
    return routers


class RouterCapture:
    """Forward hooks on the MLP routers only.

    JetMoeTopKGating.forward returns (..., logits) where logits is the raw fp32
    `self.layer(hidden_states)`. Top-2 identities are recomputed from that same tensor with
    the module's own `logits.topk(top_k)`, so identities and logits cannot disagree.
    """

    def __init__(self, routers, seq_len: int, pos: int):
        self.routers = routers
        self.seq_len = seq_len
        self.pos = pos
        self.logits: Dict[int, torch.Tensor] = {}
        self.ids: Dict[int, torch.Tensor] = {}
        self._handles = []

    def _make(self, idx: int, top_k: int):
        def hook(module, args, output):
            logits = output[-1]                      # [n_tokens, 8], fp32
            n_tok = logits.shape[0]
            bsz = n_tok // self.seq_len
            if bsz * self.seq_len != n_tok:
                raise RuntimeError(f"router saw {n_tok} tokens, not a multiple of "
                                   f"seq_len {self.seq_len}")
            g = logits.view(bsz, self.seq_len, -1)[:, self.pos, :]
            self.logits[idx] = g.detach().float().cpu()
            # The module's own selection rule, applied to the module's own logits.
            self.ids[idx] = g.detach().topk(top_k, dim=-1).indices.cpu()
        return hook

    def __enter__(self):
        for i, r in enumerate(self.routers):
            self._handles.append(r.register_forward_hook(self._make(i, r.top_k)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


# ------------------------------------------------------------------------------ data


def shard_blocks(shard: int) -> Tuple[List[List[int]], np.ndarray, np.ndarray]:
    """Rebuild this shard's blocks from the frozen manifest and verify fingerprints."""
    with open(os.path.join(ART, "data_manifest.json")) as fh:
        man = json.load(fh)

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    ds = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT,
                      revision=DATASET_REVISION)

    need = max(max(man["fit_index"]), max(man["test_index"])) + 1
    eos = tok.eos_token_id
    blocks, buf = [], []
    for rec in ds:
        if not rec["text"].strip():
            continue
        buf.extend(tok(rec["text"], add_special_tokens=False)["input_ids"])
        buf.append(eos)
        while len(buf) >= BLOCK_LEN:
            blocks.append(buf[:BLOCK_LEN])
            del buf[:BLOCK_LEN]
            if len(blocks) >= need:
                break
        if len(blocks) >= need:
            break

    fit_all = [blocks[i] for i in man["fit_index"]]
    test_all = [blocks[i] for i in man["test_index"]]
    if fingerprint(fit_all) != man["fit_fingerprint"]:
        raise SystemExit("FIT fingerprint mismatch on rebuild")
    if fingerprint(test_all) != man["test_fingerprint"]:
        raise SystemExit("TEST fingerprint mismatch on rebuild")

    # Deterministic contiguous shard slices, 128 FIT + 64 TEST each.
    f0, f1 = shard * SHARD_FIT, (shard + 1) * SHARD_FIT
    t0, t1 = shard * SHARD_TEST, (shard + 1) * SHARD_TEST
    mine = fit_all[f0:f1] + test_all[t0:t1]
    is_test = np.array([0] * SHARD_FIT + [1] * SHARD_TEST, dtype=np.int64)
    block_id = np.array(man["fit_index"][f0:f1] + man["test_index"][t0:t1], dtype=np.int64)
    return mine, is_test, block_id


# ------------------------------------------------------------------------------- run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True, choices=range(N_SHARDS))
    args = ap.parse_args()
    shard = args.shard
    os.makedirs(ART, exist_ok=True)

    log(shard, f"=== RMC-P0 extraction, shard {shard} ===")
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    dev = torch.device("cuda:0")
    log(shard, f"device {torch.cuda.get_device_name(0)}, "
               f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    blocks, is_test, block_id = shard_blocks(shard)
    log(shard, f"{len(blocks)} blocks rebuilt and fingerprint-verified "
               f"({int((is_test == 0).sum())} FIT, {int(is_test.sum())} TEST)")

    from transformers import JetMoeForCausalLM
    try:
        model = JetMoeForCausalLM.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16,
            low_cpu_mem_usage=True).to(dev)
    except torch.cuda.OutOfMemoryError as e:
        raise SystemExit(f"{MEMORY_BLOCKER}: single-GPU FP16 load failed: {e}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    assert next(model.parameters()).dtype == torch.float16
    log(shard, f"model loaded FP16, eval, frozen; "
               f"{torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB allocated")

    routers = mlp_routers(model)
    log(shard, f"MLP routers identified: {len(routers)} blocks, "
               f"{routers[0].num_experts} experts, top-{routers[0].top_k}")

    ids_out = np.zeros((len(blocks), N_LAYERS, TOP_K), dtype=np.int16)
    logits_out = np.zeros((len(blocks), N_LAYERS, NUM_EXPERTS), dtype=np.float32)

    mb = MICROBATCH
    i = 0
    t0 = time.time()
    with torch.inference_mode():
        while i < len(blocks):
            batch = blocks[i:i + mb]
            ids = torch.tensor([b[:BLOCK_LEN - 1] for b in batch], dtype=torch.long,
                               device=dev)   # 128 context tokens
            try:
                with RouterCapture(routers, ids.shape[1], EXPERIMENTAL_POS) as cap:
                    model(input_ids=ids, use_cache=False)
                    for l in range(N_LAYERS):
                        logits_out[i:i + len(batch), l, :] = cap.logits[l].numpy()
                        ids_out[i:i + len(batch), l, :] = cap.ids[l].numpy()
            except torch.cuda.OutOfMemoryError:
                if mb == 1:
                    raise SystemExit(f"{MEMORY_BLOCKER}: OOM at microbatch 1")
                torch.cuda.empty_cache()
                mb //= 2
                log(shard, f"OOM: reducing microbatch to {mb}")
                continue
            i += len(batch)
            if i % 64 == 0 or i == len(blocks):
                log(shard, f"  {i}/{len(blocks)} at microbatch {mb} "
                           f"({time.time() - t0:.1f}s)")

    # Sanity: identities must be the argsort-top2 of the captured logits.
    recomputed = np.argsort(-logits_out, axis=-1)[:, :, :TOP_K]
    agree = float((np.sort(recomputed, -1) == np.sort(ids_out, -1)).all(-1).mean())
    log(shard, f"top-{TOP_K} identity/logit agreement: {agree:.6f}")
    if agree != 1.0:
        raise SystemExit("captured identities disagree with captured logits")

    out = os.path.join(ART, f"shard_{shard}.npz")
    np.savez_compressed(
        out, ids=ids_out, logits=logits_out, is_test=is_test, block_id=block_id,
        shard=np.array([shard], dtype=np.int64),
        microbatch_final=np.array([mb], dtype=np.int64),
        agreement=np.array([agree], dtype=np.float64))
    log(shard, f"wrote {out} in {time.time() - t0:.1f}s (final microbatch {mb})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
