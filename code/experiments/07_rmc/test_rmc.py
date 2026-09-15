"""RMC-P0 implementation checks, tests A through P of protocol.md section 18.

These run before any formal R2 exists (test P is the gate inside analyze.py). Checks that
would need the 16 GB checkpoint on a GPU are done on a meta-device instantiation instead,
so structural facts are verified without touching the extraction budget.

Where a property is a claim about code rather than about numbers, the check walks the AST
for the names actually called or referenced, never the prose, so a test cannot pass or fail
on its own wording.
"""

from __future__ import annotations

import ast
import inspect
import json
import os

import numpy as np
import pytest
import torch

import analyze
import extract
import rmc
from rmc import (
    ALPHA, ART, BLOCK_LEN, DELTA4_THRESHOLD, EXPERIMENTAL_POS, FEATURE_DIM, HERE, K_VALUES,
    MODEL_ID, MODEL_REVISION, NUM_EXPERTS, N_FIT, N_LAYERS, N_TEST, PCA_COMPONENTS,
    PCA_SEED, PRIMARY_K, RECENT_DIM, TARGET_LAYERS, TOP_K, history_layers, older_layers,
)

MODULES = ("rmc", "extract", "analyze", "merge", "build_manifest", "record_provenance")
SRC = {name: open(os.path.join(HERE, f"{name}.py")).read() for name in MODULES}
TREE = {name: ast.parse(src) for name, src in SRC.items()}


def _code_only(tree: ast.AST) -> str:
    """The module with every docstring and comment removed.

    Scanning raw source for banned words makes a test fail on its own explanatory prose:
    analyze.py has to be able to say "no probability" and record `moa_router_used: False`
    without that counting as using either. Only executable code is scanned.
    """
    stripped = ast.parse(ast.unparse(tree))
    for node in ast.walk(stripped):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and ast.get_docstring(node) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(stripped)


CODE = {name: _code_only(tree) for name, tree in TREE.items()}


def call_count(tree: ast.AST, name: str) -> int:
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
               and ((isinstance(n.func, ast.Name) and n.func.id == name)
                    or (isinstance(n.func, ast.Attribute) and n.func.attr == name)))


def keyword_names(tree: ast.AST) -> set:
    return {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords
            if k.arg}


def called_names(tree: ast.AST) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def attr_names(tree: ast.AST) -> set:
    return {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}


@pytest.fixture(scope="module")
def meta_model():
    """JetMoE instantiated on meta device: real module graph, no weights, no GPU."""
    from transformers import AutoConfig, JetMoeForCausalLM
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    with torch.device("meta"):
        model = JetMoeForCausalLM(cfg)
    return model


def synth_ids(n=64, seed=0):
    """Random native-style Top-2 selections: (n, 24, 2), distinct experts per layer."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, N_LAYERS, TOP_K), dtype=np.int64)
    for i in range(n):
        for l in range(N_LAYERS):
            out[i, l] = rng.choice(NUM_EXPERTS, size=TOP_K, replace=False)
    return out


# ------------------------------------------------------------------ A, B: pinned provenance


def test_A_checkpoint_commit_is_pinned_and_base():
    assert MODEL_ID == "jetmoe/jetmoe-8b", "must be the BASE checkpoint, not SFT/chat"
    assert "sft" not in MODEL_ID.lower() and "chat" not in MODEL_ID.lower()
    assert len(MODEL_REVISION) == 40 and all(c in "0123456789abcdef" for c in MODEL_REVISION)
    assert MODEL_REVISION != "main"
    # Every load site passes the pinned revision rather than floating main.
    for name in ("extract", "build_manifest", "record_provenance"):
        src = SRC.get(name) or open(os.path.join(HERE, f"{name}.py")).read()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr == "from_pretrained":
                kw = {k.arg for k in node.keywords}
                assert "revision" in kw, f"{name}: from_pretrained without pinned revision"


def test_A2_provenance_records_the_same_commit():
    with open(os.path.join(ART, "model_provenance.json")) as fh:
        prov = json.load(fh)
    assert prov["model_id"] == MODEL_ID
    assert prov["model_revision"] == MODEL_REVISION
    assert prov["revision_is_immutable_sha"] is True
    assert prov["checkpoint_kind"].startswith("BASE")
    assert prov["quantized"] is False
    assert prov["dtype"] == "float16"
    assert prov["gradients_on_model"] is False


def test_B_implementation_is_official_and_hashed():
    """The running interpreter must be the one preregistration recorded.

    Provenance pinned an exact implementation file and its sha256. If this suite runs under
    a different environment, the pinned file is not the file that would be executed, so the
    pin is void: fail loudly rather than validate the wrong code.
    """
    import transformers
    from transformers.models.jetmoe import modeling_jetmoe
    with open(os.path.join(ART, "model_provenance.json")) as fh:
        prov = json.load(fh)
    assert prov["implementation"] == rmc.IMPLEMENTATION
    assert prov["implementation_rewritten"] is False
    assert prov["third_party_jetmoe_installed"] is False
    assert "jetmoe" in prov["implementation_file"]
    assert transformers.__version__ == prov["implementation_version"], (
        f"running transformers {transformers.__version__}, preregistration pinned "
        f"{prov['implementation_version']}; run with the recorded interpreter")
    assert modeling_jetmoe.__file__ == prov["implementation_file"], (
        f"running {modeling_jetmoe.__file__}, preregistration pinned "
        f"{prov['implementation_file']}")
    assert prov["implementation_file_sha256"] == rmc.sha256_file(modeling_jetmoe.__file__), \
        "installed JetMoE implementation changed since preregistration"


def test_B2_model_implementation_is_not_rewritten():
    """No local redefinition of JetMoE modules or monkeypatching of the official code."""
    for name, tree in TREE.items():
        classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert not any("JetMoe" in c for c in classes), f"{name} redefines a JetMoE class"
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "setattr":
                pytest.fail(f"{name} uses setattr, possible monkeypatch of the model")


# ---------------------------------------------------------------- C: frozen / eval / FP16


def test_C_model_is_frozen_eval_fp16_inference_mode():
    tree, src = TREE["extract"], SRC["extract"]
    calls, attrs = called_names(tree), attr_names(tree)
    assert "eval" in calls, "model.eval() not called"
    assert "requires_grad_" in calls, "parameters not frozen"
    assert "inference_mode" in attrs, "forward passes not under torch.inference_mode"
    assert "float16" in attrs, "FP16 dtype not requested"
    assert "no_grad" not in calls, "use inference_mode, not no_grad"
    # No training, no optimisation, no quantisation anywhere.
    for banned in ("train", "backward", "step", "zero_grad"):
        assert banned not in calls, f"extract.py calls {banned}"
    for banned in ("load_in_8bit", "load_in_4bit", "quantization_config", "bnb"):
        assert banned not in src, f"extract.py mentions {banned}: quantisation is forbidden"


def test_C2_no_model_parallel_or_device_map():
    src = SRC["extract"]
    assert "device_map" not in src, "no model parallelism without explicit authorization"
    assert "accelerate" not in src
    assert "cuda:0" in src, "each worker uses exactly one visible GPU"


# ------------------------------------------------- D: captured router is the MLP MoE, not MoA


def test_D_mlp_and_moa_routers_are_distinct_objects(meta_model):
    from transformers.models.jetmoe.modeling_jetmoe import JetMoeMoA, JetMoeMoE
    layers = meta_model.model.layers
    assert len(layers) == N_LAYERS
    for i, block in enumerate(layers):
        assert isinstance(block.mlp, JetMoeMoE)
        assert isinstance(block.self_attention.experts, JetMoeMoA)
        mlp_r = block.mlp.router
        moa_r = block.self_attention.experts.router
        assert mlp_r is not moa_r, f"block {i}: MLP and MoA share a router object"
        assert id(mlp_r) != id(moa_r)


def test_D2_shape_alone_cannot_distinguish_the_two_routers(meta_model):
    """The reason identity checks are mandatory: both routers are Linear(2048 -> 8)."""
    block = meta_model.model.layers[0]
    mlp_r, moa_r = block.mlp.router, block.self_attention.experts.router
    assert type(mlp_r) is type(moa_r), "both mixtures use JetMoeTopKGating"
    assert (mlp_r.layer.in_features, mlp_r.layer.out_features) == \
           (moa_r.layer.in_features, moa_r.layer.out_features) == (rmc.HIDDEN_SIZE, NUM_EXPERTS)


def test_D3_extractor_hooks_only_mlp_routers(meta_model):
    routers = extract.mlp_routers(meta_model)
    assert len(routers) == N_LAYERS
    mlp_set = {id(b.mlp.router) for b in meta_model.model.layers}
    moa_set = {id(b.self_attention.experts.router) for b in meta_model.model.layers}
    got = {id(r) for r in routers}
    assert got == mlp_set, "captured routers are not exactly the MLP routers"
    assert got.isdisjoint(moa_set), "an MoA router was captured"


class _Stub:
    """A stand-in model whose block structure can be deformed without touching the real one."""

    class _Inner:
        def __init__(self, layers):
            self.layers = layers

    def __init__(self, layers):
        self.model = self._Inner(layers)


class _Block:
    def __init__(self, mlp, attn_experts):
        self.mlp = mlp
        self.self_attention = type("A", (), {"experts": attn_experts})()


def test_D4_router_identification_blocks_rather_than_guesses(meta_model):
    """If the structure is not exactly as expected, extraction stops; it does not fall back.

    Four deformations, each of which a name-matching implementation would sail past.
    """
    real = list(meta_model.model.layers)
    good = [_Block(b.mlp, b.self_attention.experts) for b in real]

    # The unmodified stub structure is accepted, so the failures below are about the defect.
    assert len(extract.mlp_routers(_Stub(good))) == N_LAYERS

    class Wrong(torch.nn.Module):
        pass

    cases = {}
    bad_mlp = list(good)
    bad_mlp[3] = _Block(Wrong(), real[3].self_attention.experts)
    cases["mlp is not JetMoeMoE"] = bad_mlp

    bad_moa = list(good)
    bad_moa[5] = _Block(real[5].mlp, Wrong())
    cases["attention experts is not JetMoeMoA"] = bad_moa

    shared = list(good)                              # MoA router smuggled in as the MLP's
    shared[7] = _Block(real[7].mlp, real[7].self_attention.experts)
    shared[7].mlp = real[7].self_attention.experts    # same object for both mixtures
    cases["shared router"] = shared

    cases["wrong block count"] = good[:-1]

    for label, layers in cases.items():
        with pytest.raises(SystemExit) as e:
            extract.mlp_routers(_Stub(layers))
        assert rmc.ROUTER_BLOCKER in str(e.value), f"{label} did not raise the blocker"

    # And the router is reached by attribute, never by searching module names.
    code = CODE["extract"]
    for guess in ("named_modules", "named_children", "startswith", "re.match", "re.search"):
        assert guess not in code, f"router located by name guessing ({guess})"


GPU_EVIDENCE = os.path.join(ART, "router_identification.json")


@pytest.mark.skipif(os.environ.get("RMC_GPU_CHECK") != "1",
                    reason="GPU smoke step; run once with RMC_GPU_CHECK=1")
def test_D5_gpu_capture_equals_independently_recomputed_mlp_logits():
    """The decisive check, on real weights: recompute the MLP router logits by hand.

    For every block, the captured tensor is compared against `W_mlp_router @ x` where x is the
    hidden state actually entering that block's `.mlp`, and separately against the MoA
    router's own logits. The first difference must be exactly zero and the second must not be.
    Writes the evidence to artifacts/ so the claim survives past this process.
    """
    import torch.nn.functional as F
    from transformers import JetMoeForCausalLM

    assert torch.cuda.is_available(), "GPU check requested but no CUDA device is visible"
    model = JetMoeForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16,
        low_cpu_mem_usage=True).to("cuda:0").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    routers = extract.mlp_routers(model)
    blocks = model.model.layers

    mlp_in, moa_logits, handles = {}, {}, []
    for i, b in enumerate(blocks):
        handles.append(b.mlp.register_forward_pre_hook(
            lambda m, a, i=i: mlp_in.__setitem__(i, a[0].detach())))
        handles.append(b.self_attention.experts.router.register_forward_hook(
            lambda m, a, o, i=i: moa_logits.__setitem__(i, o[-1].detach())))

    bsz = 4
    ids = torch.randint(100, 30000, (bsz, rmc.CONTEXT_LEN), device="cuda:0")
    with torch.inference_mode():
        with extract.RouterCapture(routers, rmc.CONTEXT_LEN, EXPERIMENTAL_POS) as cap:
            model(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()

    max_mlp_diff, min_moa_diff = 0.0, float("inf")
    for i, b in enumerate(blocks):
        x = mlp_in[i].view(-1, rmc.HIDDEN_SIZE)
        mine = F.linear(x, b.mlp.router.layer.weight).float()
        mine = mine.view(bsz, rmc.CONTEXT_LEN, -1)[:, EXPERIMENTAL_POS, :].cpu()
        max_mlp_diff = max(max_mlp_diff, float((mine - cap.logits[i]).abs().max()))
        moa = moa_logits[i].view(bsz, rmc.CONTEXT_LEN, -1)[:, EXPERIMENTAL_POS, :].float().cpu()
        min_moa_diff = min(min_moa_diff, float((moa - cap.logits[i]).abs().max()))

    ids_out = np.stack([cap.ids[l].numpy() for l in range(N_LAYERS)], 1)
    lg = np.stack([cap.logits[l].numpy() for l in range(N_LAYERS)], 1)
    rec = np.argsort(-lg, -1)[:, :, :TOP_K]
    agreement = float((np.sort(rec, -1) == np.sort(ids_out, -1)).all(-1).mean())

    assert max_mlp_diff == 0.0, "captured logits are not the MLP router's own output"
    assert min_moa_diff > 1e-2, "captured logits are indistinguishable from the MoA router"
    assert agreement == 1.0

    with open(GPU_EVIDENCE, "w") as fh:
        json.dump({
            "experiment": rmc.EXPERIMENT,
            "device": torch.cuda.get_device_name(0),
            "n_blocks_checked": N_LAYERS,
            "max_abs_diff_vs_recomputed_mlp_router_logits": max_mlp_diff,
            "min_abs_diff_vs_moa_router_logits": min_moa_diff,
            "topk_identity_logit_agreement": agreement,
            "single_gpu_fp16_load": True,
            "gib_allocated": torch.cuda.memory_allocated() / 2 ** 30,
            "builtin_output_router_logits_status": (
                "raises AttributeError in transformers 4.45.1 "
                "(JetMoeForCausalLM has no attribute num_experts); path unused by design"),
        }, fh, indent=2)


def test_D6_gpu_router_evidence_is_recorded_and_exact():
    """The GPU evidence must exist and be unambiguous before results are believed."""
    if not os.path.exists(GPU_EVIDENCE):
        pytest.skip("GPU router evidence not collected yet")
    with open(GPU_EVIDENCE) as fh:
        ev = json.load(fh)
    assert ev["n_blocks_checked"] == N_LAYERS
    assert ev["max_abs_diff_vs_recomputed_mlp_router_logits"] == 0.0
    assert ev["min_abs_diff_vs_moa_router_logits"] > 1e-2
    assert ev["topk_identity_logit_agreement"] == 1.0
    assert ev["single_gpu_fp16_load"] is True


# ---------------------------------------------------- E: captured IDs reproduce native Top-2


def test_E_capture_reproduces_the_modules_own_selection():
    """Run a real JetMoeTopKGating and compare hook output against its native Top-2.

    The gating module is small enough to instantiate with real weights on CPU, so this is a
    genuine check of the capture path, not a restatement of it.
    """
    from transformers.models.jetmoe.modeling_jetmoe import JetMoeTopKGating

    torch.manual_seed(0)
    gate = JetMoeTopKGating(rmc.HIDDEN_SIZE, NUM_EXPERTS, TOP_K).eval()
    bsz, seq = 3, 16
    x = torch.randn(bsz * seq, rmc.HIDDEN_SIZE)

    with torch.inference_mode():
        native = gate(x)
    native_logits = native[-1].view(bsz, seq, NUM_EXPERTS)[:, EXPERIMENTAL_POS % seq, :]
    native_ids = native_logits.topk(TOP_K, dim=-1).indices

    cap = extract.RouterCapture([gate], seq_len=seq, pos=EXPERIMENTAL_POS % seq)
    with cap, torch.inference_mode():
        gate(x)
    assert torch.allclose(cap.logits[0], native_logits.float())
    assert torch.equal(cap.ids[0], native_ids)
    assert cap.logits[0].shape == (bsz, NUM_EXPERTS)
    assert cap.ids[0].shape == (bsz, TOP_K)


def test_E2_extractor_verifies_agreement_and_refuses_disagreement():
    src, calls = SRC["extract"], called_names(TREE["extract"])
    assert "argsort" in calls, "no independent recomputation of Top-2 from logits"
    assert "agree != 1.0" in src, "agreement is not required to be exact"


def test_E3_selection_state_matches_the_captured_ids():
    ids = synth_ids(32, seed=1)
    S = analyze.selection_state(ids, (7, 8))
    assert S.shape == (32, 2 * NUM_EXPERTS)
    for i in range(32):
        for slot, l in enumerate((7, 8)):
            block = S[i, slot * NUM_EXPERTS:(slot + 1) * NUM_EXPERTS]
            assert set(np.flatnonzero(block)) == set(ids[i, l - 1].tolist())


# ----------------------------------------------------- F: S_l is 8 dims with exactly two ones


def test_F_selection_state_is_binary_8_dims_two_ones():
    ids = synth_ids(128, seed=2)
    for l in (1, 5, 11, 19, 24):
        S = analyze.selection_state(ids, (l,))
        assert S.shape == (128, NUM_EXPERTS)
        assert set(np.unique(S)) <= {0.0, 1.0}, "indicator must be binary"
        assert np.all(S.sum(axis=1) == TOP_K), "exactly two ones per layer"


def test_F2_no_gate_values_or_ranks_enter_the_feature():
    """Only identities: the analysis never touches probabilities, gates, or rank order."""
    code, calls = CODE["analyze"], called_names(TREE["analyze"])
    for banned in ("softmax", "batch_gates", "top_k_gates", "expert_size"):
        assert banned not in code, f"analyze.py uses {banned}"
    assert "argsort" not in calls, "rank ordering must not enter the representation"
    # selection_state assigns a constant 1.0, never a magnitude.
    fn = ast.parse(inspect.getsource(analyze.selection_state))
    consts = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant)
              and isinstance(n.value, float)}
    assert consts <= {1.0, 0.0}


# -------------------------------------------------------------- G: MoA routing never an input


def test_G_moa_is_only_ever_referenced_to_exclude_it():
    """analyze.py touches no attention-mixture object at all; extract.py only guards on one."""
    a_code = CODE["analyze"]
    for banned in ("MoA", "self_attention", "JetMoe"):
        assert banned not in a_code, f"analyze.py uses {banned}"
    # In extract.py the MoA appears solely inside the identity guard.
    for node in ast.walk(TREE["extract"]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "register_forward_hook":
            assert isinstance(node.func.value, ast.Name) and node.func.value.id == "r"


def test_G2_builtin_interleaved_router_logits_path_is_unused():
    """`output_router_logits=True` interleaves MoA and MLP logits, so it is never used."""
    for name in ("extract", "analyze"):
        assert "output_router_logits" not in CODE[name], f"{name}.py uses the builtin path"
    assert "output_router_logits" not in keyword_names(TREE["extract"])
    with open(os.path.join(ART, "model_provenance.json")) as fh:
        prov = json.load(fh)
    assert prov["builtin_output_router_logits_used"] is False
    assert prov["moa_router_used"] is False
    assert prov["router_studied"].lower().startswith("mlp")


# --------------------------------------------------------- H: 512 / 256, disjoint, one manifest


def test_H_manifest_sizes_disjoint_and_fingerprinted():
    with open(os.path.join(ART, "data_manifest.json")) as fh:
        man = json.load(fh)
    fit, test = man["fit_index"], man["test_index"]
    assert man["n_fit"] == len(fit) == N_FIT
    assert man["n_test"] == len(test) == N_TEST
    assert len(set(fit)) == N_FIT and len(set(test)) == N_TEST, "duplicate blocks"
    assert set(fit).isdisjoint(set(test)), "FIT and TEST overlap"
    assert man["block_len"] == BLOCK_LEN
    assert man["experimental_pos"] == EXPERIMENTAL_POS
    assert man["sample_seed"] == rmc.SAMPLE_SEED
    assert all(len(f) == 64 for f in (man["fit_fingerprint"], man["test_fingerprint"]))


def test_H2_no_prior_project_block_indices_are_reused():
    with open(os.path.join(ART, "data_manifest.json")) as fh:
        man = json.load(fh)
    assert man["reused_prior_block_indices"] is False
    assert man["model_id"] == MODEL_ID and man["model_revision"] == MODEL_REVISION

    # The tokenizer is the pinned checkpoint's own. JetMoE ships a Llama tokenizer class, so
    # the class name proves nothing; what matters is that it came from this model_id and that
    # the blocks it produced still hash to the frozen fingerprint.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    assert type(tok).__name__ == man["tokenizer_name"]
    assert tok.eos_token_id is not None
    assert man["dataset_revision"] == rmc.DATASET_REVISION
    # No executable statement reads or writes anything belonging to a closed project.
    # Recorded provenance text may *name* a prior model to state it was not used (the manifest
    # note says OLMoE indices were not reused), so string literals that are only stored as
    # data are exempt; anything a path or an import could be built from is not.
    PRIOR = ("expert_provenance", "routing_markov_order", "cross_layer_expert_cache",
             "olmoe", "allenai")
    for name, tree in TREE.items():
        literals = {n for n in ast.walk(tree) if isinstance(n, ast.Constant)
                    and isinstance(n.value, str)}
        recorded = {n for n in literals if len(n.value.split()) > 3}   # prose, not a path
        for node in literals - recorded:
            low = node.value.lower()
            for prior in PRIOR:
                assert prior not in low, f"{name}.py has literal {node.value!r}"
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = getattr(node, "module", "") or ""
                names = [a.name for a in node.names]
                for prior in PRIOR:
                    assert prior not in mod.lower(), f"{name}.py imports {mod}"
                    assert not any(prior in n.lower() for n in names)
    # Every path this project builds is inside this project.
    for name, tree in TREE.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and node.value.startswith("/home/"):
                assert node.value.startswith(HERE), f"{name}.py hardcodes {node.value}"


def test_H3_shards_tile_fit_and_test_exactly():
    seen_fit, seen_test = [], []
    for s in range(rmc.N_SHARDS):
        seen_fit += list(range(s * rmc.SHARD_FIT, (s + 1) * rmc.SHARD_FIT))
        seen_test += list(range(s * rmc.SHARD_TEST, (s + 1) * rmc.SHARD_TEST))
    assert seen_fit == list(range(N_FIT))
    assert seen_test == list(range(N_TEST))


# ------------------------------------------------------------- I, J: target layers and target


def test_I_target_layers_are_exactly_12_and_20():
    assert TARGET_LAYERS == (12, 20)
    for m in TARGET_LAYERS:
        assert 1 <= m <= N_LAYERS
        assert m - max(K_VALUES) >= 1, f"target {m} cannot support k={max(K_VALUES)}"
    assert K_VALUES == (1, 2, 4, 8) and PRIMARY_K == 4


def test_I2_history_windows_are_contiguous_and_end_at_m_minus_1():
    for m in TARGET_LAYERS:
        for k in K_VALUES:
            h = history_layers(m, k)
            assert len(h) == k and h[-1] == m - 1
            assert list(h) == list(range(m - k, m))
            assert m not in h, "the target layer is never its own input"
            o = older_layers(m, k)
            assert (m - 1) not in o
            assert len(o) == k - 1
    assert older_layers(12, 1) == () and older_layers(20, 1) == ()


def test_J_target_is_native_mlp_logits_centred_then_fit_standardized():
    rng = np.random.default_rng(3)
    g = rng.normal(size=(40, NUM_EXPERTS)) * 3 + 7
    q = analyze.center_logits(g)
    assert np.allclose(q.mean(axis=1), 0.0), "logits must be centred per sample"
    assert q.shape == g.shape
    with pytest.raises(ValueError):
        analyze.center_logits(rng.normal(size=(4, 64)))
    # The target comes from the captured MLP logits array and nothing else.
    assert 'logits[:, m - 1, :]' in SRC["analyze"]


# --------------------------------------- K: recent-layer preprocessing fitted once and reused


def test_K_recent_scaler_is_fitted_once_per_target_and_reused():
    """One StandardScaler().fit per target for z_recent, outside the loop over k."""
    fn = next(n for n in ast.walk(TREE["analyze"])
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    target_loop = next(n for n in ast.walk(fn) if isinstance(n, ast.For)
                       and isinstance(n.target, ast.Name) and n.target.id == "m")
    k_loop = next(n for n in ast.walk(target_loop) if isinstance(n, ast.For)
                  and isinstance(n.target, ast.Name) and n.target.id == "k")
    inner = {id(n) for n in ast.walk(k_loop)}

    def fits(node):
        return [n for n in ast.walk(node) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in ("fit", "fit_transform")]

    assert not any(id(c) in inner for c in fits(target_loop)
                   if "recent" in ast.dump(c) or "rsc" in ast.dump(c)), \
        "recent-layer scaler fitted inside the k loop"
    assert len([c for c in fits(target_loop) if id(c) not in inner]) >= 2, \
        "expected the target scaler and the recent scaler fitted once per target"
    assert "rsc = StandardScaler().fit(s_recent_fit)" in SRC["analyze"]


def test_K2_recent_representation_is_identical_across_k():
    """Numerically: the first 8 feature dims do not depend on k."""
    from sklearn.preprocessing import StandardScaler
    ids = synth_ids(96, seed=4)
    m = 12
    s_recent = analyze.selection_state(ids, (m - 1,))
    z_recent = StandardScaler().fit_transform(s_recent)
    feats = []
    for k in K_VALUES:
        old = older_layers(m, k)
        if old:
            X = analyze.selection_state(ids, old)
            z_old = analyze.OlderBlock.fit(X).transform(X)
        else:
            z_old = None
        feats.append(analyze.feature(z_recent, z_old))
    for f in feats[1:]:
        assert np.array_equal(f[:, :RECENT_DIM], feats[0][:, :RECENT_DIM])


# ------------------------------------------------------------------- L, O: FIT-only fitting


def test_L_older_history_pca_uses_fit_only():
    ids = synth_ids(200, seed=5)
    X = analyze.selection_state(ids, older_layers(20, 8))
    fit, test = X[:150], X[150:]
    block = analyze.OlderBlock.fit(fit)
    assert block.pca.n_components == PCA_COMPONENTS
    assert block.pca.random_state == PCA_SEED
    assert block.pca.svd_solver == "randomized"
    # Refitting on FIT alone reproduces the transform of TEST exactly.
    again = analyze.OlderBlock.fit(fit)
    assert np.allclose(block.transform(test), again.transform(test))
    # Including TEST in the fit would change it, which is what FIT-only prevents.
    contaminated = analyze.OlderBlock.fit(X)
    assert not np.allclose(block.transform(test), contaminated.transform(test))


def test_L2_no_pca_sweep_and_no_target_in_compression():
    tree, code = TREE["analyze"], CODE["analyze"]
    assert call_count(tree, "PCA") == 1, "exactly one PCA configuration"
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "PCA":
            kw = {k.arg for k in node.keywords}
            assert kw == {"n_components", "svd_solver", "random_state"}
            assert not node.args, "PCA configured only by keyword"
    fn = _code_only(ast.parse(inspect.getsource(analyze.OlderBlock)))
    for banned in ("Y_", "q_", "logits", "y_true", "target"):
        assert banned not in fn, f"compression sees {banned}"
    for banned in ("GridSearch", "RandomizedSearch", "cross_val", "ParameterGrid"):
        assert banned not in code, f"analyze.py contains a search ({banned})"


def test_O_no_test_statistics_enter_preprocessing():
    """Every scaler/PCA fit in analyze.py is fed a FIT-only array."""
    for node in ast.walk(TREE["analyze"]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("fit", "fit_transform"):
            args = ast.dump(ast.Module(body=[ast.Expr(a) for a in node.args],
                                       type_ignores=[]))
            if "Ridge" in ast.dump(node.func.value):
                continue
            assert "test" not in args.lower(), f"preprocessing fitted on {args}"
    # Ridge is likewise only ever fitted on FIT.
    fit_probe = inspect.getsource(analyze.fit_probe)
    assert "test" not in fit_probe.lower()
    assert "model.fit(X_fit, Y_fit)" in SRC["analyze"]


def test_O2_predictions_only_ever_come_from_test_features():
    assert "preds[k] = model.predict(Xt)" in SRC["analyze"]
    assert SRC["analyze"].count(".predict(") == 1


# -------------------------------------------------------------- M, N: 16 dims, alpha = 1.0


def test_M_every_feature_is_exactly_16_dims():
    ids = synth_ids(64, seed=6)
    for m in TARGET_LAYERS:
        z_recent = analyze.selection_state(ids, (m - 1,))
        for k in K_VALUES:
            old = older_layers(m, k)
            if old:
                X = analyze.selection_state(ids, old)
                z_old = analyze.OlderBlock.fit(X).transform(X)
                assert z_old.shape[1] == PCA_COMPONENTS
            else:
                z_old = None
            F = analyze.feature(z_recent, z_old)
            assert F.shape == (64, FEATURE_DIM) == (64, 16)


def test_M2_k1_baseline_is_zero_padded_not_narrower():
    ids = synth_ids(48, seed=7)
    z_recent = analyze.selection_state(ids, (11,))
    F = analyze.feature(z_recent, None)
    assert F.shape[1] == FEATURE_DIM
    assert np.all(F[:, RECENT_DIM:] == 0.0), "k=1 padding must be exactly zeros"
    assert np.array_equal(F[:, :RECENT_DIM], z_recent)


def test_M3_feature_rejects_a_wrong_width():
    z = np.zeros((5, RECENT_DIM))
    with pytest.raises(AssertionError):
        analyze.feature(z, np.zeros((5, PCA_COMPONENTS + 1)))


def test_N_ridge_alpha_is_one_for_all_eight_probes():
    assert ALPHA == 1.0
    assert call_count(TREE["analyze"], "Ridge") == 1, \
        "one Ridge construction shared by all probes"
    for node in ast.walk(TREE["analyze"]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "Ridge":
            kw = {k.arg: k.value for k in node.keywords}
            assert set(kw) == {"alpha"} and kw["alpha"].id == "ALPHA"
    from sklearn.linear_model import Ridge
    assert Ridge(alpha=ALPHA).alpha == 1.0


def test_N2_eight_probes_exactly():
    assert len(TARGET_LAYERS) * len(K_VALUES) == 8


def test_N3_metric_is_uniform_average_r2():
    assert call_count(TREE["analyze"], "r2_score") == 1, "one scoring path for all probes"
    for node in ast.walk(TREE["analyze"]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "r2_score":
            kw = {k.arg: k.value for k in node.keywords}
            assert isinstance(kw["multioutput"], ast.Constant)
            assert kw["multioutput"].value == "uniform_average"
    # And the metric is exercised: an exact prediction scores 1.
    Y = np.random.default_rng(9).normal(size=(20, NUM_EXPERTS))
    assert analyze.score(Y, Y.copy()) == pytest.approx(1.0)


# ----------------------------------------------------------------- bootstrap and verdict


def test_bootstrap_resamples_indices_jointly_for_both_models():
    src = inspect.getsource(analyze.paired_bootstrap)
    assert "rng.integers" in src
    assert src.count("idx") >= 4
    assert "pred_k[idx]" in src and "pred_1[idx]" in src and "Y_test[idx]" in src
    tree = ast.parse(src)
    n_integers = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == "integers")
    assert n_integers == 1, "one index draw per resample, shared by both models"
    assert rmc.N_BOOTSTRAP == 10000 and rmc.BOOTSTRAP_SEED == 314159


def test_bootstrap_is_deterministic_and_paired():
    rng = np.random.default_rng(11)
    Y = rng.normal(size=(120, NUM_EXPERTS))
    p1 = Y + rng.normal(scale=1.0, size=Y.shape)
    p4 = Y + rng.normal(scale=0.5, size=Y.shape)      # strictly better model
    a = analyze.paired_bootstrap(Y, p1, p4, n_resamples=300, seed=1)
    b = analyze.paired_bootstrap(Y, p1, p4, n_resamples=300, seed=1)
    assert a == b, "bootstrap must be reproducible from its seed"
    assert a["ci_low"] < a["ci_high"]
    assert a["ci_low"] > 0, "a uniformly better model should give a positive CI"
    # Identical predictions give a degenerate zero difference, never spurious signal.
    same = analyze.paired_bootstrap(Y, p1, p1, n_resamples=200, seed=2)
    assert same["ci_low"] == same["ci_high"] == 0.0


def _target(r2_k1=0.4, delta4=0.05, ci_low=0.01):
    return {"R2": {"k1": r2_k1}, "Delta4": delta4, "bootstrap_Delta4": {"ci_low": ci_low}}


def test_verdict_rule_is_the_six_preregistered_conditions():
    """Exercise the rule itself, on synthetic inputs, before any real number exists."""
    assert DELTA4_THRESHOLD == 0.02

    passing = {m: _target() for m in TARGET_LAYERS}
    conds, verdict = analyze.verdict_conditions(passing)
    assert len(conds) == 6, "three conditions per target, two targets"
    assert set(conds) == {f"L{m}_{c}" for m in TARGET_LAYERS for c in
                          ("R2_k1_positive", "Delta4_at_least_threshold",
                           "bootstrap_ci_low_above_zero")}
    assert all(conds.values()) and verdict == rmc.REPLICATED

    # Each of the six conditions alone is sufficient to deny REPLICATED.
    for m in TARGET_LAYERS:
        for kwargs in ({"r2_k1": 0.0}, {"delta4": 0.019}, {"ci_low": 0.0}):
            case = {t: _target() for t in TARGET_LAYERS}
            case[m] = _target(**kwargs)
            c, v = analyze.verdict_conditions(case)
            assert v == rmc.NOT_REPLICATED, f"L{m} {kwargs} still gave {v}"
            assert sum(1 for x in c.values() if not x) == 1

    # The threshold is a boundary that includes equality, exactly as written.
    boundary = {m: _target(delta4=DELTA4_THRESHOLD) for m in TARGET_LAYERS}
    assert analyze.verdict_conditions(boundary)[1] == rmc.REPLICATED
    just_under = {m: _target(delta4=DELTA4_THRESHOLD - 1e-9) for m in TARGET_LAYERS}
    assert analyze.verdict_conditions(just_under)[1] == rmc.NOT_REPLICATED

    # A CI touching zero is not above zero.
    assert analyze.verdict_conditions(
        {m: _target(ci_low=0.0) for m in TARGET_LAYERS})[1] == rmc.NOT_REPLICATED


def test_verdict_reads_only_the_three_preregistered_quantities():
    """The rule cannot consult k=2, k=8, or anything else not named in section 16."""
    fn = ast.parse(inspect.getsource(analyze.verdict_conditions))
    keys = {n.slice.value for n in ast.walk(fn) if isinstance(n, ast.Subscript)
            and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}
    assert keys == {"R2", "k1", "Delta4", "bootstrap_Delta4", "ci_low"}
    assert "Delta2" not in keys and "Delta8" not in keys


def test_verdict_has_no_inconclusive_category_for_numbers():
    for name, code in CODE.items():
        assert "INCONCLUSIVE" not in code.upper(), f"{name}.py has a third numeric outcome"
    assert rmc.REPLICATED == "REPLICATED" and rmc.NOT_REPLICATED == "NOT_REPLICATED"
    # Blockers are a separate, technical channel.
    assert rmc.ROUTER_BLOCKER not in (rmc.REPLICATED, rmc.NOT_REPLICATED)
    assert rmc.MEMORY_BLOCKER not in (rmc.REPLICATED, rmc.NOT_REPLICATED)


def test_thresholds_are_module_constants_not_inline_literals():
    """A frozen threshold cannot be quietly edited at one of several call sites."""
    for node in ast.walk(TREE["analyze"]):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Subscript):
            for c in node.comparators:
                if isinstance(c, ast.Constant) and isinstance(c.value, float):
                    assert c.value == 0.0, f"inline float threshold {c.value}"


# ------------------------------------------------------------------------- P: the gate


def test_P_analysis_refuses_to_compute_before_checks_pass():
    src = SRC["analyze"]
    assert "require_checks_passed" in src
    fn = next(n for n in ast.walk(TREE["analyze"])
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    body_calls = []
    for stmt in fn.body:
        body_calls += [n.func.id for n in ast.walk(stmt)
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        if "require_checks_passed" in body_calls:
            break
    for banned in ("fit_probe", "score", "paired_bootstrap"):
        assert banned not in body_calls, f"{banned} runs before the check gate"
    gate = inspect.getsource(analyze.require_checks_passed)
    assert "pytest" in gate and "returncode" in gate and "SystemExit" in gate


def test_P2_no_statistics_exist_in_extraction_or_merge():
    for name in ("extract", "merge"):
        calls = called_names(TREE[name])
        for banned in ("Ridge", "PCA", "r2_score", "fit", "fit_transform",
                       "paired_bootstrap", "StandardScaler"):
            assert banned not in calls, f"{name}.py computes {banned}"


def test_P3_merge_validates_against_the_manifest_only():
    code, calls = CODE["merge"], called_names(TREE["merge"])
    assert "fit_index" in code and "test_index" in code, "merge does not check the manifest"
    assert "unique" in calls, "merge does not reject duplicate blocks"
    assert "R2" not in code and "r2_score" not in code
    # Merge is pure bookkeeping: concatenate, assert, write.
    assert call_count(TREE["merge"], "concatenate") >= 1
    assert sum(1 for n in ast.walk(TREE["merge"]) if isinstance(n, ast.Assert)) >= 5


def test_P4_results_absent_or_consistent_with_the_frozen_protocol():
    """Before the run there is no results.json; after it, it must match the protocol."""
    path = os.path.join(ART, "results.json")
    if not os.path.exists(path):
        pytest.skip("no results yet, as expected before the single analysis run")
    with open(path) as fh:
        res = json.load(fh)
    assert res["model_revision"] == MODEL_REVISION
    assert res["target_layers"] == list(TARGET_LAYERS)
    assert res["k_values"] == list(K_VALUES)
    assert res["delta4_threshold"] == DELTA4_THRESHOLD
    assert res["feature_dim"] == FEATURE_DIM
    assert res["verdict"] in (rmc.REPLICATED, rmc.NOT_REPLICATED)
    assert len(res["conditions"]) == 6
    assert res["moa_router_used"] is False
    assert res["verdict"] == (rmc.REPLICATED if all(res["conditions"].values())
                              else rmc.NOT_REPLICATED)
