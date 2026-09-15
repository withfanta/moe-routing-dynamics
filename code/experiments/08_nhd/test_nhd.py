"""NHD-P0 implementation checks, tests A through S of protocol.md section 12.

These run before any TEST statistic exists (test S is the gate inside analyze.py). Checks on
code properties walk the AST for names actually used, never the prose, so a test cannot pass
or fail on its own wording. Checks on numerical behaviour are exercised on synthetic arrays.
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
import nhd
from nhd import (
    ALPHA, ART, CROSSFIT_SEED, FIT_SPLIT_SEED, FULL_DIM, HERE, HIDDEN, HISTORY_DIM,
    HISTORY_OFFSETS, INIT_SEEDS, MATCHED_WIDTH, MAX_EPOCHS, NEAR_ZERO, NUM_EXPERTS, N_FIT,
    N_FOLD, N_LAYERS, N_TEST, N_TRAIN, N_VAL, PATIENCE, PERMUTE_SEED, RECENT_DIM,
    SOURCE_DIR, TARGET_LAYERS, TOP_K, history_layers, recent_layer,
)

MODULES = ("nhd", "analyze")
SRC = {name: open(os.path.join(HERE, f"{name}.py")).read() for name in MODULES}
TREE = {name: ast.parse(src) for name, src in SRC.items()}


def _code_only(tree: ast.AST) -> str:
    """The module with docstrings and comments removed.

    Scanning raw source for banned words makes a test fail on legitimate explanatory prose:
    analyze.py must be able to record `pca_used: False` without that counting as using PCA.
    """
    stripped = ast.parse(ast.unparse(tree))
    for node in ast.walk(stripped):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)) and ast.get_docstring(node) is not None:
            node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(stripped)


CODE = {name: _code_only(tree) for name, tree in TREE.items()}


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


def call_count(tree: ast.AST, name: str) -> int:
    return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
               and ((isinstance(n.func, ast.Name) and n.func.id == name)
                    or (isinstance(n.func, ast.Attribute) and n.func.attr == name)))


def synth_ids(n=64, seed=0):
    """Random native-style Top-2 selections: (n, 24, 2), two distinct experts per layer."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, N_LAYERS, TOP_K), dtype=np.int64)
    for i in range(n):
        for l in range(N_LAYERS):
            out[i, l] = rng.choice(NUM_EXPERTS, size=TOP_K, replace=False)
    return out


@pytest.fixture(scope="module")
def source():
    ids, logits, is_test = nhd.load_source()
    return ids, logits, is_test


# ------------------------------------------------------- A, B: frozen source verification


def test_A_source_repository_is_unchanged():
    clean, head = nhd.source_repo_is_clean()
    assert clean, "the closed source repository has uncommitted changes"
    assert head.startswith(nhd.SOURCE_COMMIT), \
        f"source HEAD {head} is not the pinned commit {nhd.SOURCE_COMMIT}"


def test_A2_this_project_never_writes_to_the_source():
    """Every write target is built from this project's own artifacts directory."""
    for name, tree in TREE.items():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "open":
                mode = None
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                if mode and any(c in mode for c in "wax+"):
                    dumped = ast.dump(node.args[0])
                    assert "ART" in dumped, f"{name}.py writes outside artifacts: {dumped}"
                    assert "SOURCE" not in dumped
    # No filesystem mutation aimed anywhere near the closed tree.
    for name, code in CODE.items():
        for banned in ("shutil", "os.remove", "os.rename", "rmtree", "unlink", "mkdir"):
            assert banned not in code, f"{name}.py uses {banned}"


def test_B_exact_source_artifact_hashes_match():
    assert len(nhd.SOURCE_HASHES) == 4
    for rel, expected in nhd.SOURCE_HASHES.items():
        path = os.path.join(SOURCE_DIR, rel)
        assert os.path.exists(path), f"{rel} missing"
        assert len(expected) == 64
        assert nhd.sha256_file(path) == expected, f"{rel} changed since preregistration"


def test_B2_verification_runs_before_data_is_loaded():
    """load_source cannot hand back arrays without verifying the pins first."""
    src = inspect.getsource(nhd.load_source)
    body = ast.parse(src).body[0].body
    stmts = [s for s in body if not (isinstance(s, ast.Expr)
                                     and isinstance(s.value, ast.Constant))]
    first = ast.dump(stmts[0])
    assert "verify_source" in first, "load_source does not verify before reading"


def test_B3_missing_or_wrong_artifact_stops_with_the_named_code():
    src = inspect.getsource(nhd.verify_source)
    assert nhd.NOT_AVAILABLE == "REQUIRED_ARTIFACT_NOT_AVAILABLE"
    assert src.count("NOT_AVAILABLE") >= 4, "not every absence path raises the stop code"
    assert "SystemExit" in src
    # No silent re-extraction path exists anywhere in this project.
    for name, code in CODE.items():
        for banned in ("from_pretrained", "JetMoeForCausalLM", "cuda", "load_dataset"):
            assert banned not in code, f"{name}.py could rerun inference ({banned})"


def test_B4_recorded_verification_artifact_matches_the_pins():
    path = os.path.join(ART, "source_verification.json")
    assert os.path.exists(path), "source verification was not recorded"
    with open(path) as fh:
        v = json.load(fh)
    assert v["source_experiment"] == nhd.SOURCE_EXPERIMENT
    assert v["hashes"] == nhd.SOURCE_HASHES
    assert v["source_clean"] is True
    assert v["no_new_inference"] is True and v["gpu_used"] is False
    assert v["matched_width"] == MATCHED_WIDTH


# ------------------------------------------------------------------ C, D: split and targets


def test_C_fit_and_test_are_exactly_512_and_256(source):
    ids, logits, is_test = source
    assert N_FIT == 512 and N_TEST == 256
    assert ids.shape[0] == logits.shape[0] == is_test.shape[0] == N_FIT + N_TEST
    assert int((~is_test).sum()) == N_FIT
    assert int(is_test.sum()) == N_TEST


def test_C2_same_samples_as_the_source_in_the_same_order(source):
    """No resampling, reshuffling, or re-splitting of the frozen data."""
    ids, logits, is_test = source
    with np.load(os.path.join(SOURCE_DIR, "artifacts/merged.npz")) as d:
        assert np.array_equal(ids, d["ids"].astype(np.int64))
        assert np.array_equal(is_test, d["is_test"].astype(bool))
    code = CODE["analyze"]
    for banned in ("train_test_split", "KFold", "StratifiedKFold", "shuffle("):
        assert banned not in code, f"analyze.py re-splits the data ({banned})"


def test_D_target_layers_are_exactly_12_and_20():
    assert TARGET_LAYERS == (12, 20)
    assert len(TARGET_LAYERS) == 2
    for m in TARGET_LAYERS:
        assert 1 <= m <= N_LAYERS
        assert min(history_layers(m)) >= 1
    # The loop over targets iterates the frozen tuple and nothing else.
    for node in ast.walk(TREE["analyze"]):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) \
                and node.target.id == "m":
            assert isinstance(node.iter, ast.Name) and node.iter.id == "TARGET_LAYERS"


# ------------------------------------------------------------- E, F, G, H: the variables


def test_E_R_is_the_raw_8d_binary_top2_recent_state(source):
    ids, logits, is_test = source
    for m in TARGET_LAYERS:
        assert recent_layer(m) == m - 1
        R = analyze.selection_state(ids, (recent_layer(m),))
        assert R.shape == (ids.shape[0], RECENT_DIM) == (768, 8)
        assert set(np.unique(R)) <= {0.0, 1.0}, "R must be binary"
        # And it is exactly the source's own Top-2 identities, not a re-derivation.
        for i in range(0, ids.shape[0], 97):
            assert set(np.flatnonzero(R[i])) == set(ids[i, m - 2].tolist())


def test_F_each_R_row_has_exactly_two_ones(source):
    ids, _, _ = source
    for m in TARGET_LAYERS:
        R = analyze.selection_state(ids, (recent_layer(m),))
        counts = R.sum(axis=1)
        assert np.all(counts == TOP_K) and TOP_K == 2, \
            f"row sums are {np.unique(counts)}, expected all 2"


def test_G_H_is_exactly_the_three_older_states_dim_24(source):
    ids, _, _ = source
    assert HISTORY_OFFSETS == (4, 3, 2)
    for m in TARGET_LAYERS:
        hl = history_layers(m)
        assert hl == (m - 4, m - 3, m - 2), f"L{m} history is {hl}"
        assert len(hl) == 3
        assert m - 1 not in hl, "H must exclude the recent layer"
        assert m not in hl, "H must exclude the target layer"
        H = analyze.selection_state(ids, hl)
        assert H.shape == (ids.shape[0], HISTORY_DIM) == (768, 24)
        # Each of the three 8-wide blocks is a valid Top-2 state.
        for slot in range(3):
            blk = H[:, slot * NUM_EXPERTS:(slot + 1) * NUM_EXPERTS]
            assert np.all(blk.sum(axis=1) == TOP_K)


def test_H_full_input_is_exactly_dim_32(source):
    ids, logits, is_test = source
    assert FULL_DIM == HISTORY_DIM + RECENT_DIM == 32
    for m in TARGET_LAYERS:
        v = analyze.build_variables(ids, logits, is_test, m)
        assert v["full_fit"].shape == (N_FIT, 32)
        assert v["full_test"].shape == (N_TEST, 32)
        # Ordering is [H ; R] exactly as specified.
        assert np.array_equal(v["full_fit"][:, :HISTORY_DIM], v["H_fit"])
        assert np.array_equal(v["full_fit"][:, HISTORY_DIM:], v["R_fit"])


# ------------------------------------------------------------------ I, J: what is excluded


def test_I_no_pca_and_no_compression_anywhere():
    for name, code in CODE.items():
        for banned in ("PCA", "TruncatedSVD", "decomposition", "svd", "n_components",
                       "explained_variance"):
            assert banned not in code, f"{name}.py uses {banned}"
    for name, tree in TREE.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = (getattr(node, "module", "") or "").lower()
                assert "decomposition" not in mod
    # Dimensions stay raw end to end: 8, 24, 32 with no intermediate reduction.
    assert RECENT_DIM == 8 and HISTORY_DIM == 24 and FULL_DIM == 32


def test_J_no_moa_router_data_is_used():
    for name, code in CODE.items():
        for banned in ("MoA", "self_attention", "attention", "output_router_logits",
                       "JetMoe"):
            assert banned not in code, f"{name}.py references {banned}"
    # Only 'ids' and 'logits' from the frozen file are read, both MLP-router quantities.
    assert 'd["ids"]' in SRC["nhd"] and 'd["logits"]' in SRC["nhd"]
    with open(os.path.join(SOURCE_DIR, "artifacts/model_provenance.json")) as fh:
        prov = json.load(fh)
    assert prov["moa_router_used"] is False
    assert prov["router_studied"].lower().startswith("mlp")


def test_J2_no_probabilities_ranks_or_activations():
    code, calls = CODE["analyze"], called_names(TREE["analyze"])
    for banned in ("softmax", "batch_gates", "top_k_gates", "expert_size", "hidden_states"):
        assert banned not in code, f"analyze.py uses {banned}"
    assert "argsort" not in calls, "rank ordering must not enter the representation"
    fn = ast.parse(inspect.getsource(analyze.selection_state))
    floats = {n.value for n in ast.walk(fn) if isinstance(n, ast.Constant)
              and isinstance(n.value, float)}
    assert floats <= {0.0, 1.0}, "the indicator must be a constant 1, not a magnitude"


# ----------------------------------------------------------------- K: target construction


def test_K_target_construction_matches_rmc_p0(source):
    """Centre within sample, then standardize with FIT statistics only."""
    ids, logits, is_test = source
    rng = np.random.default_rng(3)
    g = rng.normal(size=(40, NUM_EXPERTS)) * 3 + 7
    q = analyze.center_logits(g)
    assert np.allclose(q.mean(axis=1), 0.0)
    with pytest.raises(ValueError):
        analyze.center_logits(rng.normal(size=(4, 64)))

    for m in TARGET_LAYERS:
        v = analyze.build_variables(ids, logits, is_test, m)
        # FIT target is standardized on itself; TEST uses FIT statistics, so it is not
        # exactly zero-mean/unit-variance. That asymmetry is the point.
        assert np.allclose(v["Y_fit"].mean(axis=0), 0.0, atol=1e-9)
        assert np.allclose(v["Y_fit"].std(axis=0), 1.0, atol=1e-6)
        assert not np.allclose(v["Y_test"].mean(axis=0), 0.0, atol=1e-6)

    # Same source array and the same centring rule as RMC-P0.
    assert "logits[:, target - 1, :]" in SRC["analyze"]
    rmc_analyze = open(os.path.join(SOURCE_DIR, "analyze.py")).read()
    assert "g - g.mean(axis=-1, keepdims=True)" in rmc_analyze
    assert "g - g.mean(axis=-1, keepdims=True)" in SRC["analyze"]


def test_K2_target_never_enters_any_input():
    """No leakage of Y into R, H, or [H;R].

    Checked on referenced names, not on substrings: 'Y' occurs inside N_LAYERS.
    """
    fn = ast.parse(inspect.getsource(analyze.selection_state)).body[0]
    args = [a.arg for a in fn.args.args]
    assert args == ["ids", "layers_human"], f"selection_state takes {args}"
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    for banned in ("logits", "target", "Y_fit", "Y_test", "center_logits", "Y"):
        assert banned not in names, f"selection_state sees {banned}"
    assert not any(isinstance(n, ast.Attribute) and n.attr in ("Y_fit", "Y_test")
                   for n in ast.walk(fn))
    v = analyze.build_variables(*nhd.load_source(), TARGET_LAYERS[0])
    assert v["full_fit"].shape[1] == FULL_DIM
    assert set(np.unique(v["full_fit"])) <= {0.0, 1.0}, "inputs are binary; Y is not"


# ---------------------------------------------------- L, M: frozen architectures and width


def test_L_mlp_architectures_are_frozen_and_singular():
    """One architecture family, Linear -> GELU -> Linear, built in exactly one place."""
    assert call_count(TREE["analyze"], "Sequential") == 1, "more than one architecture"
    assert call_count(TREE["analyze"], "GELU") == 1
    assert call_count(TREE["analyze"], "Linear") == 2
    code = CODE["analyze"]
    for banned in ("Dropout", "BatchNorm", "LayerNorm", "ReLU", "Tanh", "Sigmoid", "SiLU",
                   "Conv", "LSTM", "GRU", "Transformer", "Attention"):
        assert banned not in code, f"analyze.py adds {banned}"
    for banned in ("RandomForest", "XGB", "GradientBoosting", "SVR", "KernelRidge",
                   "MLPRegressor", "mutual_info"):
        assert banned not in code, f"analyze.py adds a forbidden estimator ({banned})"

    # The built module is exactly the three declared layers, in order.
    model = analyze.make_mlp(FULL_DIM, HIDDEN, NUM_EXPERTS, INIT_SEEDS[0])
    kinds = [type(l).__name__ for l in model]
    assert kinds == ["Linear", "GELU", "Linear"]
    assert model[0].in_features == FULL_DIM and model[0].out_features == HIDDEN
    assert model[2].in_features == HIDDEN and model[2].out_features == NUM_EXPERTS
    assert model[0].bias is not None and model[2].bias is not None


def test_L2_fixed_optimizer_and_schedule():
    assert nhd.LR == 1e-3 and nhd.WEIGHT_DECAY == 1e-4
    assert MAX_EPOCHS == 300 and PATIENCE == 30
    assert INIT_SEEDS == (42, 123, 2026) and len(INIT_SEEDS) == 3
    code, calls = CODE["analyze"], called_names(TREE["analyze"])
    assert "AdamW" in calls and call_count(TREE["analyze"], "AdamW") == 1
    for banned in ("Adam(", "SGD", "RMSprop", "lr_scheduler", "OneCycle", "CosineAnneal"):
        assert banned not in code, f"analyze.py changes the optimizer or schedule ({banned})"
    assert "MSELoss" in calls, "early stopping must be on MSE"


def test_M_matched_width_is_determined_mechanically():
    w, mp, hp = nhd.matched_width()
    history = FULL_DIM * HIDDEN + HIDDEN + HIDDEN * NUM_EXPERTS + NUM_EXPERTS
    assert hp == history == 1320
    assert mp == RECENT_DIM * w + w + w * NUM_EXPERTS + NUM_EXPERTS
    assert w == MATCHED_WIDTH == 77 and mp == 1317
    # It is the argmin over widths, so no choice was made by hand.
    for other in range(1, 4097):
        p = RECENT_DIM * other + other + other * NUM_EXPERTS + NUM_EXPERTS
        assert abs(p - hp) >= abs(mp - hp) or other == w

    # Realised parameter counts agree with the arithmetic.
    hist_model = analyze.make_mlp(FULL_DIM, HIDDEN, NUM_EXPERTS, 42)
    matched = analyze.make_mlp(RECENT_DIM, w, NUM_EXPERTS, 42)
    plain = analyze.make_mlp(RECENT_DIM, HIDDEN, NUM_EXPERTS, 42)
    assert analyze.n_params(hist_model) == hp
    assert analyze.n_params(matched) == mp
    assert abs(analyze.n_params(matched) - analyze.n_params(hist_model)) == 3
    assert analyze.n_params(plain) < analyze.n_params(matched), \
        "the capacity control must have more parameters than plain MLP_RECENT"


def test_M2_no_width_sweep_and_no_test_dependence():
    src = inspect.getsource(nhd.matched_width)
    for banned in ("test", "Y_", "r2", "score"):
        assert banned not in src.lower().replace("out_dim", ""), \
            f"width selection sees {banned}"
    # The value is a module constant computed at import, before any data is touched.
    assert isinstance(MATCHED_WIDTH, int)
    assert nhd.matched_width()[0] == MATCHED_WIDTH


# --------------------------------------------------------------- N: TEST never tunes anything


def test_N_test_is_never_used_for_early_stopping():
    """Early stopping sees only a split carved out of the array handed to train_mlp."""
    src = inspect.getsource(analyze.train_mlp)
    for banned in ("test", "Y_test", "X_test", "R_test", "full_test", "H_test"):
        assert banned not in src, f"train_mlp sees {banned}"
    tree = ast.parse(src)
    # Every tensor it builds comes from its own X/Y arguments.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "tensor":
            dumped = ast.dump(node.args[0])
            assert "'X'" in dumped or "'Y'" in dumped, f"unexpected tensor source {dumped}"

    # And every call site passes a FIT-side array.
    for node in ast.walk(TREE["analyze"]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "train_mlp":
            for arg in node.args[:2]:
                assert "test" not in ast.dump(arg).lower(), \
                    f"train_mlp called with {ast.dump(arg)}"


def test_N2_fit_split_is_deterministic_and_the_declared_size():
    assert N_TRAIN == 384 and N_VAL == 128 and N_TRAIN + N_VAL == N_FIT
    assert FIT_SPLIT_SEED == 20260921
    tr, va = analyze.fit_split(N_FIT, FIT_SPLIT_SEED, N_TRAIN)
    assert len(tr) == N_TRAIN and len(va) == N_VAL
    assert set(tr.tolist()).isdisjoint(va.tolist()), "TRAIN and VAL overlap"
    assert sorted(tr.tolist() + va.tolist()) == list(range(N_FIT))
    again = analyze.fit_split(N_FIT, FIT_SPLIT_SEED, N_TRAIN)
    assert np.array_equal(tr, again[0]), "split is not reproducible from its seed"


def test_N3_best_validation_checkpoint_is_restored():
    src = inspect.getsource(analyze.train_mlp)
    assert "load_state_dict" in src and "best_state" in src
    assert "best_val" in src
    # Training is reproducible and actually reduces the objective.
    rng = np.random.default_rng(5)
    X = rng.normal(size=(N_FIT, RECENT_DIM))
    Y = np.tanh(X @ rng.normal(size=(RECENT_DIM, NUM_EXPERTS)))
    m1, i1 = analyze.train_mlp(X, Y, RECENT_DIM, HIDDEN, NUM_EXPERTS, 42)
    m2, i2 = analyze.train_mlp(X, Y, RECENT_DIM, HIDDEN, NUM_EXPERTS, 42)
    assert i1["best_val_mse"] == i2["best_val_mse"], "training is not deterministic"
    assert np.allclose(analyze.predict(m1, X), analyze.predict(m2, X))
    assert i1["best_epoch"] <= i1["epochs_run"] <= MAX_EPOCHS
    assert i1["n_train"] == N_TRAIN and i1["n_val"] == N_VAL
    # Different seeds give different initialisations, so the three seeds are informative.
    m3, _ = analyze.train_mlp(X, Y, RECENT_DIM, HIDDEN, NUM_EXPERTS, 2026)
    assert not np.allclose(analyze.predict(m1, X), analyze.predict(m3, X))


# ------------------------------------------------------ O, P, Q, R: the residual machinery


def test_O_crossfitting_never_predicts_a_sample_from_its_own_model():
    """The decisive property of Part B, checked by tracing which fold predicted which row."""
    src = inspect.getsource(analyze.crossfit_residuals)
    tree = ast.parse(src)
    # Structurally: the loop trains on one index set and applies to the disjoint other.
    loop = next(n for n in ast.walk(tree) if isinstance(n, ast.For))
    dumped = ast.dump(loop)
    assert "train_idx" in dumped and "apply_idx" in dumped
    assert "X[train_idx]" in src and "Y[train_idx]" in src
    assert "resid[apply_idx]" in src and "predict(model, X[apply_idx])" in src
    assert "X[apply_idx], Y[apply_idx]" not in src, "a model trained on what it predicts"

    # Numerically: fold membership is disjoint and covers FIT exactly.
    rng = np.random.default_rng(CROSSFIT_SEED)
    perm = rng.permutation(N_FIT)
    a, b = perm[:N_FOLD], perm[N_FOLD:]
    assert len(a) == len(b) == N_FOLD == 256
    assert set(a.tolist()).isdisjoint(b.tolist())
    assert sorted(a.tolist() + b.tolist()) == list(range(N_FIT))
    assert CROSSFIT_SEED == 20260922


def test_O2_crossfitted_residuals_differ_from_in_sample_ones():
    """An in-sample residual would be optimistically small; the cross-fitted one is not."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(N_FIT, RECENT_DIM))
    Y = np.tanh(X @ rng.normal(size=(RECENT_DIM, NUM_EXPERTS))) \
        + rng.normal(scale=0.1, size=(N_FIT, NUM_EXPERTS))
    eps, info = analyze.crossfit_residuals(X, Y, NUM_EXPERTS)
    assert eps.shape == (N_FIT, NUM_EXPERTS)
    assert np.all(np.abs(eps).sum(axis=1) > 0), "some rows were never residualized"

    full, _ = analyze.train_mlp(X, Y, RECENT_DIM, HIDDEN, NUM_EXPERTS, INIT_SEEDS[0])
    in_sample = Y - analyze.predict(full, X)
    assert eps.std() >= in_sample.std(), \
        "cross-fitted residuals should not be smaller than in-sample ones"
    assert len(info["folds"]["a"]) == len(info["folds"]["b"]) == N_FOLD


def test_P_both_epsilon_Y_and_epsilon_H_are_crossfitted():
    src = inspect.getsource(analyze.part_b)
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "crossfit_residuals"]
    assert len(calls) == 2, "expected one cross-fit for Y and one for H"
    targets = {ast.dump(c.args[1]) for c in calls}
    assert any("Y_fit" in t for t in targets), "epsilon_Y is not cross-fitted"
    assert any("H_fit" in t for t in targets), "epsilon_H is not cross-fitted"
    # Both are residualized on R, the recent state.
    for c in calls:
        assert "R_fit" in ast.dump(c.args[0])


def test_P2_history_residual_model_has_the_declared_shape():
    model = analyze.make_mlp(RECENT_DIM, HIDDEN, HISTORY_DIM, INIT_SEEDS[0])
    assert model[0].in_features == RECENT_DIM == 8
    assert model[0].out_features == HIDDEN == 32
    assert model[2].out_features == HISTORY_DIM == 24
    assert [type(l).__name__ for l in model] == ["Linear", "GELU", "Linear"]


def test_Q_residual_ridge_uses_crossfitted_fit_residuals_only():
    src = inspect.getsource(analyze.part_b)
    assert "probe.fit(eps_H_fit, eps_Y_fit)" in src
    for bad in ("probe.fit(eps_H_test", "probe.fit(v[", "probe.fit(H_"):
        assert bad not in src, f"residual probe fitted on {bad}"
    assert src.count("probe.fit(") == 1
    # One alpha, no search.
    assert ALPHA == 1.0
    for node in ast.walk(TREE["analyze"]):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "Ridge":
            kw = {k.arg: k.value for k in node.keywords}
            assert set(kw) == {"alpha"} and kw["alpha"].id == "ALPHA"
    code = CODE["analyze"]
    for banned in ("RidgeCV", "GridSearch", "alphas", "cross_val"):
        assert banned not in code, f"analyze.py searches alpha ({banned})"


def test_R_test_residuals_come_only_from_fit_trained_models():
    src = inspect.getsource(analyze.part_b)
    assert 'train_mlp(v["R_fit"], v["Y_fit"]' in src, "f_full is not trained on FIT"
    assert 'train_mlp(v["R_fit"], v["H_fit"]' in src, "g_full is not trained on FIT"
    assert 'v["Y_test"] - predict(f_full, v["R_test"])' in src
    assert 'v["H_test"] - predict(g_full, v["R_test"])' in src
    # No TEST array is ever an argument to training.
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("train_mlp", "crossfit_residuals"):
            for arg in node.args:
                assert "test" not in ast.dump(arg).lower()


def test_R2_permutation_control_is_descriptive_and_seeded():
    src = inspect.getsource(analyze.part_b)
    assert PERMUTE_SEED == 271828
    assert "default_rng(PERMUTE_SEED)" in src
    assert "eps_H_test[perm]" in src, "the control must permute history rows, not the target"
    assert "eps_Y_test[perm]" not in src
    # Permuting rows destroys the pairing, so a real signal should degrade.
    rng = np.random.default_rng(13)
    A = rng.normal(size=(N_TEST, HISTORY_DIM))
    B = A[:, :NUM_EXPERTS] * 0.8 + rng.normal(scale=0.2, size=(N_TEST, NUM_EXPERTS))
    from sklearn.linear_model import Ridge
    probe = Ridge(alpha=ALPHA).fit(A, B)
    real = analyze.score(B, probe.predict(A))
    perm = np.random.default_rng(PERMUTE_SEED).permutation(N_TEST)
    shuffled = analyze.score(B, probe.predict(A[perm]))
    assert real > shuffled, "the control does not behave as a control"


# --------------------------------------------------------------- classification and gate


def _target(A=0.10, B=0.10, Bm=0.10, C=0.05, b_seeds=None, bm_seeds=None):
    b_seeds = b_seeds if b_seeds is not None else [B] * 3
    bm_seeds = bm_seeds if bm_seeds is not None else [Bm] * 3
    return {"part_a": {"A": A, "B": B, "B_matched": Bm,
                       "B_per_seed": {str(s): v for s, v in zip(INIT_SEEDS, b_seeds)},
                       "B_matched_per_seed": {str(s): v for s, v in zip(INIT_SEEDS, bm_seeds)}},
            "part_b": {"residual_R2": C}}


def test_classification_criteria_behave_as_written():
    """Exercise every label on synthetic inputs, before any real number exists."""
    assert NEAR_ZERO == 0.02

    def both(**kw):
        return {m: _target(**kw) for m in TARGET_LAYERS}

    label, _ = analyze.classify(both(A=0.10, B=0.001, Bm=0.001, C=0.001,
                                     b_seeds=[0.001] * 3))
    assert label == nhd.NONLINEAR_ACCESSIBILITY

    label, _ = analyze.classify(both(A=0.10, B=0.10, Bm=0.10, C=0.05))
    assert label == nhd.RESIDUAL_HISTORY_VALUE

    label, _ = analyze.classify(both(A=0.10, B=0.10, Bm=0.10, C=0.001))
    assert label == nhd.JOINT_ONLY

    # Seeds disagreeing in sign on B is instability, whatever the means say.
    label, _ = analyze.classify(both(A=0.10, B=0.03, Bm=0.03, C=0.05,
                                     b_seeds=[0.09, -0.02, 0.02]))
    assert label == nhd.NO_CLEAR_PATTERN

    # Targets disagreeing cannot yield a clean label.
    mixed = {TARGET_LAYERS[0]: _target(A=0.10, B=0.10, Bm=0.10, C=0.05),
             TARGET_LAYERS[1]: _target(A=0.10, B=0.001, Bm=0.001, C=0.001,
                                       b_seeds=[0.001] * 3)}
    label, facts = analyze.classify(mixed)
    assert label == nhd.NO_CLEAR_PATTERN
    assert facts["targets_agree"] is False

    # A capacity-explained gain is not RESIDUAL-HISTORY-VALUE: B_matched must clear too.
    label, _ = analyze.classify(both(A=0.10, B=0.10, Bm=0.005, C=0.05,
                                     bm_seeds=[0.005] * 3))
    assert label != nhd.RESIDUAL_HISTORY_VALUE


def test_classification_reads_only_frozen_quantities():
    """Only the declared result fields reach the label; keys it writes are not reads."""
    src = inspect.getsource(analyze.classify)
    tree = ast.parse(src)
    inputs = {"per_target", "t", "a"}

    def root(node):
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None

    keys = {n.slice.value for n in ast.walk(tree)
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
            and isinstance(n.slice.value, str) and root(n) in inputs}
    assert keys == {"part_a", "part_b", "A", "B", "B_matched", "residual_R2",
                    "B_per_seed", "B_matched_per_seed"}, f"reads {sorted(keys)}"
    assert "permuted_residual_R2" not in keys, "the control must not drive the label"
    # Only the frozen threshold is compared against; no new literal cutoffs.
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for c in node.comparators:
                if isinstance(c, ast.Constant) and isinstance(c.value, float):
                    assert c.value == 0.0, f"inline float threshold {c.value}"


def test_S_no_test_metric_before_checks_pass():
    src = SRC["analyze"]
    assert "require_checks_passed" in src
    fn = next(n for n in ast.walk(TREE["analyze"])
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    seen = []
    for stmt in fn.body:
        seen += [n.func.id for n in ast.walk(stmt)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        if "require_checks_passed" in seen:
            break
    for banned in ("part_a", "part_b", "score", "classify", "build_variables", "train_mlp"):
        assert banned not in seen, f"{banned} runs before the check gate"
    gate = inspect.getsource(analyze.require_checks_passed)
    assert "pytest" in gate and "returncode" in gate and "SystemExit" in gate


def test_S2_results_absent_or_consistent_with_the_frozen_protocol():
    path = os.path.join(ART, "results.json")
    if not os.path.exists(path):
        pytest.skip("no results yet, as expected before the single analysis run")
    with open(path) as fh:
        res = json.load(fh)
    assert res["target_layers"] == list(TARGET_LAYERS)
    assert res["n_fit"] == N_FIT and res["n_test"] == N_TEST
    assert res["new_model_inference"] is False and res["gpu_used"] is False
    assert res["variables"]["pca_used"] is False
    assert res["variables"]["moa_router_used"] is False
    assert res["architectures"]["matched_width"] == MATCHED_WIDTH
    assert res["architectures"]["width_sweep"] is False
    assert res["training"]["init_seeds"] == list(INIT_SEEDS)
    assert res["training"]["test_used_for_early_stopping"] is False
    assert res["near_zero_threshold"] == NEAR_ZERO
    assert res["classification"] in (nhd.NONLINEAR_ACCESSIBILITY,
                                     nhd.RESIDUAL_HISTORY_VALUE, nhd.JOINT_ONLY,
                                     nhd.NO_CLEAR_PATTERN)
    assert res["source_verification"]["hashes"] == nhd.SOURCE_HASHES
    # The recorded label is what the frozen classifier produces from the recorded numbers.
    per = {int(k): v for k, v in res["per_target"].items()}
    assert analyze.classify(per)[0] == res["classification"]
