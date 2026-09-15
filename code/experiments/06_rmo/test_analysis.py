"""RMO-P0 minimal implementation checks — tests A-J of protocol.md.

These assert the integrity of the analysis, not its outcome. They compute no formal
statistic: nothing here evaluates the five probes on TEST or touches the R² curve.
"""

from __future__ import annotations

import ast
import os

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import analyze as A


@pytest.fixture(scope="module")
def src():
    return A.load_source()


@pytest.fixture(scope="module")
def splits(src):
    return ~src.is_test, src.is_test


# ------------------------------------------------------------------------------- A


def test_A_source_manifest_hash_matches_epd():
    """A: the source manifest hashes to EPD-P0's recorded value."""
    hashes = A.verify_source()
    assert hashes["data_manifest.json"] == A.EPD_MANIFEST_SHA256
    assert A.sha256_file(A.EPD_MANIFEST) == A.EPD_MANIFEST_SHA256


def test_A2_source_hash_mismatch_is_fatal(tmp_path, monkeypatch):
    """A: a wrong manifest must abort rather than be tolerated."""
    monkeypatch.setattr(A, "EPD_MANIFEST_SHA256", "0" * 64)
    with pytest.raises(AssertionError, match="manifest hash mismatch"):
        A.verify_source()


def test_A3_epd_project_is_not_written_to():
    """A: EPD-P0 is only ever read. Writes are allowed, but only into this project.

    Checking the mode alone would be wrong -- analyze.py legitimately appends to its own
    run.log. What matters is that no write target resolves inside the EPD-P0 tree.
    """
    source = open(os.path.join(A.HERE, "analyze.py")).read()
    tree = ast.parse(source)
    write_sites = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "open":
            mode = node.args[1].value if len(node.args) > 1 else "r"
            if mode == "r" or "rb" in mode:
                continue
            write_sites += 1
            # A write must be built from ART, never from an EPD path.
            target = ast.dump(node.args[0])
            assert "ART" in target, f"write target is not under artifacts/: {target}"
            for banned in ("EPD_DIR", "EPD_MANIFEST", "EPD_MERGED"):
                assert banned not in target, f"analyze.py writes to {banned}"
    assert write_sites >= 1, "expected at least the run.log append site"

    # Every write target sits inside this project, never inside EPD-P0.
    assert A.ART.startswith(A.HERE)
    assert not A.ART.startswith(A.EPD_DIR)
    assert A.EPD_DIR.startswith("/home/h-li/work/expert_provenance_decomposition")


# ------------------------------------------------------------------------------- B


def test_B_fit_512_test_256(src):
    """B: FIT = 512 and TEST = 256, EPD-P0's own split, not resampled."""
    assert int((~src.is_test).sum()) == 512 == A.N_FIT
    assert int(src.is_test.sum()) == 256 == A.N_TEST
    assert src.is_test.shape == (768,)
    # Disjoint and exhaustive by construction of a boolean mask.
    assert int((~src.is_test).sum()) + int(src.is_test.sum()) == 768


# ------------------------------------------------------------------------------- C


def test_C_selection_features_are_binary_64d(src):
    """C: selected-ID features are binary 64-d vectors with exactly 8 ones per layer."""
    for layers in [(11,), (10,), (8, 9, 10), tuple(range(1, 11))]:
        X = A.selection_indicator(src.hist_ids, layers)
        assert X.shape == (768, 64 * len(layers))
        assert np.isin(X, (0.0, 1.0)).all()
        per_layer = X.reshape(768, len(layers), 64).sum(axis=2)
        assert (per_layer == A.TOP_K).all()


def test_C2_indicator_matches_ids(src):
    """C: the indicator marks exactly the captured native Top-8 identities."""
    X = A.selection_indicator(src.hist_ids, (11,)).reshape(768, 64)
    for i in (0, 17, 400, 767):
        assert set(np.flatnonzero(X[i]).tolist()) == set(src.hist_ids[i, 10].tolist())


def test_C3_no_probabilities_or_content_enter_features(src):
    """C: identities only -- the indicator ignores everything but hist_ids."""
    ids = src.hist_ids.copy()
    X1 = A.selection_indicator(ids, (9, 10))
    # Permuting the within-layer rank order leaves the identity set, hence the indicator,
    # unchanged: no rank or probability information can leak in.
    perm = ids[:, :, ::-1].copy()
    X2 = A.selection_indicator(perm, (9, 10))
    assert np.array_equal(X1, X2)


# ------------------------------------------------------------------------------- D


def test_D_target_is_layer12_native_router_logits(src):
    """D: the target is exactly EPD-P0's stored native Layer-12 router logits."""
    raw = np.load(A.EPD_MERGED)
    assert np.array_equal(src.g12, raw["g12"].astype(np.float64))
    assert src.g12.shape == (768, 64)
    assert A.TARGET_LAYER_HUMAN == 12

    q = A.center_logits(src.g12)
    assert np.allclose(q.mean(axis=1), 0.0, atol=1e-10)
    # Centring only removes the per-sample mean; differences are preserved exactly.
    d_raw = src.g12[:, 3] - src.g12[:, 7]
    assert np.allclose(q[:, 3] - q[:, 7], d_raw, atol=1e-10)


def test_D2_target_softmax_reproduces_stored_top8(src):
    """D: the logits are the router's own -- their Top-8 reproduces the stored route."""
    raw = np.load(A.EPD_MERGED)
    if "top8_12" not in raw.files:
        pytest.skip("top8_12 not stored")
    top8 = raw["top8_12"].astype(np.int64)
    for i in (0, 5, 100, 500, 767):
        got = set(np.argsort(-src.g12[i])[:8].tolist())
        assert got == set(top8[i].tolist())


# ------------------------------------------------------------------------------- E


def test_E_recent_layer_preprocessing_fitted_once_and_reused(src, splits):
    """E: one Layer-11 block object produces z_recent for every model."""
    fit_i, test_i = splits
    e11_fit = A.selection_indicator(src.hist_ids[fit_i], (A.RECENT_LAYER_HUMAN,))
    e11_test = A.selection_indicator(src.hist_ids[test_i], (A.RECENT_LAYER_HUMAN,))
    block = A.FrozenBlock.fit(e11_fit)
    zf, zt = block.transform(e11_fit), block.transform(e11_test)
    assert zf.shape == (512, 16) and zt.shape == (256, 16)

    # Every model's first 16 columns are byte-identical to this one z_recent.
    for k in A.K_ORDER:
        Xf = A.probe_input(zf, None if k == 1 else zf)   # second block is irrelevant here
        assert np.array_equal(Xf[:, :16], zf)

    # analyze.main fits FrozenBlock on the Layer-11 features exactly once.
    tree = ast.parse(open(os.path.join(A.HERE, "analyze.py")).read())
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    fits = [n for n in ast.walk(main)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "fit"
            and getattr(n.func.value, "id", None) == "FrozenBlock"]
    # One for z_recent, one inside the window loop for z_old(k).
    assert len(fits) == 2, f"expected 2 FrozenBlock.fit sites in main, found {len(fits)}"
    in_loop = [n for f in ast.walk(main) if isinstance(f, ast.For)
               for n in ast.walk(f)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "fit" and getattr(n.func.value, "id", None) == "FrozenBlock"]
    assert len(in_loop) == 1, "the Layer-11 block must not be fitted inside a loop"


def test_E2_recent_block_is_deterministic(src, splits):
    """E: refitting the frozen block on the same FIT rows is reproducible, so 'fitted
    once and reused' is a real constraint rather than hidden randomness."""
    fit_i, _ = splits
    X = A.selection_indicator(src.hist_ids[fit_i], (A.RECENT_LAYER_HUMAN,))
    z1 = A.FrozenBlock.fit(X).transform(X)
    z2 = A.FrozenBlock.fit(X).transform(X)
    assert np.allclose(z1, z2, atol=1e-12)


# ------------------------------------------------------------------------------- F


def test_F_window_pca_uses_fit_only(src, splits):
    """F: each older-history PCA sees FIT rows only."""
    fit_i, test_i = splits
    for k, layers in A.WINDOWS.items():
        Xf = A.selection_indicator(src.hist_ids[fit_i], layers)
        Xt = A.selection_indicator(src.hist_ids[test_i], layers)
        block = A.FrozenBlock.fit(Xf)
        assert block.pca.n_samples_ == A.N_FIT
        assert block.scaler_in.n_samples_seen_ == A.N_FIT
        assert block.scaler_out.n_samples_seen_ == A.N_FIT
        assert block.transform(Xt).shape == (A.N_TEST, 16)


def test_F2_windows_exclude_layer_11_and_are_cumulative():
    """F: the windows are the four specified older-history sets, none containing L11."""
    assert A.WINDOWS == {2: (10,), 4: (8, 9, 10), 8: (4, 5, 6, 7, 8, 9, 10),
                         11: tuple(range(1, 11))}
    for k, layers in A.WINDOWS.items():
        assert 11 not in layers, "Layer 11 is supplied separately as z_recent"
        assert len(layers) == k - 1
        assert list(layers) == sorted(layers)
    # Cumulative: each window contains the previous one.
    ks = sorted(A.WINDOWS)
    for a, b in zip(ks, ks[1:]):
        assert set(A.WINDOWS[a]).issubset(A.WINDOWS[b])


def test_F3_transform_is_row_independent(src, splits):
    """F: applying the frozen pipeline to TEST cannot depend on which TEST rows are
    present, so no TEST statistic can enter through transform()."""
    fit_i, test_i = splits
    Xf = A.selection_indicator(src.hist_ids[fit_i], A.WINDOWS[4])
    Xt = A.selection_indicator(src.hist_ids[test_i], A.WINDOWS[4])
    block = A.FrozenBlock.fit(Xf)
    full = block.transform(Xt)
    half = block.transform(Xt[:10])
    assert np.allclose(full[:10], half, atol=1e-12)


# ------------------------------------------------------------------------------- G


def test_G_every_probe_input_has_32_dims(src, splits):
    """G: all five probe inputs are exactly 32-dimensional."""
    fit_i, test_i = splits
    e11 = A.selection_indicator(src.hist_ids[fit_i], (A.RECENT_LAYER_HUMAN,))
    block = A.FrozenBlock.fit(e11)
    z_recent = block.transform(e11)
    for k in A.K_ORDER:
        if k == 1:
            X = A.probe_input(z_recent, None)
        else:
            Xw = A.selection_indicator(src.hist_ids[fit_i], A.WINDOWS[k])
            X = A.probe_input(z_recent, A.FrozenBlock.fit(Xw).transform(Xw))
        assert X.shape == (A.N_FIT, 32) == (A.N_FIT, A.PROBE_DIM)


def test_G2_baseline_second_block_is_exactly_zero():
    """G: the k=1 baseline pads with zeros, so width is held constant without adding
    information."""
    z = np.arange(32, dtype=np.float64).reshape(2, 16)
    X = A.probe_input(z, None)
    assert X.shape == (2, 32)
    assert np.array_equal(X[:, :16], z)
    assert not X[:, 16:].any()


def test_G3_wrong_width_is_rejected():
    """G: a mis-sized probe input must raise, not be silently accepted."""
    with pytest.raises(AssertionError, match="dims"):
        A.probe_input(np.zeros((4, 16)), np.zeros((4, 8)))


# ------------------------------------------------------------------------------- H


def test_H_no_test_statistics_enter_preprocessing(src, splits):
    """H: fitting on FIT alone gives the same TEST representation as fitting on FIT while
    TEST exists -- the pipeline is a pure function of FIT."""
    fit_i, test_i = splits
    Xf = A.selection_indicator(src.hist_ids[fit_i], A.WINDOWS[11])
    Xt = A.selection_indicator(src.hist_ids[test_i], A.WINDOWS[11])
    z_a = A.FrozenBlock.fit(Xf).transform(Xt)
    # Perturbing TEST rows cannot change the fitted pipeline.
    Xt_shuffled = Xt[::-1].copy()
    z_b = A.FrozenBlock.fit(Xf).transform(Xt_shuffled)[::-1]
    assert np.allclose(z_a, z_b, atol=1e-12)


def test_H2_target_scaler_fits_on_fit_only(src, splits):
    """H: the target StandardScaler sees 512 FIT rows and nothing else."""
    fit_i, test_i = splits
    q_fit = A.center_logits(src.g12[fit_i])
    sc = StandardScaler().fit(q_fit)
    assert sc.n_samples_seen_ == A.N_FIT
    # TEST is transformed with FIT statistics, so its own mean need not be zero.
    q_test = A.center_logits(src.g12[test_i])
    Yt = sc.transform(q_test)
    assert Yt.shape == (A.N_TEST, 64)
    assert np.allclose(sc.transform(q_fit).mean(axis=0), 0.0, atol=1e-8)
    assert not np.allclose(Yt.mean(axis=0), 0.0, atol=1e-8)


def test_H3_no_target_information_enters_pca():
    """H: FrozenBlock.fit takes features only -- there is no y parameter to pass."""
    import inspect
    params = list(inspect.signature(A.FrozenBlock.fit).parameters)
    assert params == ["X_fit"], f"FrozenBlock.fit signature is {params}"


# ------------------------------------------------------------------------------- I


def test_I_alpha_is_exactly_one_for_all_models():
    """I: every probe is Ridge(alpha=1.0); nothing searches or overrides it."""
    assert A.ALPHA == 1.0
    assert Ridge(alpha=A.ALPHA).alpha == 1.0

    tree = ast.parse(open(os.path.join(A.HERE, "analyze.py")).read())
    ridges = [n for n in ast.walk(tree)
              if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "Ridge"]
    assert len(ridges) == 1, "Ridge should be constructed in exactly one place"
    kw = {k.arg: k.value for k in ridges[0].keywords}
    assert set(kw) == {"alpha"} and getattr(kw["alpha"], "id", None) == "ALPHA"
    # No search utility is imported anywhere.
    text = open(os.path.join(A.HERE, "analyze.py")).read()
    for banned in ("RidgeCV", "GridSearchCV", "RandomizedSearchCV", "cross_val"):
        assert banned not in text


def test_I2_probe_dimension_and_seed_are_frozen():
    """I: the single PCA width and seed are fixed constants, not swept."""
    assert A.PCA_COMPONENTS == 16
    assert A.PROBE_DIM == 32
    assert A.PCA_SEED == 20260920
    tree = ast.parse(open(os.path.join(A.HERE, "analyze.py")).read())
    pcas = [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "PCA"]
    assert len(pcas) == 1, "PCA should be constructed in exactly one place"
    kw = {k.arg: k.value for k in pcas[0].keywords}
    assert getattr(kw["n_components"], "id", None) == "PCA_COMPONENTS"
    assert getattr(kw["random_state"], "id", None) == "PCA_SEED"
    assert kw["svd_solver"].value == "randomized"


# ------------------------------------------------------------------------------- J


def test_J_no_statistics_computed_before_checks_pass():
    """J: importing analyze and running these checks evaluates no probe on TEST, and
    main() gates itself on this module before computing anything."""
    tree = ast.parse(open(os.path.join(A.HERE, "analyze.py")).read())
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = main.body
    # The gate call must precede every fit_and_score / load_source in main.
    def line_of(pred):
        return [n.lineno for n in ast.walk(main)
                if isinstance(n, ast.Call) and pred(n)]
    gate = line_of(lambda n: getattr(n.func, "id", None) == "require_tests_passed")
    scores = line_of(lambda n: getattr(n.func, "id", None) == "fit_and_score")
    assert len(gate) == 1, "main must call require_tests_passed exactly once"
    assert scores, "main must score the probes"
    assert gate[0] < min(scores), "the check gate must run before any statistic"

    # No module-level statistic exists: importing analyze computes nothing.
    top_calls = [n for n in tree.body if isinstance(n, ast.Expr)
                 and isinstance(n.value, ast.Call)]
    assert not top_calls, "analyze.py must not compute anything at import time"

    # This test module itself never scores a probe. Checked on the AST rather than on the
    # raw text, so mentioning a name in a comment or predicate does not count as calling it.
    t_tree = ast.parse(open(os.path.join(A.HERE, "test_analysis.py")).read())
    called = set()
    for n in ast.walk(t_tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name):
                called.add(f.id)
            elif isinstance(f, ast.Attribute):
                called.add(f.attr)
    for banned in ("fit_and_score", "r2_score", "classify_and_score"):
        assert banned not in called, f"the checks must not call {banned}"
    # Ridge is constructed here only to confirm its alpha default, never fitted on TEST.
    assert "predict" not in called


def test_J2_no_gpu_or_model_inference_referenced():
    """J: the analysis is CPU-only and loads no model."""
    text = open(os.path.join(A.HERE, "analyze.py")).read()
    for banned in ("torch", "cuda", "transformers", "from_pretrained",
                   "CUDA_VISIBLE_DEVICES", "nvidia-smi"):
        assert banned not in text, f"analyze.py references {banned}"


def test_J3_classifier_labels_are_the_four_allowed_ones():
    """J: the classifier can only emit the four protocol labels, and each is reachable."""
    allowed = {A.LOCAL_DOMINANT, A.SHORT_HISTORY, A.LONGER_HISTORY_CANDIDATE,
               A.NO_CLEAR_PATTERN}
    assert set(A.INTERPRETATION) == allowed

    def mk(r1, r2_, r4, r8, r11):
        return {1: r1, 2: r2_, 4: r4, 8: r8, 11: r11}

    # Flat curve -> LOCAL-DOMINANT.
    assert A.classify(mk(0.60, 0.601, 0.605, 0.603, 0.607))["label"] == A.LOCAL_DOMINANT
    # Recent history helps, older adds nothing -> SHORT-HISTORY.
    assert A.classify(mk(0.60, 0.65, 0.66, 0.663, 0.664))["label"] == A.SHORT_HISTORY
    # Keeps climbing -> LONGER-HISTORY-CANDIDATE.
    assert A.classify(mk(0.50, 0.55, 0.60, 0.66, 0.71))["label"] == \
        A.LONGER_HISTORY_CANDIDATE
    # Irregular -> NO-CLEAR-PATTERN. The windows are nested, so a material drop when more
    # history is available is a compression artifact, not "older history adding little".
    assert A.classify(mk(0.50, 0.60, 0.42, 0.61, 0.44))["label"] == A.NO_CLEAR_PATTERN
    # A drop within tolerance is not treated as irregular.
    assert A.classify(mk(0.60, 0.65, 0.645, 0.648, 0.650))["label"] == A.SHORT_HISTORY
    for r2v in (mk(0.6, 0.6, 0.6, 0.6, 0.6), mk(0.5, 0.55, 0.6, 0.66, 0.71)):
        assert A.classify(r2v)["label"] in allowed


def test_J5_no_label_is_forced_when_differences_are_tiny():
    """J: a curve whose every comparison is below the descriptive threshold must not be
    pushed into a directional label."""
    r2 = {1: 0.400, 2: 0.405, 4: 0.410, 8: 0.415, 11: 0.418}
    d = A.classify(r2)
    assert d["label"] == A.LOCAL_DOMINANT
    assert d["max_cumulative_gain"] < A.NOTICEABLE
    # And the threshold is descriptive: no inferential statistic is computed anywhere.
    # Checked on names actually used in code, not on prose -- the docstrings legitimately
    # say "not a preregistered significance threshold".
    tree = ast.parse(open(os.path.join(A.HERE, "analyze.py")).read())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for banned in ("p_value", "pvalue", "ttest", "bootstrap", "ci_low", "ci_high"):
        assert banned not in names, f"analyze.py computes {banned}"
    assert A.NOTICEABLE == 0.02
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert not any("scipy" in m for m in imported), "no inferential testing library"


def test_J4_gains_are_defined_as_specified():
    """J: cumulative gains are all relative to k=1; stepwise gains are consecutive."""
    r2 = {1: 0.10, 2: 0.20, 4: 0.35, 8: 0.55, 11: 0.80}
    G, S = A.cumulative_gains(r2), A.stepwise_gains(r2)
    assert G == pytest.approx({"G_2": 0.10, "G_4": 0.25, "G_8": 0.45, "G_11": 0.70})
    assert S == pytest.approx({"S_2": 0.10, "S_4": 0.15, "S_8": 0.20, "S_11": 0.25})
    assert G["G_2"] == S["S_2"]
    assert sum(S.values()) == pytest.approx(G["G_11"])
