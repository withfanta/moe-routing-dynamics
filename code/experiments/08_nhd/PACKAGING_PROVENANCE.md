# Packaging provenance — NHD-P0

**Source repository:** `/home/h-li/work/nonlinear_history_decoding`

**Source commit SHA:** `f1a7d4ab9313798a20297921dd8a8125dee71763` (branch `master`, 3 commits,
working tree clean at packaging time)

**Packaging method:** `git archive f1a7d4a | tar -x`. The frozen commit is the canonical
source; no working-tree file was packaged.

## Exact scientific status

**Result: RESIDUAL-HISTORY-VALUE.**

A mechanism-disambiguation pilot asking one question: does the RMC-P0 history gain survive a
small nonlinear recent-state predictor? It reuses frozen RMC-P0 JetMoE artifacts only — no new
model inference, no GPU, no dataset access.

Per-target facts, agreeing at both target layers:

```
                        Layer 12     Layer 20
A (nonlinear gain on R)  +0.00139     +0.00936
B                        +0.17137     +0.21861
B_matched                +0.16440     +0.21912
residual R²              +0.20549     +0.23556
nonlinear_improves        False        False
residual_positive         True         True
```

The parameter-matched control rules out "extra parameters" as the explanation: MLP_RECENT_MATCHED
has 1317 trainable parameters against MLP_HISTORY's 1320 (width 77 chosen mechanically by
parameter count before TEST was seen), and its advantage over plain MLP_RECENT is at most +0.007
at Layer 12 and −0.001 at Layer 20. B was positive across all three init seeds (42, 123, 2026)
at both targets. Residualization was cross-fitted: no sample was residualized by a model trained
on it.

Constraints on how this may be read, from the experiment's own record:

- **B is not conditional information.** Do not call it that.
- **Nothing here is a claim of information-theoretic conditional independence.**
- **Even RESIDUAL-HISTORY-VALUE does not prove that historical information is mathematically
  absent from R; it only shows that the tested nonlinear recent-state predictor does not explain
  the full held-out history gain.**
- A's small size is a statement about this predictor at this capacity on 512 FIT samples, not
  about nonlinearity in general. A larger or different nonlinear model was not tried, by
  protocol.
- No binary scientific verdict is forced; the label names a pattern, not a proof.
- `rescue_modifications: none` — hidden width, layer count, activation, dropout, learning rate,
  optimizer, history window, target layers, model, dataset, estimator family, residualization
  and seed count are all as committed before the run.

## Files included

9 files, 95,859 B total. Flat layout, no artifacts of its own beyond results.

```
README.md
protocol.md                          frozen protocol
nhd.py                               library, incl. verify_source()
analyze.py                           CPU analysis, gated on test S
test_nhd.py                          36 checks A-S, 31,828 B
artifacts/results.json               frozen result
artifacts/RESULTS.md
artifacts/source_verification.json   hash proof of the reused RMC-P0 artifacts
.gitignore
```

## Files intentionally omitted

None. `omitted_bytes_total: 0`. NHD-P0 generated no large artifacts — it read RMC-P0's small
`merged.npz` and wrote only JSON and Markdown.

Omitted by policy only: `__pycache__`, run logs, and the source `.git` directory.

## Artifact hashes

NHD-P0 pinned the RMC-P0 files it consumed and verified them before computing anything.
Recorded in `artifacts/source_verification.json`:

```
source RMC-P0 commit                cc60edd   (clean)
artifacts/merged.npz                7dafe8048772937c239bf9ab61577d9cce648847af4a6227e4ca5d86a9758726
artifacts/data_manifest.json        371601f7ef6c491d571b1528fb15ef5dcd8cd361844786aee9fc8c4aef8afe02
artifacts/model_provenance.json     e26d8ec79271f1312baad4c7993b6bd153d69f5426abb104221cae9e25585335
artifacts/results.json              595ee0b98ed70bd5a9758e8ede0042ac1ab7ccaef3192f8c44077584795c0851
```

**These pins are satisfied inside this packaged repository.** RMC-P0's `merged.npz` is packaged
(see [../07_rmc/PACKAGING_PROVENANCE.md](../07_rmc/PACKAGING_PROVENANCE.md)) and hashes to
`7dafe804…` here, matching the pin exactly. NHD-P0 is therefore the one experiment in the series
that is fully re-runnable from this repository as packaged, on CPU, with the source path
repointed at `../07_rmc`.

Variable definitions, from `source_verification.json`:

```
L12   recent layer 11,  history layers [8, 9, 10]
L20   recent layer 19,  history layers [16, 17, 18]
matched width 77 -> 1317 params vs history 1320 params (diff 3)
```

## Reproduction entry point

See [REPRODUCTION.md](REPRODUCTION.md). Summary:

```bash
PY=/home/h-li/miniconda3/envs/sensorllm/bin/python
$PY -c "import nhd; nhd.verify_source()"   # checks A, B
$PY analyze.py                             # gated on test S
```

`analyze.py` runs `test_nhd.py` as a subprocess and refuses to compute any TEST statistic unless
every check passes.
