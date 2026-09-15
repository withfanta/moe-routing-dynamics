# Beyond the Previous Layer: Residual Predictive Structure in Sparse MoE Routing

**Authors:** Hao Li, Yasuyuki Tahara, and Yuichi Sei (corresponding author).

**Format:** Yale arXiv template, 10 pt, US letter, two columns. **Final PDF: 5 pages including references.**

**Code:** https://github.com/withfanta/moe-routing-dynamics/tree/main/code

This paper uses only frozen existing experiments. No new model inference, probe fitting,
scientific experiment, or modification of experimental results is part of this manuscript.
The source archive was pinned at commit `febc169a08827239e0e2057a4f81915cfbcc4aba`.

## Build

Run in this directory with a complete TeX Live installation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

For a clean rebuild:

```bash
latexmk -C main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf
```

The class uses XCharter, mathpazo, tgpagella, inconsolata, and standard LaTeX packages.
Install missing packages in the TeX distribution; preserve `yalearxiv.cls` unchanged.
`main.bbl` is included for source submission. The archive requires no external image
assets, shell escape, custom font files, or experiment execution to compile.

## Regenerate the tables

From a checkout containing the original `code/` result files:

```bash
python3 generate_tables.py
```

This standard-library-only script checks every SHA-256 in `source_manifest.json`
and formats existing values into three tables and `numbers.tex`. It performs no
model loading, fitting, or statistical recomputation. The EPD path-plus-content
number comes from the frozen Markdown report because that post-hoc control is
not in its main JSON file.

## Contents

- `main.tex`, `main.bib`, `main.bbl`: manuscript and bibliography.
- `yalearxiv.cls`, `preamble.tex`, `command.tex`: template and paper configuration.
- `table_provenance.tex`, `table_history.tex`, `table_nonlinear.tex`, `numbers.tex`:
  generated tables and numerical macros.
- `generate_tables.py`, `source_manifest.json`: formatting and frozen-source verification.
- `REVIEW.md`: numerical sources, interpretation notes, and validation record.
- `main.pdf`: compiled five-page manuscript.

The three main tables cover EIPC/EPD, RMO/RMC, and NHD. The Discussion reports
XEC's NLL means and all three primary paired confidence intervals. Original result
reports retain supplementary implementation diagnostics and descriptive tables.

## Template attribution

`yalearxiv.cls` is copied byte-for-byte from `preprint/The Yale arXiv Paper Template/`
in the source repository. Its original Creative Commons attribution and license
comments are preserved. The paper uses the class's native author and affiliation
commands and has no template logo or institutional branding graphic.
