"""Exploratory pattern classification and the descriptive transition summary.

EPD-P0 produces no SUPPORTED/NOT_SUPPORTED verdict. It reports one of four decomposition
patterns, and declines to force a pattern when the result is ambiguous.

The 0.02 R² figure used below is a discussion aid for describing a gap as non-trivial. It is
not a preregistered significance threshold.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from . import N_TOP_TRANSITIONS, NONTRIVIAL_GAP, NUM_EXPERTS

PATH_DOMINANT = "PATH_DOMINANT"
CONTENT_DOMINANT = "CONTENT_DOMINANT"
INTERACTION = "INTERACTION"
NO_CLEAN_DECOMPOSITION = "NO_CLEAN_DECOMPOSITION"


def classify_pattern(r2: Dict[str, float], gap: float = NONTRIVIAL_GAP) -> Dict[str, object]:
    """Classify the decomposition pattern descriptively.

    "close to FULL" means within ``gap``; "substantially above" means exceeding the other
    single-factor representation by more than ``gap``.
    """
    full = r2["FULL_PROVENANCE"]
    idp = r2["ID_PATH"]
    cnt = r2["CONTENT_RANK"]
    fused = r2["FUSED"]

    # "At least as good as FULL" covers both being close to it and exceeding it. A
    # single-factor representation can exceed FULL because every representation shares one
    # compression budget: FULL is far denser and higher-dimensional, so PCA retains less of
    # its variance. That is a compression effect, not more information -- FULL provably
    # contains ID_PATH, since binarized slot occupancy reproduces ID_PATH exactly.
    id_at_least_full = (idp - full) >= -gap
    content_at_least_full = (cnt - full) >= -gap
    id_above_content = (idp - cnt) > gap
    content_above_id = (cnt - idp) > gap
    full_above_both = (full - idp) > gap and (full - cnt) > gap
    fused_matches_full = abs(full - fused) <= gap

    if full_above_both:
        pattern = INTERACTION
    elif id_at_least_full and id_above_content:
        pattern = PATH_DOMINANT
    elif content_at_least_full and content_above_id:
        pattern = CONTENT_DOMINANT
    else:
        pattern = NO_CLEAN_DECOMPOSITION

    return {
        "pattern": pattern,
        "gap_used_for_description": gap,
        "checks": {
            "id_path_at_least_full": bool(id_at_least_full),
            "content_at_least_full": bool(content_at_least_full),
            "id_path_above_content": bool(id_above_content),
            "content_above_id_path": bool(content_above_id),
            "full_above_both": bool(full_above_both),
            "fused_matches_full": bool(fused_matches_full),
        },
        "note": (
            "Exploratory description under a frozen small-sample pilot, not a "
            "hypothesis test. The 0.02 figure is a discussion aid, not a significance "
            "threshold."
        ),
    }


def pattern_interpretation(pattern: str) -> str:
    if pattern == PATH_DOMINANT:
        return ("Historical expert-selection path carries most of the signal: which experts "
                "were chosen predicts future routing, largely without their activation "
                "content.")
    if pattern == CONTENT_DOMINANT:
        return ("Expert activation content carries most of the signal: what the selected "
                "experts computed predicts future routing, largely without absolute expert "
                "identity.")
    if pattern == INTERACTION:
        return ("Future routing depends on the association between expert identity and "
                "activation content: knowing what was computed and who computed it carries "
                "more information than either alone.")
    return ("No clean decomposition emerged under this frozen pilot; the factors are not "
            "cleanly separable at this sample size and dimensional budget.")


# ------------------------------------------------------- descriptive transitions


def co_selection_enrichment(hist_ids: np.ndarray, target_ids: np.ndarray,
                            top_n: int = N_TOP_TRANSITIONS) -> Dict[str, object]:
    """Descriptive source-expert -> Layer-12-expert enrichment.

    For each historical layer, a 64x64 co-selection count matrix is built between source
    selected experts and Layer-12 selected experts; each source row is normalized to a
    probability distribution, and the Layer-12 marginal selection frequency is subtracted.
    Only the strongest enriched pairs are reported.

    No statistical testing, and these pairs never contribute to a verdict.
    """
    n, n_layers, _ = hist_ids.shape

    # Layer-12 marginal selection frequency.
    marginal = np.zeros(NUM_EXPERTS, dtype=np.float64)
    for row in range(n):
        marginal[target_ids[row].astype(np.int64)] += 1.0
    marginal /= n

    rows: List[Dict[str, object]] = []
    for lpos in range(n_layers):
        counts = np.zeros((NUM_EXPERTS, NUM_EXPERTS), dtype=np.float64)
        for row in range(n):
            src = hist_ids[row, lpos].astype(np.int64)
            tgt = target_ids[row].astype(np.int64)
            counts[np.ix_(src, tgt)] += 1.0

        row_totals = counts.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cond = np.where(row_totals > 0, counts / np.maximum(row_totals, 1e-12), 0.0)
        enrich = cond - marginal[None, :]

        support = counts.sum(axis=1)
        for src in range(NUM_EXPERTS):
            if support[src] <= 0:
                continue
            for tgt in range(NUM_EXPERTS):
                rows.append({
                    "source_layer_human": lpos + 1,
                    "source_expert": int(src),
                    "target_expert": int(tgt),
                    "conditional_prob": float(cond[src, tgt]),
                    "target_marginal": float(marginal[tgt]),
                    "enrichment": float(enrich[src, tgt]),
                    "source_support_selections": int(support[src] / 8),
                })

    rows.sort(key=lambda r: r["enrichment"], reverse=True)
    return {
        "target_marginal_mean": float(marginal.mean()),
        "top_pairs": rows[:top_n],
        "note": ("Descriptive only: no statistical testing, and these pairs do not "
                 "contribute to the reported pattern."),
    }


def layerwise_summary(r2_id: Dict[int, float], r2_content: Dict[int, float]) -> Dict[str, object]:
    """Small descriptive summary of the two 11-point curves."""
    layers = sorted(r2_id)
    ids = [r2_id[l] for l in layers]
    cnts = [r2_content[l] for l in layers]
    return {
        "layers": layers,
        "best_id_layer": int(layers[int(np.argmax(ids))]),
        "best_content_layer": int(layers[int(np.argmax(cnts))]),
        "mean_id": float(np.mean(ids)),
        "mean_content": float(np.mean(cnts)),
        "id_minus_content_mean": float(np.mean(ids) - np.mean(cnts)),
        "note": "Exploratory only; not used to create post-hoc subgroups.",
    }
