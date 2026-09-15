"""The frozen EIPC-P0 decision rule.

Two targets: Layer 8 and Layer 12.

    A_m = R2_identity(m) - R2_fused(m)      does keeping individual states beat fusion?
    B_m = R2_identity(m) - R2_shuffled(m)   does the specific expert identity matter?

PROMISING requires ALL FIVE conditions:

  1. R2_identity > 0 at BOTH targets;
  2. A_m > 0 at BOTH targets;
  3. B_m > 0 at BOTH targets;
  4. mean(A_m) >= +0.02;
  5. mean(B_m) >= +0.02.

Any failure yields NOT_PROMISING. There is no ordinary numerical INCONCLUSIVE
category; a genuine execution failure is recorded separately as TECHNICAL_BLOCKER.
"""

from __future__ import annotations

from typing import Dict, List

from . import MEAN_A_THRESHOLD, MEAN_B_THRESHOLD

PROMISING = "PROMISING"
NOT_PROMISING = "NOT_PROMISING"
TECHNICAL_BLOCKER = "TECHNICAL_BLOCKER"

TARGET_KEYS = (8, 12)


def _series(per_target: Dict[int, Dict[str, float]], key: str) -> List[float]:
    return [per_target[m][key] for m in TARGET_KEYS]


def verdict(per_target: Dict[int, Dict[str, float]]) -> str:
    """Mechanical application of the frozen five-condition rule."""
    missing = [m for m in TARGET_KEYS if m not in per_target]
    if missing:
        raise ValueError(f"missing targets {missing}")

    identity = _series(per_target, "R2_identity")
    a_vals = _series(per_target, "A")
    b_vals = _series(per_target, "B")

    if not all(v > 0.0 for v in identity):
        return NOT_PROMISING
    if not all(v > 0.0 for v in a_vals):
        return NOT_PROMISING
    if not all(v > 0.0 for v in b_vals):
        return NOT_PROMISING
    if (sum(a_vals) / len(a_vals)) < MEAN_A_THRESHOLD:
        return NOT_PROMISING
    if (sum(b_vals) / len(b_vals)) < MEAN_B_THRESHOLD:
        return NOT_PROMISING
    return PROMISING


def condition_report(per_target: Dict[int, Dict[str, float]]) -> List[Dict[str, object]]:
    """Per-condition pass/fail for reporting. Not a second rule."""
    identity = _series(per_target, "R2_identity")
    a_vals = _series(per_target, "A")
    b_vals = _series(per_target, "B")
    mean_a = sum(a_vals) / len(a_vals)
    mean_b = sum(b_vals) / len(b_vals)

    def fmt(vals):
        return ", ".join(f"L{m} {v:+.5f}" for m, v in zip(TARGET_KEYS, vals))

    return [
        {"condition": 1, "requirement": "R2_identity > 0 at both targets",
         "observed": fmt(identity), "passed": all(v > 0.0 for v in identity)},
        {"condition": 2, "requirement": "A > 0 at both targets",
         "observed": fmt(a_vals), "passed": all(v > 0.0 for v in a_vals)},
        {"condition": 3, "requirement": "B > 0 at both targets",
         "observed": fmt(b_vals), "passed": all(v > 0.0 for v in b_vals)},
        {"condition": 4, "requirement": "mean(A) >= +0.02",
         "observed": f"mean(A) = {mean_a:+.5f}", "passed": mean_a >= MEAN_A_THRESHOLD},
        {"condition": 5, "requirement": "mean(B) >= +0.02",
         "observed": f"mean(B) = {mean_b:+.5f}", "passed": mean_b >= MEAN_B_THRESHOLD},
    ]


def decision_sentence(v: str) -> str:
    if v == PROMISING:
        return (
            "Individual selected-expert provenance contains incremental held-out "
            "information about future routing states beyond early fusion and an "
            "expert-identity-destroyed control, under this frozen OLMoE/WikiText "
            "setting."
        )
    if v == NOT_PROMISING:
        return (
            "EIPC-P0 does not show that preserving individual selected-expert "
            "provenance carries incremental information about future routing states "
            "beyond early-fused history and an identity-destroyed control, under "
            "this frozen OLMoE/WikiText setting. The expert-identity-cache premise "
            "is stopped."
        )
    return "EIPC-P0 could not produce its frozen statistics due to an execution failure."


def stopping_note(v: str) -> str:
    if v == PROMISING:
        return (
            "This does NOT prove an expert cache architecture improves performance. "
            "EIPC-P0 stops here; cache attention is not implemented. A later method "
            "experiment requires explicit user authorization."
        )
    return (
        "The expert-identity-cache premise is stopped. No added target layers, no "
        "larger n, no PCA or projection-dimension change, no alpha change, no other "
        "model or dataset, no nonlinear probes, and no attention built anyway. No "
        "automatic EIPC-P1."
    )


def summarize(per_target: Dict[int, Dict], target_meta: Dict[int, Dict]) -> Dict:
    a_vals = _series(per_target, "A")
    b_vals = _series(per_target, "B")
    mean_a = sum(a_vals) / len(a_vals)
    mean_b = sum(b_vals) / len(b_vals)
    v = verdict(per_target)
    return {
        "targets": {str(m): {**target_meta[m], **per_target[m]} for m in TARGET_KEYS},
        "mean_A": mean_a,
        "mean_B": mean_b,
        "final_verdict": v,
        "decision_sentence": decision_sentence(v),
        "stopping_note": stopping_note(v),
        "conditions": condition_report(per_target),
        "judgement_rule": {
            "condition_1": "R2_identity > 0 at both targets",
            "condition_2": "A > 0 at both targets",
            "condition_3": "B > 0 at both targets",
            "condition_4": "mean(A) >= +0.02",
            "condition_5": "mean(B) >= +0.02",
            "any_failure": "NOT_PROMISING",
            "frozen_before_results": True,
        },
    }
