"""Pre-registered judgement rule for REDV-V1.

There are exactly three primary Delta_R2 values (transitions 4->5, 8->9,
12->13). Their arithmetic mean drives the verdict:

    SUPPORTED       mean Delta_R2 >= +0.02  AND  all three Delta_R2 > 0
    NOT_SUPPORTED   mean Delta_R2 <  +0.01  OR   at least two Delta_R2 <= 0
    INCONCLUSIVE    otherwise

Thresholds are frozen at preregistration. There is exactly one statistical rule.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from . import NOT_SUPPORTED_MEAN_THRESHOLD, SUPPORTED_MEAN_THRESHOLD

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
INCONCLUSIVE = "INCONCLUSIVE"


def verdict(deltas: Sequence[float]) -> str:
    """Apply the pre-registered rule to the three Delta_R2 values."""
    if len(deltas) != 3:
        raise ValueError(f"expected exactly 3 Delta_R2 values, got {len(deltas)}")

    mean_delta = sum(deltas) / 3.0
    n_nonpositive = sum(1 for d in deltas if d <= 0.0)

    # NOT_SUPPORTED is checked first: it is the disjunctive condition, and a
    # result meeting it cannot simultaneously meet the conjunctive SUPPORTED
    # condition.
    if mean_delta < NOT_SUPPORTED_MEAN_THRESHOLD or n_nonpositive >= 2:
        return NOT_SUPPORTED
    if mean_delta >= SUPPORTED_MEAN_THRESHOLD and all(d > 0.0 for d in deltas):
        return SUPPORTED
    return INCONCLUSIVE


def decision_sentence(v: str) -> str:
    """The exact sentence recorded in the decision log for each verdict."""
    if v == SUPPORTED:
        return (
            "REDV-V1 supports delayed predictive value of rejected near-miss "
            "expert evidence under the frozen OLMoE/WikiText setting."
        )
    if v == NOT_SUPPORTED:
        return (
            "REDV-V1 does not support sufficient incremental delayed predictive "
            "value under the frozen OLMoE/WikiText setting. Direction stopped."
        )
    if v == INCONCLUSIVE:
        return "REDV-V1 is inconclusive under the frozen setting. Direction stopped."
    raise ValueError(f"unknown verdict {v!r}")


def stopping_note(v: str) -> str:
    """What happens next under the pre-registered stopping rule."""
    if v == SUPPORTED:
        return (
            "The prerequisite phenomenon is supported. REDV-V1 stops here. No "
            "method is implemented. A later experiment requires explicit user "
            "authorization."
        )
    return "Direction stopped. No rescue experiment, no REDV-V2."


def summarize(per_transition: List[Dict], deltas: Sequence[float]) -> Dict:
    mean_delta = sum(deltas) / 3.0
    v = verdict(deltas)
    return {
        "transitions": per_transition,
        "mean_delta_r2": mean_delta,
        "verdict": v,
        "decision_sentence": decision_sentence(v),
        "stopping_note": stopping_note(v),
        "judgement_rule": {
            "supported": "mean Delta_R2 >= +0.02 AND all three Delta_R2 > 0",
            "not_supported": "mean Delta_R2 < +0.01 OR at least two Delta_R2 <= 0",
            "inconclusive": "otherwise",
            "frozen_before_results": True,
        },
    }
