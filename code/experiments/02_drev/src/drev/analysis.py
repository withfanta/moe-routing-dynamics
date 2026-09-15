"""The frozen DREV-P0 pilot decision rule.

Delayed horizons are Δ = 2, 4, 8 (target layers 6, 8, 12). Δ = 1 (layer 5) is the
immediate horizon.

    mean_D_delayed = mean(D_6, D_8, D_12)

PROMISING requires ALL FOUR conditions:

  1. at least two of the three delayed horizons have R2_real > 0;
  2. at least two of the three delayed horizons have D_m > 0;
  3. mean_D_delayed >= +0.02;
  4. mean_D_delayed > D_5.

Any failure yields NOT_PROMISING. This is a screening result, not a universal
theorem.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from . import DELAYED_DISTANCES, IMMEDIATE_DISTANCE, MEAN_D_DELAYED_THRESHOLD

PROMISING = "PROMISING"
NOT_PROMISING = "NOT_PROMISING"


def mean_delayed_D(per_distance: Dict[int, Dict[str, float]]) -> float:
    ds = [per_distance[d]["D"] for d in DELAYED_DISTANCES]
    return sum(ds) / len(ds)


def verdict(per_distance: Dict[int, Dict[str, float]]) -> str:
    """Mechanical application of the frozen four-condition rule."""
    missing = [d for d in (IMMEDIATE_DISTANCE,) + DELAYED_DISTANCES if d not in per_distance]
    if missing:
        raise ValueError(f"missing horizons {missing}")

    delayed_reals = [per_distance[d]["R2_real"] for d in DELAYED_DISTANCES]
    delayed_ds = [per_distance[d]["D"] for d in DELAYED_DISTANCES]
    mean_d = sum(delayed_ds) / len(delayed_ds)
    d_immediate = per_distance[IMMEDIATE_DISTANCE]["D"]

    if sum(1 for v in delayed_reals if v > 0.0) < 2:
        return NOT_PROMISING
    if sum(1 for v in delayed_ds if v > 0.0) < 2:
        return NOT_PROMISING
    if mean_d < MEAN_D_DELAYED_THRESHOLD:
        return NOT_PROMISING
    if not (mean_d > d_immediate):
        return NOT_PROMISING
    return PROMISING


def condition_report(per_distance: Dict[int, Dict[str, float]]) -> List[Dict[str, object]]:
    """Per-condition pass/fail, for reporting only. Not a second rule."""
    delayed_reals = [per_distance[d]["R2_real"] for d in DELAYED_DISTANCES]
    delayed_ds = [per_distance[d]["D"] for d in DELAYED_DISTANCES]
    mean_d = sum(delayed_ds) / len(delayed_ds)
    d_immediate = per_distance[IMMEDIATE_DISTANCE]["D"]

    n_real_pos = sum(1 for v in delayed_reals if v > 0.0)
    n_d_pos = sum(1 for v in delayed_ds if v > 0.0)

    return [
        {
            "condition": 1,
            "requirement": "at least 2 of 3 delayed horizons have R2_real > 0",
            "observed": f"{n_real_pos} of 3 positive",
            "passed": n_real_pos >= 2,
        },
        {
            "condition": 2,
            "requirement": "at least 2 of 3 delayed horizons have D > 0",
            "observed": f"{n_d_pos} of 3 positive",
            "passed": n_d_pos >= 2,
        },
        {
            "condition": 3,
            "requirement": "mean_D_delayed >= +0.02",
            "observed": f"mean_D_delayed = {mean_d:+.5f}",
            "passed": mean_d >= MEAN_D_DELAYED_THRESHOLD,
        },
        {
            "condition": 4,
            "requirement": "mean_D_delayed > D at the immediate horizon",
            "observed": f"{mean_d:+.5f} vs D_5 = {d_immediate:+.5f}",
            "passed": mean_d > d_immediate,
        },
    ]


def decision_sentence(v: str) -> str:
    if v == PROMISING:
        return (
            "Layer-4 rejected evidence contains sample-specific information about "
            "some later routing-regret horizons under this frozen pilot setting."
        )
    return (
        "DREV-P0 does not show that Layer-4 rejected evidence becomes more "
        "informative about routing regret at later depths than at the immediate "
        "next layer, under this frozen pilot setting. The delayed "
        "rejected-evidence hypothesis is stopped."
    )


def stopping_note(v: str) -> str:
    if v == PROMISING:
        return (
            "This does NOT mean cross-layer memory works. No memory and no attention "
            "is built. Any further experiment requires explicit user authorization."
        )
    return (
        "This delayed-rejected-evidence hypothesis is stopped. No added source "
        "layers or horizons, no PCA-dimension change, no alpha change, no larger n, "
        "no other model or dataset. No automatic DREV-P1."
    )


def summarize(per_distance: Dict[int, Dict], horizon_meta: Dict[int, Dict]) -> Dict:
    mean_d = mean_delayed_D(per_distance)
    v = verdict(per_distance)
    return {
        "horizons": {
            str(d): {**horizon_meta[d], **per_distance[d]}
            for d in sorted(per_distance)
        },
        "mean_D_delayed": mean_d,
        "D_immediate": per_distance[IMMEDIATE_DISTANCE]["D"],
        "final_verdict": v,
        "decision_sentence": decision_sentence(v),
        "stopping_note": stopping_note(v),
        "conditions": condition_report(per_distance),
        "judgement_rule": {
            "condition_1": "at least 2 of 3 delayed horizons have R2_real > 0",
            "condition_2": "at least 2 of 3 delayed horizons have D > 0",
            "condition_3": "mean_D_delayed >= +0.02",
            "condition_4": "mean_D_delayed > D_immediate",
            "any_failure": "NOT_PROMISING",
            "frozen_before_results": True,
        },
    }
