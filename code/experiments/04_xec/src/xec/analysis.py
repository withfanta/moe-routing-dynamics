"""Paired bootstrap and the frozen XEC-P0 decision rule.

Per TEST sample, with seed-averaged EXPERT_CACHE NLL:

    d_native_i  = mean_seeds(EXPERT_i) - native_i
    d_current_i = mean_seeds(EXPERT_i) - mean_seeds(CURRENT_ONLY_i)
    d_fused_i   = mean_seeds(EXPERT_i) - mean_seeds(FUSED_CACHE_i)

Negative means EXPERT_CACHE is better. A paired bootstrap over TEST sample indices
(10000 resamples, seed 314159) gives a 95% CI for each mean; seeds are never resampled
independently.

PROMISING requires all seven conditions:

  1/3/5. each paired mean <= -0.005 nats/token;
  2/4/6. each 95% CI upper bound < 0;
  7.     EXPERT_CACHE beats FUSED_CACHE in mean TEST NLL in ALL THREE seeds.

Any failure yields NOT_PROMISING. TECHNICAL_BLOCKER is reserved for an experiment that
cannot be executed.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from . import BOOTSTRAP_SEED, DELTA_THRESHOLD, N_BOOTSTRAP, POLICY_SEEDS

PROMISING = "PROMISING"
NOT_PROMISING = "NOT_PROMISING"
TECHNICAL_BLOCKER = "TECHNICAL_BLOCKER"

COMPARISONS = ("d_native", "d_current", "d_fused")


def paired_bootstrap_ci(d: np.ndarray, n_boot: int = N_BOOTSTRAP,
                        seed: int = BOOTSTRAP_SEED) -> Dict[str, float]:
    """Paired bootstrap 95% CI for the mean, resampling sample indices only."""
    d = np.asarray(d, dtype=np.float64)
    n = d.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
        "n_bootstrap": int(n_boot),
        "bootstrap_seed": int(seed),
    }


def build_differences(native: np.ndarray, per_seed: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Seed-average each learned variant, then form the three paired differences."""
    def avg(variant):
        stack = np.stack([per_seed[variant][s] for s in POLICY_SEEDS], axis=0)
        return stack.mean(axis=0)

    expert = avg("EXPERT_CACHE")
    return {
        "d_native": expert - np.asarray(native, dtype=np.float64),
        "d_current": expert - avg("CURRENT_ONLY"),
        "d_fused": expert - avg("FUSED_CACHE"),
    }


def per_seed_means(per_seed: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, Dict[str, float]]:
    out = {}
    for variant, by_seed in per_seed.items():
        out[variant] = {str(s): float(np.mean(by_seed[s])) for s in POLICY_SEEDS}
        out[variant]["mean"] = float(np.mean([np.mean(by_seed[s]) for s in POLICY_SEEDS]))
    return out


def expert_beats_fused_all_seeds(per_seed: Dict[str, Dict[int, np.ndarray]]) -> Dict[str, object]:
    """Condition 7: strictly lower mean TEST NLL for EXPERT_CACHE in every seed."""
    detail = {}
    all_win = True
    for s in POLICY_SEEDS:
        e = float(np.mean(per_seed["EXPERT_CACHE"][s]))
        f = float(np.mean(per_seed["FUSED_CACHE"][s]))
        win = e < f
        detail[str(s)] = {"expert": e, "fused": f, "expert_lower": win}
        all_win = all_win and win
    return {"all_seeds_expert_lower": all_win, "per_seed": detail}


def verdict(stats: Dict[str, Dict[str, float]], seed_check: Dict[str, object]) -> str:
    """Mechanical application of the frozen seven-condition rule."""
    for key in COMPARISONS:
        if key not in stats:
            raise ValueError(f"missing comparison {key}")
        if not (stats[key]["mean"] <= DELTA_THRESHOLD):
            return NOT_PROMISING
        if not (stats[key]["ci_high"] < 0.0):
            return NOT_PROMISING
    if not seed_check["all_seeds_expert_lower"]:
        return NOT_PROMISING
    return PROMISING


def condition_report(stats: Dict[str, Dict[str, float]], seed_check: Dict[str, object]) -> List[Dict]:
    rows = []
    n = 0
    for key in COMPARISONS:
        st = stats[key]
        n += 1
        rows.append({
            "condition": n,
            "requirement": f"mean {key} <= {DELTA_THRESHOLD}",
            "observed": f"mean = {st['mean']:+.6f}",
            "passed": bool(st["mean"] <= DELTA_THRESHOLD),
        })
        n += 1
        rows.append({
            "condition": n,
            "requirement": f"95% CI upper bound for {key} < 0",
            "observed": f"CI [{st['ci_low']:+.6f}, {st['ci_high']:+.6f}]",
            "passed": bool(st["ci_high"] < 0.0),
        })
    rows.append({
        "condition": 7,
        "requirement": "EXPERT_CACHE beats FUSED_CACHE in mean TEST NLL in all three seeds",
        "observed": ", ".join(
            f"seed {s}: {d['expert']:.5f} vs {d['fused']:.5f}"
            for s, d in seed_check["per_seed"].items()),
        "passed": bool(seed_check["all_seeds_expert_lower"]),
    })
    return rows


def decision_sentence(v: str) -> str:
    if v == PROMISING:
        return (
            "Under frozen OLMoE/WikiText and a restricted five-action Layer-12 "
            "intervention, dynamically retrieving historical selected-expert states "
            "provides actionable routing information that improves true next-token "
            "likelihood beyond native routing, a current-only learned policy, and a "
            "fused-history cache."
        )
    if v == NOT_PROMISING:
        return (
            "XEC-P0 does not show that historical selected-expert states can be "
            "converted into a Layer-12 routing decision that improves true next-token "
            "likelihood beyond native routing, a current-only learned policy, and a "
            "fused-history cache, under this frozen setting. The expert-cache method "
            "direction is stopped."
        )
    return "XEC-P0 could not be executed as frozen."


def stopping_note(v: str) -> str:
    if v == PROMISING:
        return (
            "This does NOT mean the final architecture is solved, that MoE improves "
            "generally, that latency or memory improves, or that other layers, datasets "
            "or models benefit. XEC-P0 stops here; a broader architecture experiment "
            "requires explicit user authorization."
        )
    return (
        "The expert-cache method direction is stopped. No more actions, no target-layer "
        "change, no cache-dimension tuning, no larger sample count, no attention heads, "
        "no optimizer change, no Layer 8, no other model or dataset, no rescue."
    )


def summarize(stats, seed_check, per_seed_nll, descriptive) -> Dict:
    v = verdict(stats, seed_check)
    return {
        "comparisons": stats,
        "seed_check": seed_check,
        "per_seed_mean_nll": per_seed_nll,
        "descriptive": descriptive,
        "final_verdict": v,
        "decision_sentence": decision_sentence(v),
        "stopping_note": stopping_note(v),
        "conditions": condition_report(stats, seed_check),
        "judgement_rule": {
            "paired_means": f"each of d_native, d_current, d_fused <= {DELTA_THRESHOLD}",
            "ci": "each 95% paired-bootstrap CI upper bound < 0",
            "seed_consistency": "EXPERT_CACHE < FUSED_CACHE mean NLL in all three seeds",
            "any_failure": "NOT_PROMISING",
            "frozen_before_results": True,
        },
    }
