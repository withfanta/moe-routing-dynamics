"""REDV-V1 — Rejected-Expert Delayed Value V1.

A single phenomenon test: does rejected near-miss expert information at MoE
layer l add held-out predictive information about counterfactual routing regret
at layer l+1?

Scope is fixed by docs/RESEARCH_CHARTER.md and protocols/REDV_V1_PREREGISTRATION.md.
"""

# Frozen constants. These are preregistered and must not be changed.
MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

SEED = 20260914
CONTEXT_LEN = 128
BLOCK_LEN = 129  # 128 context tokens + 1 next-token target
N_VALIDATION = 512
N_TEST = 512

# Human-readable 1-based transitions 4->5, 8->9, 12->13; stored 0-based.
TRANSITIONS = ((3, 4), (7, 8), (11, 12))

# Near-miss rejected experts are exactly ranks 9-12 (1-based ranks).
REJECTED_RANKS = (9, 10, 11, 12)
TOP_K = 8
NUM_EXPERTS = 64
HIDDEN_SIZE = 2048

# Pre-registered judgement thresholds.
SUPPORTED_MEAN_THRESHOLD = 0.02
NOT_SUPPORTED_MEAN_THRESHOLD = 0.01

__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "DATASET_ID",
    "DATASET_CONFIG",
    "DATASET_REVISION",
    "SEED",
    "CONTEXT_LEN",
    "BLOCK_LEN",
    "N_VALIDATION",
    "N_TEST",
    "TRANSITIONS",
    "REJECTED_RANKS",
    "TOP_K",
    "NUM_EXPERTS",
    "HIDDEN_SIZE",
    "SUPPORTED_MEAN_THRESHOLD",
    "NOT_SUPPORTED_MEAN_THRESHOLD",
]
