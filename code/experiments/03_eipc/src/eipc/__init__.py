"""EIPC-P0 — Expert-Identity Preservation Cache Pilot.

Does preserving individual selected-expert provenance across depth contain
incremental information about future routing states beyond early-fused history and
an expert-identity-destroyed control?

Scope is fixed by docs/RESEARCH_CHARTER.md and
protocols/EIPC_P0_PREREGISTRATION.md.
"""

# Frozen constants. Preregistered; do not change.
MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

# EIPC-P0 uses ONLY the WikiText train split; prior projects never touched it.
DATASET_SPLIT = "train"

SAMPLE_SEED = 20260917
PROJECTION_SEED = 20260917
PCA_SEED = 20260917
SHUFFLE_SEED = 314159

CONTEXT_LEN = 128
BLOCK_LEN = 129
N_FIT = 1024
N_TEST = 1024

# Two future routing targets: human-readable layer -> code index.
# Target A: Layer 8 (index 7), history Layers 1..7 (indices 0..6).
# Target B: Layer 12 (index 11), history Layers 1..11 (indices 0..10).
TARGETS = {
    8: {"code": 7, "history_human": tuple(range(1, 8)), "history_code": tuple(range(0, 7))},
    12: {"code": 11, "history_human": tuple(range(1, 12)), "history_code": tuple(range(0, 11))},
}

# All historical layers any target needs: Layers 1..11 -> indices 0..10.
ALL_HISTORY_CODE = tuple(range(0, 11))

TOP_K = 8
NUM_EXPERTS = 64
HIDDEN_SIZE = 2048

# Fixed non-trainable random projection R^2048 -> R^32.
PROJ_DIM = 32

# Every representation is compressed to exactly this many dimensions.
PCA_COMPONENTS = 64

# Frozen decision thresholds.
MEAN_A_THRESHOLD = 0.02
MEAN_B_THRESHOLD = 0.02

REPRESENTATIONS = ("FUSED", "EXPERT_IDENTITY", "SHUFFLED_IDENTITY")

__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "DATASET_ID",
    "DATASET_CONFIG",
    "DATASET_REVISION",
    "DATASET_SPLIT",
    "SAMPLE_SEED",
    "PROJECTION_SEED",
    "PCA_SEED",
    "SHUFFLE_SEED",
    "CONTEXT_LEN",
    "BLOCK_LEN",
    "N_FIT",
    "N_TEST",
    "TARGETS",
    "ALL_HISTORY_CODE",
    "TOP_K",
    "NUM_EXPERTS",
    "HIDDEN_SIZE",
    "PROJ_DIM",
    "PCA_COMPONENTS",
    "MEAN_A_THRESHOLD",
    "MEAN_B_THRESHOLD",
    "REPRESENTATIONS",
]
