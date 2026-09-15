"""DREV-P0 — Delayed Rejected-Evidence Value Pilot.

Does rejected near-miss expert evidence at an early MoE layer carry
sample-specific information about routing regret several layers later?

Scope is fixed by docs/RESEARCH_CHARTER.md and
protocols/DREV_P0_PREREGISTRATION.md.
"""

# Frozen constants. Preregistered; do not change.
MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

SAMPLE_SEED = 20260916
PCA_SEED = 20260916
SHUFFLE_SEED = 271828

CONTEXT_LEN = 128
BLOCK_LEN = 129
N_FIT = 512
N_TEST = 512

# Exactly one source layer: human-readable Layer 4 -> code index 3.
SOURCE_LAYER_HUMAN = 4
SOURCE_LAYER = 3

# Four fixed horizons: {distance: (human target layer, code index)}.
HORIZONS = {
    1: (5, 4),
    2: (6, 5),
    4: (8, 7),
    8: (12, 11),
}
IMMEDIATE_DISTANCE = 1
DELAYED_DISTANCES = (2, 4, 8)

REJECTED_RANKS = (9, 10, 11, 12)
TOP_K = 8
NUM_EXPERTS = 64
HIDDEN_SIZE = 2048

# Frozen unsupervised compression: 64 PCA components per block, so p = 128.
PCA_COMPONENTS = 64
PROBE_DIM = 2 * PCA_COMPONENTS

MEAN_D_DELAYED_THRESHOLD = 0.02

# Read-only provenance of the closed prior project.
REDV_ROOT = "/home/h-li/work/rejected_expert_delayed_value"

__all__ = [
    "MODEL_ID",
    "MODEL_REVISION",
    "DATASET_ID",
    "DATASET_CONFIG",
    "DATASET_REVISION",
    "SAMPLE_SEED",
    "PCA_SEED",
    "SHUFFLE_SEED",
    "CONTEXT_LEN",
    "BLOCK_LEN",
    "N_FIT",
    "N_TEST",
    "SOURCE_LAYER",
    "SOURCE_LAYER_HUMAN",
    "HORIZONS",
    "IMMEDIATE_DISTANCE",
    "DELAYED_DISTANCES",
    "REJECTED_RANKS",
    "TOP_K",
    "NUM_EXPERTS",
    "HIDDEN_SIZE",
    "PCA_COMPONENTS",
    "PROBE_DIM",
    "MEAN_D_DELAYED_THRESHOLD",
    "REDV_ROOT",
]
