"""EPD-P0 — Expert Provenance Decomposition Pilot.

Exploratory decomposition of the positive EIPC-P0 provenance signal into expert
identity/path, expert activation content, and their interaction.

Scope is fixed by docs/RESEARCH_QUESTION.md and protocols/EPD_P0_PROTOCOL.md.
"""

MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_SPLIT = "train"

# Read-only prior manifests whose blocks are excluded.
EIPC_MANIFEST = (
    "/home/h-li/work/expert_identity_preservation/artifacts/EIPC_P0/data_manifest.json")
XEC_MANIFEST = (
    "/home/h-li/work/cross_layer_expert_cache/artifacts/XEC_P0/data_manifest.json")
PRIOR_TRAIN_UNUSED_MANIFESTS = (
    "/home/h-li/work/rejected_expert_delayed_value/artifacts/REDV_V1/data_manifest.json",
    "/home/h-li/work/rejected_expert_delayed_value/artifacts/REDV_V2/data_manifest.json",
    "/home/h-li/work/delayed_rejected_evidence/artifacts/DREV_P0/data_manifest.json",
)

# The EIPC-P0 projection is reused verbatim, not regenerated.
EIPC_PROJECTION_JSON = (
    "/home/h-li/work/expert_identity_preservation/artifacts/EIPC_P0/projection.json")
EIPC_PROJECTION_SHA256 = (
    "fb9e6e9b7568f6d4a5ce74a7bee1ba2eab5a48a02fb2a93f80520e4ec0480282")
EIPC_PROJECTION_SEED = 20260917
PROJ_DIM = 32

SAMPLE_SEED = 20260919
PCA_SEED = 20260919

CONTEXT_LEN = 128
BLOCK_LEN = 129
N_FIT = 512
N_TEST = 256
N_TOTAL = N_FIT + N_TEST

# Four deterministic shards of 192 samples each.
N_SHARDS = 4
SHARD_SIZE = N_TOTAL // N_SHARDS

# Target: human-readable Layer 12 -> code 11. History: Layers 1..11 -> 0..10.
TARGET_LAYER_HUMAN = 12
TARGET_LAYER = 11
HISTORY_HUMAN = tuple(range(1, 12))
HISTORY_CODE = tuple(range(0, 11))
N_HISTORY_LAYERS = 11

TOP_K = 8
NUM_EXPERTS = 64
HIDDEN_SIZE = 2048

REPRESENTATIONS = ("FUSED", "ID_PATH", "CONTENT_RANK", "FULL_PROVENANCE")

# Raw dimensions per representation.
RAW_DIMS = {
    "FUSED": N_HISTORY_LAYERS * PROJ_DIM,                       # 352
    "ID_PATH": N_HISTORY_LAYERS * NUM_EXPERTS,                  # 704
    "CONTENT_RANK": N_HISTORY_LAYERS * TOP_K * PROJ_DIM,        # 2816
    "FULL_PROVENANCE": N_HISTORY_LAYERS * NUM_EXPERTS * PROJ_DIM,  # 22528
}

# Shared compressed budget for the main probes, and for the layerwise probes.
PCA_COMPONENTS = 32
LAYERWISE_PCA_COMPONENTS = 16

# Descriptive only: a gap of this size may be *described* as non-trivial.
NONTRIVIAL_GAP = 0.02
N_TOP_TRANSITIONS = 10

__all__ = [
    "MODEL_ID", "MODEL_REVISION", "DATASET_ID", "DATASET_CONFIG", "DATASET_REVISION",
    "DATASET_SPLIT", "EIPC_MANIFEST", "XEC_MANIFEST", "PRIOR_TRAIN_UNUSED_MANIFESTS",
    "EIPC_PROJECTION_JSON", "EIPC_PROJECTION_SHA256", "EIPC_PROJECTION_SEED", "PROJ_DIM",
    "SAMPLE_SEED", "PCA_SEED", "CONTEXT_LEN", "BLOCK_LEN", "N_FIT", "N_TEST", "N_TOTAL",
    "N_SHARDS", "SHARD_SIZE", "TARGET_LAYER_HUMAN", "TARGET_LAYER", "HISTORY_HUMAN",
    "HISTORY_CODE", "N_HISTORY_LAYERS", "TOP_K", "NUM_EXPERTS", "HIDDEN_SIZE",
    "REPRESENTATIONS", "RAW_DIMS", "PCA_COMPONENTS", "LAYERWISE_PCA_COMPONENTS",
    "NONTRIVIAL_GAP", "N_TOP_TRANSITIONS",
]
