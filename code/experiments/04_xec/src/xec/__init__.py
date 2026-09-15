"""XEC-P0 — Cross-Layer Expert Cache Actionability Pilot.

Can historical selected-expert states be converted into an actual Layer-12 routing
decision that improves true next-token likelihood, beyond native routing, a
current-only learned policy, and a fused-history cache?

Scope is fixed by docs/RESEARCH_CHARTER.md and protocols/XEC_P0_PREREGISTRATION.md.
"""

# Frozen constants. Preregistered; do not change.
MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
DATASET_SPLIT = "train"

# Read-only prior project whose blocks are excluded and whose result fixed the target.
EIPC_MANIFEST = (
    "/home/h-li/work/expert_identity_preservation/artifacts/EIPC_P0/data_manifest.json"
)

SAMPLE_SEED = 20260918
PROJECTION_SEED = 20260918
BOOTSTRAP_SEED = 314159
POLICY_SEEDS = (42, 123, 2026)

CONTEXT_LEN = 128
BLOCK_LEN = 129
N_TRAIN = 2048
N_VALIDATION = 512
N_TEST = 1024

# Target: human-readable Layer 12 -> code index 11. History: Layers 1..11 -> 0..10.
TARGET_LAYER_HUMAN = 12
TARGET_LAYER = 11
HISTORY_HUMAN = tuple(range(1, 12))
HISTORY_CODE = tuple(range(0, 11))
N_HISTORY_LAYERS = 11

TOP_K = 8
NUM_EXPERTS = 64
HIDDEN_SIZE = 2048

# Five routing actions: action 0 is native; actions 1-4 swap rank 8 for rank 9..12.
N_ACTIONS = 5
SWAP_RANKS = (9, 10, 11, 12)
MAX_RANK = 12  # native Top-12 identities/probabilities are cached

# Fixed non-trainable cache projection R^2048 -> R^64.
PROJ_DIM = 64

# Cache item layout: [LayerNorm(s) 64 ; layer one-hot 11 ; expert one-hot 64].
ITEM_DIM = PROJ_DIM + N_HISTORY_LAYERS + NUM_EXPERTS  # 139
N_EXPERT_ITEMS = N_HISTORY_LAYERS * TOP_K  # 88
N_FUSED_ITEMS = N_HISTORY_LAYERS  # 11

# Current-context vector u = [LayerNorm(x) 2048 ; centered(g) 64].
CURRENT_DIM = HIDDEN_SIZE + NUM_EXPERTS  # 2112

# Policy: single-head attention, d = 64; head sees [q ; m] = 128 dims.
ATTN_DIM = 64
POLICY_IN_DIM = 2 * ATTN_DIM  # 128

# Training.
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
BATCH_SIZE = 64
MAX_EPOCHS = 20

VARIANTS = ("CURRENT_ONLY", "FUSED_CACHE", "EXPERT_CACHE")

# Frozen decision thresholds.
DELTA_THRESHOLD = -0.005  # nats/token, for each paired mean
N_BOOTSTRAP = 10000

__all__ = [
    "MODEL_ID", "MODEL_REVISION", "DATASET_ID", "DATASET_CONFIG", "DATASET_REVISION",
    "DATASET_SPLIT", "EIPC_MANIFEST", "SAMPLE_SEED", "PROJECTION_SEED",
    "BOOTSTRAP_SEED", "POLICY_SEEDS", "CONTEXT_LEN", "BLOCK_LEN", "N_TRAIN",
    "N_VALIDATION", "N_TEST", "TARGET_LAYER_HUMAN", "TARGET_LAYER", "HISTORY_HUMAN",
    "HISTORY_CODE", "N_HISTORY_LAYERS", "TOP_K", "NUM_EXPERTS", "HIDDEN_SIZE",
    "N_ACTIONS", "SWAP_RANKS", "MAX_RANK", "PROJ_DIM", "ITEM_DIM", "N_EXPERT_ITEMS",
    "N_FUSED_ITEMS", "CURRENT_DIM", "ATTN_DIM", "POLICY_IN_DIM", "LEARNING_RATE",
    "WEIGHT_DECAY", "BATCH_SIZE", "MAX_EPOCHS", "VARIANTS", "DELTA_THRESHOLD",
    "N_BOOTSTRAP",
]
