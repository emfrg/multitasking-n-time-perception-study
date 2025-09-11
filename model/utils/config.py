# utils/config
from typing import Union, Tuple, Dict, List

SCALE = 1


def scaled(n: Union[int, float]) -> int:
    """Scale value by global multiplier."""
    return max(1, int(round(n * SCALE)))


# Production cycles
STATIC_CYCLES = {
    # Typing
    "read-word": 4,
    "extract-character": 2,
    "find-button": 4,
    "move-cursor": 6,
    "click-button": 2,
    "update-position": 2,
    # N-back task
    "attend-stimulus": 3,
    "retrieve-1back": 2,
    "retrieve-2back": 6,
    "compare-nback": 3,
    "update-nback-buffers": 2,
    "respond-same": 2,
    "respond-different": 2,
    # Task switching
    "reconfig-1": 4,
    "reconfig-2": 8,
    # Timing
    "check-buffer": 1,
}

# PS operation ranges (min, max)
PS_STORE_RANGE: Tuple[int, int] = (6, 12)
PS_RESTORE_RANGE: Tuple[int, int] = (8, 16)

# Utility values
COG_CONTROL_UTILITY = 1.2
NBACK_UTILITY = 1.0
TYPING_UTILITY = 0.9
TIMING_UTILITY = 1.0

# Production system parameters
PRODUCTION_NOISE_SD = 0.1

# Weber fractions for time estimation
NBACK_WEBER_EASY = None
NBACK_WEBER_HARD = None
TYPING_WEBER = None

# N-back task parameters
NBACK_CYCLES_PER_STIM = 24
NBACK_NUM_STIMULI_MIN = 4
NBACK_NUM_STIMULI_MAX = 9
NBACK_TARGET_RATE = 0.3

# Task switching parameters
TASK_SWITCH_PHASE1_CYCLES = 20  # Fixed interruption lag

# Safety limits
MAX_CYCLES_PER_WORD = 10000

# Simulation parameters
PRINT_SUMMARIES = True

# Single task simulation parameters
TYPING_SIM_PARTICIPANTS = 10
TYPING_SIM_TRIALS = 40
NBACK_SIM_PARTICIPANTS = 10
NBACK_SIM_TRIALS = 20  # per difficulty

# Single task seed configuration
TYPING_SIM_SEED_OFFSET = 300
NBACK_SIM_STIMULI_SEED = 42
NBACK_SIM_SEED_OFFSET_1BACK = 100
NBACK_SIM_SEED_OFFSET_2BACK = 200

# Experiment parameters
DEFAULT_NUM_PARTICIPANTS = 26
DEFAULT_TRIALS_PER_CONDITION = 10

# Experiment seed configuration
SEED_NBACK_OFFSET = 1
SEED_PARTICIPANT_MULTIPLIER = 1000
SEED_BLOCK_MULTIPLIER = 100

# Typing word bank
TYPING_WORD_BANK: Dict[int, List[str]] = {
    6: [
        "HANDLE",
        "MIDDLE",
        "SIMPLE",
        "TEMPLE",
        "CANDLE",
        "BUNDLE",
    ],
    7: [
        "EXAMPLE",
        "FOREVER",
        "WITHOUT",
        "ANOTHER",
        "FACTORY",
    ],
    8: [
        "ELEPHANT",
        "COMPUTER",
        "ANYTHING",
        "HOMEMADE",
        "NOTEBOOK",
    ],
    9: [
        "CHARACTER",
        "IMPORTANT",
        "UNDERSTAND",
        "DIAMETERS",
        "NOTEWORTHY",
    ],
}
