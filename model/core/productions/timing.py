# productions/timing
from model.core.productions.base import Production, CognitiveState
from model.utils.config import TIMING_UTILITY


class CheckBuffer(Production):
    """Sample internal clock to track time."""

    def __init__(self):
        super().__init__("check-buffer", utility=TIMING_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return state.timing_active

    def _complete_operation(self, state: CognitiveState) -> None:
        pass
