# productions/reconfiguration
from model.core.productions.base import Production, CognitiveState, TaskPhase
from model.utils.config import COG_CONTROL_UTILITY


class Reconfig1(Production):
    """Typing to N-Back reconfiguration."""

    def __init__(self):
        super().__init__("reconfig-1", utility=COG_CONTROL_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.TASK_SWITCHING_1
            and state.switch_stage == "to_nback"
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        state.switch_stage = "done"


class Reconfig2(Production):
    """N-Back to Typing reconfiguration."""

    def __init__(self):
        super().__init__("reconfig-2", utility=COG_CONTROL_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.TASK_SWITCHING_2
            and state.switch_stage == "to_typing"
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        state.switch_stage = "done"
