# productions/interruption
from model.core.productions.base import Production, CognitiveState, TaskPhase
import random
from model.utils.config import (
    scaled,
    COG_CONTROL_UTILITY,
    PS_STORE_RANGE,
    PS_RESTORE_RANGE,
)


class StoreProblemState(Production):
    """Store problem state during task interruption."""

    def __init__(self):
        super().__init__("ps-store", utility=COG_CONTROL_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.TASK_SWITCHING_1
            and state.switch_stage == "ps_store"
        )

    def _start_operation(self, state: CognitiveState) -> None:
        random_cycles = scaled(random.randint(*PS_STORE_RANGE))
        state.operation_cycles_total = random_cycles
        state.operation_cycles_remaining = random_cycles

    def _complete_operation(self, state: CognitiveState) -> None:
        state.ps_stored_contents = state.ps_slots.copy()
        state.switch_stage = "done"


class RestoreProblemState(Production):
    """Restore problem state after interruption."""

    def __init__(self):
        super().__init__("ps-restore", utility=COG_CONTROL_UTILITY, cycles=None)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.TASK_SWITCHING_2
            and state.switch_stage == "ps_restore"
        )

    def _start_operation(self, state: CognitiveState) -> None:
        random_cycles = scaled(random.randint(*PS_RESTORE_RANGE))
        state.operation_cycles_total = random_cycles
        state.operation_cycles_remaining = random_cycles

    def _complete_operation(self, state: CognitiveState) -> None:
        if (
            hasattr(state, "ps_stored_contents")
            and state.ps_stored_contents is not None
        ):
            state.ps_slots = state.ps_stored_contents.copy()
            if "position" in state.ps_slots:
                state.typing_position = state.ps_slots["position"]

        state.switch_stage = "done"
