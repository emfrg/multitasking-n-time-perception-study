# productions/nback
from model.core.productions.base import Production, CognitiveState, TaskPhase
from model.utils.config import NBACK_UTILITY


class AttendStimulus(Production):
    """Attend to current stimulus."""

    def __init__(self):
        super().__init__("attend-stimulus", utility=NBACK_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.NBACK
            and state.nback_current is not None
            and not state.nback_attended
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        state.nback_attended = True


class Retrieve1Back(Production):
    """Retrieve n-1 value."""

    def __init__(self):
        super().__init__("retrieve-1back", utility=NBACK_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.NBACK
            and state.nback_difficulty == "easy"
            and state.nback_attended
            and not state.nback_retrieved
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        value = state.ps_slots.get("n-1")
        state.nback_retrieved_value = value
        state.nback_retrieved = True


class Retrieve2Back(Production):
    """Retrieve n-2 value."""

    def __init__(self):
        super().__init__("retrieve-2back", utility=NBACK_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.NBACK
            and state.nback_difficulty == "hard"
            and state.nback_attended
            and not state.nback_retrieved
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        value = state.ps_slots.get("n-2")
        state.nback_retrieved_value = value
        state.nback_retrieved = True


class CompareNBack(Production):
    """Compare stimulus with retrieved value."""

    def __init__(self):
        super().__init__("compare-nback", utility=NBACK_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.NBACK
            and state.nback_retrieved
            and state.nback_response is None
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        retrieved = state.nback_retrieved_value
        if retrieved is None:
            state.nback_response = "different"
        else:
            state.nback_response = (
                "same" if state.nback_current == retrieved else "different"
            )


class RespondSame(Production):
    """Press SPACE key for same."""

    def __init__(self):
        super().__init__("respond-same", utility=NBACK_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.NBACK
            and state.nback_response == "same"
            and not state.nback_responded
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        state.nback_responded = True


class RespondDifferent(Production):
    """Press x key for different."""

    def __init__(self):
        super().__init__("respond-different", utility=NBACK_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.NBACK
            and state.nback_response == "different"
            and not state.nback_responded
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        state.nback_responded = True


class UpdateNBackBuffers(Production):
    """Shift values in memory."""

    def __init__(self):
        super().__init__("update-nback-buffers", utility=NBACK_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.NBACK
            and state.nback_responded
            and not state.nback_buffers_updated
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        if state.nback_difficulty == "easy":
            state.ps_slots["n-1"] = state.nback_current
        else:
            state.ps_slots["n-2"] = state.ps_slots.get("n-1")
            state.ps_slots["n-1"] = state.nback_current

        state.nback_retrieved_value = None
        state.nback_buffers_updated = True
