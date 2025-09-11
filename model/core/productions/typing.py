# productions/typing
from model.core.productions.base import Production, CognitiveState, TaskPhase
from model.utils.config import TYPING_UTILITY


class ReadWord(Production):
    """Read complete word from display."""

    def __init__(self):
        super().__init__("read-word", utility=TYPING_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.TYPING
            and state.typing_position == 0
            and "word" not in state.ps_slots
            and state.typing_word != ""
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        state.ps_slots["word"] = state.typing_word
        state.ps_slots["position"] = 0


class ExtractCharacter(Production):
    """Extract character at current position."""

    def __init__(self):
        super().__init__("extract-character", utility=TYPING_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.TYPING
            and "word" in state.ps_slots
            and state.current_character is None
            and not state.button_clicked
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        word = state.ps_slots["word"]
        pos = state.typing_position
        if pos < len(word):
            state.current_character = word[pos]


class FindButton(Production):
    """Visually locate button."""

    def __init__(self):
        super().__init__("find-button", utility=TYPING_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.TYPING
            and state.current_character is not None
            and not state.cursor_at_target
            and not state.button_found
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        state.button_found = True


class MoveCursor(Production):
    """Move mouse cursor."""

    def __init__(self):
        super().__init__("move-cursor", utility=TYPING_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.TYPING
            and state.current_character is not None
            and not state.cursor_at_target
            and state.button_found
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        state.cursor_at_target = True


class ClickButton(Production):
    """Click mouse button."""

    def __init__(self):
        super().__init__("click-button", utility=TYPING_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return (
            state.current_phase == TaskPhase.TYPING
            and state.cursor_at_target
            and not state.button_clicked
        )

    def _complete_operation(self, state: CognitiveState) -> None:
        state.button_clicked = True


class UpdatePosition(Production):
    """Increment position and reset."""

    def __init__(self):
        super().__init__("update-position", utility=TYPING_UTILITY)

    def _specific_matches(self, state: CognitiveState) -> bool:
        return state.current_phase == TaskPhase.TYPING and state.button_clicked

    def _complete_operation(self, state: CognitiveState) -> None:
        state.typing_position += 1
        state.ps_slots["position"] = state.typing_position

        state.current_character = None
        state.cursor_at_target = False
        state.button_clicked = False
        state.button_found = False
