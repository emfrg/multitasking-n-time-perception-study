# productions/base
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np

from model.utils.config import STATIC_CYCLES, scaled

DEFAULT_CYCLES = 1000


class TaskPhase(Enum):
    """Experimental trial phases."""

    TYPING = "typing"
    NBACK = "nback"
    TASK_SWITCHING_1 = "task_switching_1"
    TASK_SWITCHING_2 = "task_switching_2"


class Production:
    """Base production class for cognitive architecture."""

    def __init__(self, name: str, utility: float = 1.0, cycles: Optional[int] = None):
        self.name = name
        self.base_utility = utility
        base = STATIC_CYCLES.get(name, DEFAULT_CYCLES) if cycles is None else cycles
        self.cycles = scaled(base)

    def matches(self, state: "CognitiveState") -> bool:
        """Check if production can fire given current state."""
        if state.current_operation and state.current_operation != self.name:
            return False

        if state.current_operation == self.name and state.operation_started:
            return True

        return self._specific_matches(state)

    def _specific_matches(self, state: "CognitiveState") -> bool:
        """Production-specific matching logic."""
        raise NotImplementedError(
            f"Production {self.name} must implement _specific_matches()"
        )

    def fire(self, state: "CognitiveState") -> None:
        """Execute production, handling multi-cycle operations."""
        if not state.operation_started:
            state.current_operation = self.name
            state.operation_cycles_total = self.cycles
            state.operation_cycles_remaining = self.cycles
            state.operation_started = True
            self._start_operation(state)

        state.operation_cycles_remaining -= 1
        self._continue_operation(state)

        if state.operation_cycles_remaining == 0:
            self._complete_operation(state)
            state.current_operation = None
            state.operation_started = False

    def _start_operation(self, state: "CognitiveState") -> None:
        """Initialize multi-cycle operation."""
        pass

    def _continue_operation(self, state: "CognitiveState") -> None:
        """Process ongoing operation."""
        pass

    def _complete_operation(self, state: "CognitiveState") -> None:
        """Finalize operation completion."""
        raise NotImplementedError(
            f"Production {self.name} must implement _complete_operation()"
        )

    def get_utility(self, state: "CognitiveState") -> float:
        """Return utility value for production competition."""
        return self.base_utility


@dataclass
class CognitiveState:
    """Central cognitive state tracking all task-relevant information."""

    # Task state
    current_phase: TaskPhase = TaskPhase.TYPING
    nback_difficulty: str = "easy"

    # Timing state
    timing_active: bool = True
    current_time: float = 0.0

    # Typing task state
    typing_word: str = ""
    typing_position: int = 0
    current_character: Optional[str] = None
    cursor_at_target: bool = False
    button_clicked: bool = False
    button_found: bool = False

    # N-back state
    nback_current: Optional[str] = None
    nback_response: Optional[str] = None
    nback_attended: bool = False
    nback_retrieved: bool = False
    nback_retrieved_value: Optional[str] = None
    nback_responded: bool = False
    nback_buffers_updated: bool = False

    # Multi-cycle operation tracking
    current_operation: Optional[str] = None
    operation_cycles_total: int = 0
    operation_cycles_remaining: int = 0
    operation_started: bool = False

    # Problem state
    ps_slots: Dict[str, Any] = field(default_factory=dict)
    ps_stored_contents: Optional[Dict[str, Any]] = None

    # Control state
    cycle_count: int = 0
    productions_fired: List[str] = field(default_factory=list)
    last_production: Optional[str] = None
    switch_stage: Optional[str] = None

    def log_production(self, production_name: str):
        """Record production firing for analysis."""
        self.productions_fired.append(f"{self.cycle_count}:{production_name}")
        self.last_production = production_name
