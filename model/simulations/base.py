# simulations/base
import numpy as np
import random
from typing import List, Optional
from model.core.productions.base import CognitiveState, Production
from model.core.systems.timing import TimingSystem
from model.utils.config import PRODUCTION_NOISE_SD


class BaseSimulation:
    """Base class for cognitive simulations with common functionality."""

    def __init__(self, seed=None, use_fixed_seed=False):
        """Initialize random number generators."""
        if use_fixed_seed and seed is not None:
            self.rng = np.random.default_rng(seed)
            random.seed(seed)
        else:
            self.rng = np.random.default_rng()
            random.seed()

        self.seed = seed
        self.use_fixed_seed = use_fixed_seed

        # To be set by subclasses
        self.productions = []
        self.state = CognitiveState()
        self.timing_system = None

    def find_matching_productions(self) -> List[Production]:
        """Find all productions that match current state."""
        return [p for p in self.productions if p.matches(self.state)]

    def select_production(self, matching: List[Production]) -> Optional[Production]:
        """Select winner with noise."""
        if not matching:
            return None

        utilities = []
        for prod in matching:
            base_util = prod.get_utility(self.state)
            noise = self.rng.normal(0, PRODUCTION_NOISE_SD)
            utilities.append(base_util + noise)

        winner_idx = np.argmax(utilities)
        return matching[winner_idx]

    def cycle(self):
        """Run one 50ms cycle. Override in subclasses if needed."""
        matching = self.find_matching_productions()
        winner = self.select_production(matching)

        if winner:
            winner.fire(self.state)
            if winner.name == "check-buffer":
                self.timing_system.cycle(production_winner="check-buffer")
            else:
                self.timing_system.cycle(production_winner=winner.name)
        else:
            self.timing_system.cycle(production_winner=None)

        self.state.cycle_count += 1
        self.state.current_time += 50

        return winner.name if winner else None
