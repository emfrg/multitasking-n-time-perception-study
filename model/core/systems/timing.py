# systems/timing
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class Pulse:
    """Single pulse emitted by the pacemaker."""

    time: float
    id: int


class Pacemaker:
    """Fixed-rate pacemaker with noisy intervals based on AGM model."""

    def __init__(
        self,
        mean_interval: float = 120.0,
        noise_sd: float = 30.0,
        seed: Optional[int] = None,
    ):
        self.mean_interval = mean_interval
        self.noise_sd = noise_sd
        self.rng = np.random.default_rng(seed)

        self.current_time = 0.0
        self.next_pulse_time = 0.0
        self.pulse_count = 0
        self.pulse_history: List[Pulse] = []

        self._schedule_next_pulse()

    def _schedule_next_pulse(self):
        """Schedule next pulse with lognormal noise."""
        mean = self.mean_interval
        sd = self.noise_sd

        mu = np.log(mean / np.sqrt(1 + (sd / mean) ** 2))
        sigma = np.sqrt(np.log(1 + (sd / mean) ** 2))

        interval = self.rng.lognormal(mu, sigma)
        self.next_pulse_time = self.current_time + interval

    def tick(self, dt: float = 50.0) -> Optional[Pulse]:
        """Advance time and emit pulse if scheduled."""
        self.current_time += dt

        if self.current_time >= self.next_pulse_time:
            pulse = Pulse(time=self.next_pulse_time, id=self.pulse_count)
            self.pulse_count += 1
            self.pulse_history.append(pulse)
            self._schedule_next_pulse()
            return pulse

        return None

    def reset(self):
        """Reset pacemaker to initial state."""
        self.current_time = 0.0
        self.next_pulse_time = 0.0
        self.pulse_count = 0
        self.pulse_history = []
        self._schedule_next_pulse()


@dataclass
class TemporalBuffer:
    """Buffer holding one unread pulse at a time."""

    def __init__(self, pacemaker: Pacemaker):
        self.pacemaker = pacemaker
        self.current_count = 0
        self.pulse_available = False
        self.current_pulse: Optional[Pulse] = None

    def update(self):
        """Advance pacemaker and check for new pulse."""
        pulse = self.pacemaker.tick(50.0)
        if pulse:
            self.current_count += 1
            self.current_pulse = pulse
            self.pulse_available = True

    def is_pulse_available(self) -> bool:
        """Check if unread pulse is available."""
        return self.pulse_available

    def read(self) -> int:
        """Consume current pulse."""
        self.pulse_available = False
        self.current_pulse = None
        return self.current_count

    def reset(self):
        """Reset for new timing period."""
        self.current_count = 0
        self.pulse_available = False
        self.current_pulse = None
        self.pacemaker.reset()


class TimingSystem:
    """Models subjective time perception through pulse encoding."""

    def __init__(self, seed=None):
        self.pacemaker = Pacemaker(mean_interval=120.0, noise_sd=30.0, seed=seed)
        self.temporal_buffer = TemporalBuffer(self.pacemaker)

        self.is_active = False
        self.encoded_count = 0
        self.simulation_time = 0.0
        self.k = 1.6

    def start(self):
        """Begin timing period."""
        self.is_active = True
        self.encoded_count = 0
        self.simulation_time = 0.0
        self.temporal_buffer.reset()

    def cycle(self, production_winner: Optional[str] = None) -> bool:
        """Run one 50ms cycle, return True if pulse encoded."""
        if not self.is_active:
            return False

        self.temporal_buffer.update()

        pulse_encoded = False
        if (
            production_winner == "check-buffer"
            and self.temporal_buffer.is_pulse_available()
        ):
            self.temporal_buffer.read()
            self.encoded_count += 1
            pulse_encoded = True

        self.simulation_time += 50.0
        return pulse_encoded

    def apply_weber_noise(self, weber_fraction: float) -> Dict[str, float]:
        """Apply Weber fraction noise to time estimate."""
        base_estimate = self.get_time_estimate()

        noisy_count = base_estimate["pulses_encoded"] + self.pacemaker.rng.normal(
            0, weber_fraction * base_estimate["pulses_encoded"]
        )
        noisy_count = max(0, noisy_count)

        final_subjective = (
            noisy_count * base_estimate["k"] * base_estimate["mean_interval"] / 1000.0
        )

        percentage = (
            (final_subjective / base_estimate["actual_seconds"] * 100)
            if base_estimate["actual_seconds"] > 0
            else 0
        )

        return {
            "actual_seconds": base_estimate["actual_seconds"],
            "subjective_seconds": final_subjective,
            "underestimation_seconds": base_estimate["actual_seconds"]
            - final_subjective,
            "percentage_of_actual": percentage,
            "pulses_encoded": base_estimate["pulses_encoded"],
            "pulses_emitted": base_estimate["pulses_emitted"],
        }

    def get_time_estimate(
        self, weber_fraction: Optional[float] = None
    ) -> Dict[str, float]:
        """Convert encoded pulses to subjective time estimate."""
        actual_seconds = self.simulation_time / 1000.0

        base_subjective_ms = self.encoded_count * self.k * self.pacemaker.mean_interval
        base_subjective_seconds = base_subjective_ms / 1000.0

        if weber_fraction is not None:
            return self.apply_weber_noise(weber_fraction)

        percentage = (
            (base_subjective_seconds / actual_seconds * 100)
            if actual_seconds > 0
            else 0
        )

        return {
            "actual_seconds": actual_seconds,
            "subjective_seconds": base_subjective_seconds,
            "underestimation_seconds": actual_seconds - base_subjective_seconds,
            "percentage_of_actual": percentage,
            "pulses_encoded": self.encoded_count,
            "pulses_emitted": self.temporal_buffer.current_count,
            "k": self.k,
            "mean_interval": self.pacemaker.mean_interval,
        }
