# simulations/tasks
import numpy as np
import random
import statistics
from typing import List, Dict, Optional, Sequence
from model.core.productions.base import CognitiveState, Production, TaskPhase
from model.core.productions.nback import *
from model.core.productions.typing import *
from model.core.productions.timing import CheckBuffer
from model.core.productions.interruption import StoreProblemState, RestoreProblemState
from model.core.productions.reconfiguration import Reconfig1, Reconfig2
from model.core.systems.timing import TimingSystem
from model.simulations.base import BaseSimulation
from model.utils.config import (
    scaled,
    PRODUCTION_NOISE_SD,
    # Weber fractions
    NBACK_WEBER_EASY,
    NBACK_WEBER_HARD,
    TYPING_WEBER,
    # N-back parameters
    NBACK_CYCLES_PER_STIM,
    NBACK_NUM_STIMULI_MIN,
    NBACK_NUM_STIMULI_MAX,
    NBACK_TARGET_RATE,
    # Task switching
    TASK_SWITCH_PHASE1_CYCLES,
    # Simulation parameters
    PRINT_SUMMARIES,
    TYPING_SIM_PARTICIPANTS,
    TYPING_SIM_TRIALS,
    NBACK_SIM_PARTICIPANTS,
    NBACK_SIM_TRIALS,
    # Seeds
    TYPING_SIM_SEED_OFFSET,
    NBACK_SIM_STIMULI_SEED,
    NBACK_SIM_SEED_OFFSET_1BACK,
    NBACK_SIM_SEED_OFFSET_2BACK,
    # Safety
    MAX_CYCLES_PER_WORD,
    # Word bank
    TYPING_WORD_BANK,
)

import pandas as pd
import os


class TypingSimulation(BaseSimulation):
    """Focused simulation for typing with timing."""

    def __init__(self, seed=None, use_fixed_seed=False):
        super().__init__(seed, use_fixed_seed)

        self.productions = [
            ReadWord(),
            ExtractCharacter(),
            FindButton(),
            MoveCursor(),
            ClickButton(),
            UpdatePosition(),
            CheckBuffer(),
        ]

        self.state = CognitiveState()
        self.state.ps_slots = {}

        self.timing_system = TimingSystem(seed=seed if use_fixed_seed else None)

        self.letters_typed = 0
        self.letters_correct = 0
        self.words_completed = 0

    def generate_typing_word(self, length: Optional[int] = None) -> str:
        """Return a random word for the typing task."""
        if length is None:
            length = random.choice(list(TYPING_WORD_BANK.keys()))

        if length not in TYPING_WORD_BANK:
            raise ValueError("length must be an int between 6 and 9, inclusive")

        return random.choice(TYPING_WORD_BANK[length])

    def run_word(self, word: str, max_cycles: int = MAX_CYCLES_PER_WORD):
        """Type one word and track performance."""
        self.state.typing_word = word
        self.state.typing_position = 0
        self.state.current_character = None
        self.state.cursor_at_target = False
        self.state.button_clicked = False
        self.state.button_found = False

        if "word" in self.state.ps_slots:
            del self.state.ps_slots["word"]
        if "position" in self.state.ps_slots:
            del self.state.ps_slots["position"]

        productions_fired = []
        cycles_used = 0

        prod_counts = {}
        last_winner = None

        for _ in range(max_cycles):
            winner_name = self.cycle()
            if winner_name:
                productions_fired.append(winner_name)
                if winner_name != last_winner:
                    prod_counts[winner_name] = prod_counts.get(winner_name, 0) + 1
                    last_winner = winner_name
            cycles_used += 1

            if self.state.typing_position >= len(word):
                self.words_completed += 1
                break

        self.letters_typed += self.state.typing_position
        self.letters_correct += self.state.typing_position

        return {
            "word": word,
            "letters_typed": self.state.typing_position,
            "completed": self.state.typing_position >= len(word),
            "cycles_used": cycles_used,
            "productions": productions_fired,
        }

    def run_typing_block(
        self, n_words: int = 10, max_cycles_per_word: int = MAX_CYCLES_PER_WORD
    ):
        """Run a set of typing trials."""
        self.state.current_phase = TaskPhase.TYPING

        self.letters_typed = 0
        self.letters_correct = 0
        self.words_completed = 0

        words = [self.generate_typing_word() for _ in range(n_words)]

        trial_results = []
        for i, word in enumerate(words):
            result = self.run_word(word, max_cycles_per_word)
            trial_results.append(result)

        timing = self.timing_system.get_time_estimate(weber_fraction=TYPING_WEBER)

        accuracy = (
            self.letters_correct / self.letters_typed if self.letters_typed > 0 else 0
        )
        completion_rate = self.words_completed / n_words

        return {
            "accuracy": accuracy,
            "completion_rate": completion_rate,
            "timing": timing,
            "trial_results": trial_results,
            "words_completed": self.words_completed,
        }

    def run_participant_trials(self, num_trials: int, n_words: int = 10):
        """Run multiple typing trials for one participant."""
        participant_results = []

        for trial in range(num_trials):
            self.state = CognitiveState()
            self.state.ps_slots = {}

            if (
                hasattr(self, "use_fixed_seed")
                and self.use_fixed_seed
                and hasattr(self, "seed")
            ):
                trial_seed = self.seed + trial * 1000
            else:
                trial_seed = None

            self.timing_system = TimingSystem(seed=trial_seed)
            self.timing_system.start()

            block_result = self.run_typing_block(n_words)
            participant_results.append(block_result)

        subjective_times = [
            r["timing"]["subjective_seconds"] for r in participant_results
        ]
        accuracies = [r["accuracy"] for r in participant_results]
        mean_so_ratios = [
            r["timing"]["percentage_of_actual"] for r in participant_results
        ]

        mean_time = statistics.mean(subjective_times)
        sd_time = statistics.stdev(subjective_times) if len(subjective_times) > 1 else 0
        participant_cv = sd_time / mean_time if mean_time > 0 else 0

        return {
            "trials": participant_results,
            "participant_cv": participant_cv,
            "mean_accuracy": statistics.mean(accuracies),
            "mean_so_ratio": statistics.mean(mean_so_ratios),
            "subjective_times": subjective_times,
        }


class NBackSimulation(BaseSimulation):
    """Focused simulation for n-back with timing."""

    def __init__(self, seed=None, use_fixed_seed=False):
        super().__init__(seed, use_fixed_seed)

        self.productions = [
            AttendStimulus(),
            Retrieve1Back(),
            Retrieve2Back(),
            CompareNBack(),
            UpdateNBackBuffers(),
            RespondSame(),
            RespondDifferent(),
            CheckBuffer(),
        ]

        self.state = CognitiveState()
        self.state.ps_slots = {}

        self.timing_system = TimingSystem(
            seed=seed if use_fixed_seed else None,
        )

        self.nback_correct = 0
        self.nback_total = 0
        self.nback_missed = 0

    def generate_nback_sequence(
        self, n_stimuli: int, n_back: int, target_rate: float = NBACK_TARGET_RATE
    ):
        """Generate n-back sequence with controlled target rate."""
        sequence = []
        n_targets = int(n_stimuli * target_rate)

        possible_positions = list(range(n_back, n_stimuli))
        if possible_positions:
            target_positions = random.sample(
                possible_positions, min(n_targets, len(possible_positions))
            )
        else:
            target_positions = []

        for i in range(n_stimuli):
            if i < n_back:
                sequence.append(str(random.randint(0, 9)))
            elif i in target_positions:
                sequence.append(sequence[i - n_back])
            else:
                choices = [str(d) for d in range(10) if str(d) != sequence[i - n_back]]
                sequence.append(random.choice(choices))

        return sequence

    def run_stimulus(
        self, stimulus: str, correct_response: str, cycles: int = NBACK_CYCLES_PER_STIM
    ):
        """Present one stimulus and track performance."""
        self.state.nback_current = stimulus
        self.state.nback_response = None

        self.state.nback_attended = False
        self.state.nback_retrieved = False
        self.state.nback_responded = False
        self.state.nback_buffers_updated = False
        self.state.nback_retrieved_value = None

        productions_fired = []

        for _ in range(cycles):
            winner_name = self.cycle()
            if winner_name:
                productions_fired.append(winner_name)

        while self.state.operation_started:
            winner_name = self.cycle()
            if winner_name:
                productions_fired.append(winner_name)

        self.nback_total += 1
        if self.state.nback_response is None:
            self.nback_missed += 1
            result = "MISSED"
        elif self.state.nback_response == correct_response:
            self.nback_correct += 1
            result = "CORRECT"
        else:
            result = "ERROR"

        return {
            "stimulus": stimulus,
            "response": self.state.nback_response,
            "correct": correct_response,
            "result": result,
            "productions": productions_fired,
        }

    def run_nback_block(
        self,
        difficulty: str,
        n_stimuli: int,
        cycles_per_stim: int = NBACK_CYCLES_PER_STIM,
    ):
        """Run a block of n-back trials."""
        self.state.current_phase = TaskPhase.NBACK
        self.state.nback_difficulty = difficulty

        self.nback_correct = 0
        self.nback_total = 0
        self.nback_missed = 0

        n_back = 1 if difficulty == "easy" else 2
        stimuli = self.generate_nback_sequence(n_stimuli, n_back)

        correct_responses = []
        for i, stim in enumerate(stimuli):
            if i < n_back:
                correct_responses.append("different")
            else:
                if stim == stimuli[i - n_back]:
                    correct_responses.append("same")
                else:
                    correct_responses.append("different")

        n_targets = correct_responses.count("same")

        trial_results = []
        for i, (stim, correct) in enumerate(zip(stimuli, correct_responses)):
            result = self.run_stimulus(stim, correct, cycles_per_stim)
            trial_results.append(result)

        if difficulty == "easy":
            timing = self.timing_system.get_time_estimate(
                weber_fraction=NBACK_WEBER_EASY
            )
        else:
            timing = self.timing_system.get_time_estimate(
                weber_fraction=NBACK_WEBER_HARD
            )

        accuracy = self.nback_correct / self.nback_total if self.nback_total > 0 else 0

        hits = sum(
            1
            for r in trial_results
            if r["correct"] == "same" and r["response"] == "same"
        )
        misses = sum(
            1
            for r in trial_results
            if r["correct"] == "same" and r["response"] != "same"
        )
        false_alarms = sum(
            1
            for r in trial_results
            if r["correct"] == "different" and r["response"] == "same"
        )
        correct_rejections = sum(
            1
            for r in trial_results
            if r["correct"] == "different" and r["response"] == "different"
        )

        return {
            "difficulty": difficulty,
            "accuracy": accuracy,
            "missed_rate": self.nback_missed / self.nback_total,
            "timing": timing,
            "trial_results": trial_results,
            "n_targets": n_targets,
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "correct_rejections": correct_rejections,
            "n_stimuli": n_stimuli,
        }

    def run_participant_trials(
        self,
        difficulty: str,
        num_trials: int,
        n_stimuli: int,
        cycles_per_stim: int = NBACK_CYCLES_PER_STIM,
    ):
        """Run multiple n-back trials for one participant."""
        participant_results = []

        for trial in range(num_trials):
            self.state = CognitiveState()
            self.state.ps_slots = {}

            if (
                hasattr(self, "use_fixed_seed")
                and self.use_fixed_seed
                and hasattr(self, "seed")
            ):
                trial_seed = self.seed + trial * 1000
            else:
                trial_seed = None

            self.timing_system = TimingSystem(seed=trial_seed)
            self.timing_system.start()

            block_result = self.run_nback_block(difficulty, n_stimuli, cycles_per_stim)
            participant_results.append(block_result)

        subjective_times = [
            r["timing"]["subjective_seconds"] for r in participant_results
        ]
        accuracies = [r["accuracy"] for r in participant_results]
        mean_so_ratios = [
            r["timing"]["percentage_of_actual"] for r in participant_results
        ]

        mean_time = statistics.mean(subjective_times)
        sd_time = statistics.stdev(subjective_times) if len(subjective_times) > 1 else 0
        participant_cv = sd_time / mean_time if mean_time > 0 else 0

        return {
            "difficulty": difficulty,
            "trials": participant_results,
            "participant_cv": participant_cv,
            "mean_accuracy": statistics.mean(accuracies),
            "mean_so_ratio": statistics.mean(mean_so_ratios),
            "subjective_times": subjective_times,
        }


class TaskSwitchSimulation(BaseSimulation):
    """Task switching simulation with its own cycle implementation."""

    def __init__(self, state: CognitiveState, timing_system, rng=None):
        """Initialize with existing state and timing system."""
        self.state = state
        self.timing_system = timing_system
        self.rng = rng or np.random.default_rng()
        self.seed = None
        self.use_fixed_seed = False

        self.productions = [
            Reconfig1(),
            Reconfig2(),
            StoreProblemState(),
            RestoreProblemState(),
            CheckBuffer(),
        ]

    def cycle(self):
        """Run one 50ms cycle with production logging."""
        matches = self.find_matching_productions()
        winner = self.select_production(matches)

        if winner:
            winner.fire(self.state)
            self.state.log_production(winner.name)
            role = "check-buffer" if winner.name == "check-buffer" else winner.name
            self.timing_system.cycle(production_winner=role)
        else:
            self.timing_system.cycle(production_winner=None)

        self.state.cycle_count += 1
        self.state.current_time += 50

    def run_phase1(
        self, include_ps_store=False, fixed_cycles=TASK_SWITCH_PHASE1_CYCLES
    ):
        """Run task_switching_1 phase with fixed duration."""
        self.state.current_phase = TaskPhase.TASK_SWITCHING_1

        need_reconfig = True
        need_ps_store = include_ps_store

        for cycle in range(fixed_cycles):
            if self.state.switch_stage == "done":
                if need_reconfig:
                    need_reconfig = False
                    if need_ps_store:
                        self.state.switch_stage = "ps_store"
                    else:
                        self.state.switch_stage = None
                elif need_ps_store:
                    need_ps_store = False
                    self.state.switch_stage = None
            elif need_reconfig and self.state.switch_stage != "to_nback":
                self.state.switch_stage = "to_nback"

            self.cycle()

    def run_phase2(self, include_ps_restore=False):
        """Run task_switching_2 phase."""
        self.state.current_phase = TaskPhase.TASK_SWITCHING_2

        self.state.switch_stage = "to_typing"
        while self.state.switch_stage != "done":
            self.cycle()

        if include_ps_restore:
            self.state.switch_stage = "ps_restore"
            while self.state.switch_stage != "done":
                self.cycle()


def save_typing_participant_data(
    participant_id: str, participant_results: Dict, output_dir: str
):
    """Save typing simulation data for one participant to CSV."""
    csv_columns = [
        "GLOBAL_TRIAL_NUM",
        "participant_id",
        "trial_num",
        "word",
        "word_length",
        "letters_typed",
        "completed",
        "accuracy",
        "completion_rate",
        "cycles_used",
        "actual_duration_s",
        "subjective_duration_s",
        "so_ratio",
        "time_per_letter",
        "participant_cv",
    ]

    output_rows = []
    trials = participant_results["trials"]
    participant_cv = participant_results["participant_cv"]

    for trial_num, trial_data in enumerate(trials, 1):
        word_result = trial_data["trial_results"][0]

        actual_s = trial_data["timing"]["actual_seconds"]
        subjective_s = trial_data["timing"]["subjective_seconds"]
        so_ratio = trial_data["timing"]["percentage_of_actual"]

        time_per_letter = (
            actual_s / word_result["letters_typed"]
            if word_result["letters_typed"] > 0
            else 0
        )

        row = {
            "GLOBAL_TRIAL_NUM": trial_num,
            "participant_id": participant_id,
            "trial_num": trial_num,
            "word": word_result["word"],
            "word_length": len(word_result["word"]),
            "letters_typed": word_result["letters_typed"],
            "completed": word_result["completed"],
            "accuracy": trial_data["accuracy"],
            "completion_rate": trial_data["completion_rate"],
            "cycles_used": word_result["cycles_used"],
            "actual_duration_s": actual_s,
            "subjective_duration_s": subjective_s,
            "so_ratio": so_ratio,
            "time_per_letter": time_per_letter,
            "participant_cv": participant_cv,
        }
        output_rows.append(row)

    df = pd.DataFrame(output_rows)
    df = df.reindex(columns=csv_columns)

    filepath = os.path.join(output_dir, f"participant_{participant_id}_output.csv")
    df.to_csv(filepath, index=False)


def save_nback_participant_data(
    participant_id: str, difficulty: str, participant_results: Dict, output_dir: str
):
    """Save n-back simulation data for one participant to CSV."""
    csv_columns = [
        "GLOBAL_TRIAL_NUM",
        "participant_id",
        "trial_num",
        "difficulty",
        "nback_level",
        "n_stimuli",
        "n_targets",
        "accuracy",
        "missed_rate",
        "hits",
        "misses",
        "false_alarms",
        "correct_rejections",
        "actual_duration_s",
        "subjective_duration_s",
        "so_ratio",
        "participant_cv",
    ]

    output_rows = []
    trials = participant_results["trials"]
    participant_cv = participant_results["participant_cv"]

    for trial_num, trial_data in enumerate(trials, 1):
        actual_s = trial_data["timing"]["actual_seconds"]
        subjective_s = trial_data["timing"]["subjective_seconds"]
        so_ratio = trial_data["timing"]["percentage_of_actual"]

        row = {
            "GLOBAL_TRIAL_NUM": trial_num,
            "participant_id": participant_id,
            "trial_num": trial_num,
            "difficulty": difficulty,
            "nback_level": 1 if difficulty == "easy" else 2,
            "n_stimuli": trial_data["n_stimuli"],
            "n_targets": trial_data["n_targets"],
            "accuracy": trial_data["accuracy"],
            "missed_rate": trial_data["missed_rate"],
            "hits": trial_data["hits"],
            "misses": trial_data["misses"],
            "false_alarms": trial_data["false_alarms"],
            "correct_rejections": trial_data["correct_rejections"],
            "actual_duration_s": actual_s,
            "subjective_duration_s": subjective_s,
            "so_ratio": so_ratio,
            "participant_cv": participant_cv,
        }
        output_rows.append(row)

    df = pd.DataFrame(output_rows)
    df = df.reindex(columns=csv_columns)

    filepath = os.path.join(
        output_dir, f"participant_{participant_id}_{difficulty}_output.csv"
    )
    df.to_csv(filepath, index=False)


def run_typing_sims(use_fixed_seed=False, print_summary=False):
    """Run focused typing simulations."""
    output_dir = os.path.join(
        os.path.dirname(__file__), "data", "single_tasks", "typing_simulation"
    )
    os.makedirs(output_dir, exist_ok=True)

    num_participants = TYPING_SIM_PARTICIPANTS
    num_trials = TYPING_SIM_TRIALS

    expected_duration = num_trials * 1.8

    all_results = []

    for p in range(num_participants):
        participant_id = f"sim_{p + 1}"

        seed = TYPING_SIM_SEED_OFFSET + p if use_fixed_seed else None

        sim = TypingSimulation(seed=seed, use_fixed_seed=use_fixed_seed)
        sim.seed = seed
        sim.use_fixed_seed = use_fixed_seed
        results = sim.run_participant_trials(num_trials, n_words=1)
        all_results.append(results)

        save_typing_participant_data(participant_id, results, output_dir)

    participant_cvs = [r["participant_cv"] for r in all_results]
    mean_accuracies = [r["mean_accuracy"] for r in all_results]
    mean_so_ratios = [r["mean_so_ratio"] for r in all_results]

    summary = {
        "mean_cv": statistics.mean(participant_cvs),
        "mean_accuracy": statistics.mean(mean_accuracies) * 100,
        "mean_so_ratio": statistics.mean(mean_so_ratios),
    }

    if print_summary:
        print("\n=== TYPING SIMULATION SUMMARY ===")
        print(f"Participants: {num_participants}, Trials per participant: {num_trials}")
        print(f"Mean Typing Accuracy: {summary['mean_accuracy']:.2f}%")
        print(f"Mean SO Ratio: {summary['mean_so_ratio']:.2f}%")
        print(f"Mean CV: {summary['mean_cv']:.3f}")

    return summary


def run_nback_sims(use_fixed_seed=False, print_summary=False):
    """Run focused n-back simulations."""
    output_dir = os.path.join(
        os.path.dirname(__file__), "data", "single_tasks", "nback_simulation"
    )
    os.makedirs(output_dir, exist_ok=True)

    num_participants = NBACK_SIM_PARTICIPANTS
    num_trials = NBACK_SIM_TRIALS

    cycles_per_stim = NBACK_CYCLES_PER_STIM
    rng = np.random.default_rng(NBACK_SIM_STIMULI_SEED)
    n_stimuli = rng.integers(NBACK_NUM_STIMULI_MIN, NBACK_NUM_STIMULI_MAX + 1)

    nback_stim_dur = NBACK_CYCLES_PER_STIM / 1000
    expected_duration = num_trials * n_stimuli * nback_stim_dur

    all_results = {"1back": [], "2back": []}

    for p in range(num_participants):
        participant_id = f"sim_{p + 1}"

        seed_1back = NBACK_SIM_SEED_OFFSET_1BACK + p if use_fixed_seed else None
        seed_2back = NBACK_SIM_SEED_OFFSET_2BACK + p if use_fixed_seed else None

        sim = NBackSimulation(seed=seed_1back, use_fixed_seed=use_fixed_seed)
        sim.seed = seed_1back
        sim.use_fixed_seed = use_fixed_seed
        results_1back = sim.run_participant_trials(
            "easy", num_trials, n_stimuli, cycles_per_stim
        )
        all_results["1back"].append(results_1back)

        save_nback_participant_data(participant_id, "1back", results_1back, output_dir)

        sim = NBackSimulation(seed=seed_2back, use_fixed_seed=use_fixed_seed)
        sim.seed = seed_2back
        sim.use_fixed_seed = use_fixed_seed
        results_2back = sim.run_participant_trials(
            "hard", num_trials, n_stimuli, cycles_per_stim
        )
        all_results["2back"].append(results_2back)

        save_nback_participant_data(participant_id, "2back", results_2back, output_dir)

    summaries = {}
    for task in ["1back", "2back"]:
        participant_cvs = [r["participant_cv"] for r in all_results[task]]
        mean_accuracies = [r["mean_accuracy"] for r in all_results[task]]
        mean_so_ratios = [r["mean_so_ratio"] for r in all_results[task]]

        all_subjective_times = []
        for participant in all_results[task]:
            all_subjective_times.extend(participant["subjective_times"])

        summaries[task] = {
            "mean_cv": statistics.mean(participant_cvs),
            "mean_accuracy": statistics.mean(mean_accuracies) * 100,
            "mean_so_ratio": statistics.mean(mean_so_ratios),
        }

    if print_summary:
        print("\n=== N-BACK SIMULATION SUMMARY ===")
        print(f"Participants: {num_participants}, Trials per participant: {num_trials}")
        print(f"Stimuli per trial: {n_stimuli}")
        for task in ["1back", "2back"]:
            print(f"\n{task.upper()}:")
            print(f"  Mean N-back Accuracy: {summaries[task]['mean_accuracy']:.2f}%")
            print(f"  Mean SO Ratio: {summaries[task]['mean_so_ratio']:.2f}%")
            print(f"  Mean CV: {summaries[task]['mean_cv']:.3f}")

    return summaries


def run_task_switch_sims(
    seed=None, use_fixed_seed=False, num_participants=20, num_trials_per_participant=10
):
    """Run task switching simulations across multiple scenarios."""
    scenarios = [
        ("Phase 1 only (reconfig-1)", True, False, False, False),
        ("Phase 1 + PS store", True, True, False, False),
        ("Phase 2 only (reconfig-2)", False, False, True, False),
        ("Phase 2 + PS restore", False, False, True, True),
        ("Full sequence", True, True, True, True),
    ]

    all_scenario_results = []

    for scenario_name, run_p1, store_ps, run_p2, restore_ps in scenarios:
        participant_results = []

        for p in range(num_participants):
            trial_results = []

            for trial in range(num_trials_per_participant):
                state = CognitiveState()
                state.ps_slots = {"word": "EXAMPLE", "position": 3}

                if use_fixed_seed:
                    trial_seed = (
                        seed + p * 10000 + trial * 100
                        if seed
                        else p * 10000 + trial * 100
                    )
                else:
                    trial_seed = None

                ts = TimingSystem(seed=trial_seed)
                ts.start()

                ts_sim = TaskSwitchSimulation(
                    state,
                    ts,
                    rng=np.random.default_rng(trial_seed) if use_fixed_seed else None,
                )

                initial_cycles = state.cycle_count

                if run_p1:
                    ts_sim.run_phase1(include_ps_store=store_ps)

                if run_p2:
                    ts_sim.run_phase2(include_ps_restore=restore_ps)

                total_cycles = state.cycle_count - initial_cycles

                est = ts.get_time_estimate()

                trial_results.append({"cycles": total_cycles, "timing": est})

            p_cycles = [r["cycles"] for r in trial_results]
            p_subjective_times = [
                r["timing"]["subjective_seconds"] for r in trial_results
            ]
            p_so_ratios = [r["timing"]["percentage_of_actual"] for r in trial_results]

            p_mean_time = statistics.mean(p_subjective_times)
            p_sd_time = (
                statistics.stdev(p_subjective_times)
                if len(p_subjective_times) > 1
                else 0
            )
            p_cv = p_sd_time / p_mean_time if p_mean_time > 0 else 0

            participant_results.append(
                {
                    "participant": p + 1,
                    "mean_cycles": statistics.mean(p_cycles),
                    "mean_subjective_time": p_mean_time,
                    "mean_so_ratio": statistics.mean(p_so_ratios),
                    "participant_cv": p_cv,
                    "trials": trial_results,
                }
            )

        all_participant_cvs = [p["participant_cv"] for p in participant_results]
        all_mean_cycles = [p["mean_cycles"] for p in participant_results]
        all_mean_so_ratios = [p["mean_so_ratio"] for p in participant_results]

        all_actual_times = []
        all_subjective_times = []
        for p_result in participant_results:
            for trial in p_result["trials"]:
                all_actual_times.append(trial["timing"]["actual_seconds"])
                all_subjective_times.append(trial["timing"]["subjective_seconds"])

        all_scenario_results.append(
            {
                "name": scenario_name,
                "mean_cycles": statistics.mean(all_mean_cycles),
                "sd_cycles": (
                    statistics.stdev(all_mean_cycles) if len(all_mean_cycles) > 1 else 0
                ),
                "mean_seconds": statistics.mean(all_actual_times),
                "mean_so_ratio": statistics.mean(all_mean_so_ratios),
                "mean_within_cv": statistics.mean(all_participant_cvs),
                "sd_within_cv": (
                    statistics.stdev(all_participant_cvs)
                    if len(all_participant_cvs) > 1
                    else 0
                ),
            }
        )

    for scenario in all_scenario_results:
        print(f"\n{scenario['name']}:")
        print(f"  Mean cycles: {scenario['mean_cycles']:.1f}")
        print(f"  Mean SO ratio: {scenario['mean_so_ratio']:.2f}%")
        print(f"  Mean within-CV: {scenario['mean_within_cv']:.3f}")

    return all_scenario_results


if __name__ == "__main__":

    typing_results = run_typing_sims(
        use_fixed_seed=False, print_summary=PRINT_SUMMARIES
    )
    nback_results = run_nback_sims(use_fixed_seed=False, print_summary=PRINT_SUMMARIES)

    # Optional: check how task switching works in isolation
    # task_switch_results = run_task_switch_sims(
    #     seed=42, use_fixed_seed=True, num_participants=10, num_trials_per_participant=10
    # )

    print("\nDone running single task simulations.")
    print("Data saved to data/single_tasks")
