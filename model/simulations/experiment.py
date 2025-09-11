# simulations/experiment
import numpy as np
import random
import statistics
from typing import List, Dict, Optional
from dataclasses import dataclass
import pandas as pd
import os

from model.simulations.tasks import (
    TypingSimulation,
    NBackSimulation,
    TaskSwitchSimulation,
)
from model.utils.config import (
    # Weber fractions
    NBACK_WEBER_EASY,
    NBACK_WEBER_HARD,
    TYPING_WEBER,
    # N-back parameters
    NBACK_NUM_STIMULI_MIN,
    NBACK_NUM_STIMULI_MAX,
    NBACK_CYCLES_PER_STIM,
    # Simulation parameters
    PRINT_SUMMARIES,
    # Experiment parameters
    DEFAULT_NUM_PARTICIPANTS,
    DEFAULT_TRIALS_PER_CONDITION,
    # Experiment seeds
    SEED_NBACK_OFFSET,
    SEED_PARTICIPANT_MULTIPLIER,
    SEED_BLOCK_MULTIPLIER,
)
from model.core.systems.timing import TimingSystem
from model.core.productions.base import CognitiveState, TaskPhase
from model.core.productions.reconfiguration import Reconfig1, Reconfig2
from model.core.productions.interruption import StoreProblemState, RestoreProblemState


def run_trial(
    condition: str,
    participant_id: str,
    global_trial_num: int,
    block_num: int,
    trial_in_block: int,
    seed: Optional[int] = None,
    use_fixed_seed: bool = False,
) -> Dict:
    """Run one complete trial for any condition."""

    rng = np.random.default_rng(seed)

    n_stimuli = rng.integers(NBACK_NUM_STIMULI_MIN, NBACK_NUM_STIMULI_MAX + 1)

    is_interrupted = condition.startswith("int_")
    nback_difficulty = "easy" if "1back" in condition else "hard"

    state = CognitiveState()
    state.ps_slots = {}

    timing_system = TimingSystem(seed=seed)
    timing_system.start()

    typing_sim = TypingSimulation(seed=seed, use_fixed_seed=use_fixed_seed)
    typing_sim.state = state
    typing_sim.timing_system = timing_system
    typing_sim.productions.extend(
        [Reconfig1(), Reconfig2(), StoreProblemState(), RestoreProblemState()]
    )

    nback_sim = NBackSimulation(
        seed=(seed + SEED_NBACK_OFFSET) if seed else None, use_fixed_seed=use_fixed_seed
    )
    nback_sim.state = state
    nback_sim.timing_system = timing_system
    nback_sim.productions.extend(
        [Reconfig1(), Reconfig2(), StoreProblemState(), RestoreProblemState()]
    )

    word = typing_sim.generate_typing_word()

    start_time = timing_system.simulation_time

    split_point = None

    if is_interrupted:
        if len(word) > 4:
            split_point = rng.integers(2, len(word) - 2, endpoint=True)
        else:
            split_point = 2

        state.current_phase = TaskPhase.TYPING
        state.typing_word = word
        state.typing_position = 0

        while state.typing_position < split_point:
            typing_sim.cycle()
    else:
        typing_result = typing_sim.run_word(word)

    switch_1_start_time = timing_system.simulation_time

    switch_sim = TaskSwitchSimulation(state, timing_system)
    use_ps_store = condition == "int_2back"
    switch_sim.run_phase1(include_ps_store=use_ps_store)

    switch_1_end_time = timing_system.simulation_time

    nback_start_time = timing_system.simulation_time

    nback_result = nback_sim.run_nback_block(
        difficulty=nback_difficulty,
        n_stimuli=n_stimuli,
        cycles_per_stim=NBACK_CYCLES_PER_STIM,
    )

    nback_end_time = timing_system.simulation_time

    if is_interrupted:
        state.ps_slots.pop("n-1", None)
        state.ps_slots.pop("n-2", None)

        use_ps_restore = condition == "int_2back"
        switch_sim.run_phase2(include_ps_restore=use_ps_restore)

        state.current_phase = TaskPhase.TYPING

        while state.typing_position < len(word):
            typing_sim.cycle()

    weber = NBACK_WEBER_HARD if nback_difficulty == "hard" else NBACK_WEBER_EASY
    timing_estimate = timing_system.get_time_estimate(weber_fraction=weber)

    trial_duration_ms = timing_system.simulation_time - start_time
    trial_duration_s = trial_duration_ms / 1000.0

    typing_correct = True
    entered_text = word

    nback_duration_s = (nback_end_time - nback_start_time) / 1000.0
    task_switch_1_duration_s = (switch_1_end_time - switch_1_start_time) / 1000.0
    time_on_primary_task_s = (
        trial_duration_s - nback_duration_s - task_switch_1_duration_s
    )
    word_length = len(word)
    time_per_letter = time_on_primary_task_s / word_length if word_length > 0 else 0

    return {
        "participant_id": participant_id,
        "global_trial_num": global_trial_num,
        "block_num": block_num,
        "trial_in_block": trial_in_block,
        "condition": condition,
        "word": word,
        "split_point": split_point,
        "nback_difficulty": nback_difficulty,
        "nback_result": nback_result,
        "timing_estimate": timing_estimate,
        "trial_duration_s": trial_duration_s,
        "typing_correct": typing_correct,
        "entered_text": entered_text,
        "n_stimuli": n_stimuli,
        "nback_duration_s": nback_duration_s,
        "time_on_primary_task_s": time_on_primary_task_s,
        "time_per_letter": time_per_letter,
    }


def save_participant_data(
    participant_id: str, participant_trials: List[Dict], output_dir: str
):
    """Save participant data to CSV file."""

    csv_columns = [
        "GLOBAL_TRIAL_NUM",
        "COND_trial",
        "COND_practice",
        "COND_block",
        "COND_trial_in_block",
        "COND_word",
        "COND_letters",
        "COND_interrupt_pos",
        "COND_interruption_condition",
        "COND_nback_level",
        "COND_stim_list",
        "COND_stim_durations",
        "COND_match_positions",
        "COND_isi_times",
        "COND_num_stims",
        "COND_nback_total_duration",
        "COND_short_break",
        "COND_long_break",
        "OUT_experiment_phase",
        "OUT_block_number",
        "OUT_target_word",
        "OUT_target_word_length",
        "OUT_interruption_condition",
        "OUT_interrupt_positions",
        "OUT_resumption_lag",
        "OUT_entered_text",
        "OUT_typing_correct",
        "OUT_time_on_primary_task",
        "OUT_nback_hits",
        "OUT_nback_misses",
        "OUT_nback_false_alarms",
        "OUT_nback_correct_rejections",
        "OUT_nback_accuracy",
        "OUT_time_estimate_seconds",
        "OUT_actual_trial_duration",
        "OUT_actual_trial_duration_sec",
        "OUT_time_estimation_ratio",
        "OUT_gender",
        "OUT_age",
        "OUT_normalized_absolute_error",
        "OUT_estimation_direction",
        "OUT_time_per_letter",
        "OUT_typing_distance",
    ]

    output_rows = []
    for trial_data in participant_trials:
        is_interrupted = trial_data["condition"].startswith("int_")

        actual_s = trial_data["timing_estimate"]["actual_seconds"]
        subjective_s = trial_data["timing_estimate"]["subjective_seconds"]
        ratio = subjective_s / actual_s if actual_s > 0 else 0
        norm_abs_error = abs(subjective_s - actual_s) / actual_s if actual_s > 0 else 0
        est_direction = (
            1 if subjective_s > actual_s else -1 if subjective_s < actual_s else 0
        )
        time_per_letter = trial_data["time_per_letter"]
        time_on_primary_task_s = trial_data["time_on_primary_task_s"]
        nback_duration_s = trial_data["nback_duration_s"]

        row = {
            "GLOBAL_TRIAL_NUM": trial_data["global_trial_num"],
            "COND_trial": trial_data["global_trial_num"],
            "COND_practice": False,
            "COND_block": trial_data["block_num"],
            "COND_trial_in_block": trial_data["trial_in_block"],
            "COND_word": trial_data["word"],
            "COND_letters": list(trial_data["word"]),
            "COND_interrupt_pos": [trial_data["split_point"]] if is_interrupted else [],
            "COND_interruption_condition": (
                "interrupted" if is_interrupted else "sequential"
            ),
            "COND_nback_level": 1 if trial_data["nback_difficulty"] == "easy" else 2,
            "COND_stim_list": [],
            "COND_stim_durations": [],
            "COND_match_positions": [],
            "COND_isi_times": [],
            "COND_num_stims": trial_data["n_stimuli"],
            "COND_nback_total_duration": nback_duration_s * 1000,
            "COND_short_break": False,
            "COND_long_break": False,
            "OUT_experiment_phase": "main",
            "OUT_block_number": trial_data["block_num"],
            "OUT_target_word": trial_data["word"],
            "OUT_target_word_length": len(trial_data["word"]),
            "OUT_interruption_condition": (
                "interrupted" if is_interrupted else "sequential"
            ),
            "OUT_interrupt_positions": (
                [trial_data["split_point"]] if is_interrupted else []
            ),
            "OUT_resumption_lag": None,
            "OUT_entered_text": trial_data["entered_text"],
            "OUT_typing_correct": trial_data["typing_correct"],
            "OUT_time_on_primary_task": time_on_primary_task_s,
            "OUT_nback_hits": trial_data["nback_result"].get("hits", None),
            "OUT_nback_misses": trial_data["nback_result"].get("misses", None),
            "OUT_nback_false_alarms": trial_data["nback_result"].get(
                "false_alarms", None
            ),
            "OUT_nback_correct_rejections": trial_data["nback_result"].get(
                "correct_rejections", None
            ),
            "OUT_nback_accuracy": trial_data["nback_result"]["accuracy"],
            "OUT_time_estimate_seconds": subjective_s,
            "OUT_actual_trial_duration": trial_data["trial_duration_s"] * 1000,
            "OUT_actual_trial_duration_sec": actual_s,
            "OUT_time_estimation_ratio": ratio,
            "OUT_gender": None,
            "OUT_age": None,
            "OUT_normalized_absolute_error": norm_abs_error,
            "OUT_estimation_direction": est_direction,
            "OUT_time_per_letter": time_per_letter,
            "OUT_typing_distance": 0,
        }
        output_rows.append(row)

    df = pd.DataFrame(output_rows)
    df = df.reindex(columns=csv_columns)

    filepath = os.path.join(output_dir, f"participant_{participant_id}_output.csv")
    df.to_csv(filepath, index=False)


def run_experiment(
    num_participants: int,
    num_trials_per_condition: int,
    use_fixed_seed: bool = False,
    seed_base: int = 42,
) -> pd.DataFrame:

    output_dir = os.path.join(
        os.path.dirname(__file__), "data", "simulated_participants_raw"
    )
    os.makedirs(output_dir, exist_ok=True)

    conditions = ["seq_1back", "seq_2back", "int_1back", "int_2back"]
    all_results = []
    global_trial_counter = 0

    for p_id in range(num_participants):
        participant_id_str = f"sim_{p_id + 1}"
        participant_trials = []

        for block_num, condition in enumerate(conditions, 1):
            for trial_in_block, trial in enumerate(range(num_trials_per_condition), 1):
                global_trial_counter += 1
                seed = None
                if use_fixed_seed:
                    seed = (
                        seed_base
                        + p_id * SEED_PARTICIPANT_MULTIPLIER
                        + (block_num - 1) * SEED_BLOCK_MULTIPLIER
                        + trial
                    )

                result = run_trial(
                    condition,
                    participant_id_str,
                    global_trial_counter,
                    block_num,
                    trial_in_block,
                    seed,
                    use_fixed_seed,
                )
                participant_trials.append(result)

        save_participant_data(participant_id_str, participant_trials, output_dir)
        all_results.extend(participant_trials)

    return pd.DataFrame(all_results)


def summarize_results(df: pd.DataFrame, print_summary: bool = False):
    df["so_ratio"] = df.apply(
        lambda row: row["timing_estimate"]["percentage_of_actual"], axis=1
    )
    df["subjective_seconds"] = df.apply(
        lambda row: row["timing_estimate"]["subjective_seconds"], axis=1
    )
    df["nback_accuracy"] = df.apply(lambda row: row["nback_result"]["accuracy"], axis=1)

    condition_order = ["seq_1back", "seq_2back", "int_1back", "int_2back"]

    cv_stats = {}
    for condition in condition_order:
        cond_df = df[df["condition"] == condition]
        participant_cvs = []

        for pid in cond_df["participant_id"].unique():
            p_data = cond_df[cond_df["participant_id"] == pid]
            if len(p_data) > 1:
                mean_subj = p_data["subjective_seconds"].mean()
                sd_subj = p_data["subjective_seconds"].std()
                cv = sd_subj / mean_subj if mean_subj > 0 else 0
                participant_cvs.append(cv)

        cv_stats[condition] = {
            "mean": np.mean(participant_cvs) if participant_cvs else 0,
            "std": np.std(participant_cvs) if participant_cvs else 0,
        }

    summary_data = []

    for condition in condition_order:
        cond_df = df[df["condition"] == condition]
        summary_data.append(
            {
                "condition": condition,
                "so_ratio_mean": cond_df["so_ratio"].mean(),
                "within_cv_mean": cv_stats[condition]["mean"],
                "subjective_s_mean": cond_df["subjective_seconds"].mean(),
                "trial_duration_mean": cond_df["trial_duration_s"].mean(),
                "time_per_letter_mean": cond_df["time_per_letter"].mean(),
                "nback_accuracy": cond_df["nback_accuracy"].mean() * 100,
            }
        )

    summary_df = pd.DataFrame(summary_data).set_index("condition")

    seq_data = df[df["condition"].str.startswith("seq")]
    int_data = df[df["condition"].str.startswith("int")]

    one_back = df[df["condition"].str.contains("1back")]
    two_back = df[df["condition"].str.contains("2back")]

    duration_summary = (
        df.groupby("condition")["trial_duration_s"].agg(["mean", "std"]).round(2)
    )
    duration_summary = duration_summary.reindex(condition_order)

    if print_summary:
        print("\n=== EXPERIMENT SUMMARY ===")
        print(f"Total trials: {len(df)}")
        print(f"Participants: {df['participant_id'].nunique()}")

        print("\n--- Condition-wise Results ---")

        display_df = pd.DataFrame(
            {
                "SO Ratio (%)": [
                    f"{row['so_ratio_mean']:.2f}" for _, row in summary_df.iterrows()
                ],
                "N-back Acc (%)": [
                    f"{row['nback_accuracy']:.2f}" for _, row in summary_df.iterrows()
                ],
                "Trial Dur (s)": [
                    f"{row['trial_duration_mean']:.2f}"
                    for _, row in summary_df.iterrows()
                ],
                "Time/Letter (s)": [
                    f"{row['time_per_letter_mean']:.3f}"
                    for _, row in summary_df.iterrows()
                ],
                "Within-CV": [
                    f"{row['within_cv_mean']:.3f}" for _, row in summary_df.iterrows()
                ],
            },
            index=summary_df.index,
        )

        print(display_df.to_string())

        print("\n--- Task Type Comparisons ---")
        print(f"Sequential tasks - Mean SO Ratio: {seq_data['so_ratio'].mean():.2f}%")
        print(f"Interrupted tasks - Mean SO Ratio: {int_data['so_ratio'].mean():.2f}%")
        print(
            f"1-back tasks - Mean Accuracy: {one_back['nback_accuracy'].mean()*100:.2f}%"
        )
        print(
            f"2-back tasks - Mean Accuracy: {two_back['nback_accuracy'].mean()*100:.2f}%"
        )

    return summary_df


if __name__ == "__main__":

    results_df = run_experiment(
        num_participants=DEFAULT_NUM_PARTICIPANTS,
        num_trials_per_condition=DEFAULT_TRIALS_PER_CONDITION,
        use_fixed_seed=False,
    )

    summarize_results(results_df, print_summary=PRINT_SUMMARIES)
    print("\nDone running experiment simulation.")
    print("Data saved to data/simulated_participants_raw")
