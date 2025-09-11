# utils/create_conditions
"""Create combined experimental conditions for text entry + n-back tasks."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import List, Dict, Any, Tuple

import pandas as pd

from .generate_text_entry_conditions import load_wordbank, choose_interruptions
from .generate_nback_conditions import (
    generate_trial as generate_nback_trial,
    parse_range_arg,
    LETTERS,
    DIGITS,
)

OUTPUT_DIR: pathlib.Path = pathlib.Path("conditions")


def generate_text_entry_data(
    word_lengths: List[int], num_trials: int
) -> List[Dict[str, Any]]:
    """Generate text entry trial data."""
    word_pools = {}
    for length in word_lengths:
        word_pools[length] = load_wordbank(length)

    words_per_length = num_trials // len(word_lengths)
    remainder = num_trials % len(word_lengths)

    selected_words = []
    for i, length in enumerate(word_lengths):
        n_words = words_per_length + (1 if i < remainder else 0)
        selected_words.extend(word_pools[length][:n_words])

    random.shuffle(selected_words)

    n_interrupted = num_trials // 2
    conditions = ["interrupted"] * n_interrupted + ["sequential"] * (
        num_trials - n_interrupted
    )
    random.shuffle(conditions)

    text_data = []
    for word, cond in zip(selected_words, conditions):
        interrupt_positions = (
            choose_interruptions(len(word), 1) if cond == "interrupted" else []
        )

        text_data.append(
            {
                "word": word.upper(),
                "letters": json.dumps(list(word.upper())),
                "interrupt_pos": json.dumps(interrupt_positions),
                "interruption_condition": cond,
            }
        )

    return text_data


def generate_nback_data(
    nback_levels: List[int],
    num_trials: int,
    duration_range: Tuple[float, float],
    stim_duration: float,
    isi_range: Tuple[float, float],
    match_rate: float,
    stim_type: str,
    quantum: float,
) -> List[Dict[str, Any]]:
    """Generate n-back trial data."""
    stim_set = LETTERS if stim_type == "letters" else DIGITS

    base_count = num_trials // len(nback_levels)
    remainder = num_trials % len(nback_levels)

    level_sequence = []
    for i, level in enumerate(nback_levels):
        count = base_count + (1 if i < remainder else 0)
        level_sequence.extend([level] * count)

    random.shuffle(level_sequence)

    nback_data = []
    for n_back in level_sequence:
        trial = generate_nback_trial(
            n_back=n_back,
            duration_range=duration_range,
            stim_duration_range=(stim_duration, stim_duration),
            isi_range=isi_range,
            match_rate=match_rate,
            stim_set=stim_set,
            quantum=quantum,
        )

        nback_data.append(
            {
                "n_back": trial["n_back"],
                "stim_list": json.dumps(trial["stim_list"]),
                "stim_durations": json.dumps(trial["stim_durations"]),
                "match_positions": json.dumps(trial["match_positions"]),
                "isi_times": json.dumps(trial["isi_times"]),
                "num_stims": trial["num_stims"],
                "nback_total_duration": trial["total_duration"],
            }
        )

    return nback_data


def add_break_indicators(
    rows: List[Dict[str, Any]], block_size: int, is_practice: bool = False
) -> None:
    """Add break indicators to trials."""
    num_trials = len(rows)

    for i, row in enumerate(rows):
        row["short_break"] = False
        row["long_break"] = False

        if not is_practice and i < num_trials - 1:
            if (i + 1) % block_size == 0 and (i + 1) != num_trials // 2:
                row["short_break"] = True

            if (i + 1) == num_trials // 2:
                row["long_break"] = True


def create_combined_trials(
    text_data: List[Dict[str, Any]],
    nback_data: List[Dict[str, Any]],
    block_size: int,
    is_practice: bool = False,
) -> List[Dict[str, Any]]:
    """Combine text entry and n-back data into complete trials."""
    combined = []

    indices = list(range(len(text_data)))
    random.shuffle(indices)

    for i, idx in enumerate(indices):
        if is_practice:
            block_num = 0
            trial_pos = i + 1
        else:
            block_num = (i // block_size) + 1
            trial_pos = (i % block_size) + 1

        trial = {
            "trial": i + 1,
            "practice": is_practice,
            "block": block_num,
            "trial_in_block": trial_pos,
            **text_data[idx],
            **nback_data[idx],
        }
        combined.append(trial)

    add_break_indicators(combined, block_size, is_practice)

    return combined


def generate_participant_conditions(
    participant_id: int, args: argparse.Namespace
) -> pd.DataFrame:
    """Generate conditions for a single participant."""
    if args.seed is not None:
        random.seed(args.seed + participant_id)

    all_trials = []

    if args.practice_trials > 0:
        practice_text = generate_text_entry_data(
            args.text_lengths, args.practice_trials
        )
        practice_nback = generate_nback_data(
            args.nback_levels,
            args.practice_trials,
            args.duration_range,
            args.stim_duration,
            args.isi,
            args.match_rate,
            args.stimuli,
            args.quantum,
        )

        practice_combined = create_combined_trials(
            practice_text, practice_nback, args.block_size, is_practice=True
        )
        all_trials.extend(practice_combined)

    main_text = generate_text_entry_data(args.text_lengths, args.trials)
    main_nback = generate_nback_data(
        args.nback_levels,
        args.trials,
        args.duration_range,
        args.stim_duration,
        args.isi,
        args.match_rate,
        args.stimuli,
        args.quantum,
    )

    main_combined = create_combined_trials(
        main_text, main_nback, args.block_size, is_practice=False
    )
    all_trials.extend(main_combined)

    return pd.DataFrame(all_trials)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate combined text entry + n-back experimental conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python create_conditions.py --trials 60 --practice-trials 10 --participants 5
  python create_conditions.py --trials 90 --block-size 30 --text-lengths 7 8 9 --nback-levels 1 2
        """,
    )

    parser.add_argument(
        "--trials", type=int, required=True, help="Number of main experimental trials"
    )
    parser.add_argument(
        "--practice-trials",
        type=int,
        default=0,
        help="Number of practice trials (default: 0)",
    )
    parser.add_argument(
        "--block-size", type=int, default=20, help="Trials per block (default: 20)"
    )
    parser.add_argument(
        "--participants",
        type=int,
        default=1,
        help="Number of participant files to generate (default: 1)",
    )
    parser.add_argument("--seed", type=int, help="Base random seed")

    parser.add_argument(
        "--text-lengths",
        type=int,
        nargs="+",
        default=[7, 8, 9],
        help="Word lengths for text entry (default: 7 8 9)",
    )

    parser.add_argument(
        "--nback-levels",
        type=int,
        nargs="+",
        default=[1, 2],
        help="N-back levels (default: 1 2)",
    )
    parser.add_argument(
        "--duration-range",
        type=float,
        nargs="+",
        default=[6, 14],
        help="N-back trial duration range in seconds (default: 6 14)",
    )
    parser.add_argument(
        "--stim-duration",
        type=float,
        default=1.2,
        help="Stimulus duration in seconds (default: 1.2)",
    )
    parser.add_argument(
        "--isi",
        type=float,
        nargs="+",
        default=[0.25, 0.4],
        help="Inter-stimulus interval range in seconds (default: 0.25 0.4)",
    )
    parser.add_argument(
        "--match-rate",
        type=float,
        default=30,
        help="N-back target match percentage (default: 30)",
    )
    parser.add_argument(
        "--stimuli",
        choices=["letters", "digits"],
        default="digits",
        help="N-back stimulus type (default: letters)",
    )
    parser.add_argument(
        "--quantum",
        type=float,
        default=0.01,
        help="Time quantization step in seconds (default: 0.01)",
    )

    args = parser.parse_args()

    args.duration_range = parse_range_arg(args.duration_range)
    args.isi = parse_range_arg(args.isi)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for p_id in range(1, args.participants + 1):
        df = generate_participant_conditions(p_id, args)

        filename = f"participant_{p_id:03d}.csv"
        filepath = OUTPUT_DIR / filename
        df.to_csv(filepath, index=False)

        print(f"Generated {filepath}")

        if p_id == 1:
            print(f"\nSummary for {filename}:")
            print(f"- Practice trials: {args.practice_trials}")
            print(f"- Main trials: {args.trials}")
            print(f"- Block size: {args.block_size}")
            print(f"- Text lengths: {args.text_lengths}")
            print(f"- N-back levels: {args.nback_levels}")
            print(f"- Total rows: {len(df)}")


if __name__ == "__main__":
    main()
