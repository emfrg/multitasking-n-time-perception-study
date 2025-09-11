# utils/generate_nback_conditions
"""Generate experimental conditions for n-back task."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import List, Tuple, Set, Optional, Dict, Any

import pandas as pd

LETTERS: List[str] = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K"]
DIGITS: List[str] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

OUTPUT_DIR: pathlib.Path = pathlib.Path("conditions", "nback")


def parse_range_arg(arg_value: List[float]) -> Tuple[float, float]:
    """Convert argument list to range tuple."""
    if len(arg_value) == 1:
        return (arg_value[0], arg_value[0])
    elif len(arg_value) == 2:
        return (arg_value[0], arg_value[1])
    else:
        raise ValueError("Range must be 1 or 2 values")


def calculate_num_matches(n_back: int, num_stims: int, match_rate: float) -> int:
    """Calculate number of matches based on eligible positions."""
    eligible = num_stims - n_back
    return int(round(match_rate / 100 * eligible))


def select_match_positions(n_back: int, num_stims: int, num_matches: int) -> Set[int]:
    """Select positions for matches, avoiding consecutive matches."""
    if num_matches == 0:
        return set()

    eligible = list(range(n_back, num_stims))

    # Try to avoid consecutive matches
    for _ in range(1_000):
        picks = sorted(random.sample(eligible, num_matches))
        if all(picks[i + 1] - picks[i] > 1 for i in range(len(picks) - 1)):
            return set(picks)

    return set(random.sample(eligible, num_matches))


def feasible_isi_total(
    num_stims: int, isi_range: Tuple[float, float]
) -> Tuple[float, float]:
    """Return min and max sum of ISIs for given stimulus count."""
    min_total = num_stims * isi_range[0]
    max_total = num_stims * isi_range[1]
    return min_total, max_total


def generate_isi_times(
    num_stims: int, target_total: float, isi_range: Tuple[float, float], quantum: float
) -> List[float]:
    """Generate ISI times that sum to target total."""
    isi_min, isi_max = isi_range
    capacity_quanta = int(round((isi_max - isi_min) / quantum))

    baseline_total = num_stims * isi_min
    remaining = target_total - baseline_total
    quanta_needed = int(round(remaining / quantum))

    extras_quanta = [0] * num_stims
    while quanta_needed > 0:
        idx = random.randrange(num_stims)
        if extras_quanta[idx] < capacity_quanta:
            extras_quanta[idx] += 1
            quanta_needed -= 1

    return [round(isi_min + q * quantum, 3) for q in extras_quanta]


def choose_num_stims_for_target(
    n_back: int,
    target_total: float,
    stim_duration_range: Tuple[float, float],
    isi_range: Tuple[float, float],
) -> Optional[int]:
    """Find stimulus count that makes target duration feasible."""
    feasible: List[int] = []

    avg_stim_duration = (stim_duration_range[0] + stim_duration_range[1]) / 2

    for num in range(n_back + 2, 60):
        min_isi_total, max_isi_total = feasible_isi_total(num, isi_range)
        min_possible = num * stim_duration_range[0] + min_isi_total
        max_possible = num * stim_duration_range[1] + max_isi_total

        if min_possible <= target_total <= max_possible:
            feasible.append(num)

    return random.choice(feasible) if feasible else None


def generate_stimulus_durations(
    num_stims: int, stim_duration_range: Tuple[float, float], quantum: float
) -> List[float]:
    """Generate stimulus durations within specified range."""
    if stim_duration_range[0] == stim_duration_range[1]:
        return [stim_duration_range[0]] * num_stims
    else:
        durations = []
        for _ in range(num_stims):
            dur = random.uniform(*stim_duration_range)
            quantized = round(dur / quantum) * quantum
            quantized = max(
                stim_duration_range[0], min(stim_duration_range[1], quantized)
            )
            durations.append(round(quantized, 3))
        return durations


def generate_trial(
    n_back: int,
    duration_range: Tuple[float, float],
    stim_duration_range: Tuple[float, float],
    isi_range: Tuple[float, float],
    match_rate: float,
    stim_set: List[str],
    quantum: float,
) -> Dict[str, Any]:
    """Generate one n-back trial with specified parameters."""
    low, high = duration_range

    while True:
        target_total = random.uniform(low, high)
        num_stims = choose_num_stims_for_target(
            n_back, target_total, stim_duration_range, isi_range
        )

        if num_stims is None:
            continue

        stim_durations = generate_stimulus_durations(
            num_stims, stim_duration_range, quantum
        )
        total_stim_duration = sum(stim_durations)

        target_isi_total = target_total - total_stim_duration
        min_isi_total, max_isi_total = feasible_isi_total(num_stims, isi_range)

        if not (min_isi_total <= target_isi_total <= max_isi_total):
            continue

        num_matches = calculate_num_matches(n_back, num_stims, match_rate)
        match_pos = select_match_positions(n_back, num_stims, num_matches)

        stim_list: List[str] = []
        for i in range(num_stims):
            if i in match_pos:
                stim = stim_list[i - n_back]
            else:
                forbidden = stim_list[i - n_back] if i >= n_back else None
                choices = [s for s in stim_set if s != forbidden]
                stim = random.choice(choices)
            stim_list.append(stim)

        isi_times = generate_isi_times(num_stims, target_isi_total, isi_range, quantum)
        total_duration = round(total_stim_duration + sum(isi_times), 3)

        if not (low - 1e-6 <= total_duration <= high + 1e-6):
            continue

        return {
            "n_back": n_back,
            "stim_list": stim_list,
            "stim_durations": stim_durations,
            "match_positions": sorted(match_pos),
            "isi_times": isi_times,
            "num_stims": num_stims,
            "total_duration": total_duration,
        }


def generate_level_sequence(levels: List[int], num_trials: int) -> List[int]:
    """Generate sequence of n-back levels for trials."""
    if len(levels) == 1:
        return levels * num_trials

    base_count = num_trials // len(levels)
    remainder = num_trials % len(levels)

    sequence = []
    for i, level in enumerate(levels):
        count = base_count + (1 if i < remainder else 0)
        sequence.extend([level] * count)

    random.shuffle(sequence)
    return sequence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate n-back task conditions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_nback_conditions.py --trials 100 --levels 2 3
  python generate_nback_conditions.py --trials 60 --levels 1 2 --duration-range 8 12
  python generate_nback_conditions.py --trials 40 --levels 2 --stim-duration 1.0 --isi 0.5
        """,
    )

    parser.add_argument(
        "--trials", type=int, required=True, help="Total number of trials"
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        required=True,
        help="N-back levels (e.g., 1 2 or just 2)",
    )
    parser.add_argument(
        "--duration-range",
        type=float,
        nargs="+",
        default=[9, 12],
        help="Trial duration range in seconds (default: 9 12)",
    )
    parser.add_argument(
        "--stim-duration",
        type=float,
        nargs="+",
        default=[1.2],
        help="Stimulus duration(s) in seconds (default: 1.2)",
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
        help="Target match percentage (default: 30)",
    )
    parser.add_argument(
        "--quantum",
        type=float,
        default=0.01,
        help="Time quantization step in seconds (default: 0.01)",
    )
    parser.add_argument(
        "--stimuli",
        choices=["letters", "digits"],
        default="letters",
        help="Stimulus type (default: letters)",
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    duration_range = parse_range_arg(args.duration_range)
    stim_duration_range = parse_range_arg(args.stim_duration)
    isi_range = parse_range_arg(args.isi)

    stim_set = LETTERS if args.stimuli == "letters" else DIGITS

    level_sequence = generate_level_sequence(args.levels, args.trials)

    rows = []
    for trial_num, n_back in enumerate(level_sequence, start=1):
        trial = generate_trial(
            n_back=n_back,
            duration_range=duration_range,
            stim_duration_range=stim_duration_range,
            isi_range=isi_range,
            match_rate=args.match_rate,
            stim_set=stim_set,
            quantum=args.quantum,
        )

        rows.append(
            {
                "trial": trial_num,
                "n_back": trial["n_back"],
                "stim_list": json.dumps(trial["stim_list"]),
                "stim_durations": json.dumps(trial["stim_durations"]),
                "match_positions": json.dumps(trial["match_positions"]),
                "isi_times": json.dumps(trial["isi_times"]),
                "num_stims": trial["num_stims"],
                "total_duration": trial["total_duration"],
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    levels_str = "_".join(map(str, sorted(set(args.levels))))
    filename = f"{args.stimuli}_{levels_str}back_{args.trials}trials.csv"
    if args.seed is not None:
        filename = filename.replace(".csv", f"_seed{args.seed}.csv")

    filepath = OUTPUT_DIR / filename
    pd.DataFrame(rows).to_csv(filepath, index=False)

    print(f"Saved conditions to {filepath.resolve()}")

    df = pd.DataFrame(rows)
    durations = df["total_duration"].tolist()
    print(f"\nSummary:")
    print(f"- Total trials: {len(df)}")
    print(f"- N-back levels: {sorted(set(args.levels))}")
    print(f"- Duration: {min(durations):.2f}s - {max(durations):.2f}s")
    print(f"- Match rate: {args.match_rate}%")


if __name__ == "__main__":
    main()
