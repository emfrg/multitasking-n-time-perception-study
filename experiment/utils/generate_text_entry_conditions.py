# utils/generate_text_entry_conditions
"""Generate trial conditions for text-entry experiment."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import Final, List, Sequence

import pandas as pd

BANK_PATH: Final[pathlib.Path] = pathlib.Path("resources", "word_sets.json")
DEFAULT_INTERRUPTIONS: Final[int] = 1


def load_wordbank(length: int) -> List[str]:
    """Return list of words of requested length."""
    try:
        with BANK_PATH.open(encoding="utf-8") as fh:
            bank: dict[str, List[str]] = json.load(fh)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Word bank not found at {BANK_PATH}. Run build_word_sets.py first."
        ) from exc

    words = bank.get(str(length))
    if not words:
        raise SystemExit(f"No entry for word length {length} in word bank.")
    return words


def choose_interruptions(word_length: int, n_interruptions: int) -> Sequence[int]:
    """Return interruption position for interrupted trials."""
    if n_interruptions > 0:
        valid_positions = list(range(2, word_length))
        return [random.choice(valid_positions)]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=90, help="number of trials")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[7, 8, 9],
        help="word lengths to use (e.g., --lengths 8 or --lengths 7 8 9)",
    )
    parser.add_argument(
        "--interruptions",
        type=int,
        default=DEFAULT_INTERRUPTIONS,
        help="interruptions per trial",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    word_pools = {}
    for length in args.lengths:
        word_pools[length] = load_wordbank(length)

    words_per_length = args.trials // len(args.lengths)
    remainder = args.trials % len(args.lengths)

    selected_words = []
    for i, length in enumerate(args.lengths):
        n_words = words_per_length
        if i < remainder:
            n_words += 1

        if n_words > len(word_pools[length]):
            raise SystemExit(
                f"Not enough {length}-letter words. "
                f"Requested {n_words} but only {len(word_pools[length])} available."
            )

        selected_words.extend(word_pools[length][:n_words])

    random.shuffle(selected_words)

    n_interrupted = args.trials // 2
    conditions = ["interrupted"] * n_interrupted + ["sequential"] * (
        args.trials - n_interrupted
    )
    random.shuffle(conditions)

    rows = []
    for idx, (word, cond) in enumerate(zip(selected_words, conditions), start=1):
        interrupt_positions = (
            choose_interruptions(len(word), args.interruptions)
            if cond == "interrupted"
            else []
        )
        rows.append(
            {
                "trial": idx,
                "word": word.upper(),
                "letters": json.dumps(list(word.upper())),
                "interrupt_pos": json.dumps(interrupt_positions),
                "interruption_condition": cond,
            }
        )

    outdir = pathlib.Path("conditions", "text_entry")
    outdir.mkdir(parents=True, exist_ok=True)

    lengths_str = "_".join(map(str, sorted(args.lengths)))
    outfile = outdir / f"conditions_len{lengths_str}_{args.trials}trials.csv"

    pd.DataFrame(rows).to_csv(outfile, index=False)
    print(f"Saved conditions to {outfile.resolve()}")


if __name__ == "__main__":
    main()
