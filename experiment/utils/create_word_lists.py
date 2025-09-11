# utils/create_word_lists
"""Generate word lists of specific lengths from frequency corpus."""

from __future__ import annotations
import json
import pathlib
from typing import Final, List, Dict
from wordfreq import top_n_list

# Word length ranges: 6 to 10 letters
WORD_LENGTHS: Final[List[int]] = [6, 7, 8, 9, 10]
# Number of words to collect for each length
WORDS_PER_LENGTH: Final[Dict[int, int]] = {
    6: 100,
    7: 100,
    8: 100,
    9: 100,
    10: 100,
}
OUTFILE: Final[pathlib.Path] = pathlib.Path("resources", "word_sets.json")


def collect_words(length: int, n: int) -> List[str]:
    """Return n most frequent English words of given length."""
    pool = top_n_list("en", 50_000)

    filtered_words = []
    for w in pool:
        if len(w) == length:
            has_double = any(w[i] == w[i + 1] for i in range(len(w) - 1))
            only_letters = w.isalpha()

            if not has_double and only_letters:
                filtered_words.append(w)

    return filtered_words[:n]


def main() -> None:
    OUTFILE.parent.mkdir(parents=True, exist_ok=True)

    wordbank: dict[str, List[str]] = {
        str(length): collect_words(length, WORDS_PER_LENGTH[length])
        for length in WORD_LENGTHS
    }

    with OUTFILE.open("w", encoding="utf-8") as fh:
        json.dump(wordbank, fh, ensure_ascii=False, indent=2)

    print(f"Saved word bank to {OUTFILE.resolve()}")


if __name__ == "__main__":
    main()
