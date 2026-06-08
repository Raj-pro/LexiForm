"""
Mine a phrase-level translation table from clean.jsonl.

For every (src, tgt) pair, each source n-gram (n=1..max_n) is associated
with the *bag* of target token IDs that appear in the matching target sentence.
After add-1 smoothing and a minimum-count filter, the result is:

    table: dict[tuple[int,...], dict[int, float]]
               src n-gram     → sparse log-probs over vocab

This is intentionally "bag of words" alignment (not word-aligned), which is
robust at 62K pairs and fast to build.  The table is queried at each decode
step by the cross-attention peak to add a small bias toward substitutions
the training data endorses.

Usage:
    python3 build_phrase_table.py \\
        --data data/clean.jsonl \\
        --tok  tokenizer/tokenizer.model \\
        --out  phrase_table.pkl
"""
import argparse
import json
import math
import pickle
from collections import defaultdict, Counter
from pathlib import Path

from tokenizer.tokenizer import Tokenizer


def build(
    jsonl_path: Path,
    tok: Tokenizer,
    max_n: int = 3,
    min_count: int = 3,
    bidirectional: bool = True,
) -> dict[tuple, dict[int, float]]:
    """Return a sparse phrase table mined from *jsonl_path*."""
    special = {tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id}
    counts: dict[tuple, Counter] = defaultdict(Counter)

    def _process(src_text: str, tgt_text: str) -> None:
        src_t = [t for t in tok.encode("<paraphrase> " + src_text) if t not in special]
        tgt_t = [t for t in tok.encode(tgt_text) if t not in special]
        if not src_t or not tgt_t:
            return
        # bag-of-words: each tgt token counted once per pair per src n-gram
        tgt_bag = set(tgt_t)
        for n in range(1, max_n + 1):
            for i in range(len(src_t) - n + 1):
                ng = tuple(src_t[i : i + n])
                for t in tgt_bag:
                    counts[ng][t] += 1

    with open(jsonl_path) as f:
        for line in f:
            row = json.loads(line)
            _process(row["src"], row["tgt"])
            if bidirectional:
                _process(row["tgt"], row["src"])

    V = tok.vocab_size
    table: dict[tuple, dict[int, float]] = {}
    for ng, counter in counts.items():
        total = sum(counter.values())
        if total < min_count:
            continue
        # add-1 smoothed log-probs (sparse: only store observed target tokens)
        log_z = math.log(total + V)
        table[ng] = {t: math.log(c + 1) - log_z for t, c in counter.items()}

    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      default="data/clean.jsonl", type=Path)
    parser.add_argument("--tok",       default="tokenizer/tokenizer.model")
    parser.add_argument("--out",       default="phrase_table.pkl", type=Path)
    parser.add_argument("--max_n",     default=3, type=int,
                        help="Maximum source n-gram order (1–3 recommended)")
    parser.add_argument("--min_count", default=3, type=int,
                        help="Drop n-grams seen fewer than this many times")
    parser.add_argument("--no_bidir",  action="store_true",
                        help="Skip reverse-direction (tgt→src) pairs")
    args = parser.parse_args()

    tok = Tokenizer(args.tok)
    print(f"Mining phrase table from {args.data} …")
    table = build(
        args.data, tok,
        max_n=args.max_n,
        min_count=args.min_count,
        bidirectional=not args.no_bidir,
    )

    unigrams = sum(1 for k in table if len(k) == 1)
    bigrams  = sum(1 for k in table if len(k) == 2)
    trigrams = sum(1 for k in table if len(k) == 3)
    print(f"  {len(table):,} entries  "
          f"(1-gram: {unigrams:,}  2-gram: {bigrams:,}  3-gram: {trigrams:,})")

    with open(args.out, "wb") as f:
        pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
