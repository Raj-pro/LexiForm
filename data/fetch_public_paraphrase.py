"""
Fetch public paraphrase datasets (PAWS-Wiki + Quora QQP) and convert to the
project's `(src, tgt)` JSONL schema.

Both are direct downloads — no `datasets` library dependency.
Outputs are filtered to PARAPHRASE-POSITIVE pairs only (label/is_duplicate=1).

Run:
    .venv/bin/python -m data.fetch_public_paraphrase --out data/public_paraphrase.jsonl

Sources:
    PAWS-Wiki:  https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz
                (CC-BY-SA-4.0, Google Research)
    Quora QQP:  http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv
                (Quora terms; public release for non-commercial NLP research)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.request
from pathlib import Path


QUORA_URL = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"


def _http_get(url: str, retries: int = 5, backoff: float = 2.0, timeout: int = 600) -> bytes:
    last_err: Exception | None = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "llm-paraphraser-trainer/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            last_err = e
            time.sleep(backoff ** i)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_err}")


def fetch_paws() -> list[tuple[str, str]]:
    """Return PAWS-Wiki paraphrase pairs (label=1 only) via HuggingFace datasets.

    PAWS' direct Google Cloud bucket URL now returns 403 for anonymous clients,
    so we use the HF mirror at `google-research-datasets/paws`, config
    `labeled_final` (the human-validated 'final' subset). Combines train + dev
    + test splits, filters to label=1.
    """
    from datasets import load_dataset
    print("Loading PAWS-Wiki via HuggingFace (google-research-datasets/paws) ...",
          file=sys.stderr)
    pairs: list[tuple[str, str]] = []
    for split in ("train", "validation", "test"):
        ds = load_dataset("google-research-datasets/paws", "labeled_final", split=split)
        before = len(pairs)
        for row in ds:
            if row.get("label") != 1:
                continue
            s1 = (row.get("sentence1") or "").strip()
            s2 = (row.get("sentence2") or "").strip()
            if s1 and s2 and s1 != s2:
                pairs.append((s1, s2))
        print(f"  PAWS {split}: +{len(pairs) - before:,} paraphrase pairs",
              file=sys.stderr)
    return pairs


def fetch_quora() -> list[tuple[str, str]]:
    """Return Quora QQP duplicates (is_duplicate=1).
    Each row: id, qid1, qid2, question1, question2, is_duplicate.
    """
    print(f"Downloading Quora QQP ({QUORA_URL}) ...", file=sys.stderr)
    blob = _http_get(QUORA_URL)
    text = blob.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    pairs: list[tuple[str, str]] = []
    for row in reader:
        if row.get("is_duplicate") != "1":
            continue
        q1, q2 = (row.get("question1") or "").strip(), (row.get("question2") or "").strip()
        if q1 and q2 and q1 != q2:
            pairs.append((q1, q2))
    print(f"  Quora: +{len(pairs):,} duplicate pairs", file=sys.stderr)
    return pairs


def _good_pair(s: str, t: str, min_words: int = 4, max_words: int = 64) -> bool:
    """Sanity-filter: skip very short / very long pairs, drop near-identical."""
    sw, tw = s.split(), t.split()
    if not (min_words <= len(sw) <= max_words and min_words <= len(tw) <= max_words):
        return False
    # Drop pairs whose normalized forms match (already-identical, contributes nothing).
    if s.lower().strip() == t.lower().strip():
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path("data/public_paraphrase.jsonl"), type=Path)
    ap.add_argument("--no_paws",  action="store_true", help="Skip PAWS")
    ap.add_argument("--no_quora", action="store_true", help="Skip Quora")
    ap.add_argument("--min_words", default=4,  type=int)
    ap.add_argument("--max_words", default=64, type=int)
    args = ap.parse_args()

    all_pairs: list[tuple[str, str, str]] = []  # (src, tgt, source_label)

    if not args.no_paws:
        for s, t in fetch_paws():
            all_pairs.append((s, t, "paws"))
    if not args.no_quora:
        for s, t in fetch_quora():
            all_pairs.append((s, t, "quora"))

    # Sanity-filter and dedup by (src, tgt) tuple.
    seen: set[tuple[str, str]] = set()
    kept: list[tuple[str, str, str]] = []
    dropped_filter = 0
    dropped_dup    = 0
    for s, t, src_lbl in all_pairs:
        if not _good_pair(s, t, args.min_words, args.max_words):
            dropped_filter += 1
            continue
        key = (s, t)
        if key in seen:
            dropped_dup += 1
            continue
        seen.add(key)
        kept.append((s, t, src_lbl))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for s, t, src_lbl in kept:
            f.write(json.dumps({"src": s, "tgt": t, "source": src_lbl}) + "\n")

    n_paws  = sum(1 for _, _, l in kept if l == "paws")
    n_quora = sum(1 for _, _, l in kept if l == "quora")
    print(f"\nKept {len(kept):,} pairs  (PAWS {n_paws:,}, Quora {n_quora:,})", file=sys.stderr)
    print(f"Dropped {dropped_filter:,} by length/identity filter, "
          f"{dropped_dup:,} as duplicates.", file=sys.stderr)
    print(f"Wrote -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
