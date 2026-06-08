"""
Tier 4 — filter noisy paraphrase pairs.

Drops any (src, tgt) pair where the two sides disagree on any of:
  • Named entities (PERSON, ORG, GPE, LOC, DATE, MONEY, PERCENT, ...)
  • Numeric content (regex over digits / currency / percent)
  • Sentiment polarity (DistilBERT SST-2)
  • Negation parity (count of negation tokens mod 2)

These are the four failure modes the audit identified (F1, F2, F3, F5).
A pair where src and tgt disagree on any of them is a noisy training signal —
the model would learn to *invent* such drift. Removing them sharpens the
training distribution.

Run:
    .venv/bin/python -m data.filter \\
        --inp data/clean.jsonl --out data/clean_filtered.jsonl
"""
import argparse
import json
import re
import sys
from pathlib import Path

import spacy
from transformers import pipeline


NUM_RE = re.compile(r"\b\d[\d,\.]*\b|\$\d+|\d+%")

NEG = {
    "not", "n't", "never", "no", "without", "nor",
    "cannot", "can't", "won't", "didn't", "doesn't",
    "isn't", "aren't", "wasn't", "weren't",
    "shouldn't", "wouldn't", "couldn't", "hasn't", "haven't", "hadn't",
}

ENT_LABELS = {
    "PERSON", "ORG", "GPE", "LOC", "DATE", "TIME",
    "MONEY", "PERCENT", "QUANTITY", "CARDINAL", "ORDINAL", "NORP",
}


def _ent_set(doc) -> set[tuple[str, str]]:
    return {(e.text.lower(), e.label_) for e in doc.ents if e.label_ in ENT_LABELS}


def _neg_count(text: str) -> int:
    toks = re.findall(r"[A-Za-z']+", text.lower())
    return sum(1 for t in toks if t in NEG)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/clean.jsonl", type=Path)
    ap.add_argument("--out", default="data/clean_filtered.jsonl", type=Path)
    ap.add_argument("--batch", default=64, type=int,
                    help="Batch size for sentiment pipeline and spaCy nlp.pipe.")
    ap.add_argument("--no_sentiment", action="store_true",
                    help="Skip the (slow) DistilBERT sentiment check.")
    ap.add_argument("--limit", default=0, type=int,
                    help="Process only the first N rows. 0 = all.")
    args = ap.parse_args()

    print("Loading spaCy en_core_web_sm...", file=sys.stderr)
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])

    sent = None
    if not args.no_sentiment:
        print("Loading DistilBERT SST-2 (sentiment)...", file=sys.stderr)
        sent = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=1,
            device=-1,  # CPU; we just need polarity labels, not speed
        )

    rows = []
    with args.inp.open() as f:
        for i, line in enumerate(f):
            if args.limit and i >= args.limit:
                break
            rows.append(json.loads(line))
    print(f"Read {len(rows):,} pairs from {args.inp}", file=sys.stderr)

    srcs = [r["src"] for r in rows]
    tgts = [r["tgt"] for r in rows]

    # spaCy in one pipe pass each side
    print("NER pass (src)...", file=sys.stderr)
    src_docs = list(nlp.pipe(srcs, batch_size=args.batch))
    print("NER pass (tgt)...", file=sys.stderr)
    tgt_docs = list(nlp.pipe(tgts, batch_size=args.batch))

    src_sent_lbl: list[str] = []
    tgt_sent_lbl: list[str] = []
    if sent is not None:
        print("Sentiment pass (src)...", file=sys.stderr)
        for i in range(0, len(srcs), args.batch):
            chunk = srcs[i : i + args.batch]
            preds = sent(chunk, truncation=True, max_length=128)
            src_sent_lbl.extend(p[0]["label"] for p in preds)
            if (i // args.batch) % 20 == 0:
                print(f"  {i+len(chunk)}/{len(srcs)}", file=sys.stderr)
        print("Sentiment pass (tgt)...", file=sys.stderr)
        for i in range(0, len(tgts), args.batch):
            chunk = tgts[i : i + args.batch]
            preds = sent(chunk, truncation=True, max_length=128)
            tgt_sent_lbl.extend(p[0]["label"] for p in preds)
            if (i // args.batch) % 20 == 0:
                print(f"  {i+len(tgts)}/{len(tgts)}", file=sys.stderr)

    drop_entity   = 0
    drop_number   = 0
    drop_sentiment = 0
    drop_negation = 0
    kept_rows: list[dict] = []

    for i, row in enumerate(rows):
        src, tgt = row["src"], row["tgt"]
        if _ent_set(src_docs[i]) != _ent_set(tgt_docs[i]):
            drop_entity += 1
            continue
        if set(NUM_RE.findall(src)) != set(NUM_RE.findall(tgt)):
            drop_number += 1
            continue
        if sent is not None and src_sent_lbl[i] != tgt_sent_lbl[i]:
            drop_sentiment += 1
            continue
        if (_neg_count(src) % 2) != (_neg_count(tgt) % 2):
            drop_negation += 1
            continue
        kept_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in kept_rows:
            f.write(json.dumps(r) + "\n")

    total = len(rows)
    kept  = len(kept_rows)
    print(file=sys.stderr)
    print(f"Total:           {total:,}", file=sys.stderr)
    print(f"Kept:            {kept:,}  ({100*kept/total:.1f}%)", file=sys.stderr)
    print(f"Dropped (entity):    {drop_entity:,}", file=sys.stderr)
    print(f"Dropped (number):    {drop_number:,}", file=sys.stderr)
    print(f"Dropped (sentiment): {drop_sentiment:,}", file=sys.stderr)
    print(f"Dropped (negation):  {drop_negation:,}", file=sys.stderr)
    print(f"\nWrote -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
