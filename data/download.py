"""
Download and convert paraphrase datasets to JSONL.
Run: python3 -m data.download --out data/raw
"""
import json
import random
import argparse
from pathlib import Path
from datasets import load_dataset
from data.clean import is_valid_pair, basic_clean


SEED = 42


def save_jsonl(pairs: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(pairs):,} pairs -> {path}")


def load_paws(split: str = "train") -> list[dict]:
    ds = load_dataset("paws", "labeled_final", split=split)
    pairs = []
    for row in ds:
        if row["label"] == 1:
            src, tgt = basic_clean(row["sentence1"]), basic_clean(row["sentence2"])
            if is_valid_pair(src, tgt):
                pairs.append({"src": src, "tgt": tgt})
    return pairs


def load_qqp(split: str = "train", cap: int | None = 150_000) -> list[dict]:
    """
    Previously truncated to the first 150k rows mid-iteration, which is biased
    toward whatever order the source returns. Now we collect ALL valid pairs,
    shuffle deterministically, and only then take the cap.
    """
    ds = load_dataset("glue", "qqp", split=split)
    pairs = []
    for row in ds:
        if row["label"] != 1:
            continue
        q1, q2 = row["question1"], row["question2"]
        if not q1 or not q2:
            continue
        src, tgt = basic_clean(q1), basic_clean(q2)
        if is_valid_pair(src, tgt):
            pairs.append({"src": src, "tgt": tgt})

    rng = random.Random(SEED)
    rng.shuffle(pairs)
    if cap is not None and len(pairs) > cap:
        pairs = pairs[:cap]
    return pairs


def load_mrpc(split: str = "train") -> list[dict]:
    ds = load_dataset("glue", "mrpc", split=split)
    pairs = []
    for row in ds:
        if row["label"] == 1:
            src, tgt = basic_clean(row["sentence1"]), basic_clean(row["sentence2"])
            if is_valid_pair(src, tgt):
                pairs.append({"src": src, "tgt": tgt})
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",     default="data/raw", type=Path)
    parser.add_argument("--qqp_cap", default=150_000,    type=int)
    args = parser.parse_args()

    print("Downloading PAWS...")
    paws = load_paws()
    save_jsonl(paws, args.out / "paws.jsonl")

    print("Downloading MRPC...")
    mrpc = load_mrpc()
    save_jsonl(mrpc, args.out / "mrpc.jsonl")

    print(f"Downloading QQP (shuffle+cap {args.qqp_cap:,})...")
    qqp = load_qqp(cap=args.qqp_cap)
    save_jsonl(qqp, args.out / "qqp.jsonl")

    total = len(paws) + len(mrpc) + len(qqp)
    print("\nPer-source counts:")
    for name, n in (("PAWS", len(paws)), ("MRPC", len(mrpc)), ("QQP", len(qqp))):
        pct = 100.0 * n / max(total, 1)
        print(f"  {name:6s} {n:>7,}  ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
