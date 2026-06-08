"""
Deduplicate JSONL pairs using MinHash LSH.
Run: python3 -m data.dedup --inp data/raw --out data/clean.jsonl
"""
import json
import argparse
from pathlib import Path
from datasketch import MinHash, MinHashLSH


def minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for word in text.lower().split():
        m.update(word.encode())
    return m


def dedup(paths: list[Path], threshold: float = 0.92) -> list[dict]:
    # 0.92 was 0.8: the prior threshold dropped genuine paraphrases that
    # happened to share many tokens. 0.92 only drops near-identical pairs.
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    kept = []
    for path in paths:
        with open(path) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                key = f"{path.stem}_{i}"
                m = minhash(row["src"] + " " + row["tgt"])
                if not lsh.query(m):
                    lsh.insert(key, m)
                    kept.append(row)
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp",       default="data/raw",        type=Path)
    parser.add_argument("--out",       default="data/clean.jsonl", type=Path)
    parser.add_argument("--threshold", default=0.92,              type=float)
    args = parser.parse_args()

    paths = sorted(args.inp.glob("*.jsonl"))
    print(f"Deduplicating {len(paths)} files at threshold {args.threshold}...")
    pairs = dedup(paths, threshold=args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"After dedup: {len(pairs):,} pairs → {args.out}")


if __name__ == "__main__":
    main()
