"""
Load all book CSVs under a directory, concatenate chapter text, and split into
overlapping fixed-length token windows for pretraining.

Used by `training/pretrain_dataset.py` (BookSpanCorruptionDataset).

Each input CSV uses the project's standard schema (`no, story`) — see
[fetch_gutenberg.py](fetch_gutenberg.py) and the user-provided book CSVs.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from tokenizer.tokenizer import Tokenizer


# Each row's story column can be megabytes long for un-chaptered books.
csv.field_size_limit(sys.maxsize)


def _iter_chapter_text(book_dir: Path) -> list[str]:
    """Yield every chapter's text from every CSV in `book_dir`, in stable order
    (sorted by filename). Skips manifests and other helper files.
    """
    out: list[str] = []
    for path in sorted(book_dir.glob("*.csv")):
        if path.name.startswith("_"):  # skip _manifest.jsonl, _se_manifest.jsonl
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "story" not in reader.fieldnames:
                continue
            for row in reader:
                story = (row.get("story") or "").strip()
                if story:
                    out.append(story)
    return out


def build_chunks(
    book_dir: str | Path,
    tokenizer: Tokenizer,
    seq_len: int = 128,
    stride: int = 64,
    min_chunk_tokens: int = 32,
) -> list[list[int]]:
    """Tokenize every chapter, slice into overlapping seq_len-windows with the
    given stride, drop windows shorter than `min_chunk_tokens` (the trailing
    tail of a chapter).

    Returns a list of token-id lists. Each window is *plain* tokens — no BOS/
    EOS. The span-corruption dataset adds sentinel structure on top.
    """
    book_dir = Path(book_dir)
    chapters = _iter_chapter_text(book_dir)
    if not chapters:
        return []

    chunks: list[list[int]] = []
    special_skip = {tokenizer.pad_id, tokenizer.bos_id,
                    tokenizer.eos_id, tokenizer.unk_id}
    for chap in chapters:
        ids = tokenizer.encode(chap)
        ids = [t for t in ids if t not in special_skip]  # defensive
        for start in range(0, len(ids), stride):
            window = ids[start : start + seq_len]
            if len(window) >= min_chunk_tokens:
                chunks.append(window)
            if start + seq_len >= len(ids):
                break
    return chunks


def corpus_stats(book_dir: str | Path, tokenizer: Tokenizer) -> dict:
    """One-pass corpus tally — total chapters, total characters, total tokens,
    byte-fallback token count. Used by Phase 1 verification.
    """
    book_dir = Path(book_dir)
    chapters = _iter_chapter_text(book_dir)
    total_chars = sum(len(c) for c in chapters)

    total_tokens   = 0
    bytefallbacks  = 0
    for chap in chapters:
        # Encode in str-mode to detect byte-fallback pieces (form "<0xAB>").
        pieces = tokenizer.sp.encode(chap, out_type=str)
        total_tokens  += len(pieces)
        bytefallbacks += sum(1 for p in pieces if p.startswith("<0x"))

    return {
        "chapters":         len(chapters),
        "total_chars":      total_chars,
        "total_tokens":     total_tokens,
        "byte_fallback":    bytefallbacks,
        "bf_ratio":         bytefallbacks / max(total_tokens, 1),
        "chars_per_token":  total_chars   / max(total_tokens, 1),
    }


if __name__ == "__main__":
    # Quick verification CLI: `python -m data.books data/books/`
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("book_dir", type=Path, nargs="?", default=Path("data/books"))
    ap.add_argument("--tok",    default="tokenizer/tokenizer.model")
    ap.add_argument("--seq_len", default=128, type=int)
    ap.add_argument("--stride",  default=64,  type=int)
    args = ap.parse_args()

    tok = Tokenizer(args.tok)
    stats = corpus_stats(args.book_dir, tok)
    print(f"chapters:        {stats['chapters']:,}")
    print(f"total chars:     {stats['total_chars']:,}")
    print(f"total tokens:    {stats['total_tokens']:,}")
    print(f"byte-fallback:   {stats['byte_fallback']:,}  "
          f"({100 * stats['bf_ratio']:.3f}% — target ≤0.5%)")
    print(f"chars/token:     {stats['chars_per_token']:.2f}")

    chunks = build_chunks(args.book_dir, tok, seq_len=args.seq_len, stride=args.stride)
    print(f"chunks (seq_len={args.seq_len}, stride={args.stride}): {len(chunks):,}")
