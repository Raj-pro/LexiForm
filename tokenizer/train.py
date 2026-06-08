"""
Train a 16K BPE SentencePiece tokenizer on the cleaned corpus.
Run: python3 -m tokenizer.train --data data/clean.jsonl --out tokenizer/
"""
import json
import argparse
import tempfile
from pathlib import Path
import sentencepiece as spm


def build_corpus(jsonl_path: Path, corpus_path: Path):
    with open(jsonl_path) as fin, open(corpus_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            fout.write(row["src"] + "\n")
            fout.write(row["tgt"] + "\n")


def train(corpus_path: Path, model_prefix: str, vocab_size: int = 16000):
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=0.9999,
        model_type="bpe",
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        user_defined_symbols=["<paraphrase>"],
        # Byte-fallback degrades rare / OOV characters to UTF-8 byte tokens
        # instead of collapsing them to <unk>, which is much more robust on
        # user input that contains uncommon proper nouns or punctuation.
        byte_fallback=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/clean.jsonl", type=Path)
    parser.add_argument("--out", default="tokenizer", type=Path)
    parser.add_argument("--vocab_size", default=16000, type=int)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    corpus = args.out / "corpus.txt"

    print("Building tokenizer corpus...")
    build_corpus(args.data, corpus)

    model_prefix = str(args.out / "tokenizer")
    print(f"Training tokenizer (vocab={args.vocab_size})...")
    train(corpus, model_prefix, args.vocab_size)
    print(f"Saved → {model_prefix}.model")


if __name__ == "__main__":
    main()
