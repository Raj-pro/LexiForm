"""
Batch paraphrase a text file, sentence by sentence.

Reads --input, splits each paragraph into sentences, runs the paraphraser
on each, and writes:
  • --out      a side-by-side log (SRC / PAR per sentence + paragraph breaks)
  • --rewrite  a coherent paraphrased copy of the input (paragraph structure preserved)

Run:
    .venv/bin/python paraphrase_file.py \
        --input sample.txt \
        --out   sample_log.txt \
        --rewrite sample_paraphrased.txt
"""
import argparse
import re
import sys
from pathlib import Path

import torch

from inference.infer import load_model, paraphrase, get_reranker, load_phrase_table


# Naïve sentence splitter — good enough for prose, avoids an nltk dep.
# Splits on .?! followed by whitespace and an opening capital / quote.
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'“])')


def split_sentences(paragraph: str) -> list[str]:
    p = paragraph.strip()
    if not p:
        return []
    parts = [s.strip() for s in SENT_SPLIT_RE.split(p) if s.strip()]
    return parts or [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",   required=True, type=Path)
    ap.add_argument("--out",     default=Path("sample_log.txt"), type=Path,
                    help="Side-by-side SRC/PAR log.")
    ap.add_argument("--rewrite", default=Path("sample_paraphrased.txt"), type=Path,
                    help="Coherent paraphrased rewrite (paragraph structure preserved).")
    ap.add_argument("--ckpt",    default="checkpoints/best.pt")
    ap.add_argument("--tok",     default="tokenizer/tokenizer.model")
    ap.add_argument("--beams",   default=12, type=int)
    ap.add_argument("--groups",  default=4,  type=int)
    ap.add_argument("--diversity_lambda", default=0.6, type=float)
    ap.add_argument("--phrase_table", default="", type=str)
    ap.add_argument("--phrase_beta",  default=0.2, type=float)
    ap.add_argument("--min_edit_ratio", default=0.15, type=float)
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", file=sys.stderr)

    model, tok, config = load_model(args.ckpt, args.tok, device)
    ptable = load_phrase_table(args.phrase_table) if args.phrase_table else None
    get_reranker()

    text = args.input.read_text()
    paragraphs = text.split("\n\n")

    log_lines:     list[str] = []
    rewrite_paras: list[str] = []

    n_total = 0
    n_echoed = 0
    for pi, para in enumerate(paragraphs):
        sents = split_sentences(para)
        if not sents:
            rewrite_paras.append("")
            log_lines.append("")
            continue

        rewritten: list[str] = []
        log_lines.append(f"--- paragraph {pi} ---")
        for sent in sents:
            n_total += 1
            outs = paraphrase(
                sent, model, tok, config, device,
                num_outputs=1, num_beams=args.beams,
                num_groups=args.groups, diversity_lambda=args.diversity_lambda,
                phrase_table=ptable, phrase_beta=args.phrase_beta,
                min_edit_ratio=args.min_edit_ratio,
            )
            pred = outs[0] if outs else sent
            if pred.strip() == sent.strip():
                n_echoed += 1
                tag = "ECHO"
            else:
                tag = "PAR "
            log_lines.append(f"SRC : {sent}")
            log_lines.append(f"{tag}: {pred}")
            log_lines.append("")
            rewritten.append(pred)
            print(f"[{pi}] {tag} {sent[:60]}{'…' if len(sent)>60 else ''}",
                  file=sys.stderr)

        rewrite_paras.append(" ".join(rewritten))

    args.out.write_text("\n".join(log_lines))
    args.rewrite.write_text("\n\n".join(rewrite_paras))

    print(f"\nProcessed {n_total} sentences  echoed: {n_echoed}  "
          f"paraphrased: {n_total - n_echoed}", file=sys.stderr)
    print(f"Log:     {args.out}", file=sys.stderr)
    print(f"Rewrite: {args.rewrite}", file=sys.stderr)


if __name__ == "__main__":
    main()
