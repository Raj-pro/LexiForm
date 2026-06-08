"""
Evaluate a checkpoint on the validation set, using the SAME beam search +
reranker as production inference so reported metrics reflect deployment behavior.

Run: python3 -m eval.evaluate --ckpt checkpoints/best.pt --tok tokenizer/tokenizer.model --data data/clean.jsonl
"""
import argparse
from pathlib import Path

import torch
from torch.utils.data import random_split
from evaluate import load as hf_load
import bert_score

from model.config import ModelConfig
from model.model import ParaphraseModel
from tokenizer.tokenizer import Tokenizer
from training.dataset import ParaphraseDataset
from inference.infer import beam_search, rerank, get_reranker


def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok    = Tokenizer(args.tok)

    ckpt   = torch.load(args.ckpt, map_location=device, weights_only=True)
    config = ModelConfig(**ckpt["config"])
    model  = ParaphraseModel(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset  = ParaphraseDataset(args.data, tok, max_len=config.max_seq_len)
    val_size = max(2000, int(0.05 * len(dataset)))
    split_gen = torch.Generator().manual_seed(args.seed)
    _, val_ds = random_split(dataset, [len(dataset) - val_size, val_size], generator=split_gen)
    print(f"Eval set: {len(val_ds):,} pairs (seed={args.seed})")

    if args.limit:
        val_ds = torch.utils.data.Subset(val_ds, list(range(min(args.limit, len(val_ds)))))
        print(f"Limiting eval to {len(val_ds):,} pairs")

    get_reranker()  # warm

    bleu_metric  = hf_load("sacrebleu")
    rouge_metric = hf_load("rouge")

    preds, refs, srcs = [], [], []
    for i in range(len(val_ds)):
        item = val_ds[i]
        src_text = item["src_text"]
        tgt_text = item["tgt_text"]

        src_ids = torch.tensor(
            [tok.encode("<paraphrase> " + src_text, max_length=config.max_seq_len)],
            dtype=torch.long, device=device,
        )
        candidates = beam_search(model, src_ids, tok, config,
                                 num_beams=args.beams, num_return=10)
        ranked = rerank(src_text, candidates, num_return=1)
        pred   = ranked[0] if ranked else (candidates[0] if candidates else "")

        preds.append(pred)
        refs.append(tgt_text)
        srcs.append(src_text)

        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(val_ds)}")

    bleu  = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
    rouge = rouge_metric.compute(predictions=preds, references=refs)
    _, _, F = bert_score.score(preds, refs, lang="en", verbose=False)

    copy_rates = []
    self_bleu  = []
    for src, pred in zip(srcs, preds):
        src_toks  = set(src.lower().split())
        pred_toks = pred.lower().split()
        if pred_toks:
            copy_rates.append(sum(1 for t in pred_toks if t in src_toks) / len(pred_toks))
        # self-BLEU vs source (high means output is too similar to input)
        self_bleu.append(bleu_metric.compute(predictions=[pred], references=[[src]])["score"])

    print(f"BLEU vs ref:      {bleu['score']:.2f}")
    print(f"ROUGE-L:          {rouge['rougeL']:.4f}")
    print(f"BERTScore F:      {F.mean().item():.4f}")
    print(f"Copy Rate:        {sum(copy_rates)/max(len(copy_rates),1):.4f}   (want < 0.4)")
    print(f"Self-BLEU (src):  {sum(self_bleu)/max(len(self_bleu),1):.2f}   (want lower — paraphrase, not copy)")

    if args.samples:
        print("\n--- Sample outputs ---")
        for i in range(min(args.samples, len(preds))):
            print(f"IN:  {srcs[i]}")
            print(f"OUT: {preds[i]}")
            print(f"REF: {refs[i]}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default="checkpoints/best.pt")
    parser.add_argument("--tok",     default="tokenizer/tokenizer.model")
    parser.add_argument("--data",    default="data/clean.jsonl", type=Path)
    parser.add_argument("--beams",   default=12, type=int)
    parser.add_argument("--samples", default=5,  type=int)
    parser.add_argument("--limit",   default=0,  type=int, help="0 = full val set")
    parser.add_argument("--seed",    default=42, type=int)
    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()
