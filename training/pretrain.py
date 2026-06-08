"""
Unsupervised pretraining (T5-style span corruption) on book corpus.

Reuses the same encoder–decoder architecture and optimizer / scheduler / EMA /
loss as [training/train.py](training/train.py); only the dataset is swapped
out for [training/pretrain_dataset.py](training/pretrain_dataset.py)'s
`BookSpanCorruptionDataset`.

Run:
    .venv/bin/python -m training.pretrain \\
        --data data/books/ \\
        --tok  tokenizer/tokenizer.model \\
        --ckpt_dir checkpoints/pretrain/ \\
        --epochs 20 --batch_size 64 --grad_accum 4 \\
        --lr 5e-4 --warmup 500

After pretraining, fine-tune on the paraphrase pairs:
    .venv/bin/python -m training.train --init_from checkpoints/pretrain/best.pt
"""
import argparse
import math
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from model.config import ModelConfig
from model.model import ParaphraseModel
from tokenizer.tokenizer import Tokenizer
from training.dataset import collate_fn, LengthBucketSampler
from training.pretrain_dataset import load_pretrain_dataset
from training.ema import EMA
from training.loss import LabelSmoothingLoss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.01):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def subset_lengths(subset, all_lengths: list[int]) -> list[int]:
    return [all_lengths[i] for i in subset.indices]


def pretrain(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}  seed: {args.seed}")

    tok = Tokenizer(args.tok)
    config = ModelConfig(
        vocab_size=tok.vocab_size,
        pad_id=tok.pad_id,
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
    )

    dataset = load_pretrain_dataset(
        args.data, tok,
        num_sentinels=config.num_sentinels,
        seq_len=config.max_seq_len,
        stride=args.stride,
        mask_ratio=args.mask_ratio,
        mean_span_len=args.mean_span_len,
        seed=args.seed,
    )
    val_size  = max(1000, int(0.02 * len(dataset)))
    train_size = len(dataset) - val_size
    split_gen  = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=split_gen)
    print(f"Chunks  total={len(dataset):,}  train={train_size:,}  val={val_size:,}")

    train_collate = partial(collate_fn, pad_id=tok.pad_id)  # no encoder noise — span corruption already corrupts
    val_collate   = partial(collate_fn, pad_id=tok.pad_id)

    train_sampler = LengthBucketSampler(
        lengths=subset_lengths(train_ds, dataset.lengths),
        batch_size=args.batch_size, shuffle=True, seed=args.seed,
    )
    val_sampler = LengthBucketSampler(
        lengths=subset_lengths(val_ds, dataset.lengths),
        batch_size=args.batch_size, shuffle=False, seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              collate_fn=train_collate, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_sampler=val_sampler,
                              collate_fn=val_collate,   num_workers=2)

    model = ParaphraseModel(config).to(device)
    print(f"Parameters: {model.param_count():,}")

    no_decay_names = {"embedding.weight"}
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in no_decay_names or p.ndim <= 1:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,    "weight_decay": 0.01},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.98), eps=1e-8,
    )
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = min(args.warmup, max(1, steps_per_epoch // 2))
    print(f"steps/epoch={steps_per_epoch}  warmup={warmup_steps}  total={total_steps}")
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
    criterion = LabelSmoothingLoss(config.vocab_size, smoothing=0.1, ignore_index=config.pad_id)
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"EMA enabled (decay={args.ema_decay})")

    # During pretraining the model's decode() returns log-probs only when copy
    # is enabled — same as paraphrase training. Pass through unchanged.
    loss_in_log_probs = config.use_copy

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step     = 0
    best_val = float("inf")
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        epoch_loss_t = torch.zeros((), device=device)

        for batch in train_loader:
            src_ids   = batch["src_ids"].to(device,   non_blocking=True)
            dec_input = batch["dec_input"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device,    non_blocking=True)

            autocast_kw = dict(device_type=device.type, dtype=torch.bfloat16,
                               enabled=(device.type == "cuda"))
            with torch.autocast(**autocast_kw):
                scores = model(src_ids, dec_input)
                loss   = criterion(scores, labels, input_is_log_probs=loss_in_log_probs) / args.grad_accum

            loss.backward()
            step += 1

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            epoch_loss_t += loss.detach() * args.grad_accum

            if step % 100 == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"epoch={epoch} step={step} loss={loss.item()*args.grad_accum:.4f} lr={lr:.2e}")

        optimizer.zero_grad(set_to_none=True)
        avg_loss = (epoch_loss_t / max(len(train_loader), 1)).item()

        # Validation on EMA weights if EMA is on.
        model.eval()
        if ema is not None:
            ema.apply_to(model)
        val_loss_t = torch.zeros((), device=device)
        with torch.no_grad():
            for batch in val_loader:
                src_ids   = batch["src_ids"].to(device,   non_blocking=True)
                dec_input = batch["dec_input"].to(device, non_blocking=True)
                labels    = batch["labels"].to(device,    non_blocking=True)
                scores    = model(src_ids, dec_input)
                val_loss_t += criterion(scores, labels, input_is_log_probs=loss_in_log_probs).detach()
        val_loss = (val_loss_t / max(len(val_loader), 1)).item()

        print(f"=== epoch={epoch} train_loss={avg_loss:.4f} val_loss={val_loss:.4f} ===")

        ckpt = {
            "epoch": epoch,
            "step": step,
            "model_state": model.state_dict(),  # EMA-swapped if ema is not None
            "optimizer_state": optimizer.state_dict(),
            "config": config.__dict__,
            "val_loss": val_loss,
            "stage": "pretrain",
        }
        if ema is not None:
            ema.restore(model)
            ckpt["raw_model_state"] = model.state_dict()
            ckpt["ema_state"] = ema.state_dict()
            ema.apply_to(model)

        torch.save(ckpt, ckpt_dir / f"checkpoint_epoch{epoch}.pt")
        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  Best model saved (val_loss={best_val:.4f})")
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  Val loss plateaued for {patience} epochs — early stopping.")
                if ema is not None:
                    ema.restore(model)
                break

        if ema is not None:
            ema.restore(model)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/books/", type=Path)
    p.add_argument("--tok",        default="tokenizer/tokenizer.model")
    p.add_argument("--ckpt_dir",   default="checkpoints/pretrain/", type=Path)
    p.add_argument("--epochs",     default=20, type=int)
    p.add_argument("--batch_size", default=64, type=int)
    p.add_argument("--grad_accum", default=4,  type=int)
    p.add_argument("--warmup",     default=500, type=int)
    p.add_argument("--lr",         default=5e-4, type=float)
    p.add_argument("--seed",       default=42, type=int)
    p.add_argument("--ema_decay",  default=0.999, type=float)
    p.add_argument("--mask_ratio", default=0.15, type=float)
    p.add_argument("--mean_span_len", default=3.0, type=float)
    p.add_argument("--stride",     default=64, type=int,
                   help="Token stride between successive chunks (overlap = seq_len - stride).")
    p.add_argument("--patience",   default=3, type=int,
                   help="Early-stop after this many epochs without val_loss improvement.")
    args = p.parse_args()
    pretrain(args)


if __name__ == "__main__":
    main()
