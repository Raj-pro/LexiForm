"""
Train the paraphrase model.
Run: python3 -m training.train --data data/clean.jsonl --tok tokenizer/tokenizer.model
"""
import argparse
import math
import os
import random
from pathlib import Path
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, random_split

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from model.config import ModelConfig
from model.model import ParaphraseModel
from training.dataset import ParaphraseDataset, collate_fn, LengthBucketSampler
from training.ema import EMA
from training.loss import LabelSmoothingLoss
from tokenizer.tokenizer import Tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_scheduler(optimizer, warmup_steps: int, total_steps: int, min_ratio: float = 0.01):
    """Linear warmup then cosine decay to `min_ratio` × peak LR.
    Floor lowered from 0.1 to 0.01 — the dataset is small enough that
    late-stage refinement benefits more from a lower LR than from preserving
    headroom against overshoot.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(min_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def subset_lengths(subset: Subset, all_lengths: list[int]) -> list[int]:
    return [all_lengths[i] for i in subset.indices]


def train(args):
    set_seed(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}  seed: {args.seed}")

    use_wandb = _WANDB_AVAILABLE and bool(args.wandb_project)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or None,
            config=vars(args),
        )
        print(f"W&B run: {wandb.run.url}")
    elif args.wandb_project:
        print("Warning: wandb not installed — run `pip install wandb` to enable tracking.")

    tok = Tokenizer(args.tok)

    config = ModelConfig(
        vocab_size=tok.vocab_size,
        pad_id=tok.pad_id,
        bos_id=tok.bos_id,
        eos_id=tok.eos_id,
    )

    dataset   = ParaphraseDataset(args.data, tok, max_len=config.max_seq_len)
    val_size  = max(2000, int(0.05 * len(dataset)))
    train_size = len(dataset) - val_size
    split_gen  = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=split_gen)
    print(f"Train: {train_size:,}  Val: {val_size:,}")

    train_collate = partial(
        collate_fn, pad_id=tok.pad_id,
        noise_prob=args.encoder_noise, noise_mask_id=tok.unk_id,
    )
    val_collate = partial(collate_fn, pad_id=tok.pad_id)  # noise off for val

    train_sampler = LengthBucketSampler(
        lengths=subset_lengths(train_ds, dataset.lengths),
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    val_sampler = LengthBucketSampler(
        lengths=subset_lengths(val_ds, dataset.lengths),
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        collate_fn=train_collate, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler,
        collate_fn=val_collate, num_workers=2,
    )

    model = ParaphraseModel(config).to(device)
    print(f"Parameters: {model.param_count():,}")

    if args.init_from:
        # Load matching parameter tensors from a pretrained checkpoint
        # (e.g. checkpoints/pretrain/best.pt). Optimizer / EMA / scheduler are
        # intentionally NOT loaded — fine-tune starts a fresh optimizer state.
        ckpt = torch.load(args.init_from, map_location=device, weights_only=True)
        src_state = ckpt.get("model_state", ckpt)
        cur_state = model.state_dict()
        loaded, skipped_shape, skipped_missing = 0, [], []
        for k, v in src_state.items():
            if k not in cur_state:
                skipped_missing.append(k)
                continue
            if cur_state[k].shape != v.shape:
                skipped_shape.append(f"{k} {tuple(v.shape)}→{tuple(cur_state[k].shape)}")
                continue
            cur_state[k] = v
            loaded += 1
        model.load_state_dict(cur_state)
        print(f"Initialized from {args.init_from}: loaded {loaded} tensors"
              + (f", skipped {len(skipped_shape)} shape mismatches" if skipped_shape else "")
              + (f", skipped {len(skipped_missing)} missing keys"   if skipped_missing else ""))
        if skipped_shape[:3]:
            for s in skipped_shape[:3]:
                print(f"  shape mismatch: {s}")

    # AdamW param groups: weight decay applies to matmul weights only.
    # Embeddings (tied with the output projection — decaying pulls logits
    # toward zero), RMSNorm scales, and biases are kept decay-free, matching
    # T5 / LLaMA-style conventions.
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
        [
            {"params": decay_params,    "weight_decay": 0.01},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr, betas=(0.9, 0.98), eps=1e-8,
    )
    print(f"AdamW groups: decay={sum(p.numel() for p in decay_params):,} "
          f"no_decay={sum(p.numel() for p in no_decay_params):,}")
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps     = steps_per_epoch * args.epochs
    warmup_steps    = min(args.warmup, steps_per_epoch // 2)
    print(f"Optimizer steps/epoch: {steps_per_epoch}  warmup: {warmup_steps}  total: {total_steps}")
    scheduler = make_scheduler(optimizer, warmup_steps, total_steps)
    criterion = LabelSmoothingLoss(config.vocab_size, smoothing=args.label_smoothing, ignore_index=config.pad_id)
    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"EMA enabled (decay={args.ema_decay})")

    # When copy is enabled, decode() returns log-probs; otherwise raw logits.
    loss_in_log_probs = config.use_copy

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    step        = 0
    best_val    = float("inf")
    no_improve  = 0  # epochs without val_loss improvement (for early stopping)

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        # 0-d tensor accumulator avoids a GPU->CPU sync every batch.
        epoch_loss_t = torch.zeros((), device=device)

        for batch in train_loader:
            src_ids   = batch["src_ids"].to(device,   non_blocking=True)
            dec_input = batch["dec_input"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device,    non_blocking=True)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
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
                batch_loss = loss.item() * args.grad_accum
                # Single sync every 100 steps is tolerable for logging.
                print(f"epoch={epoch} step={step} loss={batch_loss:.4f} lr={lr:.2e}")
                if use_wandb:
                    wandb.log({"train/loss": batch_loss, "train/lr": lr, "step": step})

        # Discard residual gradients from a partial final accumulation window
        # so they don't bleed into the next epoch's first optimizer step.
        optimizer.zero_grad(set_to_none=True)

        avg_loss = (epoch_loss_t / max(len(train_loader), 1)).item()

        # Validation runs on EMA weights when EMA is on (they're what we save
        # as the "deployable" model and what inference loads).
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
        if use_wandb:
            wandb.log({"epoch/train_loss": avg_loss, "epoch/val_loss": val_loss, "epoch": epoch})

        # Save: model_state is EMA weights (deployable); raw_model_state is
        # the live training weights, kept so training can be resumed exactly.
        ckpt = {
            "epoch": epoch,
            "step": step,
            "model_state": model.state_dict(),  # currently EMA-swapped if ema is not None
            "optimizer_state": optimizer.state_dict(),
            "config": config.__dict__,
            "val_loss": val_loss,
        }
        if ema is not None:
            # Temporarily restore to capture raw weights too, then re-apply EMA
            # so the saved model_state remains the EMA snapshot.
            ema.restore(model)
            ckpt["raw_model_state"] = model.state_dict()
            ckpt["ema_state"] = ema.state_dict()
            ema.apply_to(model)

        torch.save(ckpt, ckpt_dir / f"checkpoint_epoch{epoch}.pt")

        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  Best model saved (val_loss={best_val:.4f})")
            if use_wandb:
                wandb.summary["best_val_loss"] = best_val
                wandb.summary["best_epoch"]    = epoch
        else:
            no_improve += 1
            print(f"  No improvement for {no_improve} epoch(s)"
                  + (f" (patience={args.patience})" if args.patience > 0 else ""))

        # Restore live weights for the next training epoch.
        if ema is not None:
            ema.restore(model)

        if args.patience > 0 and no_improve >= args.patience:
            print(f"Early stopping: val_loss has not improved for {args.patience} epochs.")
            break

    if use_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="data/clean.jsonl", type=Path)
    parser.add_argument("--tok",        default="tokenizer/tokenizer.model")
    parser.add_argument("--ckpt_dir",   default="checkpoints")
    parser.add_argument("--epochs",     default=20,   type=int)
    parser.add_argument("--batch_size", default=64,   type=int)
    parser.add_argument("--grad_accum", default=4,    type=int)
    parser.add_argument("--warmup",     default=200,  type=int)
    parser.add_argument("--lr",         default=3e-4, type=float)
    parser.add_argument("--seed",       default=42,   type=int)
    parser.add_argument("--ema_decay",  default=0.999, type=float,
                        help="EMA decay; 0 disables. Saved best.pt uses EMA weights.")
    parser.add_argument("--encoder_noise", default=0.10, type=float,
                        help="Per-token probability of replacing a non-pad source "
                             "token with <unk> at training time (encoder-side "
                             "regularisation). 0 disables.")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Label smoothing epsilon. Lower (e.g. 0.05) can help "
                             "when fine-tuning from a strong pretrained checkpoint.")
    parser.add_argument("--patience",   default=5,    type=int,
                        help="Early-stopping patience: stop after this many epochs "
                             "with no val_loss improvement. 0 disables.")
    parser.add_argument("--init_from", default="", type=str,
                        help="Path to a pretrained checkpoint (e.g. "
                             "checkpoints/pretrain/best.pt). Loads matching "
                             "parameter tensors; optimizer / EMA / scheduler "
                             "state is NOT carried over.")
    parser.add_argument("--wandb_project", default="", type=str,
                        help="Weights & Biases project name. Omit to disable W&B logging.")
    parser.add_argument("--wandb_run",     default="", type=str,
                        help="W&B run name (optional). Defaults to auto-generated name.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
