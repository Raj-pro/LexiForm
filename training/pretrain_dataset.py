"""
T5-style span-corruption dataset for unsupervised pretraining on book text.

Each example is a (corrupted_input, sentinel_target) pair derived on the fly
from a tokenized chunk:

    ORIGINAL:  [a, b, c, d, e, f, g, h, i, j]
    CORRUPTED: [a, b, <X0>,       e, <X1>,    h, i, j]                 # encoder input
    TARGET:    [<X0>, c, d, <X1>, f, g, <X2>]                          # decoder labels

Sentinel ids occupy ids [vocab_size, vocab_size + num_sentinels) — appended
*after* the BPE vocab so they never collide with real tokens. The model's
embedding and output projection are sized to include them (see
[model/config.py](../model/config.py) num_sentinels).

The dataset emits `{src_ids, dec_input, labels}` tensors in the same shape
the existing paraphrase training loop ([training/train.py](training/train.py),
[training/dataset.py](training/dataset.py)) consumes, so collation, length
bucketing, and the AdamW + cosine + EMA loop are reused unchanged.

A fresh random mask is sampled every `__getitem__`, so each epoch sees a
different corruption of the same chunks (T5 strategy).
"""
from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset

from data.books import build_chunks
from tokenizer.tokenizer import Tokenizer


def sentinel_ids(tokenizer: Tokenizer, num_sentinels: int) -> list[int]:
    """Return the sentinel ids, which sit *after* the regular BPE vocab.

    The tokenizer never emits these ids; the model embedding is sized to
    include them.
    """
    V = tokenizer.vocab_size
    return list(range(V, V + num_sentinels))


def _sample_span_boundaries(
    n_tokens: int,
    mask_ratio: float,
    mean_span_len: float,
    rng: random.Random,
) -> list[tuple[int, int]]:
    """Place non-overlapping spans of geometric length so that *approximately*
    `mask_ratio` fraction of tokens land inside a span. Returns sorted
    [(start, end_exclusive), …].
    """
    if n_tokens < 4 or mask_ratio <= 0:
        return []

    target_masked = max(1, int(round(n_tokens * mask_ratio)))
    spans: list[tuple[int, int]] = []
    masked_so_far = 0
    attempts = 0
    # Geometric: p such that mean = 1/p. Mean span len 3 → p ≈ 1/3.
    p_geom = 1.0 / max(mean_span_len, 1.0)

    while masked_so_far < target_masked and attempts < 200:
        attempts += 1
        length = 1
        # Truncated geometric on [1, 8].
        while length < 8 and rng.random() > p_geom:
            length += 1
        start = rng.randrange(0, max(1, n_tokens - length))
        end = start + length
        # Reject if overlapping an existing span.
        if any(not (end <= s or start >= e) for s, e in spans):
            continue
        spans.append((start, end))
        masked_so_far += length

    spans.sort()
    return spans


def corrupt(
    tokens: list[int],
    sentinels: list[int],
    mask_ratio: float,
    mean_span_len: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Returns (encoder_input_ids, target_ids).

    Encoder input replaces each masked span with a single sentinel. Target is
    the concatenation of (sentinel_i, original_span_i, sentinel_i+1, …),
    ending with a final closing sentinel.
    """
    spans = _sample_span_boundaries(len(tokens), mask_ratio, mean_span_len, rng)
    if not spans:
        return list(tokens), list(tokens)  # degenerate — no masking happened

    # Truncate to fit our sentinel budget.
    spans = spans[: len(sentinels) - 1]

    enc: list[int] = []
    tgt: list[int] = []
    cursor = 0
    for i, (s, e) in enumerate(spans):
        enc.extend(tokens[cursor:s])
        enc.append(sentinels[i])
        tgt.append(sentinels[i])
        tgt.extend(tokens[s:e])
        cursor = e
    enc.extend(tokens[cursor:])
    tgt.append(sentinels[len(spans)])  # closing sentinel

    return enc, tgt


class BookSpanCorruptionDataset(Dataset):
    """Wraps a list of pre-tokenized chunks; samples a fresh span corruption
    each time `__getitem__` is called.

    Length signal for the existing `LengthBucketSampler` is the chunk length
    (constant ≈ seq_len), so bucketing is essentially a no-op here — but the
    sampler is reused for code symmetry with [training/dataset.py](training/dataset.py).
    """
    def __init__(
        self,
        chunks: list[list[int]],
        tokenizer: Tokenizer,
        num_sentinels: int = 32,
        mask_ratio: float = 0.15,
        mean_span_len: float = 3.0,
        seed: int = 0,
    ):
        self.chunks        = chunks
        self.tokenizer     = tokenizer
        self.mask_ratio    = mask_ratio
        self.mean_span_len = mean_span_len
        self.sentinels     = sentinel_ids(tokenizer, num_sentinels)
        self.base_seed     = seed

        # Stored for LengthBucketSampler (combined src+dec length).
        self.lengths = [len(c) for c in chunks]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        # Hash-seeded RNG so each (epoch, idx) is reproducibly random. Python's
        # random.Random doesn't accept tuples directly — hash to an int first.
        rng = random.Random(hash((self.base_seed, idx, torch.initial_seed() & 0xFFFF)))
        chunk = self.chunks[idx]

        enc, tgt = corrupt(
            chunk,
            self.sentinels,
            mask_ratio=self.mask_ratio,
            mean_span_len=self.mean_span_len,
            rng=rng,
        )

        bos, eos = self.tokenizer.bos_id, self.tokenizer.eos_id
        return {
            "src_ids":   torch.tensor(enc,           dtype=torch.long),
            "dec_input": torch.tensor([bos] + tgt,   dtype=torch.long),
            "labels":    torch.tensor(tgt   + [eos], dtype=torch.long),
            "src_text":  "",  # populated only for ParaphraseDataset; kept for collate symmetry
            "tgt_text":  "",
        }


def load_pretrain_dataset(
    book_dir: str,
    tokenizer: Tokenizer,
    num_sentinels: int = 32,
    seq_len: int = 128,
    stride: int = 64,
    mask_ratio: float = 0.15,
    mean_span_len: float = 3.0,
    seed: int = 0,
) -> BookSpanCorruptionDataset:
    chunks = build_chunks(book_dir, tokenizer, seq_len=seq_len, stride=stride)
    if not chunks:
        raise RuntimeError(f"No chunks built from {book_dir!r}. Did the Phase 1 "
                           "fetch produce CSVs there?")
    return BookSpanCorruptionDataset(
        chunks, tokenizer,
        num_sentinels=num_sentinels,
        mask_ratio=mask_ratio,
        mean_span_len=mean_span_len,
        seed=seed,
    )
