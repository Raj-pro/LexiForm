# Architecture

## Project Overview

A T5-style encoder-decoder Transformer (~13.5M parameters) trained from scratch to generate English paraphrases. Given a sentence, it outputs semantically equivalent rewrites with different surface form.

**Current training stage:** Phase 2 — pretraining on book corpus (span corruption objective, 20 epochs, run in progress). Fine-tune on 63K paraphrase pairs follows next.

---

## Project Structure

```
llm/
├── model/
│   ├── config.py       — ModelConfig dataclass (all hyperparameters)
│   ├── attention.py    — MultiHeadAttention (RoPE), CrossAttention
│   ├── blocks.py       — EncoderBlock, DecoderBlock, RMSNorm, SwiGLU
│   └── model.py        — ParaphraseModel: CopyGate, encode, decode, generate
├── tokenizer/
│   ├── train.py        — builds corpus.txt from JSONL, trains SentencePiece .model
│   └── tokenizer.py    — thin wrapper around SentencePieceProcessor
├── data/
│   ├── download.py              — downloads PAWS / MRPC / QQP via HuggingFace datasets
│   ├── clean.py                 — length, BLEU, language filters
│   ├── dedup.py                 — deduplication
│   ├── filter.py                — additional filtering utilities
│   ├── clean.jsonl              — final ~62K source→target paraphrase pairs
│   ├── fetch_gutenberg.py       — [NEW] fetches public-domain books via gutendex.com API
│   ├── fetch_standard_ebooks.py — [NEW] fetches books from Standard Ebooks JSON feed
│   ├── books.py                 — [NEW] loads/chunks CSVs in data/books/ for pretraining
│   └── books/                   — [NEW] ~250 Project Gutenberg CSV files (~18M+ tokens)
├── training/
│   ├── dataset.py               — ParaphraseDataset + LengthBucketSampler + collate_fn
│   ├── loss.py                  — LabelSmoothingLoss
│   ├── ema.py                   — [NEW] Exponential Moving Average weight shadow
│   ├── pretrain_dataset.py      — [NEW] BookSpanCorruptionDataset (T5-style span masking)
│   ├── pretrain.py              — [NEW] pretraining loop (book corpus, span corruption)
│   └── train.py                 — fine-tuning loop (AdamW, scheduler, checkpointing)
├── inference/
│   └── infer.py        — beam search + semantic reranking
├── eval/
│   └── evaluate.py     — evaluation metrics
├── checkpoints/
│   ├── best.pt                  — best fine-tune checkpoint (val_loss=2.83, epoch 10)
│   ├── checkpoint_epoch{1-10}.pt — fine-tune checkpoints (paraphrase pairs)
│   └── pretrain/
│       ├── best.pt              — best pretrain checkpoint so far
│       └── checkpoint_epoch{1-3}.pt — pretrain checkpoints (run in progress)
├── export_onnx.py      — export to ONNX
└── upload_to_hf.py     — push to HuggingFace Hub
```

---

## Model Architecture

### Configuration ([model/config.py](model/config.py))

| Hyperparameter | Value |
|---|---|
| `vocab_size` | 16,000 |
| `d_model` | 256 |
| `num_heads` | 4 |
| `head_dim` | 64 (= d_model / num_heads) |
| `d_ff` | 1,024 |
| `num_encoder_layers` | 4 |
| `num_decoder_layers` | 4 |
| `max_seq_len` | 128 |
| `dropout` | 0.1 |
| `use_copy` | True |

### Tokenizer ([tokenizer/](tokenizer/))

SentencePiece BPE with 16K vocabulary, trained on the cleaned paraphrase corpus. Special token IDs: `<pad>=0`, `<unk>=1`, `<s>=2`, `</s>=3`, plus a task-prefix symbol `<paraphrase>`.

Inputs are prefixed with `<paraphrase>` at both training and inference time (like T5 task prefixes), so the model knows the generation mode.

> **Pretraining note:** The same tokenizer is reused for span-corruption pretraining. The 32 rarest BPE IDs are repurposed as sentinel tokens (`<extra_id_0>` … `<extra_id_31>`) at runtime; no tokenizer retraining is needed.

### Positional Encoding — RoPE ([model/attention.py](model/attention.py):7)

Rotary Position Embeddings are applied directly to Q and K inside every self-attention head, not added to the input embeddings. Frequencies are precomputed once (`precompute_rope`) and cached as non-trainable buffers. This gives relative position awareness without extra parameters.

### Encoder ([model/blocks.py](model/blocks.py):30)

Four identical `EncoderBlock` layers. Each block uses **pre-norm** (RMSNorm before the sublayer, not after):

```
x = x + dropout(SelfAttention(RMSNorm(x)))   # bidirectional, no causal mask
x = x + dropout(SwiGLU_FFN(RMSNorm(x)))
```

Output is passed through a final `RMSNorm` before being handed to the decoder.

### Decoder ([model/blocks.py](model/blocks.py):45)

Four identical `DecoderBlock` layers, each with three sublayers:

```
x = x + dropout(CausalSelfAttention(RMSNorm(x)))     # masked, can't look ahead
x = x + dropout(CrossAttention(RMSNorm(x), enc_out))  # attends to encoder
x = x + dropout(SwiGLU_FFN(RMSNorm(x)))
```

`CrossAttention` returns the full per-head attention tensor `(B, H, T_tgt, T_src)`; head aggregation is done at the model level so the copy mechanism can pick how it consumes them (currently a mean over heads).

Both self-attention paths use `F.scaled_dot_product_attention` (Flash-Attention kernels when available). Cross-attention is computed manually so we can return the softmaxed weights for the copy mechanism.

**Padding masks** are built from `src_ids == pad_id` and applied as an additive `-inf` mask in encoder self-attention and decoder cross-attention. This stops both modules from attending to padded positions — a quiet bug the original code did not guard against.

**KV caching**. Both self- and cross-attention support an optional per-layer cache dict. During incremental decoding the encoder K/V are computed once and cached forever; decoder self-attention K/V grow by one row per step. RoPE is applied with an explicit `start_pos` so positions stay correct in single-step mode.

### Feed-Forward — SwiGLU ([model/blocks.py](model/blocks.py):18)

```
SwiGLU(x) = down(SiLU(gate(x)) * up(x))
```

`gate` and `up` project `d_model → d_ff`; `down` projects back. No bias. This is the same FFN used in LLaMA/PaLM.

### Normalization — RMSNorm ([model/blocks.py](model/blocks.py):7)

Root-mean-square layer norm. No mean centering (simpler than LayerNorm, similar empirical performance):

```
RMSNorm(x) = weight * x / sqrt(mean(x²) + ε)
```

### Pointer-Generator Copy Mechanism ([model/model.py](model/model.py):8)

When `use_copy=True`, the decoder blends two distributions at each output position:

1. **Generate distribution** — softmax over the full vocabulary from the decoder hidden state.
2. **Copy distribution** — attention weights from the last cross-attention layer, scattered back onto vocabulary positions by the source token IDs.

A learned gate `p_gen ∈ (0,1)` controls the mixture, applied **in log space** via `logaddexp` to avoid the numerical instability of `log(p + ε)` on tiny blended probabilities:

```
p_gen     = sigmoid(W · [context ; decoder_state ; decoder_input])
log_final = logaddexp(log(p_gen)   + log_softmax(vocab_logits),
                      log(1-p_gen) + log(copy_probs + ε))
```

`p_gen → 1`: generate freely from vocabulary.  `p_gen → 0`: copy tokens directly from the source.

The CopyGate is now also fed `src_ids` during inference, so the trained gate weights actually influence generation. (Previously, the inference path silently dropped the copy distribution — a train/test mismatch.)

### Weight Tying ([model/model.py](model/model.py):39)

The output projection `(d_model → vocab_size)` shares weights with the input embedding matrix. This halves the parameter cost at that layer and regularises the model.

---

## Data Pipeline

### Paraphrase Pairs (Fine-tuning Data)

#### Sources

| Dataset | Description | Label used |
|---|---|---|
| PAWS | Adversarial paraphrase pairs | `label == 1` |
| MRPC | Microsoft Research Paraphrase Corpus | `label == 1` |
| QQP | Quora Question Pairs (capped 150K) | `label == 1` |

#### Filters ([data/clean.py](data/clean.py))

Three independent filters, all must pass:

1. **Length filter** — each sentence 5–100 words; length ratio ≤ 2.5.
2. **BLEU/copy filter** — sentence BLEU between **0.25** and **0.80**. Lower bound raised from 0.10 (pairs that low are essentially unrelated sentences, not paraphrases) and upper nudged from 0.85 to 0.80 to drop near-copies.
3. **Language filter** — both sentences detected as English via `langdetect`.

QQP is shuffled with a fixed seed before the 150K cap so the kept slice isn't biased toward whatever order the source happened to enumerate. MinHash dedup threshold raised from 0.8 to **0.92** so genuine paraphrases with high lexical overlap are no longer thrown away.

Final cleaned dataset is stored in `data/clean.jsonl` as `{"src": "...", "tgt": "..."}`.

### Book Corpus (Pretraining Data) — [NEW]

#### Fetching ([data/fetch_gutenberg.py](data/fetch_gutenberg.py), [data/fetch_standard_ebooks.py](data/fetch_standard_ebooks.py))

- Queries `gutendex.com` by topic (`fantasy`, `mythology`, `fairy-tale`) and downloads plain-text UTF-8 books.
- Strips `*** START / END OF THE PROJECT GUTENBERG ***` headers/footers.
- Segments by chapter regex `^(?:CHAPTER|Chapter)\s+[IVXLC0-9]+`.
- Emits one CSV per book in `data/books/` with `(no, story)` schema.
- Standard Ebooks feed provides additional curated public-domain texts.

**Status:** ✅ Complete. `data/books/` contains ~250 CSVs; corpus reaches ≥18M tokens.

#### Loading & Chunking ([data/books.py](data/books.py))

Loads all CSVs under `data/books/`, concatenates chapter text, splits into `seq_len=128` chunks with `stride=64`. Returns a flat list of token-ID chunks ready for `BookSpanCorruptionDataset`.

---

## Training

### Two-Stage Training Plan

```
Stage 1 — Pretrain (span corruption):
    data/books/  →  training/pretrain_dataset.py  →  training/pretrain.py
    Saves: checkpoints/pretrain/best.pt

Stage 2 — Fine-tune (paraphrase pairs):
    data/clean.jsonl  →  training/dataset.py  →  training/train.py --init_from checkpoints/pretrain/best.pt
    Saves: checkpoints/best.pt
```

### Pretraining Dataset ([training/pretrain_dataset.py](training/pretrain_dataset.py)) — [NEW]

`BookSpanCorruptionDataset` wraps the chunked book text and applies **T5-style span corruption** on the fly per epoch (so masks differ across epochs):

| Parameter | Value |
|---|---|
| Mask ratio | 0.15 |
| Mean span length | 3 tokens (geometric distribution) |
| Sentinel tokens | up to 32, reusing the 32 rarest BPE IDs |

Emits `{src_ids, dec_input, labels}` in exactly the same shape as `ParaphraseDataset`, so the existing collate function, `LengthBucketSampler`, and training loop ingest it unchanged.

### EMA — Exponential Moving Average ([training/ema.py](training/ema.py)) — [NEW]

A shadow copy of model weights is maintained with `decay=0.999`. During pretraining validation, EMA weights are swapped in for inference (`ema.apply_to(model)`), then restored afterwards (`ema.restore(model)`). Checkpoints store both raw and EMA weights.

### Pretraining Loop ([training/pretrain.py](training/pretrain.py)) — [NEW]

Thin wrapper that points the standard training machinery at `BookSpanCorruptionDataset`:

- Same `LabelSmoothingLoss`, AdamW, cosine LR schedule, gradient accumulation, bfloat16 autocast, and gradient clipping as `training/train.py`.
- Peak LR `5e-4` (higher than fine-tune `3e-4` — pretraining benefits from larger steps).
- Linear warmup (500 steps) → cosine decay to 1% of peak.
- EMA enabled (`decay=0.999`) — EMA weights used for validation and saved to `best.pt`.
- Early stopping: patience=3 epochs without val_loss improvement.
- Saves `checkpoints/pretrain/checkpoint_epochN.pt` + `checkpoints/pretrain/best.pt`.

**Run command:**
```bash
.venv/bin/python -m training.pretrain --data data/books/ --epochs 20 \
    --batch_size 64 --grad_accum 4 --lr 5e-4 --warmup 500 \
    --ckpt_dir checkpoints/pretrain/
```

**Current status:** 🔄 In progress — epoch 3/20 complete (`best.pt` saved at each improvement).

### Fine-tuning Dataset ([training/dataset.py](training/dataset.py))

Each sample is prepared as a seq2seq pair:

```
src_ids   = encode("<paraphrase> " + src_text)
dec_input = [BOS] + encode(tgt_text)      # teacher-forced decoder input
labels    =         encode(tgt_text) + [EOS]   # shifted right by one
```

Items are pre-tokenized once at dataset construction time and held in memory (no per-step retokenization).

**Length bucketing** ([training/dataset.py](training/dataset.py) — `LengthBucketSampler`): instead of plain random shuffling, the sampler shuffles, slices into large buckets, sorts each bucket by `len(src) + len(tgt)`, and yields batches from those sorted chunks. Result: padding within a batch drops dramatically (every batch is near-uniform in length), reclaiming ~30–40% of the FLOPs that previously paid for `<pad>` tokens.

### Loss — Label Smoothing ([training/loss.py](training/loss.py))

Standard NLL over non-pad positions, with label smoothing (ε = 0.1):

```
loss = (1 - ε) * NLL_loss + ε * uniform_smooth_loss
```

The caller passes an explicit `input_is_log_probs` flag (driven by `config.use_copy`). The previous heuristic "is `x.max() <= 0`?" both forced a GPU→CPU sync every step and could misfire on early-training logits.

### Optimizer & Scheduler

| Setting | Pretrain | Fine-tune |
|---|---|---|
| Optimizer | AdamW | AdamW |
| Peak LR | 5e-4 | 3e-4 |
| β | (0.9, 0.98) | (0.9, 0.98) |
| ε | 1e-8 | 1e-8 |
| weight_decay | 0.01 | 0.01 |
| Schedule | linear warmup 500 → cosine to 1% | linear warmup 200 → cosine to 10% |
| Gradient accumulation | 4 steps | 4 steps (effective batch = 256) |
| Gradient clipping | max norm 1.0 | max norm 1.0 |
| Mixed precision | bfloat16 (CUDA only) | bfloat16 (CUDA only) |
| EMA | decay=0.999 | — |

### Checkpointing

A checkpoint is saved after every epoch. `best.pt` is overwritten whenever validation loss improves. Each checkpoint stores model weights, optimizer state, config dict, epoch, step, and val loss, so training can be resumed exactly.

**Pretrain checkpoints** additionally store `ema_state` and `raw_model_state` alongside the EMA-applied `model_state`.

#### Current checkpoint state

| Location | Epochs saved | Best val_loss | Notes |
|---|---|---|---|
| `checkpoints/pretrain/` | 1–3 | improving | Pretraining on books, run in progress |
| `checkpoints/` | 1–10 | 2.831 (epoch 10) | Paraphrase fine-tune baseline (pre-pretrain) |

---

## Inference

### Beam Search ([inference/infer.py](inference/infer.py))

Beam search is fully batched and KV-cached:

- The encoder runs **once** per source; its K/V are cached in every cross-attention layer so each decode step only computes Q.
- Decoder self-attention K/V grow by one row per step (no re-encoding of the prefix). RoPE receives the absolute position via `start_pos`.
- All `num_beams` beams advance in a **single** `decode_step` call per timestep — one batched decoder forward instead of `num_beams` Python-level calls.
- When the top-K reorders beams, the per-layer KV caches are reordered with `index_select(dim=0)` so each beam's cache stays consistent with its history.
- The CopyGate distribution is included at inference (the old path silently dropped `src_ids`, defeating the trained gate).

Decoding constraints:

- **No-repeat n-gram** (n=3): per-beam, tokens that would form an already-seen 3-gram are banned (`-inf` in the logit).
- **Length penalty**: only applied when a beam finishes or at the end; uses the Google NMT formula `((5 + length) / 6) ^ α` (α=1.0), which is better behaved than `length ^ α` for short outputs.

### Semantic Reranking

After beam search, the source and all candidates are encoded in a single batched call to `all-MiniLM-L6-v2`:

```
score = cosine_similarity(source, candidate) - 0.3 * max(0, word_overlap - 0.85)
```

This selects outputs that preserve meaning (high similarity) while actually rephrasing (penalises near-copies with > 85% word overlap). The top `num_outputs` (default 3) are returned. The reranker is loaded lazily and cached across calls.

### Evaluation ([eval/evaluate.py](eval/evaluate.py))

Eval now runs the **same** `beam_search` + `rerank` pipeline as production inference (previously it used greedy `model.generate()`, so reported numbers diverged from what users actually saw). Metrics: BLEU vs. reference, ROUGE-L, BERTScore, copy-rate, and self-BLEU vs. source (so we can spot near-copy regressions).

---

## Upcoming Phases

| Phase | What | Status |
|---|---|---|
| Phase 2 — Pretrain | Span corruption on 18M+ book tokens, 20 epochs | 🔄 Running (epoch 3/20) |
| Phase 2 — Fine-tune | Load pretrained weights → fine-tune on clean.jsonl | ⏳ Waiting on pretrain |
| Phase 3 — B.4 Wrappers | WordNet synonym bias + voice transform + LanguageTool fix | 📋 Planned |
| Phase 4 — B.3 Edit Decoder | Levenshtein edit-op decoder (architectural change) | 📋 Planned |
| Phase 5 — B.2 kNN-LM | FAISS datastore + retrieval-augmented decoding | 📋 Planned |
| Phase 6 — B.5 RL Polish | PPO fine-tune on composite reward | 📋 Planned |

Expected quality after Phase 2: **~5.0/10 literary** (up from current 1.5/10 baseline). Target after all phases: **~8.0/10 literary**.
