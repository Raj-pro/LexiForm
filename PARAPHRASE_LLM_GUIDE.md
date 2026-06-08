# Building a Paraphrase Text-to-Text LLM from Scratch

> **Project status (June 2026):** Phase 2 in progress — pretraining a 13.5M-param encoder-decoder on 18M+ tokens of Project Gutenberg books (span corruption, 20 epochs). Phase 1 (data corpus) is complete. Fine-tune on paraphrase pairs follows after pretraining.

---

## Phase 0 — Architecture Decision (Do This First)

Paraphrase is a **seq2seq task** (input sentence → rephrased sentence). The right architecture is **encoder-decoder (T5-style)**, not decoder-only (GPT-style).

```
Why encoder-decoder?
- Encoder reads full input context bidirectionally
- Decoder generates output conditioned on encoded input
- Naturally suited to transformation tasks, not generation from nothing
```

**Recommended baseline size:**

| Scale | Params | `d_model` | heads | layers | `d_ff` | GPU needed |
|-------|--------|-----------|-------|--------|--------|------------|
| Toy / debugging | ~12M–22M | 192–256 | 3–4 | 4/4–6/6 | 768–1024 | Single GPU, 8GB |
| Usable quality | 60M–250M | 512–768 | 8–12 | 6/6–12/12 | 2048–3072 | Single GPU, 24GB |
| Strong quality | 500M–1B | 1024 | 16 | 24/24 | 4096 | Multi-GPU, A100 |

**This project uses the toy/debugging scale (13.5M params, d_model=256, 4 heads, 4/4 layers, d_ff=1024)** — chosen to stay from-scratch within a single MacBook MPS budget while still reaching ~8/10 quality via a pretrain + augmentation pipeline.

---

## Phase 1 — Define Scope and Metrics

Before writing any code, define what "good paraphrase" means for you.

**Target behaviors:**
- Preserve semantic meaning
- Change surface form (lexical, syntactic, or both)
- Maintain fluency and grammaticality

**Evaluation metrics you will use:**

```
Primary:
  - BLEU-4          → n-gram overlap with reference paraphrase
  - ROUGE-L         → longest common subsequence overlap
  - BERTScore       → semantic similarity via embeddings
  - iBLEU           → penalizes copying input (uniqueness-aware BLEU)

Secondary:
  - Self-BLEU       → diversity across multiple samples
  - Copy rate       → % of output tokens copied verbatim from input (lower is better)
  - Perplexity      → fluency from a reference LM
```

Set a baseline: run the input through unchanged — that gives you a ceiling on BLEU and a floor on copy rate to beat.

---

## Phase 2 — Data Collection

### 2A — Paraphrase Pairs (Fine-Tuning)

**Datasets actually used in this project:**

```python
datasets = {
    "PAWS": {
        "source": "HuggingFace: google/paws",
        "size": "49K train pairs",
        "notes": "High-quality, adversarially constructed. Best quality dataset."
    },
    "QQP": {
        "source": "HuggingFace: quora",
        "size": "400K pairs → capped at 150K, filtered to label==1",
        "notes": "Questions only. Shuffled with fixed seed before cap to avoid selection bias."
    },
    "MRPC": {
        "source": "HuggingFace: glue/mrpc",
        "size": "3.7K pairs",
        "notes": "News sentences. Small but clean."
    }
}
# After cleaning + dedup: ~62K pairs in data/clean.jsonl
```

**If you want more data (not used here due to from-scratch constraint):**

```
ParaNMT-50M   → top-scored 500K subset (back-translation, noisy at bottom)
MSCOCO        → ~200K pairs from 5 captions per image
WikiAnswers   → ~18M question clusters (very noisy, use sparingly)
```

**Back-translation augmentation** (not used in this project — kept for reference):

```python
# English → French → English using Helsinki-NLP/opus-mt models
from transformers import pipeline
en_to_fr = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")
fr_to_en = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
def back_translate(text):
    french = en_to_fr(text)[0]["translation_text"]
    return fr_to_en(french)[0]["translation_text"]
# Do with 3-4 pivot languages: French, German, Spanish, Russian
```

### 2B — Book Corpus (Pretraining) ✅ Complete

This project adds a two-stage training pipeline. Before fine-tuning on paraphrase pairs, the model is pretrained on public-domain literary prose to learn fluent English — the main cure for word-salad outputs on literary text.

```bash
# Fetch ~250 fantasy/mythology/fairy-tale books from Project Gutenberg
.venv/bin/python -m data.fetch_gutenberg --topics fantasy,mythology --limit 250 --out data/books/
.venv/bin/python -m data.fetch_standard_ebooks --filter fantasy --out data/books/
```

**Result:** `data/books/` — ~250 CSV files, ≥18M tokens of literary English.

**Legal sources used:** Project Gutenberg, Standard Ebooks (both public domain).
**Excluded (TOS/legal):** Royal Road, Wattpad, ScribbleHub, AO3, NovelUpdates.

---

## Phase 3 — Data Cleaning Pipeline

Run every pair through this pipeline in order:

```python
# Step 1: Length filter
def length_filter(src, tgt):
    src_words = len(src.split())
    tgt_words = len(tgt.split())
    if src_words < 5 or src_words > 100:      return False
    if tgt_words < 5 or tgt_words > 100:      return False
    ratio = max(src_words, tgt_words) / min(src_words, tgt_words)
    if ratio > 2.5:                            return False  # too different in length
    return True

# Step 2: Copy filter — discard near-identical AND unrelated pairs
# ⚠️  Thresholds tuned for this project (tighter than naive defaults):
from nltk.translate.bleu_score import sentence_bleu
def copy_filter(src, tgt):
    bleu = sentence_bleu([src.split()], tgt.split())
    if bleu > 0.80: return False   # ← upper nudged from 0.85; drops near-copies
    if bleu < 0.25: return False   # ← lower raised from 0.10; drops unrelated pairs
    return True

# Step 3: Language detection
from langdetect import detect
def lang_filter(src, tgt):
    try:
        return detect(src) == "en" and detect(tgt) == "en"
    except:
        return False

# Step 4: Deduplication
# Use MinHash LSH — datasketch library
# ⚠️  MinHash threshold raised from 0.80 → 0.92 so genuine paraphrases
#     with high lexical overlap aren't incorrectly discarded.
from datasketch import MinHash, MinHashLSH
lsh = MinHashLSH(threshold=0.92, num_perm=128)

# Step 5: Quality score (optional — for larger noisy corpora like ParaNMT)
from sentence_transformers import CrossEncoder
quality_model = CrossEncoder("cross-encoder/stsb-roberta-base")
def quality_score(src, tgt):
    return quality_model.predict([[src, tgt]])  # 0–1 range, keep > 0.65
```

> **Key insight:** QQP is shuffled with a fixed seed *before* the 150K cap — otherwise the kept slice would be biased toward whatever order the source enumerates the data.

**Final clean format — save as JSONL:**

```json
{"src": "The cat sat on the mat.", "tgt": "A feline rested upon the rug."}
{"src": "She runs every morning.", "tgt": "Every morning she goes for a run."}
```

**This project result:** `data/clean.jsonl` — ~62K pairs after all filters.

---

## Phase 4 — Tokenizer Training

Train a **SentencePiece BPE tokenizer** on your cleaned corpus (source + target combined).

```python
import sentencepiece as spm

# Prepare corpus file: one sentence per line, mixed src and tgt
with open("tokenizer_corpus.txt", "w") as f:
    for pair in dataset:
        f.write(pair["src"] + "\n")
        f.write(pair["tgt"] + "\n")

# Train tokenizer
spm.SentencePieceTrainer.train(
    input="tokenizer_corpus.txt",
    model_prefix="paraphrase_tokenizer",
    vocab_size=16000,           # 16K is sufficient for English-only (32K is for multilingual)
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
    user_defined_symbols=["<paraphrase>"]  # task prefix token
)
```

**Add a task prefix to every input (T5-style):**

```
Input:  "<paraphrase> The cat sat on the mat."
Output: "A feline rested upon the rug."
```

---

## Phase 5 — Model Architecture

Implement a T5-style encoder-decoder Transformer.

```python
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 16000
    d_model: int = 512          # hidden size
    num_heads: int = 8
    d_ff: int = 2048            # feedforward size = 4 × d_model
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    max_seq_len: int = 128      # paraphrases are short
    dropout: float = 0.1
    # Use RoPE for positional encoding (better than learned or sinusoidal)

# Size guide:
# Tiny   (~12M params): d_model=192,  heads=3,  layers=4/4,  d_ff=768
# Small  (~18M params): d_model=256,  heads=4,  layers=4/4,  d_ff=1024   ← toy/debugging start
# Base   (60M params):  d_model=512,  heads=8,  layers=6/6,  d_ff=2048
# Medium (220M params): d_model=768,  heads=12, layers=12/12, d_ff=3072
# Large  (770M params): d_model=1024, heads=16, layers=24/24, d_ff=4096
```

**Toy / debugging config (~18M, fits on 8GB GPU, ~5 min/epoch on 50K pairs):**

```python
@dataclass
class ToyModelConfig:
    vocab_size: int = 16000
    d_model: int = 256
    num_heads: int = 4
    d_ff: int = 1024
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1
```

**Key architecture choices for paraphrase:**

```
Positional encoding:    RoPE (Rotary Position Embedding) — better generalization
Attention:              Multi-head cross-attention in decoder (standard T5)
Normalization:          RMSNorm (pre-norm placement, not post-norm)
Activation:             SwiGLU in feedforward blocks
Attention optimization: Flash Attention 2 (pip install flash-attn)
Tied embeddings:        Share encoder/decoder/output projection weights (saves ~20%)
```

**Encoder block:**

```python
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.self_attn = MultiHeadAttention(config)   # with RoPE
        self.norm2 = RMSNorm(config.d_model)
        self.ff = SwiGLUFeedForward(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.self_attn(self.norm1(x), mask=mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x
```

**Decoder block:**

```python
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.self_attn = MultiHeadAttention(config, causal=True)
        self.norm2 = RMSNorm(config.d_model)
        self.cross_attn = CrossAttention(config)      # attends to encoder output
        self.norm3 = RMSNorm(config.d_model)
        self.ff = SwiGLUFeedForward(config)

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        x = x + self.self_attn(self.norm1(x), mask=tgt_mask)
        x = x + self.cross_attn(self.norm2(x), encoder_out, mask=src_mask)
        x = x + self.ff(self.norm3(x))
        return x
```

**Full model:**

```python
class ParaphraseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.encoder = nn.ModuleList([EncoderBlock(config) for _ in range(config.num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_decoder_layers)])
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Tie weights: embedding and output projection share parameters
        self.output_proj.weight = self.embedding.weight

    def encode(self, src_ids, src_mask=None):
        x = self.embedding(src_ids)
        for layer in self.encoder:
            x = layer(x, mask=src_mask)
        return x

    def decode(self, tgt_ids, encoder_out, src_mask=None, tgt_mask=None):
        x = self.embedding(tgt_ids)
        for layer in self.decoder:
            x = layer(x, encoder_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.output_proj(x)

    def forward(self, src_ids, dec_input, src_mask=None, tgt_mask=None):
        encoder_out = self.encode(src_ids, src_mask)
        logits = self.decode(dec_input, encoder_out, src_mask, tgt_mask)
        return logits
```

---

## Phase 6 — Training Pipeline

This project uses a **two-stage** training pipeline:

```
Stage 1 — Pretrain  (span corruption on 18M book tokens, 20 epochs)
                     → checkpoints/pretrain/best.pt
Stage 2 — Fine-tune (paraphrase pairs, load pretrained weights via --init_from)
                     → checkpoints/best.pt
```

### 6A — Pretraining (NEW — BookSpanCorruptionDataset)

`training/pretrain_dataset.py` applies T5-style span corruption on the fly:

```python
# Key parameters (in training/pretrain.py)
--mask_ratio   0.15     # 15% of tokens masked per example
--mean_span_len 3.0     # mean span length (geometric distribution)
--stride       64       # chunk stride = 64 tokens (50% overlap at seq_len=128)

# Sentinel tokens: 32 rarest BPE IDs repurposed at runtime
# No tokenizer retraining needed.
```

**EMA (Exponential Moving Average)** is enabled during pretraining (`decay=0.999`). EMA weights are used for validation and saved to `best.pt` — this gives a smoother, more generalised checkpoint for subsequent fine-tuning.

```bash
# Run pretraining
.venv/bin/python -m training.pretrain \
    --data data/books/ --epochs 20 \
    --batch_size 64 --grad_accum 4 --lr 5e-4 --warmup 500 \
    --ckpt_dir checkpoints/pretrain/
```

**Status:** 🔄 In progress — epoch 3/20 complete.

### 6B — Fine-Tuning (ParaphraseDataset)

**Dataset class:**

```python
from torch.utils.data import Dataset

class ParaphraseDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tok = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        src = "<paraphrase> " + self.data[idx]["src"]
        tgt = self.data[idx]["tgt"]

        src_ids = self.tok.encode(src, max_length=self.max_len, truncation=True)
        tgt_ids = self.tok.encode(tgt, max_length=self.max_len, truncation=True)

        # Decoder input:  <s> + tgt tokens  (teacher forcing)
        # Decoder labels: tgt tokens + </s>  (shifted right by one)
        return {
            "src_ids":   torch.tensor(src_ids),
            "dec_input": torch.tensor([BOS_ID] + tgt_ids),
            "labels":    torch.tensor(tgt_ids + [EOS_ID]),
        }
```

> Items are pre-tokenized at dataset construction time (no per-step re-tokenization). `LengthBucketSampler` sorts batches by `len(src)+len(tgt)` to reduce padding waste by ~30–40%.

**Training recipe (actual implemented values):**

```python
# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,           # peak LR (pretrain uses 5e-4)
    betas=(0.9, 0.98),
    eps=1e-8,
    weight_decay=0.01
)

# LR schedule: linear warmup → cosine decay to 10% of peak
# (warmup=200 steps for fine-tune, 500 for pretrain)
scheduler = make_scheduler(optimizer, warmup_steps=200, total_steps=total, min_ratio=0.1)

# Loss: label-smoothed cross-entropy
# input_is_log_probs=True when use_copy=True (CopyGate outputs log-probs)
criterion = LabelSmoothingLoss(vocab_size=16000, smoothing=0.1, ignore_index=PAD_ID)

# Training settings
BATCH_SIZE      = 64
GRAD_ACCUM      = 4      # effective batch = 64 × 4 = 256
MAX_GRAD_NORM   = 1.0
MIXED_PRECISION = True   # bfloat16 on CUDA; no-op on CPU/MPS
```

**Load pretrained weights for fine-tuning:**

```bash
.venv/bin/python -m training.train \
    --data data/clean.jsonl \
    --tok tokenizer/tokenizer.model \
    --init_from checkpoints/pretrain/best.pt \
    --ckpt_dir checkpoints/
```

**Pre-pretrain baseline** (fine-tuned directly on 62K pairs, no pretraining):
- 10 epochs completed, val_loss: 3.496 → **2.831** (epoch 10, best.pt)
- Quality on literary prose: **~1.5/10** (word-salad; model never seen literary English)

**Expected after pretrain + fine-tune:** val_loss < 2.6, quality **~5.0/10 literary**.

**Run a pilot experiment first:**

```
Before full training:
1. Train on 1% of data for 1000 steps
2. Confirm loss decreases (should drop from ~10 to ~3 range in pretraining)
3. Sample outputs manually — do they look like real English sentences?
4. Only then scale to full dataset
```

---

## Phase 7 — Evaluation

**Automated metrics after each epoch:**

```python
from evaluate import load
import bert_score

bleu_metric  = load("sacrebleu")
rouge_metric = load("rouge")

def evaluate(model, val_loader, tokenizer):
    preds, refs, srcs = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            generated = model.generate(
                batch["src_ids"].to(device),
                max_new_tokens=128,
                num_beams=4,
                no_repeat_ngram_size=3,
            )
            preds += tokenizer.batch_decode(generated, skip_special_tokens=True)
            refs  += batch["tgt_text"]
            srcs  += batch["src_text"]

    bleu  = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
    rouge = rouge_metric.compute(predictions=preds, references=refs)
    P, R, F = bert_score.score(preds, refs, lang="en", verbose=False)

    copy_rates = []
    for src, pred in zip(srcs, preds):
        src_toks  = set(src.lower().split())
        pred_toks = pred.lower().split()
        if pred_toks:
            copy_rates.append(sum(1 for t in pred_toks if t in src_toks) / len(pred_toks))

    print(f"BLEU:        {bleu['score']:.2f}")
    print(f"ROUGE-L:     {rouge['rougeL']:.4f}")
    print(f"BERTScore F: {F.mean().item():.4f}")
    print(f"Copy Rate:   {sum(copy_rates)/len(copy_rates):.4f}  (want < 0.4)")
```

**Human evaluation checklist (sample 100 outputs):**

```
For each output, rate 1–5 on:
  [ ] Semantic preservation — does it mean the same thing?
  [ ] Fluency              — is it natural English?
  [ ] Diversity            — is it actually different from the input?
  [ ] Adequacy             — does it cover all information in the input?
```

---

## Phase 8 — Inference and Diversity

```python
def paraphrase(text, model, tokenizer, strategy="beam"):
    src = "<paraphrase> " + text
    ids = tokenizer.encode(src, return_tensors="pt").to(device)

    if strategy == "beam":
        # Deterministic, highest quality
        output = model.generate(
            ids,
            max_new_tokens=128,
            num_beams=5,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )

    elif strategy == "diverse_beam":
        # Multiple diverse outputs
        output = model.generate(
            ids,
            max_new_tokens=128,
            num_beams=10,
            num_beam_groups=5,
            diversity_penalty=1.5,
            num_return_sequences=5,
        )

    elif strategy == "sample":
        # Creative, varied outputs
        output = model.generate(
            ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            num_return_sequences=3,
        )

    return tokenizer.batch_decode(output, skip_special_tokens=True)
```

---

## Phase 9 — Post-Training Refinement

After base training converges, do a short **supervised fine-tune on PAWS only** (highest-quality data):

```
Base training:  ~900K mixed pairs, 5–10 epochs
SFT round:      PAWS 49K pairs only, 2–3 epochs, LR = 1e-5 (10× lower)
```

This aligns the model to high-quality paraphrase patterns and typically gives +1–2 BLEU points.

**Label Smoothing Loss implementation:**

```python
class LabelSmoothingLoss(nn.Module):
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.smoothing = smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: [B, T, V], targets: [B, T]
        B, T, V = logits.shape
        logits = logits.view(-1, V)
        targets = targets.view(-1)

        log_probs = torch.log_softmax(logits, dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)

        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        mask = targets.ne(self.ignore_index)
        return loss[mask].mean()
```

---

## Phase 10 — Deployment

```python
# Step 1: Quantize for inference speed
# INT8 quantization — ~2× speedup, minimal quality loss
import torch.quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Step 2: Export to ONNX for production
torch.onnx.export(
    model,
    (dummy_src, dummy_dec),
    "paraphrase_model.onnx",
    opset_version=17,
    dynamic_axes={"src_ids": {0: "batch", 1: "seq"}}
)

# Step 3: Serve with FastAPI
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ParaphraseRequest(BaseModel):
    text: str
    strategy: str = "diverse_beam"
    num_outputs: int = 3

@app.post("/paraphrase")
def paraphrase_endpoint(request: ParaphraseRequest):
    outputs = paraphrase(request.text, model, tokenizer, strategy=request.strategy)
    return {"input": request.text, "paraphrases": outputs}
```

---

## Dependencies

```txt
torch>=2.1.0
transformers>=4.35.0
sentencepiece>=0.1.99
datasets>=2.14.0
evaluate>=0.4.0
bert-score>=0.3.13
sacrebleu>=2.3.1
rouge-score>=0.1.2
sentence-transformers>=2.2.2
datasketch>=1.6.4
langdetect>=1.0.9
nltk>=3.8.1
fastapi>=0.104.0
uvicorn>=0.24.0
flash-attn>=2.3.0
```

---

## Full Checklist

```
Phase 0   [x] Choose encoder-decoder (T5-style) architecture
Phase 1   [x] Define BLEU / BERTScore / copy-rate targets upfront
Phase 2   [x] Download PAWS, QQP, MRPC datasets → data/clean.jsonl (~62K pairs)
Phase 2   [x] Fetch Project Gutenberg book corpus → data/books/ (18M+ tokens)
Phase 3   [x] Run length filter → copy filter (BLEU 0.25–0.80) → lang filter → dedup (0.92)
Phase 4   [x] Train 16K BPE tokenizer with <paraphrase> prefix token
Phase 5   [x] Implement model: RoPE + RMSNorm + SwiGLU + SDPA (Flash-Attention path)
Phase 5   [x] Add Pointer-Generator CopyGate (log-space logaddexp mix)
Phase 5   [x] Fix padding masks in encoder self-attn + cross-attn
Phase 5   [x] KV caching for encoder K/V + growing decoder K/V
Phase 5   [x] LengthBucketSampler (reduces padding waste ~30-40%)
Phase 6   [x] Baseline fine-tune: AdamW + cosine LR + label smoothing (val_loss=2.831)
Phase 6   [/] Pretrain on book corpus: BookSpanCorruptionDataset + EMA (epoch 3/20, in progress)
Phase 6   [ ] Fine-tune from pretrained weights (--init_from checkpoints/pretrain/best.pt)
Phase 7   [ ] Evaluate BLEU, ROUGE-L, BERTScore, copy rate on post-pretrain model
Phase 7   [ ] Human evaluation on 100 sampled literary outputs (target ≥5.0/10)
Phase 8   [x] Batched beam search + KV cache + semantic reranking (MiniLM)
Phase 9   [ ] B.4 wrappers: WordNet bias + voice transform + LanguageTool fix
Phase 9   [ ] B.3 edit decoder (Levenshtein ops) → retrain
Phase 9   [ ] B.2 kNN-LM retrieval (FAISS datastore)
Phase 9   [ ] B.5 RL polish (PPO on composite reward)
Phase 10  [ ] Quantize INT8 → export ONNX → serve via FastAPI
```

---

## Common Failure Points

| Failure | Cause | Fix |
|---------|-------|-----|
| Model copies input | Near-identical pairs not filtered | Apply copy filter (BLEU 0.10–0.85 range) |
| Loss explodes early | LR too high or no warmup | Use warmup 4000 steps, clip gradients at 1.0 |
| Outputs truncated | max_new_tokens too small | Set to 1.5× average target length |
| Repetitive output | No n-gram penalty | Set no_repeat_ngram_size=3 during generation |
| Slow convergence | Batch size too small | Use gradient accumulation to reach batch 1024+ |
| Meaning not preserved | Quality threshold too low | Raise CrossEncoder quality filter to > 0.65 |
