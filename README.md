# LexiForm

A T5-style encoder-decoder Transformer trained from scratch for English paraphrase generation. Given a sentence, the model outputs semantically equivalent rewrites with varied surface form.

## Model Details

| | |
|---|---|
| Architecture | Encoder-Decoder Transformer |
| Parameters | ~13.5M |
| Vocab size | 16,000 (BPE SentencePiece) |
| d_model | 256 |
| Layers | 4 encoder / 4 decoder |
| Attention heads | 4 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Feed-forward | SwiGLU |
| Copy mechanism | Pointer-Generator gate |
| Fine-tune data | PAWS + MRPC + QQP (~62K pairs) |
| Pretrain data | ~250 Project Gutenberg books (18M+ tokens) |
| Best fine-tune val loss | 2.831 (epoch 10) |

## Project Structure

```
llm/
├── model/          — ModelConfig, MultiHeadAttention (RoPE), Encoder/DecoderBlock, ParaphraseModel
├── tokenizer/      — SentencePiece BPE trainer and wrapper
├── data/           — download, clean, dedup, filter scripts + cleaned JSONL + book CSVs
├── training/       — fine-tune loop, pretrain loop, dataset, loss, EMA
├── inference/      — beam search with KV cache + semantic reranking
├── eval/           — BLEU, ROUGE-L, BERTScore evaluation
├── checkpoints/    — saved model weights (.pt)
├── run.sh          — end-to-end pipeline script
├── export_onnx.py  — export to ONNX
└── upload_to_hf.py — push to HuggingFace Hub
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed breakdown of every module.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch sentencepiece transformers datasets sentence-transformers sacrebleu rouge-score bert-score langdetect wandb
```

## Quickstart

**Run inference on a single sentence:**

```bash
python3 -m inference.infer \
    --ckpt checkpoints/best.pt \
    --tok  tokenizer/tokenizer.model \
    --text "The dog ran quickly across the yard."
```

**Paraphrase a file line-by-line:**

```bash
python3 paraphrase_file.py \
    --ckpt checkpoints/best.pt \
    --tok  tokenizer/tokenizer.model \
    --input sample.txt \
    --output output.txt
```

**Evaluate on the cleaned dataset:**

```bash
python3 -m eval.evaluate \
    --ckpt checkpoints/best.pt \
    --tok  tokenizer/tokenizer.model \
    --data data/clean.jsonl
```

## Training Pipeline

Training is two-stage. Run the full pipeline with:

```bash
bash run.sh
```

Or run each stage manually:

### Stage 1 — Pretrain (span corruption on book corpus)

```bash
python3 -m training.pretrain \
    --data      data/books/ \
    --epochs    20 \
    --batch_size 64 \
    --grad_accum 4 \
    --lr        5e-4 \
    --warmup    500 \
    --ckpt_dir  checkpoints/pretrain/
```

### Stage 2 — Fine-tune (paraphrase pairs)

```bash
python3 -m training.train \
    --data            data/clean_combined.jsonl \
    --tok             tokenizer/tokenizer.model \
    --init_from       checkpoints/pretrain/best.pt \
    --ckpt_dir        checkpoints/finetune/ \
    --epochs          20 \
    --lr              1e-4 \
    --warmup          500 \
    --label_smoothing 0.05 \
    --patience        5 \
    --wandb_project   lexiform \
    --wandb_run       stage2-finetune
```

### Data preparation (if starting from scratch)

```bash
python3 -m data.download   --out data/raw
python3 -m data.dedup      --inp data/raw --out data/clean.jsonl
python3 -m tokenizer.train --data data/clean.jsonl
```

## Limitations

- Small model (13M) — outputs may hallucinate or repeat on complex inputs
- English only
- Best on short sentences (5–30 words)
- Pretraining is still in progress — quality will improve after Phase 2 completes

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 2 — Pretrain | Span corruption on 18M+ book tokens | Running |
| 2 — Fine-tune | Load pretrained weights → fine-tune | Waiting |
| 3 | WordNet synonym bias + voice transform | Planned |
| 4 | Levenshtein edit-op decoder | Planned |
| 5 | FAISS kNN-LM retrieval-augmented decoding | Planned |
| 6 | PPO RL fine-tuning on composite reward | Planned |
