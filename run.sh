#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Step 1: Download data ==="
python3 -m data.download --out data/raw

echo "=== Step 2: Deduplicate ==="
python3 -m data.dedup --inp data/raw --out data/clean.jsonl

echo "=== Step 3: Train tokenizer ==="
python3 -m tokenizer.train --data data/clean.jsonl

echo "=== Step 4: Train model ==="
python3 -m training.train --data data/clean.jsonl --tok tokenizer/tokenizer.model

echo "=== Step 5: Evaluate ==="
python3 -m eval.evaluate --ckpt checkpoints/best.pt --tok tokenizer/tokenizer.model --data data/clean.jsonl

echo "=== Done ==="
