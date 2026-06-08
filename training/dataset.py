import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tokenizer.tokenizer import Tokenizer


class ParaphraseDataset(Dataset):
    """
    Pre-tokenizes all pairs at construction. Exposes `.lengths` so a
    length-bucketed sampler can group similar-length examples together.

    When `bidirectional=True` (default), every (s, t) pair is also added as
    (t, s) — paraphrase is a symmetric relation, so this doubles the effective
    training signal without collecting new data. Exact duplicates are skipped
    via a set on (src, tgt) so symmetric pairs already present in the source
    don't get double-counted.
    """
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Tokenizer,
        max_len: int = 128,
        bidirectional: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.src_ids:   list[list[int]] = []
        self.dec_input: list[list[int]] = []
        self.labels:    list[list[int]] = []
        self.src_text:  list[str] = []
        self.tgt_text:  list[str] = []

        seen: set[tuple[str, str]] = set()
        with open(jsonl_path) as f:
            for line in f:
                row = json.loads(line)
                self._add_pair(row["src"], row["tgt"], seen)
                if bidirectional:
                    self._add_pair(row["tgt"], row["src"], seen)

        # combined length drives bucketing — both src and tgt contribute to per-batch cost
        self.lengths = [len(s) + len(d) for s, d in zip(self.src_ids, self.dec_input)]

    def _add_pair(self, src_text: str, tgt_text: str, seen: set[tuple[str, str]]) -> None:
        key = (src_text, tgt_text)
        if key in seen:
            return
        seen.add(key)
        bos, eos = self.tokenizer.bos_id, self.tokenizer.eos_id
        src    = "<paraphrase> " + src_text
        src_t  = self.tokenizer.encode(src,      max_length=self.max_len)
        tgt_t  = self.tokenizer.encode(tgt_text, max_length=self.max_len)
        self.src_ids.append(src_t)
        self.dec_input.append([bos] + tgt_t)
        self.labels.append(tgt_t + [eos])
        self.src_text.append(src_text)
        self.tgt_text.append(tgt_text)

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "src_ids":   torch.tensor(self.src_ids[idx],   dtype=torch.long),
            "dec_input": torch.tensor(self.dec_input[idx], dtype=torch.long),
            "labels":    torch.tensor(self.labels[idx],    dtype=torch.long),
            "src_text":  self.src_text[idx],
            "tgt_text":  self.tgt_text[idx],
        }


def collate_fn(
    batch: list[dict],
    pad_id: int,
    noise_prob: float = 0.0,
    noise_mask_id: int | None = None,
) -> dict:
    src_ids   = pad_sequence([b["src_ids"]   for b in batch], batch_first=True, padding_value=pad_id)
    dec_input = pad_sequence([b["dec_input"] for b in batch], batch_first=True, padding_value=pad_id)
    labels    = pad_sequence([b["labels"]    for b in batch], batch_first=True, padding_value=pad_id)

    # Encoder-side noise: with prob `noise_prob`, replace a non-pad source
    # token with `noise_mask_id`. The leading task-prefix token (position 0)
    # is preserved so the model still sees the `<paraphrase>` cue. This is a
    # mild regulariser; the model has to recover meaning from partial input,
    # which curbs overfit on a 60K-pair dataset.
    if noise_prob > 0.0 and noise_mask_id is not None:
        mask = torch.rand(src_ids.shape) < noise_prob
        mask[:, 0] = False
        mask = mask & (src_ids != pad_id)
        src_ids = torch.where(mask, torch.full_like(src_ids, noise_mask_id), src_ids)

    return {
        "src_ids":   src_ids,
        "dec_input": dec_input,
        "labels":    labels,
        "src_text":  [b["src_text"] for b in batch],
        "tgt_text":  [b["tgt_text"] for b in batch],
    }


class LengthBucketSampler(torch.utils.data.Sampler):
    """
    Yields lists of indices (batches) where samples within a batch have similar
    combined src+tgt length, dramatically reducing wasted padding.

    Algorithm:
      1. Shuffle the full index list.
      2. Slice into large "buckets" of `batch_size * bucket_multiplier`.
      3. Sort each bucket by length, slice into batch-sized chunks.
      4. Shuffle the order in which batches are yielded.
    """
    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_multiplier: int = 100,
        seed: int = 0,
    ):
        self.lengths           = lengths
        self.batch_size        = batch_size
        self.shuffle           = shuffle
        self.bucket_size       = batch_size * bucket_multiplier
        self.seed              = seed
        self.epoch             = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        n = len(self.lengths)
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        order = torch.randperm(n, generator=g).tolist() if self.shuffle else list(range(n))

        batches: list[list[int]] = []
        for bs in range(0, n, self.bucket_size):
            bucket = order[bs:bs + self.bucket_size]
            bucket.sort(key=lambda i: self.lengths[i])
            for b in range(0, len(bucket), self.batch_size):
                chunk = bucket[b:b + self.batch_size]
                if chunk:
                    batches.append(chunk)

        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in perm]

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size
