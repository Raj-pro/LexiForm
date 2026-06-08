from pathlib import Path
import sentencepiece as spm


class Tokenizer:
    def __init__(self, model_path: str | Path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    @property
    def pad_id(self) -> int: return self.sp.pad_id()

    @property
    def bos_id(self) -> int: return self.sp.bos_id()

    @property
    def eos_id(self) -> int: return self.sp.eos_id()

    @property
    def unk_id(self) -> int: return self.sp.unk_id()

    def encode(self, text: str, max_length: int | None = None) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if max_length:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    def batch_decode(self, batch: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        results = []
        for ids in batch:
            if skip_special_tokens:
                ids = [i for i in ids if i not in (self.pad_id, self.bos_id, self.eos_id)]
            results.append(self.decode(ids))
        return results
