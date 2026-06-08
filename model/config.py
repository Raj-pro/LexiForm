from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    d_model: int = 256
    num_heads: int = 4
    d_ff: int = 1024
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    max_seq_len: int = 128
    dropout: float = 0.1

    use_copy: bool = True   # pointer-generator copy mechanism

    # Sentinels appended after the BPE vocab for T5-style span corruption.
    # Their ids are [vocab_size, vocab_size + num_sentinels). At zero this is
    # a no-op; at 32 the effective embedding / output projection grows by
    # 32 rows (~8K extra params at d_model=256).
    num_sentinels: int = 32

    # special token ids — set after tokenizer is trained
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3

    @property
    def effective_vocab_size(self) -> int:
        return self.vocab_size + self.num_sentinels
