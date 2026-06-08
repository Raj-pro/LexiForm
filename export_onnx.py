"""
Export the trained model to ONNX format (encoder + decoder separately).
Run: python3 export_onnx.py --ckpt checkpoints/best.pt --tok tokenizer/tokenizer.model
"""
import argparse
import torch
from pathlib import Path
from model.config import ModelConfig
from model.model import ParaphraseModel
from tokenizer.tokenizer import Tokenizer


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, src_ids):
        return self.model.encode(src_ids)


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, dec_ids, encoder_out):
        return self.model.decode(dec_ids, encoder_out)


def export(ckpt_path: str, tok_path: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    tok  = Tokenizer(tok_path)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ModelConfig(**ckpt["config"])
    model  = ParaphraseModel(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # --- export encoder ---
    encoder = EncoderWrapper(model)
    dummy_src = torch.randint(0, config.vocab_size, (1, 32))

    torch.onnx.export(
        encoder,
        (dummy_src,),
        str(out_dir / "encoder.onnx"),
        input_names=["src_ids"],
        output_names=["encoder_out"],
        dynamic_axes={
            "src_ids":     {0: "batch", 1: "src_len"},
            "encoder_out": {0: "batch", 1: "src_len"},
        },
        opset_version=17,
    )
    print("Exported encoder.onnx")

    # --- export decoder ---
    decoder = DecoderWrapper(model)
    dummy_dec = torch.randint(0, config.vocab_size, (1, 16))
    dummy_enc = torch.zeros(1, 32, config.d_model)

    torch.onnx.export(
        decoder,
        (dummy_dec, dummy_enc),
        str(out_dir / "decoder.onnx"),
        input_names=["dec_ids", "encoder_out"],
        output_names=["logits"],
        dynamic_axes={
            "dec_ids":     {0: "batch", 1: "tgt_len"},
            "encoder_out": {0: "batch", 1: "src_len"},
            "logits":      {0: "batch", 1: "tgt_len"},
        },
        opset_version=17,
    )
    print("Exported decoder.onnx")
    print(f"\nONNX files saved to: {out_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default="checkpoints/best.pt")
    parser.add_argument("--tok",     default="tokenizer/tokenizer.model")
    parser.add_argument("--out_dir", default="onnx", type=Path)
    args = parser.parse_args()
    export(args.ckpt, args.tok, args.out_dir)


if __name__ == "__main__":
    main()
