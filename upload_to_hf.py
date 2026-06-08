"""
Upload model to Hugging Face Hub.
Run: python3 upload_to_hf.py --repo YOUR_USERNAME/paraphrase-llm-13m
"""
import argparse
from huggingface_hub import HfApi, login


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo",  required=True, help="e.g. yourname/paraphrase-llm-13m")
    parser.add_argument("--token", default=None,  help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    login(token=args.token)  # prompts if token not provided
    api = HfApi()

    # create repo if it doesn't exist
    api.create_repo(repo_id=args.repo, exist_ok=True)

    files = {
        "checkpoints/best.pt":          "best.pt",
        "onnx/encoder.onnx":            "encoder.onnx",
        "onnx/decoder.onnx":            "decoder.onnx",
        "tokenizer/tokenizer.model":    "tokenizer.model",
        "README.md":                    "README.md",
        "model/config.py":              "model/config.py",
        "model/model.py":               "model/model.py",
        "model/attention.py":           "model/attention.py",
        "model/blocks.py":              "model/blocks.py",
        "inference/infer.py":           "inference/infer.py",
        "tokenizer/tokenizer.py":       "tokenizer/tokenizer.py",
    }

    for local_path, repo_path in files.items():
        print(f"Uploading {local_path} → {repo_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=args.repo,
        )

    print(f"\nDone! View at: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
