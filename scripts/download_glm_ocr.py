from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download GLM-OCR model files")
    parser.add_argument("--repo", default="zai-org/GLM-OCR", help="Hugging Face repo id")
    parser.add_argument("--output", default="models/GLM-OCR", help="local model directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    snapshot_download(
        repo_id=args.repo,
        local_dir=args.output,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"model downloaded to {args.output}")


if __name__ == "__main__":
    main()
