#!/usr/bin/env python3
"""Download a model from ModelScope to local cache.

Usage:
    python download_model.py                                          # default: Qwen2.5-3B-Instruct
    python download_model.py LLM-Research/Meta-Llama-3.1-8B-Instruct  # Llama 3.1 8B
    python download_model.py Qwen/Qwen2.5-7B-Instruct                # Qwen 7B
"""

import sys
import time

from modelscope import snapshot_download


RECOMMENDED_MODELS = {
    "qwen3b":   "Qwen/Qwen2.5-3B-Instruct",
    "qwen7b":   "Qwen/Qwen2.5-7B-Instruct",
    "llama3b":  "LLM-Research/Llama-3.2-3B-Instruct",
    "llama8b":  "LLM-Research/Meta-Llama-3.1-8B-Instruct",
}


def main():
    if len(sys.argv) < 2:
        print("Available shortcuts:")
        for k, v in RECOMMENDED_MODELS.items():
            print(f"  python download_model.py {k}  ->  {v}")
        print("\nOr pass a full model name:")
        print("  python download_model.py Qwen/Qwen2.5-3B-Instruct")
        return

    name = sys.argv[1]
    model_id = RECOMMENDED_MODELS.get(name, name)

    print(f"Downloading: {model_id}")
    print("(modelscope will show progress per file)\n")

    t0 = time.time()
    local_dir = snapshot_download(
        model_id,
        ignore_file_pattern=["*.pth", "*.bin", "original/*"],
    )
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.0f}s")
    print(f"Cached at: {local_dir}")


if __name__ == "__main__":
    main()
