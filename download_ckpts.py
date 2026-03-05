#!/usr/bin/env python3
"""Download WorldFM weights from HuggingFace Hub to local weights/ directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "inspatio/worldfm"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

FILES = [
    "worldfm_1-step.pth",
    "worldfm_2-step.pth",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download WorldFM weights from HuggingFace Hub")
    p.add_argument("--repo-id", type=str, default=REPO_ID,
                    help=f"HuggingFace repo id (default: {REPO_ID})")
    p.add_argument("--weights-dir", type=str, default=str(WEIGHTS_DIR),
                    help="Local destination directory (default: ./weights)")
    p.add_argument("--revision", type=str, default="main",
                    help="Branch or tag to download from (default: main)")
    p.add_argument("--token", type=str, default=None,
                    help="HuggingFace token (or set HF_TOKEN env var)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dest = Path(args.weights_dir)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from https://huggingface.co/{args.repo_id} → {dest}\n")

    for filename in FILES:
        local_path = dest / filename
        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filename} already exists ({size_mb:,.1f} MB), skipping")
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  ↓ {filename} ...", end="", flush=True)
        hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            revision=args.revision,
            local_dir=str(dest),
            token=args.token,
        )
        print(" done")

    print(f"\nAll weights ready at {dest}")


if __name__ == "__main__":
    main()
