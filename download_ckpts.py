#!/usr/bin/env python3
"""Download WorldFM weights from HuggingFace Hub to local weights/ directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import os

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "inspatio/worldfm"
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"

FLUX_REPO_ID = "black-forest-labs/FLUX.1-Fill-dev"

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
    p.add_argument("--flux", action="store_true",
                    help=f"Also download {FLUX_REPO_ID} into the HF cache "
                         "(gated — requires HF login and accepted licence)")
    return p.parse_args()


def _download_flux(token: str | None) -> None:
    print(f"\nDownloading {FLUX_REPO_ID} into HF cache …")
    print("  (this is a ~25 GB model — make sure you have accepted the licence at")
    print(f"   https://huggingface.co/{FLUX_REPO_ID} )\n")
    snapshot_download(
        repo_id=FLUX_REPO_ID,
        token=token or os.environ.get("HF_TOKEN"),
        ignore_patterns=["*.gguf"],
    )
    print(f"  ✓ {FLUX_REPO_ID} cached")


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

    if args.flux:
        _download_flux(args.token)


if __name__ == "__main__":
    main()
