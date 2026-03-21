#!/usr/bin/env python3
"""
WorldFM VBench i2v runner.

Iterates over VBench2 i2v images, runs the WorldFM pipeline for each,
and saves outputs as  results_vbench/videos/{caption}_s{idx}.mp4
matching the DeepVerse naming convention.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

WORLDFM_ROOT = Path(__file__).resolve().parent
DEMO_META = WORLDFM_ROOT / "demo" / "meta_vbench.json"
VBENCH_DEFAULT_JSON = (
    WORLDFM_ROOT.parent / "VBench" / "vbench2_beta_i2v" /
    "vbench2_beta_i2v" / "data" / "i2v-bench-info.json"
)
VBENCH_DEFAULT_CROP = (
    WORLDFM_ROOT.parent / "VBench" / "vbench2_beta_i2v" /
    "vbench2_beta_i2v" / "data" / "crop" / "1-1"
)


def _make_k(width: int, height: int, fov_deg: float = 90.0) -> list:
    """Pinhole K for a square/portrait/landscape image at the given FOV."""
    # Use the smaller dimension for FOV so the full image fits in view
    focal_px = (min(width, height) / 2) / math.tan(math.radians(fov_deg / 2))
    cx, cy = width / 2.0, height / 2.0
    return [[focal_px, 0.0, cx], [0.0, focal_px, cy], [0.0, 0.0, 1.0]]


def _load_demo_c2w() -> list:
    """Return the canonical walk-through trajectory from demo/meta.json."""
    with open(DEMO_META) as f:
        return json.load(f)["c2w"]


def run_vbench(
    vbench_json: Path,
    vbench_crop: Path,
    output_dir: Path,
    tmp_dir: Path,
    config: str,
    num_samples: int,
    skip_existing: bool,
    extra_args: list[str],
) -> None:
    with open(vbench_json) as f:
        entries = json.load(f)

    c2w = _load_demo_c2w()
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for entry in entries:
        if num_samples > 0 and processed >= num_samples:
            break

        caption = entry.get("caption", Path(entry["file_name"]).stem)
        file_name = entry["file_name"]
        img_path = vbench_crop / file_name

        if not img_path.exists():
            print(f"[skip] image not found: {img_path}", flush=True)
            continue

        out_video = output_dir / f"{caption}_s0.mp4"
        if skip_existing and out_video.exists():
            print(f"[skip] already exists: {out_video.name}", flush=True)
            processed += 1
            continue

        # --- Derive K from image size ---
        try:
            from PIL import Image as PILImage
            img = PILImage.open(img_path)
            w, h = img.size
        except Exception:
            w, h = 512, 512
        K = _make_k(w, h)

        # --- Write temp meta.json ---
        safe_name = caption[:80].replace("/", "_").replace("\\", "_")
        meta_dir = tmp_dir / safe_name
        meta_dir.mkdir(parents=True, exist_ok=True)
        img_copy = meta_dir / file_name
        if not img_copy.exists():
            shutil.copy2(img_path, img_copy)

        meta = {"name": safe_name, "image": file_name, "K": K, "c2w": c2w}
        meta_path = meta_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

        pipeline_out = tmp_dir / "pipeline_out"
        pipeline_out.mkdir(exist_ok=True)

        # --- Run pipeline ---
        cmd = [
            sys.executable, str(WORLDFM_ROOT / "run_pipeline.py"),
            "--meta", str(meta_path),
            "--output_dir", str(pipeline_out),
        ]
        if config:
            cmd += ["--config", config]
        cmd += extra_args

        print(f"\n[{processed + 1}] {caption}", flush=True)
        print(f"    image  : {img_path}", flush=True)
        print(f"    output : {out_video}", flush=True)
        ret = subprocess.call(cmd)

        if ret != 0:
            print(f"[warn] pipeline returned {ret} for: {caption}", flush=True)
        else:
            # Pipeline saves to pipeline_out/{safe_name}/output.mp4
            src = pipeline_out / safe_name / "output.mp4"
            if src.exists():
                shutil.move(str(src), str(out_video))
                print(f"    saved  : {out_video.name}", flush=True)
            else:
                print(f"[warn] output not found: {src}", flush=True)

        processed += 1

    print(f"\nDone. {processed} samples processed -> {output_dir}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="WorldFM VBench i2v batch runner")
    p.add_argument("--vbench-json", type=Path, default=VBENCH_DEFAULT_JSON,
                   help="Path to i2v-bench-info.json")
    p.add_argument("--vbench-crop", type=Path, default=VBENCH_DEFAULT_CROP,
                   help="Directory containing 1-1 cropped VBench images")
    p.add_argument("--output-dir", type=Path,
                   default=WORLDFM_ROOT / "results_vbench" / "videos",
                   help="Where to save output MP4s")
    p.add_argument("--tmp-dir", type=Path,
                   default=WORLDFM_ROOT / "_vbench_tmp",
                   help="Temp directory for meta.json and intermediate outputs")
    p.add_argument("--config", type=str, default=str(WORLDFM_ROOT / "vbench.yaml"),
                   help="Config YAML to pass to run_pipeline.py")
    p.add_argument("--num-samples", type=int, default=5,
                   help="Max number of VBench images to process (0 = all)")
    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip images whose output MP4 already exists")
    args, extra = p.parse_known_args()

    run_vbench(
        vbench_json=args.vbench_json,
        vbench_crop=args.vbench_crop,
        output_dir=args.output_dir,
        tmp_dir=args.tmp_dir,
        config=args.config,
        num_samples=args.num_samples,
        skip_existing=args.skip_existing,
        extra_args=extra,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
