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



def _load_demo_meta() -> tuple[list, list]:
    """Return (K, c2w) from demo/meta_vbench.json — identical to the demo run."""
    with open(DEMO_META) as f:
        m = json.load(f)
    return m["K"], m["c2w"]


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

    K, c2w = _load_demo_meta()
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
            print(f"[error] pipeline returned {ret} for: {caption}", flush=True)
            sys.exit(1)

        import cv2 as _cv2
        import numpy as _np

        def _check_brightness(path: Path, label: str, is_video: bool) -> None:
            if not path.exists():
                print(f"[error] {label} not found: {path}", flush=True)
                sys.exit(1)
            if is_video:
                cap = _cv2.VideoCapture(str(path))
                n = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
                means = []
                for fi in [int(n * k / 5) for k in range(5)]:
                    cap.set(_cv2.CAP_PROP_POS_FRAMES, fi)
                    ok, frame = cap.read()
                    if ok:
                        means.append(float(frame.mean()))
                cap.release()
                brightness = sum(means) / len(means) if means else 0.0
            else:
                img = _cv2.imread(str(path))
                brightness = float(img.mean()) if img is not None else 0.0
            if brightness < 5.0:
                print(f"[error] all-black {label}: {path.name} brightness={brightness:.2f}", flush=True)
                sys.exit(1)
            print(f"    {label}: brightness={brightness:.1f}", flush=True)

        # Check panorama
        pano_path = pipeline_out / safe_name / "panorama.png"
        _check_brightness(pano_path, "panorama", is_video=False)

        # Check output video
        src = pipeline_out / safe_name / "output.mp4"
        _check_brightness(src, "video", is_video=True)
        shutil.move(str(src), str(out_video))
        print(f"    saved  : {out_video.name}", flush=True)

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
