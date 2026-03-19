#!/usr/bin/env python3
"""
WorldFM end-to-end pipeline.

Input:  meta.json  (name, image, K, c2w)
Output: generated images at the specified camera poses.

Steps:
  1. Perspective image  ->  panorama           (modules.panogen)
  2. Panorama           ->  depth/PLY/conditions  (modules.moge_pano + pano_postprocess)
  3. Target pose        ->  condition_render + condition_nearest  (modules.point_renderer + depth_selector)
  4. Condition pair     ->  final generated image  (modules.worldfm_infer)

All inter-step data is passed in memory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Disable xformers memory-efficient attention — its CUDA kernels don't support
# compute capability >= 12.0 (Blackwell / RTX 5090).  DINOv2 in MoGe checks
# this env var at import time and falls back to standard attention.
os.environ.setdefault("XFORMERS_DISABLED", "1")

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import trange

WORLDFM_ROOT = Path(__file__).resolve().parent
SUBMODULES = WORLDFM_ROOT / "submodules"
DEFAULT_CFG = OmegaConf.load(str(WORLDFM_ROOT / "default.yaml"))

# ---------------------------------------------------------------------------
# Local modules — moge_pano and panogen use try/except for external deps,
# so importing them here is safe even before setup_external_repos().
# MoGeModel etc. are accessed via moge_pano.<attr> because ensure_moge()
# rebinds the module-level globals at runtime.
# ---------------------------------------------------------------------------
import modules.moge_pano as moge_pano
from modules.moge_pano import (
    ensure_moge,
    select_tier,
    _get_panorama_cameras,
)
from modules.panogen import ensure_hy3dworld, Image2PanoramaDemo
from modules.pano_postprocess import postprocess_panorama
from modules.point_renderer import TorchPointCloudRenderer
from modules.depth_selector import (
    build_condition_db_in_memory,
    select_best_condition_index,
)
from modules.worldfm_infer import WorldFMInprocessConfig, WorldFMTriConditionInprocess


# ============================== helpers ======================================

def _load_meta(meta_path: str) -> dict:
    """Load and validate meta.json. Normalises c2w to a list of 4×4 matrices."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    required = ("name", "image", "K", "c2w")
    for k in required:
        if k not in meta:
            raise KeyError(f"meta.json missing required key: {k}")
    c2w = np.asarray(meta["c2w"], dtype=np.float64)
    if c2w.ndim == 2:
        meta["c2w"] = [c2w.tolist()]
    elif c2w.ndim != 3:
        raise ValueError(f"c2w must be (4,4) or (N,4,4), got shape {c2w.shape}")
    return meta


def _log(step: str, msg: str) -> None:
    print(f"[WorldFM][{step}] {msg}", flush=True)


def setup_external_repos(*, hw_path: str = "", moge_path: str = "") -> None:
    """Register external repo paths on sys.path **before** any model imports.

    Must be called once at the very beginning of the pipeline, because
    hy3dworld internally imports moge, so MoGe must be available first.
    Additionally, hy3dworld uses realesrgan and zim_anything, so their
    repos (Real-ESRGAN, ZIM) must also be on sys.path.
    """
    resolved_moge = moge_path or str(SUBMODULES / "MoGe")
    if Path(resolved_moge).exists():
        _log("Init", f"ensure_moge({resolved_moge})")
        ensure_moge(resolved_moge)

    for dep in ("Real-ESRGAN", "ZIM"):
        dep_path = str(SUBMODULES / dep)
        if Path(dep_path).exists() and dep_path not in sys.path:
            sys.path.insert(0, dep_path)
            _log("Init", f"sys.path += {dep_path}")

    resolved_hw = hw_path or str(SUBMODULES / "HunyuanWorld-1.0")
    if Path(resolved_hw).exists():
        _log("Init", f"ensure_hy3dworld({resolved_hw})")
        ensure_hy3dworld(resolved_hw)


# ============================== Step 1 =======================================

def step1_panogen(image_path: str, output_dir: Path, *, cfg=None, prompt: str = ""):
    """Perspective image -> panorama (PIL Image).

    Returns PIL.Image.Image (panorama).
    Requires setup_external_repos() to have been called first.
    """
    pcfg = (cfg or DEFAULT_CFG).panogen
    _log("Step1", f"Generating panorama from {image_path}")

    pano_disk = output_dir / "panorama.png"
    if pano_disk.exists():
        _log("Step1", f"Panorama already exists, loading: {pano_disk}")
        return Image.open(pano_disk).convert("RGB")

    class _Args:
        fp8_attention = bool(pcfg.fp8_attention)
        fp8_gemm = bool(pcfg.fp8_gemm)
        cache = bool(pcfg.cache)

    demo = Image2PanoramaDemo(_Args())
    demo.num_inference_steps = int(pcfg.num_inference_steps)

    output_dir.mkdir(parents=True, exist_ok=True)
    pano_img = demo.run(
        prompt=prompt,
        negative_prompt="",
        image_path=str(image_path),
        seed=int(pcfg.seed),
        save_to_disk=False,
        output_path=None,
    )

    _log("Step1", f"Panorama generated: {np.array(pano_img).shape}")
    return pano_img


# ============================== Step 2 =======================================

def step2_moge_pipeline(panorama_img, output_dir: Path, *, cfg=None, pretrained: str = ""):
    """Panorama image -> depth + PLY arrays + condition images + transforms.

    Returns modules.pano_postprocess.PostProcessResult.
    Requires setup_external_repos() to have been called first.
    """
    mcfg = (cfg or DEFAULT_CFG).moge
    pretrained = pretrained or mcfg.pretrained
    resolution_level = int(mcfg.resolution_level)
    fov_deg = float(mcfg.fov_deg)
    num_views = int(mcfg.num_views)
    merge_max_w = int(mcfg.merge_max_width)
    merge_max_h = int(mcfg.merge_max_height)
    batch_size = int(mcfg.batch_size)

    _log("Step2", "Running MoGe + postprocess")
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

    image_rgb = np.array(panorama_img)
    if image_rgb.ndim == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    tier = select_tier(orig_w)
    tgt_w, tgt_h = tier["width"], tier["height"]
    split_resolution = tier["split_res"]
    _log("Step2", f"tier={tier['name']} ({tgt_w}x{tgt_h}), input={orig_w}x{orig_h}")

    if orig_w != tgt_w or orig_h != tgt_h:
        interp = cv2.INTER_AREA if tgt_w < orig_w else cv2.INTER_LINEAR
        image_rgb = cv2.resize(image_rgb, (tgt_w, tgt_h), interpolation=interp)
    height, width = image_rgb.shape[:2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = moge_pano.MoGeModel.from_pretrained(pretrained).to(device).eval()
    _log("Step2", f"MoGe model loaded on {device}")

    import utils3d
    extrinsics, intrinsics = _get_panorama_cameras(num_views, fov_deg)
    splitted_images = moge_pano.split_panorama_image(image_rgb, extrinsics, intrinsics, split_resolution)

    splitted_dist, splitted_masks = [], []
    for i in trange(0, len(splitted_images), batch_size, desc="MoGe Infer", leave=False):
        batch = np.stack(splitted_images[i:i + batch_size])
        tensor = torch.tensor(batch / 255, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        fov_x, _ = np.rad2deg(utils3d.numpy.intrinsics_to_fov(np.array(intrinsics[i:i + batch_size])))
        fov_x_t = torch.tensor(fov_x, dtype=torch.float32, device=device)
        out = model.infer(tensor, resolution_level=resolution_level, fov_x=fov_x_t, apply_mask=False)
        splitted_dist.extend(list(out["points"].norm(dim=-1).cpu().numpy()))
        splitted_masks.extend(list(out["mask"].cpu().numpy()))

    _log("Step2", "Merge panorama depth")
    merging_w = min(merge_max_w, width)
    merging_h = min(merge_max_h, height)
    panorama_depth, panorama_mask = moge_pano.merge_panorama_depth(
        merging_w, merging_h, splitted_dist, splitted_masks, extrinsics, intrinsics,
    )
    panorama_depth = panorama_depth.astype(np.float32)
    panorama_depth = cv2.resize(panorama_depth, (width, height), cv2.INTER_LINEAR)
    panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (width, height), cv2.INTER_NEAREST) > 0

    depth_raw = panorama_depth.copy()
    if panorama_mask.any():
        depth_raw[~panorama_mask] = panorama_depth[panorama_mask].max()
    depth_raw = depth_raw / 100.0

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    pano_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    result = postprocess_panorama(pano_bgr, depth_raw, save_dir=None)
    _log("Step2", f"PLY: {result.ply_xyz.shape[0]:,} points, conditions: {len(result.condition_images)}")
    return result


# ============================== Step 3 =======================================

def step3_init(pp_result, *, cfg=None, render_size: int = 0):
    """Create renderer and condition DB (heavy objects, reusable across frames).

    Returns (renderer, cond_db, rcfg, render_size).
    """
    rcfg = (cfg or DEFAULT_CFG).render
    S = render_size or int(rcfg.render_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    renderer = TorchPointCloudRenderer(
        points_xyz=pp_result.ply_xyz,
        points_rgb=pp_result.ply_rgb / 255.0 if pp_result.ply_rgb.dtype == np.uint8 else pp_result.ply_rgb,
        width=S, height=S, device=str(device), mode="fast",
    )

    cond_db = build_condition_db_in_memory(
        condition_images=pp_result.condition_images,
        transforms_dict=pp_result.transforms,
        torch_renderer=renderer,
        device=device,
    )

    return renderer, cond_db, rcfg, S


def step3_render_one(
    renderer,
    cond_db,
    pp_result,
    K: np.ndarray,
    c2w: np.ndarray,
    *,
    rcfg=None,
    render_size: int = 512,
):
    """Render condition pair for a single target pose.

    Returns (render_rgb_u8: torch.Tensor, cond_nearest_resized: np.ndarray).
    """
    rcfg = rcfg or DEFAULT_CFG.render
    S = render_size

    K_arr = np.asarray(K, dtype=np.float64)
    c2w_arr = np.asarray(c2w, dtype=np.float64)

    out = renderer.render_torch(K_3x3=K_arr, c2w_4x4=c2w_arr, c2w_is_camera_to_world=True)
    rgb_u8 = out.rgb_u8
    depth = out.depth_f32

    idx, hits, samples = select_best_condition_index(
        depth_cur=depth,
        K_cur=K_arr, c2w_cur=c2w_arr,
        cond_db=cond_db,
        sample_grid=int(rcfg.sample_grid),
        center_grid=int(rcfg.center_grid),
        center_frac=float(rcfg.center_frac),
        eps_rel=float(rcfg.eps_rel),
        eps_abs=float(rcfg.eps_abs),
        px_radius=int(rcfg.px_radius),
        max_view_angle_deg=float(rcfg.max_view_angle_deg),
        use_distance_weight=bool(rcfg.use_distance_weight),
        dist_min_m=float(rcfg.dist_min_m),
        dist_max_m=float(rcfg.dist_max_m),
        weight_near=float(rcfg.weight_near),
        weight_far=float(rcfg.weight_far),
    )
    _log("Step3", f"Selected condition: idx={idx} hits={hits}/{samples}")

    cond_nearest_rgb = pp_result.condition_images[int(idx)]
    cond_nearest_resized = np.array(
        Image.fromarray(cond_nearest_rgb, "RGB").resize((S, S), resample=Image.BILINEAR)
    )

    return rgb_u8, cond_nearest_resized


# ============================== Step 4 =======================================

def step4_init(*, cfg=None):
    """Load WorldFM inference service (heavy, reusable across frames).

    Returns (svc, wcfg).
    """
    wcfg = (cfg or DEFAULT_CFG).worldfm
    model_path = str(wcfg.model_path)
    vae_path = str(wcfg.vae_path)
    image_size = int(wcfg.image_size)
    image_height = int(getattr(wcfg, "image_height", image_size))
    image_width = int(getattr(wcfg, "image_width", image_size))
    step = int(wcfg.step)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = f"cuda:{torch.cuda.current_device()}" if device.type == "cuda" else "cpu"

    model_p = Path(model_path).resolve()
    vae_p = Path(vae_path)
    if not vae_p.is_absolute():
        vae_p = (WORLDFM_ROOT / vae_p).resolve()

    svc = WorldFMTriConditionInprocess(
        WorldFMInprocessConfig(
            model_path=str(model_p),
            vae_path=str(vae_p),
            image_size=image_size,
            image_height=image_height,
            image_width=image_width,
            version=str(wcfg.version),
            disable_cross_attn=True,
            step=(step if step in (1, 2) else 2),
            mid_t=int(wcfg.mid_t), cfg_scale=0.0,
            device=device_str,
            weight_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
    )

    return svc, wcfg


def step4_infer_one(
    svc,
    render_rgb_u8,
    cond_nearest_rgb: np.ndarray,
    *,
    wcfg=None,
) -> np.ndarray:
    """Run WorldFM inference for a single frame.

    Returns (H, W, 3) uint8 numpy array (RGB).
    """
    wcfg = wcfg or DEFAULT_CFG.worldfm
    step = int(wcfg.step)
    cfg_scale = float(wcfg.cfg_scale)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    svc.set_cond2_from_array(cond_nearest_rgb)

    if isinstance(render_rgb_u8, torch.Tensor):
        render_u8 = render_rgb_u8
    else:
        render_u8 = torch.from_numpy(
            np.asarray(render_rgb_u8, dtype=np.uint8)
        ).to(device=device, dtype=torch.uint8)

    if step in (1, 2):
        decoded = svc.infer_from_render_u8(render_u8)
    else:
        decoded = svc.infer_from_render_u8_multistep(
            render_u8, sample_steps=step, cfg_scale=cfg_scale,
        )

    out_u8 = (
        torch.clamp(127.5 * decoded[0] + 128.0, 0, 255)
        .permute(1, 2, 0).to(torch.uint8).cpu().numpy()
    )
    return out_u8


# ============================== main =========================================

def build_parser() -> argparse.ArgumentParser:
    d = DEFAULT_CFG
    p = argparse.ArgumentParser(
        description="WorldFM pipeline: perspective image + target poses -> generated images",
    )
    p.add_argument("--config", type=str, default="",
                   help="Override config YAML (merged on top of default.yaml)")
    p.add_argument("--meta", type=str, required=True,
                   help="Path to meta.json (name, image, K, c2w)")
    p.add_argument("--output_dir", type=str, default=d.pipeline.output_dir,
                   help="Base output directory")

    p.add_argument("--hw_path", type=str, default=d.submodules.hw_path,
                   help="HunyuanWorld-1.0 repo path (auto-detect if empty)")
    p.add_argument("--moge_path", type=str, default=d.submodules.moge_path,
                   help="MoGe repo path (auto-detect if empty)")

    p.add_argument("--moge_pretrained", type=str, default=d.moge.pretrained,
                   help="MoGe pretrained model path")
    p.add_argument("--render_size", type=int, default=d.render.render_size,
                   help="Point-cloud render resolution (square)")

    p.add_argument("--model_path", type=str, default=d.worldfm.model_path,
                   help="WorldFM model checkpoint path")
    p.add_argument("--vae_path", type=str, default=d.worldfm.vae_path,
                   help="Path to VAE directory (AutoencoderKL)")
    p.add_argument("--image_size", type=int, default=d.worldfm.image_size,
                   help="WorldFM inference resolution (square, overridden by --image_height/--image_width)")
    p.add_argument("--image_height", type=int, default=None,
                   help="WorldFM output height in pixels (overrides --image_size)")
    p.add_argument("--image_width", type=int, default=None,
                   help="WorldFM output width in pixels (overrides --image_size)")
    p.add_argument("--step", type=int, default=d.worldfm.step,
                   help="WorldFM inference steps (1 or 2)", choices=[1, 2])
    p.add_argument("--mid_t", type=int, default=d.worldfm.mid_t,
                       help="Intermediate noise timestep for DMD 2-step sampling (default: %(default)s)")
    p.add_argument("--cfg_scale", type=float, default=d.worldfm.cfg_scale,
                   help="CFG scale for multi-step sampling")
    p.add_argument("--gpu_index", type=int, default=d.pipeline.gpu_index,
                   help="CUDA device index")
    p.add_argument("--save_mode", type=str, default="video",
                   choices=["image", "video"],
                   help="Output format: 'image' saves per-frame PNGs, 'video' saves MP4 (default: video)")
    p.add_argument("--fps", type=int, default=30,
                   help="Video frame rate when --save_mode=video (default: 30)")
    return p


def _load_config(args) -> OmegaConf:
    """Merge default.yaml <- user config <- CLI overrides."""
    cfg = OmegaConf.create(DEFAULT_CFG)
    if args.config:
        user_cfg = OmegaConf.load(args.config)
        cfg = OmegaConf.merge(cfg, user_cfg)
    cli_overrides = OmegaConf.create({
        "pipeline": {"output_dir": args.output_dir,
                      "gpu_index": args.gpu_index},
        "submodules": {"hw_path": args.hw_path, "moge_path": args.moge_path},
        "moge": {"pretrained": "Ruicheng/moge-2-vitl-normal" if args.moge_pretrained is None else args.moge_pretrained},
        "render": {"render_size": args.render_size},
        "worldfm": {"model_path": args.model_path,
                     "vae_path": args.vae_path,
                     "image_size": args.image_size,
                     "image_height": args.image_height if args.image_height is not None else args.image_size,
                     "image_width": args.image_width if args.image_width is not None else args.image_size,
                     "step": args.step,
                     "mid_t": args.mid_t, "cfg_scale": args.cfg_scale},
    })
    cfg = OmegaConf.merge(cfg, cli_overrides)
    return cfg


def main() -> int:
    args = build_parser().parse_args()
    cfg = _load_config(args)

    if cfg.pipeline.gpu_index >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(int(cfg.pipeline.gpu_index))

    meta_path = Path(args.meta).resolve()
    meta = _load_meta(str(meta_path))
    meta_dir = meta_path.parent

    name = meta["name"]
    image_path = (meta_dir / meta["image"]).resolve()
    K = np.asarray(meta["K"], dtype=np.float64)
    c2w_list = [np.asarray(c, dtype=np.float64) for c in meta["c2w"]]

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    base_output = Path(str(cfg.pipeline.output_dir))
    if not base_output.is_absolute():
        base_output = (WORLDFM_ROOT / base_output).resolve()
    output_dir = base_output / name
    output_dir.mkdir(parents=True, exist_ok=True)

    _log("Main", f"name={name}")
    _log("Main", f"image={image_path}")
    _log("Main", f"output_dir={output_dir}")
    _log("Main", f"poses={len(c2w_list)}")

    # ---- Setup external repos (MoGe first, then HunyuanWorld) ----
    setup_external_repos(
        hw_path=str(cfg.submodules.hw_path),
        moge_path=str(cfg.submodules.moge_path),
    )

    # ---- Step 1: Perspective -> Panorama (PIL Image) ----
    panorama_img = step1_panogen(
        image_path=str(image_path),
        output_dir=output_dir,
        cfg=cfg,
    )

    # ---- Step 2: Panorama -> depth/PLY/conditions (in memory) ----
    pp_result = step2_moge_pipeline(
        panorama_img=panorama_img,
        output_dir=output_dir,
        cfg=cfg,
    )

    # ---- Step 3 init: renderer + condition DB (once) ----
    _log("Step3", "Initializing renderer and condition DB")
    renderer, cond_db, rcfg, S = step3_init(pp_result, cfg=cfg)

    # ---- Step 4 init: WorldFM service (once) ----
    _log("Step4", "Loading WorldFM inference service")
    svc, wcfg = step4_init(cfg=cfg)

    # ---- Generate for each target pose ----
    save_mode = args.save_mode
    frames: list[np.ndarray] = []
    for i, c2w in enumerate(c2w_list):
        _log("Main", f"Generating frame {i + 1}/{len(c2w_list)}")

        render_u8, cond_nearest_rgb = step3_render_one(
            renderer, cond_db, pp_result, K, c2w,
            rcfg=rcfg, render_size=S,
        )

        frame = step4_infer_one(svc, render_u8, cond_nearest_rgb, wcfg=wcfg)

        if save_mode == "image":
            out_name = "output.png" if len(c2w_list) == 1 else f"output_{i:04d}.png"
            out_path = output_dir / out_name
            Image.fromarray(frame, mode="RGB").save(str(out_path))
            _log("Main", f"Saved: {out_path}")
        else:
            frames.append(frame)

    # ---- Save video ----
    if save_mode == "video" and frames:
        video_path = output_dir / "output.mp4"
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w, h))
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        _log("Main", f"Video saved: {video_path} ({len(frames)} frames, {args.fps} fps)")

    # ---- Cleanup ----
    del renderer, cond_db, svc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    n = len(c2w_list)
    _log("Main", f"Pipeline complete: {n} frames ({'video' if save_mode == 'video' else 'images'}) in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
