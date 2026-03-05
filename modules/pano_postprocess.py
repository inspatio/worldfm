"""
Panorama post-processing: depth -> PLY + condition images + transforms JSON.

Pure numpy / cv2 — no external-repo dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ====================== data structures ======================================

@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class PostProcessResult:
    """In-memory result of panorama post-processing (Step 2 output)."""

    pano_bgr: np.ndarray          # (H,W,3) uint8
    depth: np.ndarray              # (H,W) float32, same resolution as pano_bgr
    ply_xyz: np.ndarray            # (N,3) float32  OpenCV coords
    ply_rgb: np.ndarray            # (N,3) uint8
    condition_images: List[np.ndarray]  # list of (cH,cW,3) uint8 RGB
    transforms: Dict               # {"frames": [...]}


# ====================== helpers ==============================================

def fit_within(w: int, h: int, max_w: int, max_h: int) -> Tuple[int, int]:
    sw = max_w / max(1, w)
    sh = max_h / max(1, h)
    s = min(1.0, sw, sh)
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def resize_panorama(img_bgr_u8: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = img_bgr_u8.shape[:2]
    out_w, out_h = fit_within(w, h, max_w, max_h)
    if out_w == w and out_h == h:
        return img_bgr_u8
    interp = cv2.INTER_AREA if (out_w < w or out_h < h) else cv2.INTER_LINEAR
    return cv2.resize(img_bgr_u8, (out_w, out_h), interpolation=interp)


def load_depth_npy(
    p: Path,
    depth_scale: float,
    *,
    allow_pickle: bool = False,
) -> np.ndarray:
    if p.suffix.lower() not in (".npy", ".npz"):
        raise ValueError(f"Expected .npy/.npz, got: {p}")
    arr = np.load(str(p), allow_pickle=bool(allow_pickle)).astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Depth must be 2D, got shape={arr.shape}")
    if depth_scale != 1.0:
        arr = arr * float(depth_scale)
    valid = np.isfinite(arr) & (arr > 0)
    return np.where(valid, arr, 0.0).astype(np.float32, copy=False)


def upsample_depth_masked(depth: np.ndarray, tgt_w: int, tgt_h: int) -> np.ndarray:
    src_h, src_w = depth.shape
    if src_w == tgt_w and src_h == tgt_h:
        return depth
    mask = (depth > 0).astype(np.float32)
    up_depth = cv2.resize(depth * mask, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    up_mask = cv2.resize(mask, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out = np.zeros((tgt_h, tgt_w), dtype=np.float32)
    m = up_mask > 1e-6
    out[m] = up_depth[m] / up_mask[m]
    return out


def fill_invalid_depth(depth: np.ndarray, *, far_depth: float) -> np.ndarray:
    far = float(far_depth)
    if not (far > 0) or not math.isfinite(far):
        raise ValueError(f"far_depth must be finite and > 0, got {far_depth}")
    d = depth.astype(np.float32, copy=False)
    invalid = (~np.isfinite(d)) | (d <= 0)
    if np.any(invalid):
        d = d.copy()
        d[invalid] = far
    return d


def prepare_depth(
    depth_raw: np.ndarray,
    depth_scale: float,
    pano_w: int,
    pano_h: int,
) -> np.ndarray:
    """Scale -> upsample -> fill invalid. Works on in-memory arrays."""
    if depth_scale != 1.0:
        depth_raw = depth_raw.astype(np.float32, copy=False) * float(depth_scale)
    valid_mask = np.isfinite(depth_raw) & (depth_raw > 0)
    depth_raw = np.where(valid_mask, depth_raw, 0.0).astype(np.float32)
    depth_up = upsample_depth_masked(depth_raw, pano_w, pano_h)
    valid = depth_up[np.isfinite(depth_up) & (depth_up > 0)]
    if valid.size == 0:
        raise RuntimeError("No valid depth values")
    return fill_invalid_depth(depth_up, far_depth=float(np.max(valid)))


# ====================== PLY generation =======================================

def compute_ply_arrays(
    pano_bgr_u8: np.ndarray,
    depth: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (xyz, rgb) arrays in OpenCV coords, same as _write_ply_opencv."""
    H, W = depth.shape
    assert pano_bgr_u8.shape[:2] == (H, W)

    x_idx = np.arange(W, dtype=np.float32)
    theta = (x_idx / W) * (2.0 * math.pi) - math.pi
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    y_idx = np.arange(H, dtype=np.float32)
    phi = (y_idx / H) * math.pi
    sin_p = np.sin(phi)[:, None]
    cos_p = np.cos(phi)[:, None]

    r = depth
    X = (r * sin_p * cos_t[None, :]).reshape(-1)
    Y = (-(r * cos_p)).reshape(-1)
    Z = (-(r * sin_p * sin_t[None, :])).reshape(-1)

    xyz = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    bgr = pano_bgr_u8.reshape(-1, 3)
    rgb = bgr[:, ::-1].copy()
    return xyz, rgb


def write_ply(
    out_path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
) -> int:
    """Write binary_little_endian PLY from (N,3) xyz float32 + (N,3) rgb uint8."""
    count = xyz.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {count}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    vtx_dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    verts = np.empty(count, dtype=vtx_dtype)
    verts["x"] = xyz[:, 0]
    verts["y"] = xyz[:, 1]
    verts["z"] = xyz[:, 2]
    verts["red"] = rgb[:, 0]
    verts["green"] = rgb[:, 1]
    verts["blue"] = rgb[:, 2]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(header)
        f.write(verts.tobytes())
    return count


# ====================== Condition generation =================================

def _views_yaw_pitch() -> List[Tuple[float, float]]:
    yaws = [
        0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4,
        math.pi, -3 * math.pi / 4, -math.pi / 2, -math.pi / 4,
    ]
    pitches = [0.0, math.pi / 4, -math.pi / 4, math.pi / 6, -math.pi / 6]
    views: List[Tuple[float, float]] = []
    for pitch in pitches:
        for yaw in yaws:
            views.append((yaw, pitch))
    views.append((0.0, math.pi / 2))
    views.append((0.0, -math.pi / 2))
    return views


def _basis_from_yaw_pitch(yaw: float, pitch: float):
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    f = np.array([cyaw * cp, sp, syaw * cp], dtype=np.float32)
    fl = float(np.linalg.norm(f))
    f = f / fl if fl > 1e-8 else np.array([0, 0, 1], dtype=np.float32)

    world_up = np.array([0, 1, 0], dtype=np.float32)
    r = np.cross(f, world_up).astype(np.float32)
    rl = float(np.linalg.norm(r))
    r = r / rl if rl > 1e-8 else np.array([0, 0, 1], dtype=np.float32)

    u = np.cross(r, f).astype(np.float32)
    ul = float(np.linalg.norm(u))
    u = u / ul if ul > 1e-8 else np.array([0, 1, 0], dtype=np.float32)
    return r, u, f


def _bilinear_sample_pano_rgb(pano_rgb_u8: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    H, W = pano_rgb_u8.shape[:2]
    x0 = np.mod(np.floor(x).astype(np.int64), W)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, H - 1)
    x1 = np.mod(x0 + 1, W)
    y1 = np.clip(y0 + 1, 0, H - 1)
    fx = (x - x0.astype(np.float32)).astype(np.float32)
    fy = (y - y0.astype(np.float32)).astype(np.float32)
    x0i, y0i = x0.astype(np.int32), y0.astype(np.int32)
    x1i, y1i = x1.astype(np.int32), y1.astype(np.int32)
    out = (
        pano_rgb_u8[y0i, x0i].astype(np.float32) * ((1 - fx) * (1 - fy))[:, None]
        + pano_rgb_u8[y0i, x1i].astype(np.float32) * (fx * (1 - fy))[:, None]
        + pano_rgb_u8[y1i, x0i].astype(np.float32) * ((1 - fx) * fy)[:, None]
        + pano_rgb_u8[y1i, x1i].astype(np.float32) * (fx * fy)[:, None]
    )
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def _generate_condition_image(
    pano_rgb_u8: np.ndarray,
    intr: Intrinsics,
    yaw: float,
    pitch: float,
) -> np.ndarray:
    H_p, W_p = pano_rgb_u8.shape[:2]
    W, H = int(intr.width), int(intr.height)
    r, u, f = _basis_from_yaw_pitch(yaw, pitch)
    rX, rY, rZ = float(r[0]), float(r[1]), float(r[2])
    uX, uY, uZ = float(u[0]), float(u[1]), float(u[2])
    fX, fY, fZ = float(f[0]), float(f[1]), float(f[2])

    uu = np.arange(W, dtype=np.float32)
    vv = np.arange(H, dtype=np.float32)
    U, V = np.meshgrid(uu, vv)
    xCam = (U - float(intr.cx)) / float(intr.fx)
    yCam = -((V - float(intr.cy)) / float(intr.fy))
    zCam = np.ones_like(xCam, dtype=np.float32)
    inv_len = 1.0 / np.sqrt(xCam ** 2 + yCam ** 2 + zCam ** 2)
    dx, dy, dz = xCam * inv_len, yCam * inv_len, zCam * inv_len
    wx = rX * dx + uX * dy + fX * dz
    wy = rY * dx + uY * dy + fY * dz
    wz = rZ * dx + uZ * dy + fZ * dz

    theta = np.arctan2(wz, wx).astype(np.float32)
    phi = np.arccos(np.clip(wy, -1, 1).astype(np.float32))
    panoX = np.mod(((theta + math.pi) / (2 * math.pi)) * W_p, float(W_p)).astype(np.float32)
    panoY = np.clip((phi / math.pi) * H_p, 0, H_p - 1e-6).astype(np.float32)

    samp = _bilinear_sample_pano_rgb(pano_rgb_u8, panoX.reshape(-1), panoY.reshape(-1))
    return samp.reshape(H, W, 3)


def _c2w_opencv_rows(yaw: float, pitch: float) -> List[List[float]]:
    r, u, f = _basis_from_yaw_pitch(yaw, pitch)
    z = (-f).astype(np.float32)
    M = np.eye(4, dtype=np.float32)
    M[:3, 0], M[:3, 1], M[:3, 2] = r, u, z
    A = np.diag(np.array([1, -1, -1, 1], dtype=np.float32))
    Mcv = A @ M @ A
    return [[float(Mcv[i, j]) for j in range(4)] for i in range(4)]


def generate_conditions(
    pano_bgr_u8: np.ndarray,
    intr: Intrinsics,
    *,
    save_dir: Optional[Path] = None,
) -> Tuple[List[np.ndarray], Dict]:
    """Generate condition images and transforms dict.

    Returns (condition_images_rgb, transforms_dict).
    If *save_dir* is given, also writes images and JSON to disk.
    """
    pano_rgb = pano_bgr_u8[:, :, ::-1].copy()
    views = _views_yaw_pitch()
    frames: List[Dict] = []
    images: List[np.ndarray] = []

    if save_dir is not None:
        cond_dir = save_dir / "conditions"
        cond_dir.mkdir(parents=True, exist_ok=True)

    for idx, (yaw, pitch) in enumerate(views):
        img_rgb = _generate_condition_image(pano_rgb, intr, yaw, pitch)
        images.append(img_rgb)

        filename = f"{idx:04d}.png"
        if save_dir is not None:
            cv2.imwrite(str(save_dir / "conditions" / filename), img_rgb[:, :, ::-1])

        c2w = _c2w_opencv_rows(yaw, pitch)
        frames.append({
            "id": idx + 1,
            "path": f"conditions/{filename}",
            "width": int(intr.width),
            "height": int(intr.height),
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.cx),
            "cy": float(intr.cy),
            "K": [
                [float(intr.fx), 0, float(intr.cx)],
                [0, float(intr.fy), float(intr.cy)],
                [0, 0, 1],
            ],
            "c2w": c2w,
        })

    frames.sort(key=lambda x: str(x["path"]))
    for i, fr in enumerate(frames):
        fr["id"] = i + 1

    transforms = {"frames": frames}
    if save_dir is not None:
        (save_dir / "transforms_condition.json").write_text(
            json.dumps(transforms, ensure_ascii=False, indent=2), encoding="utf-8",
        )

    return images, transforms


# ====================== High-level API =======================================

def postprocess_panorama(
    pano_bgr_u8: np.ndarray,
    depth_raw: np.ndarray,
    *,
    depth_scale: float = 100.0,
    pano_max_width: int = 8192,
    pano_max_height: int = 4096,
    cond_size: int = 504,
    cond_fx: float = 320.0,
    cond_fy: float = 320.0,
    save_dir: Optional[Path] = None,
) -> PostProcessResult:
    """Full post-processing: panorama + depth -> PLY arrays + conditions.

    When *save_dir* is given, PLY / conditions / JSON are also written to disk.
    """
    pano = resize_panorama(pano_bgr_u8, pano_max_width, pano_max_height)
    pano_h, pano_w = pano.shape[:2]
    depth = prepare_depth(depth_raw, depth_scale, pano_w, pano_h)

    xyz, rgb = compute_ply_arrays(pano, depth)
    if save_dir is not None:
        write_ply(save_dir / "pointcloud.ply", xyz, rgb)

    intr = Intrinsics(
        width=cond_size, height=cond_size,
        fx=cond_fx, fy=cond_fy,
        cx=cond_size / 2.0, cy=cond_size / 2.0,
    )
    cond_images, transforms = generate_conditions(pano, intr, save_dir=save_dir)

    return PostProcessResult(
        pano_bgr=pano,
        depth=depth,
        ply_xyz=xyz,
        ply_rgb=rgb,
        condition_images=cond_images,
        transforms=transforms,
    )
