"""
Depth-based condition-view selector.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F

from modules.transforms_io import (
    CameraFrame,
    load_camera_frames,
    load_camera_frames_from_dict,
    scale_K_for_resize,
)


@dataclass(frozen=True)
class ConditionDB:
    cond_paths: List[str]     # len=V
    P_views: torch.Tensor     # (V,3,4) float32 on device
    depth_views: torch.Tensor  # (V,H,W) float32 on device
    C_views: torch.Tensor     # (V,3) float32 on device
    width: int
    height: int


def _as_4x4(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float32)
    if m.shape == (4, 4):
        return m
    if m.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = m
        return out
    raise ValueError(f"Expected (4,4) or (3,4), got {m.shape}")


def _P_from_K_c2w(K: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    c2w4 = _as_4x4(c2w)
    w2c = np.linalg.inv(c2w4).astype(np.float32)
    return (K.astype(np.float32) @ w2c[:3, :4]).astype(np.float32)


# ---------------------------------------------------------------------------
# build_condition_db: file-based variant (original interface)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def build_condition_db(
    *,
    scene_dir: str,
    transforms_condition_json: str,
    torch_renderer,
    device: torch.device,
    max_views: int = 0,
    profile: bool = False,
) -> ConditionDB:
    frames = load_camera_frames(transforms_condition_json)
    scene_dir_p = Path(scene_dir)
    cond_dir = scene_dir_p / "conditions"
    width = int(torch_renderer.width)
    height = int(torch_renderer.height)

    cond_paths: List[str] = []
    P_list: List[torch.Tensor] = []
    depth_list: List[torch.Tensor] = []
    C_list: List[torch.Tensor] = []
    skipped = 0

    t0 = time.perf_counter()
    with open(transforms_condition_json, "r", encoding="utf-8") as f:
        raw_frames = json.load(f).get("frames", [])

    for idx, fr in enumerate(frames):
        if 0 < max_views <= len(cond_paths):
            break
        pv = str(raw_frames[idx].get("path", "")) if idx < len(raw_frames) else ""
        bn = Path(pv).name
        p = cond_dir / bn
        if not p.exists():
            p2 = scene_dir_p / pv
            if p2.exists():
                p = p2
            else:
                skipped += 1
                continue
        cond_paths.append(str(p))

        K_use = scale_K_for_resize(fr.K_3x3, src_wh=(fr.width, fr.height), dst_wh=(width, height))
        P_list.append(torch.from_numpy(_P_from_K_c2w(K_use, fr.c2w_4x4)).to(device=device))
        C_list.append(torch.from_numpy(_as_4x4(fr.c2w_4x4)[:3, 3].astype(np.float32)).to(device=device))

        out_t = torch_renderer.render_torch(K_3x3=fr.K_3x3, c2w_4x4=fr.c2w_4x4, c2w_is_camera_to_world=True)
        depth_list.append(out_t.depth_f32)

    if not cond_paths:
        raise RuntimeError("No valid condition views found.")
    if skipped:
        print(f"[CondDB] skipped_missing_images={skipped}")

    P_views = torch.stack(P_list)
    depth_views = torch.stack(depth_list)
    C_views = torch.stack(C_list)

    if profile:
        torch.cuda.synchronize(device=device)
        print(f"[CondDB] views={len(cond_paths)} size=({height},{width}) "
              f"build_ms={(time.perf_counter()-t0)*1000:.2f}")

    return ConditionDB(cond_paths=cond_paths, P_views=P_views,
                       depth_views=depth_views, C_views=C_views,
                       width=width, height=height)


# ---------------------------------------------------------------------------
# build_condition_db_in_memory: works with in-memory condition images + transforms
# ---------------------------------------------------------------------------

@torch.inference_mode()
def build_condition_db_in_memory(
    *,
    condition_images: List[np.ndarray],
    transforms_dict: Dict,
    torch_renderer,
    device: torch.device,
) -> ConditionDB:
    """Build ConditionDB from in-memory data (no disk I/O).

    *condition_images*: list of (H,W,3) uint8 RGB arrays
    *transforms_dict*: {"frames": [...]} with K, c2w, width, height per frame
    """
    frames = load_camera_frames_from_dict(transforms_dict)
    width = int(torch_renderer.width)
    height = int(torch_renderer.height)

    P_list: List[torch.Tensor] = []
    depth_list: List[torch.Tensor] = []
    C_list: List[torch.Tensor] = []

    for fr in frames:
        K_use = scale_K_for_resize(fr.K_3x3, src_wh=(fr.width, fr.height), dst_wh=(width, height))
        P_list.append(torch.from_numpy(_P_from_K_c2w(K_use, fr.c2w_4x4)).to(device=device))
        C_list.append(torch.from_numpy(_as_4x4(fr.c2w_4x4)[:3, 3].astype(np.float32)).to(device=device))
        out_t = torch_renderer.render_torch(K_3x3=fr.K_3x3, c2w_4x4=fr.c2w_4x4, c2w_is_camera_to_world=True)
        depth_list.append(out_t.depth_f32)

    return ConditionDB(
        cond_paths=[f"mem:{i}" for i in range(len(frames))],
        P_views=torch.stack(P_list),
        depth_views=torch.stack(depth_list),
        C_views=torch.stack(C_list),
        width=width, height=height,
    )


# ---------------------------------------------------------------------------
# select_best_condition_index
# ---------------------------------------------------------------------------

@torch.inference_mode()
def select_best_condition_index(
    *,
    depth_cur: torch.Tensor,
    K_cur: Union[np.ndarray, torch.Tensor],
    c2w_cur: Union[np.ndarray, torch.Tensor],
    cond_db: ConditionDB,
    sample_grid: int = 10,
    center_grid: int = 15,
    center_frac: float = 0.5,
    uniform_sampling: bool = False,
    eps_rel: float = 0.02,
    eps_abs: float = 0.0,
    px_radius: int = 0,
    max_view_angle_deg: float = 180.0,
    use_distance_weight: bool = True,
    dist_min_m: float = 1.0,
    dist_max_m: float = 5.0,
    weight_near: float = 1.5,
    weight_far: float = 0.5,
    profile: bool = False,
) -> tuple:
    device = depth_cur.device
    H, W = int(depth_cur.shape[0]), int(depth_cur.shape[1])
    assert H == cond_db.height and W == cond_db.width

    t0 = time.perf_counter()

    def _to_np_f32(x) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    K_cur_np = _to_np_f32(K_cur)
    c2w_cur_np = _to_np_f32(c2w_cur)

    g_out = int(sample_grid)
    g_ctr = int(center_grid)
    frac = max(0.0, min(1.0, float(center_frac)))

    def _grid(x0, x1, nx, y0, y1, ny):
        xs = torch.linspace(float(x0), float(x1), int(nx), device=device)
        ys = torch.linspace(float(y0), float(y1), int(ny), device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.round(xx).to(torch.int64).view(-1), torch.round(yy).to(torch.int64).view(-1)

    if uniform_sampling or frac <= 0 or g_ctr <= 0:
        ui, vi = _grid(0, W - 1, g_out, 0, H - 1, g_out)
    else:
        ui_o, vi_o = _grid(0, W - 1, g_out, 0, H - 1, g_out)
        x0 = (1 - frac) * 0.5 * (W - 1)
        x1 = (1 + frac) * 0.5 * (W - 1)
        y0 = (1 - frac) * 0.5 * (H - 1)
        y1 = (1 + frac) * 0.5 * (H - 1)
        inside = (ui_o >= round(x0)) & (ui_o <= round(x1)) & (vi_o >= round(y0)) & (vi_o <= round(y1))
        ui_o, vi_o = ui_o[~inside], vi_o[~inside]
        ui_c, vi_c = _grid(x0, x1, g_ctr, y0, y1, g_ctr)
        ui = torch.cat([ui_o, ui_c])
        vi = torch.cat([vi_o, vi_c])
        idx = vi * W + ui
        uniq = torch.unique(idx)
        ui, vi = uniq % W, uniq // W

    depth_samples = depth_cur[vi, ui]
    valid = depth_samples > 0
    ui, vi, d = ui[valid], vi[valid], depth_samples[valid]
    if ui.numel() == 0:
        return 0, 0, 0

    K = torch.from_numpy(K_cur_np).to(device=device)
    K_inv = torch.linalg.inv(K)
    pix = torch.stack([ui.float(), vi.float(), torch.ones_like(d)], dim=1)
    X_cam = (pix @ K_inv.T) * d[:, None]

    c2w_t = torch.from_numpy(_as_4x4(c2w_cur_np)).to(device=device)
    w2c = torch.linalg.inv(c2w_t)
    R, t_vec = w2c[:3, :3], w2c[:3, 3]
    X_world = (X_cam - t_vec[None, :]) @ R

    M = X_world.shape[0]
    Xw_h = torch.cat([X_world, torch.ones((M, 1), device=device, dtype=torch.float32)], dim=1)

    P = cond_db.P_views
    proj = torch.einsum("vij,mj->vmi", P, Xw_h)
    z = proj[:, :, 2]
    u_proj = proj[:, :, 0] / (z + 1e-8)
    v_proj = proj[:, :, 1] / (z + 1e-8)
    ui2 = torch.round(u_proj).to(torch.int64)
    vi2 = torch.round(v_proj).to(torch.int64)
    inside_mask = (ui2 >= 0) & (ui2 < W) & (vi2 >= 0) & (vi2 < H) & (z > 0)

    depth_flat = cond_db.depth_views.view(cond_db.depth_views.shape[0], -1)
    r = int(px_radius)
    if r <= 0:
        idx_flat = (vi2 * W + ui2).clamp(0, H * W - 1)
        depth_at = torch.gather(depth_flat, 1, idx_flat)
        depth_ok = depth_at > 0
        min_abs_diff = torch.abs(depth_at - z)
        inside_any, depth_ok_any = inside_mask, depth_ok
    else:
        diffs, insides, oks = [], [], []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                un, vn = ui2 + dx, vi2 + dy
                ins_n = (un >= 0) & (un < W) & (vn >= 0) & (vn < H) & (z > 0)
                idn = (vn * W + un).clamp(0, H * W - 1)
                dn = torch.gather(depth_flat, 1, idn)
                ok_n = dn > 0
                diff_n = torch.where(ins_n & ok_n, torch.abs(dn - z), torch.full_like(dn, 1e9))
                diffs.append(diff_n); insides.append(ins_n); oks.append(ok_n)
        min_abs_diff = torch.stack(diffs, dim=-1).min(dim=-1).values
        inside_any = torch.stack(insides, dim=-1).any(dim=-1)
        depth_ok_any = torch.stack(oks, dim=-1).any(dim=-1)

    tol = torch.maximum(eps_rel * z, torch.full_like(z, eps_abs))
    consistent = min_abs_diff < tol

    if max_view_angle_deg >= 179.999:
        angle_ok = torch.ones_like(consistent, dtype=torch.bool)
    else:
        cos_thr = float(np.cos(np.deg2rad(max_view_angle_deg)))
        C_cur = c2w_t[:3, 3]
        v_cur = F.normalize(X_world - C_cur[None, :], dim=1, eps=1e-8)
        v_view = F.normalize(X_world[None, :, :] - cond_db.C_views[:, None, :], dim=2, eps=1e-8)
        angle_ok = (v_view * v_cur[None, :, :]).sum(dim=2) >= cos_thr

    hit = inside_any & depth_ok_any & consistent & angle_ok
    counts_i64 = hit.sum(dim=1)

    counts = counts_i64.float()
    if use_distance_weight:
        C_cur = c2w_t[:3, 3]
        dist = torch.linalg.norm(cond_db.C_views - C_cur[None, :], dim=1)
        d0, d1 = float(dist_min_m), float(dist_max_m)
        if d1 <= d0:
            d1 = d0 + 1e-6
        t_w = torch.clamp((dist - d0) / (d1 - d0), 0, 1)
        weight = float(weight_near) + (float(weight_far) - float(weight_near)) * t_w
        score = counts * weight
    else:
        score = counts

    best = int(torch.argmax(score).item())
    best_hits = int(counts_i64[best].item())
    samples = int(M)

    if profile:
        torch.cuda.synchronize(device=device)
        print(f"[CondSelect] views={P.shape[0]} samples={samples} best={best} "
              f"hits={best_hits} ms={(time.perf_counter()-t0)*1000:.2f}")

    return best, best_hits, samples
