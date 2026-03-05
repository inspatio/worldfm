"""
CUDA point-cloud renderer (pure PyTorch, no EGL/OpenGL).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch


@dataclass(frozen=True)
class RenderOutput:
    rgb_u8: np.ndarray       # (H,W,3) uint8
    depth_f32: np.ndarray    # (H,W) float32, 0 = invalid


@dataclass(frozen=True)
class RenderOutputTorch:
    rgb_u8: torch.Tensor     # (H,W,3) uint8 on device
    depth_f32: torch.Tensor  # (H,W) float32 on device


class TorchPointCloudRenderer:
    """Fast splatting/projection renderer for conditioning images + depth."""

    def __init__(
        self,
        *,
        points_xyz: np.ndarray,
        points_rgb: np.ndarray,
        width: int,
        height: int,
        device: str = "cuda",
        near: float = 1e-3,
        far: float = 1e6,
        z_soft_alpha: float = 500.0,
        mode: str = "fast",
        use_fp16_cache: bool = True,
        axis_flip: Optional[np.ndarray] = None,
        max_points: Optional[int] = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.device = torch.device(device)
        self.near = float(near)
        self.far = float(far)
        self.z_soft_alpha = float(z_soft_alpha)
        self.mode = str(mode).lower()
        if self.mode not in ("fast", "softmin"):
            raise ValueError("mode must be: fast or softmin")
        self.use_fp16_cache = bool(use_fp16_cache)

        xyz = np.asarray(points_xyz, dtype=np.float32)
        rgb = np.asarray(points_rgb, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"points_xyz must be (N,3), got {xyz.shape}")
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(f"points_rgb must be (N,3), got {rgb.shape}")
        if xyz.shape[0] != rgb.shape[0]:
            raise ValueError(f"N mismatch: {xyz.shape[0]} vs {rgb.shape[0]}")

        self.num_points_total = int(xyz.shape[0])
        if max_points is not None and xyz.shape[0] > int(max_points):
            step = int(np.ceil(xyz.shape[0] / int(max_points)))
            xyz = xyz[::step]
            rgb = rgb[::step]
        self.num_points_used = int(xyz.shape[0])

        rgb = np.clip(rgb, 0.0, 1.0)
        if self.use_fp16_cache:
            self._xyz = torch.from_numpy(xyz).to(self.device).half()
            self._rgb = torch.from_numpy(rgb).to(self.device).half()
        else:
            self._xyz = torch.from_numpy(xyz).to(self.device)
            self._rgb = torch.from_numpy(rgb).to(self.device)

        if axis_flip is None:
            self._axis_flip = None
        else:
            A = np.asarray(axis_flip, dtype=np.float32)
            if A.shape != (4, 4):
                raise ValueError(f"axis_flip must be (4,4), got {A.shape}")
            self._axis_flip = torch.from_numpy(A).to(self.device)

    @torch.inference_mode()
    def render(
        self,
        *,
        K_3x3: np.ndarray,
        c2w_4x4: np.ndarray,
        c2w_is_camera_to_world: bool = True,
        return_torch: bool = False,
        point_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> Union[RenderOutput, RenderOutputTorch]:
        K = torch.tensor(np.asarray(K_3x3, dtype=np.float32), device=self.device)
        if K.shape != (3, 3):
            raise ValueError(f"K must be (3,3), got {tuple(K.shape)}")
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        c2w = torch.tensor(np.asarray(c2w_4x4, dtype=np.float32), device=self.device)
        if c2w.shape == (3, 4):
            c2w4 = torch.eye(4, device=self.device, dtype=torch.float32)
            c2w4[:3, :4] = c2w
            c2w = c2w4
        if c2w.shape != (4, 4):
            raise ValueError(f"c2w must be (4,4) or (3,4), got {tuple(c2w.shape)}")

        w2c = torch.linalg.inv(c2w) if c2w_is_camera_to_world else c2w
        if self._axis_flip is not None:
            w2c = self._axis_flip @ w2c

        R = w2c[:3, :3]
        t = w2c[:3, 3]

        xyz = self._xyz
        rgb0 = self._rgb
        if point_ranges is not None:
            parts_xyz, parts_rgb = [], []
            for s, e in point_ranges:
                s_i, e_i = max(0, int(s)), min(int(xyz.shape[0]), int(e))
                if e_i > s_i:
                    parts_xyz.append(xyz[s_i:e_i])
                    parts_rgb.append(rgb0[s_i:e_i])
            if not parts_xyz:
                z_img = torch.zeros((self.height, self.width, 3), device=self.device, dtype=torch.uint8)
                z_d = torch.zeros((self.height, self.width), device=self.device, dtype=torch.float32)
                return RenderOutputTorch(rgb_u8=z_img, depth_f32=z_d) if return_torch else RenderOutput(
                    rgb_u8=z_img.cpu().numpy(), depth_f32=z_d.cpu().numpy())
            xyz = parts_xyz[0] if len(parts_xyz) == 1 else torch.cat(parts_xyz)
            rgb0 = parts_rgb[0] if len(parts_rgb) == 1 else torch.cat(parts_rgb)

        if xyz.dtype == torch.float16 and R.dtype != torch.float16:
            X = (xyz @ R.half().T) + t.half()
        else:
            X = (xyz @ R.T) + t
        z = X[:, 2]
        valid = (z > self.near) & (z < self.far) & torch.isfinite(z)
        X, z, rgb_v = X[valid], z[valid], rgb0[valid]

        z_f = z.float()
        inv_z = 1.0 / z_f
        u = fx * (X[:, 0].float() * inv_z) + cx
        v = fy * (X[:, 1].float() * inv_z) + cy
        ui = torch.round(u).to(torch.int64)
        vi = torch.round(v).to(torch.int64)
        inside = (ui >= 0) & (ui < self.width) & (vi >= 0) & (vi < self.height)
        ui, vi, z_f, rgb_v = ui[inside], vi[inside], z_f[inside], rgb_v[inside]

        H, W = self.height, self.width
        P = H * W
        idx = vi * W + ui

        if self.mode == "softmin":
            inf = torch.tensor(float("inf"), device=self.device, dtype=torch.float32)
            min_z = torch.full((P,), inf, device=self.device, dtype=torch.float32)
            min_z.scatter_reduce_(0, idx, z_f, reduce="amin", include_self=True)
            dz = torch.clamp(z_f - min_z[idx], min=0.0)
            w = torch.exp(-self.z_soft_alpha * dz).float()
            acc_rgb = torch.zeros((P, 3), device=self.device, dtype=torch.float32)
            acc_w = torch.zeros((P,), device=self.device, dtype=torch.float32)
            acc_rgb.index_add_(0, idx, rgb_v.float() * w[:, None])
            acc_w.index_add_(0, idx, w)
            rgb_img = torch.clamp(acc_rgb / (acc_w[:, None] + 1e-8), 0, 1)
            rgb_img = (rgb_img.view(H, W, 3) * 255).to(torch.uint8)
            depth = min_z.view(H, W)
            depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))
        else:
            image_flat = torch.zeros((P, 3), device=self.device, dtype=torch.uint8)
            inf = torch.tensor(float("inf"), device=self.device, dtype=torch.float32)
            depth_flat = torch.full((P,), inf, device=self.device, dtype=torch.float32)
            depth_flat.scatter_reduce_(0, idx, z_f, reduce="amin", include_self=True)
            minz_at = depth_flat[idx]
            update = torch.abs(z_f - minz_at) <= 1e-4
            if update.any():
                upd_idx = idx[update]
                upd_rgb = (rgb_v[update].float().clamp(0, 1) * 255).to(torch.uint8)
                for c in range(3):
                    image_flat[:, c].scatter_(0, upd_idx, upd_rgb[:, c])
            rgb_img = image_flat.view(H, W, 3)
            depth = depth_flat.view(H, W)
            depth = torch.where(torch.isfinite(depth), depth, torch.zeros_like(depth))

        if return_torch:
            return RenderOutputTorch(rgb_u8=rgb_img, depth_f32=depth)
        return RenderOutput(
            rgb_u8=rgb_img.detach().cpu().numpy(),
            depth_f32=depth.detach().cpu().numpy().astype(np.float32),
        )

    @torch.inference_mode()
    def render_torch(
        self,
        *,
        K_3x3: np.ndarray,
        c2w_4x4: np.ndarray,
        c2w_is_camera_to_world: bool = True,
        point_ranges: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> RenderOutputTorch:
        out = self.render(
            K_3x3=K_3x3, c2w_4x4=c2w_4x4,
            c2w_is_camera_to_world=c2w_is_camera_to_world,
            return_torch=True, point_ranges=point_ranges,
        )
        assert isinstance(out, RenderOutputTorch)
        return out
