from __future__ import annotations

from typing import Tuple

import torch


def compute_plucker_rays(
    *,
    w2c: torch.Tensor,  # (B, V, 4, 4) camera<-world
    K: torch.Tensor,  # (B, V, 3, 3) pixel intrinsics in the SAME image space as image_hw
    image_hw: torch.Tensor,  # (B, 2) [H, W] in pixels
    token_hw: Tuple[int, int],  # (h_tokens, w_tokens_per_view)
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Return per-token default Plücker features (o x d, d) in world coordinates.

    Output:
      plucker: (B, V, h_tokens, w_tokens_per_view, 6)

    Notes:
    - Token centers are placed uniformly across image pixels:
        u = (x + 0.5) * (W / w_tokens)
        v = (y + 0.5) * (H / h_tokens)
      This matches patch-grid centers after resize+center-crop.
    - We assume w2c is camera<-world. We compute:
        Rcw = w2c[:3,:3], tcw = w2c[:3,3]
        cam_origin_world = -Rcw^T @ tcw
        d_world = normalize(Rcw^T @ d_cam)
    """
    B, V, _, _ = w2c.shape
    h_tokens, w_tokens = int(token_hw[0]), int(token_hw[1])
    assert w2c.shape == (B, V, 4, 4)
    assert K.shape == (B, V, 3, 3)
    assert image_hw.shape == (B, 2)

    # Build pixel grid in (u,v) with shape (h_tokens, w_tokens).
    # torch.linalg.inv doesn't support fp16 on some builds; do geometry in fp32 and cast back.
    compute_dtype = torch.float32 if dtype == torch.float16 else dtype

    H = image_hw[:, 0].to(device=device, dtype=compute_dtype).view(B, 1, 1)  # (B,1,1)
    W = image_hw[:, 1].to(device=device, dtype=compute_dtype).view(B, 1, 1)
    xs = (torch.arange(w_tokens, device=device, dtype=compute_dtype) + 0.5).view(1, 1, w_tokens)
    ys = (torch.arange(h_tokens, device=device, dtype=compute_dtype) + 0.5).view(1, h_tokens, 1)
    u = xs * (W / float(w_tokens))  # (B,1,w)
    v = ys * (H / float(h_tokens))  # (B,h,1)
    u = u.expand(B, h_tokens, w_tokens)
    v = v.expand(B, h_tokens, w_tokens)
    ones = torch.ones((B, h_tokens, w_tokens), device=device, dtype=compute_dtype)
    pix = torch.stack([u, v, ones], dim=-1)  # (B,h,w,3)

    # Per-view transforms.
    Rcw = w2c[..., :3, :3].to(device=device, dtype=compute_dtype)  # (B,V,3,3)
    tcw = w2c[..., :3, 3].to(device=device, dtype=compute_dtype)  # (B,V,3)
    Rwc = Rcw.transpose(-1, -2)  # (B,V,3,3)
    cam_o = -(Rwc @ tcw.unsqueeze(-1)).squeeze(-1)  # (B,V,3)

    Kinv = torch.linalg.inv(K.to(device=device, dtype=compute_dtype))  # (B,V,3,3)

    # d_cam: (B,V,h,w,3) = normalize(Kinv @ pix)
    # einsum: (B,V,3,3) x (B,h,w,3) -> (B,V,h,w,3)
    d_cam = torch.einsum("bvij,bhwj->bvhwi", Kinv, pix)
    d_cam = d_cam / (d_cam.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    d_world = torch.einsum("bvij,bvhwj->bvhwi", Rwc, d_cam)
    d_world = d_world / (d_world.norm(dim=-1, keepdim=True).clamp_min(1e-8))

    o = cam_o[:, :, None, None, :].expand(B, V, h_tokens, w_tokens, 3)
    o_cross_d = torch.cross(o, d_world, dim=-1)
    plucker = torch.cat([o_cross_d, d_world], dim=-1)  # (B,V,h,w,6)
    return plucker.to(dtype=dtype)


