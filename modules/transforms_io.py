"""
Camera-frame / transforms JSON I/O utilities.

Pure numpy — no external-repo dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    id: int
    width: int
    height: int
    K_3x3: np.ndarray   # float64 (3,3)
    c2w_4x4: np.ndarray  # float64 (4,4)


def scale_K_for_resize(
    K_3x3: np.ndarray,
    *,
    src_wh: tuple,
    dst_wh: tuple,
) -> np.ndarray:
    K = np.asarray(K_3x3, dtype=np.float64).copy()
    if K.shape != (3, 3):
        raise ValueError(f"K must be (3,3), got {K.shape}")
    sx = float(dst_wh[0]) / float(src_wh[0])
    sy = float(dst_wh[1]) / float(src_wh[1])
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return K


def _as_4x4(mat) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64)
    if m.shape == (4, 4):
        return m
    if m.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = m
        return out
    raise ValueError(f"Expected (4,4) or (3,4), got {m.shape}")


def _frame_K(frame: Dict[str, Any]) -> np.ndarray:
    if "K" in frame and frame["K"] is not None:
        K = np.asarray(frame["K"], dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"frame['K'] must be (3,3), got {K.shape}")
        return K
    return np.asarray([
        [float(frame["fx"]), 0.0, float(frame["cx"])],
        [0.0, float(frame["fy"]), float(frame["cy"])],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def load_camera_frames(transforms_json_path: str) -> List[CameraFrame]:
    """Load transforms_*.json -> list of CameraFrame."""
    p = Path(transforms_json_path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", [])
    out: List[CameraFrame] = []
    for i, fr in enumerate(frames):
        try:
            fid = int(fr.get("id", i + 1))
            w = int(fr["width"])
            h = int(fr["height"])
            K = _frame_K(fr)
            c2w = _as_4x4(fr["c2w"])
        except Exception as e:
            raise ValueError(f"Bad frame index={i} in {transforms_json_path}: {e}") from e
        out.append(CameraFrame(id=fid, width=w, height=h, K_3x3=K, c2w_4x4=c2w))
    return out


def load_camera_frames_from_dict(transforms_dict: Dict) -> List[CameraFrame]:
    """Same as load_camera_frames but from in-memory dict (no file I/O)."""
    frames = transforms_dict.get("frames", [])
    out: List[CameraFrame] = []
    for i, fr in enumerate(frames):
        fid = int(fr.get("id", i + 1))
        w = int(fr["width"])
        h = int(fr["height"])
        K = _frame_K(fr)
        c2w = _as_4x4(fr["c2w"])
        out.append(CameraFrame(id=fid, width=w, height=h, K_3x3=K, c2w_4x4=c2w))
    return out
