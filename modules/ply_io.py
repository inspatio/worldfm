"""
PLY point-cloud I/O.

Pure numpy — no external-repo dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal, Optional

import numpy as np

_PLY_SCALAR_TO_DTYPE = {
    "char": ("i1", 1), "int8": ("i1", 1),
    "uchar": ("u1", 1), "uint8": ("u1", 1),
    "short": ("i2", 2), "int16": ("i2", 2),
    "ushort": ("u2", 2), "uint16": ("u2", 2),
    "int": ("i4", 4), "int32": ("i4", 4),
    "uint": ("u4", 4), "uint32": ("u4", 4),
    "float": ("f4", 4), "float32": ("f4", 4),
    "double": ("f8", 8), "float64": ("f8", 8),
}


@dataclass(frozen=True)
class _PlyHeader:
    fmt: Literal["ascii", "binary_little_endian", "binary_big_endian"]
    vertex_count: int
    vertex_props: list  # [(dtype_name, prop_name), ...]
    header_bytes: int


def _readline_ascii(f: BinaryIO) -> str:
    b = f.readline()
    return b.decode("utf-8", errors="replace").rstrip("\r\n") if b else ""


def _parse_header(ply_path: str) -> _PlyHeader:
    p = Path(ply_path)
    with p.open("rb") as f:
        first = _readline_ascii(f).strip()
        if first != "ply":
            raise ValueError(f"Not a PLY file: {ply_path}")

        fmt: Optional[str] = None
        vertex_count: Optional[int] = None
        in_vertex = False
        vertex_props: list = []

        while True:
            line = _readline_ascii(f)
            if line == "":
                raise ValueError(f"Unexpected EOF in PLY header: {ply_path}")
            if line.startswith("format "):
                m = re.match(r"^format\s+(\S+)\s+(\S+)\s*$", line)
                if not m:
                    raise ValueError(f"Bad format line: {line!r}")
                fmt = m.group(1)
                if fmt not in ("ascii", "binary_little_endian", "binary_big_endian"):
                    raise ValueError(f"Unsupported PLY format: {fmt}")
            elif line.startswith("element "):
                m = re.match(r"^element\s+(\S+)\s+(\d+)\s*$", line)
                if not m:
                    raise ValueError(f"Bad element line: {line!r}")
                in_vertex = (m.group(1) == "vertex")
                if in_vertex:
                    vertex_count = int(m.group(2))
            elif line.startswith("property ") and in_vertex:
                if line.startswith("property list "):
                    continue
                m = re.match(r"^property\s+(\S+)\s+(\S+)\s*$", line)
                if not m:
                    continue
                dtype_name, prop_name = m.group(1), m.group(2)
                if dtype_name in _PLY_SCALAR_TO_DTYPE:
                    vertex_props.append((dtype_name, prop_name))
            elif line == "end_header":
                header_bytes = f.tell()
                break

    if fmt is None:
        raise ValueError(f"Missing format in PLY: {ply_path}")
    if vertex_count is None or vertex_count <= 0:
        raise ValueError(f"Missing/invalid vertex count: {ply_path}")
    if not vertex_props:
        raise ValueError(f"No vertex properties: {ply_path}")
    return _PlyHeader(fmt=fmt, vertex_count=vertex_count,  # type: ignore[arg-type]
                      vertex_props=vertex_props, header_bytes=header_bytes)


def _dtype_for_vertex_props(props: list, *, endian: str) -> np.dtype:
    fields = []
    for dtype_name, prop_name in props:
        code, _ = _PLY_SCALAR_TO_DTYPE[dtype_name]
        fields.append((prop_name, endian + code))
    return np.dtype(fields)


def load_ply_xyz_rgb(ply_path: str) -> tuple:
    """Load PLY -> (xyz float32 (N,3), rgb float32 (N,3) in [0,1])."""
    hdr = _parse_header(ply_path)
    props = hdr.vertex_props
    names = [n for _, n in props]
    if not all(k in names for k in ("x", "y", "z")):
        raise ValueError(f"PLY missing x/y/z: {ply_path}")

    color_keys = None
    for cand in (("red", "green", "blue"), ("r", "g", "b")):
        if all(k in names for k in cand):
            color_keys = cand
            break
    if color_keys is None:
        raise ValueError(f"PLY missing RGB color: {ply_path}")

    if hdr.fmt == "ascii":
        xyz_list, rgb_list = [], []
        idx_x, idx_y, idx_z = names.index("x"), names.index("y"), names.index("z")
        idx_r, idx_g, idx_b = (names.index(color_keys[0]),
                                names.index(color_keys[1]),
                                names.index(color_keys[2]))
        with Path(ply_path).open("rb") as f:
            f.seek(hdr.header_bytes)
            for _ in range(hdr.vertex_count):
                line = _readline_ascii(f)
                if not line:
                    break
                parts = line.split()
                if len(parts) < len(names):
                    continue
                xyz_list.append((float(parts[idx_x]), float(parts[idx_y]), float(parts[idx_z])))
                rgb_list.append((float(parts[idx_r]), float(parts[idx_g]), float(parts[idx_b])))
        xyz = np.asarray(xyz_list, dtype=np.float32)
        rgb = np.asarray(rgb_list, dtype=np.float32)
    else:
        endian = "<" if hdr.fmt == "binary_little_endian" else ">"
        dt = _dtype_for_vertex_props(props, endian=endian)
        with Path(ply_path).open("rb") as f:
            f.seek(hdr.header_bytes)
            data = np.fromfile(f, dtype=dt, count=hdr.vertex_count)
        xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
        r, g, b = data[color_keys[0]], data[color_keys[1]], data[color_keys[2]]
        rgb = np.stack([r, g, b], axis=1).astype(np.float32)

    if rgb.size == 0:
        raise ValueError(f"Empty vertex data: {ply_path}")
    if float(np.nanmax(rgb)) > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    return xyz, rgb
