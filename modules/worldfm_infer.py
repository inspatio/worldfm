"""
WorldFM tri-condition in-process inference.

Model components are provided by the worldfm.diffusion package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

if not hasattr(torch, "xpu"):
    torch.xpu = SimpleNamespace()
    torch.xpu.empty_cache = lambda: None
    torch.xpu.device_count = lambda: 0
    torch.xpu.is_available = lambda: False
    torch.xpu.synchronize = lambda: None
    torch.xpu.reset_peak_memory_stats = lambda x=None: None
    torch.xpu.max_memory_allocated = lambda x=None: 0
    torch.xpu.manual_seed = lambda x: None
    torch.xpu.manual_seed_all = lambda x: None
    torch.xpu._is_in_bad_fork = lambda: False

from diffusers.models import AutoencoderKL
from PIL import Image
from torchvision.transforms.functional import (
    InterpolationMode,
    center_crop,
    normalize,
    resize as tv_resize,
    to_tensor,
)

from worldfm.diffusion import DPMS, IDDPM
from worldfm.diffusion.model.nets import PixArtWorldFM_XL_2, PixArtWorldFMMS_XL_2
from worldfm.download import find_model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_pil_to_tensor(
    img: Image.Image,
    *,
    target_size_hw: tuple,
    interpolation: InterpolationMode = InterpolationMode.BICUBIC,
) -> torch.Tensor:
    """PIL -> (3,H,W) tensor in [-1,1]."""
    img = img.convert("RGB")
    w, h = img.size
    tgt_h, tgt_w = int(target_size_hw[0]), int(target_size_hw[1])
    scale = max(tgt_h / h, tgt_w / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    img = tv_resize(img, [new_h, new_w], interpolation=interpolation)
    img = center_crop(img, [tgt_h, tgt_w])
    t = to_tensor(img)
    return normalize(t, [0.5], [0.5])


def _preprocess_u8_tensor(
    rgb_u8: torch.Tensor,
    *,
    target_size_hw: tuple,
) -> torch.Tensor:
    """(H,W,3) uint8 GPU tensor -> (3,tgt_h,tgt_w) float in [-1,1]."""
    if rgb_u8.dtype != torch.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError(f"Expected (H,W,3) uint8, got {tuple(rgb_u8.shape)} {rgb_u8.dtype}")
    x = rgb_u8.permute(2, 0, 1).float() / 255.0
    x = x.unsqueeze(0)
    H, W = int(x.shape[2]), int(x.shape[3])
    tgt_h, tgt_w = int(target_size_hw[0]), int(target_size_hw[1])
    scale = max(tgt_h / H, tgt_w / W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    x = torch.nn.functional.interpolate(x, size=(new_h, new_w), mode="bicubic", align_corners=False)
    crop_y = max(0, (new_h - tgt_h) // 2)
    crop_x = max(0, (new_w - tgt_w) // 2)
    x = x[:, :, crop_y:crop_y + tgt_h, crop_x:crop_x + tgt_w].squeeze(0)
    return (x - 0.5) / 0.5


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WorldFMInprocessConfig:
    model_path: str
    vae_path: str
    image_size: int = 512
    image_height: int = 0   # 0 → falls back to image_size
    image_width: int = 0    # 0 → falls back to image_size
    version: str = "sigma"
    disable_cross_attn: bool = True
    step: int = 2
    mid_t: int = 200
    cfg_scale: float = 0.0
    device: str = "cuda"
    weight_dtype: torch.dtype = torch.float16
    profile: bool = False


# ---------------------------------------------------------------------------
# Inference service
# ---------------------------------------------------------------------------

class WorldFMTriConditionInprocess:
    """Minimal in-process tri-condition inference (bs=1)."""

    def __init__(self, cfg: WorldFMInprocessConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        _h = int(cfg.image_height) if cfg.image_height else int(cfg.image_size)
        _w = int(cfg.image_width) if cfg.image_width else int(cfg.image_size)
        self.target_hw = (_h, _w)

        max_sequence_length = {"alpha": 120, "sigma": 300}[cfg.version]
        latent_size = int(cfg.image_size // 8)
        pe_interpolation = cfg.image_size / 512
        micro_condition = cfg.version == "alpha" and cfg.image_size == 1024

        if cfg.image_size in (512, 1024, 2048) or cfg.version == "sigma":
            model = PixArtWorldFMMS_XL_2(
                input_size=latent_size,
                pe_interpolation=pe_interpolation,
                micro_condition=micro_condition,
                model_max_length=max_sequence_length,
                disable_cross_attn=cfg.disable_cross_attn,
                use_mask_channel=False,
            ).to(self.device)
        else:
            model = PixArtWorldFM_XL_2(
                input_size=latent_size,
                pe_interpolation=pe_interpolation,
                model_max_length=max_sequence_length,
                disable_cross_attn=cfg.disable_cross_attn,
                use_mask_channel=False,
            ).to(self.device)

        state_dict = find_model(cfg.model_path)
        model_sd = state_dict["state_dict"] if isinstance(state_dict, dict) and "state_dict" in state_dict else state_dict
        if isinstance(model_sd, dict) and "pos_embed" in model_sd:
            del model_sd["pos_embed"]
        model.load_state_dict(model_sd, strict=False)
        model.eval().to(cfg.weight_dtype)
        self.model = model

        vae = AutoencoderKL.from_pretrained(cfg.vae_path).to(self.device).to(cfg.weight_dtype)
        vae.eval()
        self.vae = vae

        self.max_sequence_length = max_sequence_length
        self._cond2_cached: Optional[torch.Tensor] = None
        self._cond2_latent_cached: Optional[torch.Tensor] = None
        self._cond2_candidates_paths: list = []
        self._cond2_candidates_tensor: Optional[torch.Tensor] = None
        self._cond2_candidates_latent: Optional[torch.Tensor] = None
        self._last_profile: dict = {}
        self._last_cond1_tensor: Optional[torch.Tensor] = None
        self._last_cond2_tensor: Optional[torch.Tensor] = None

    # ---- cond2 setters ----

    @torch.inference_mode()
    def set_cond2_from_path(self, cond2_path: str) -> None:
        img = Image.open(cond2_path).convert("RGB")
        t = _preprocess_pil_to_tensor(img, target_size_hw=self.target_hw)
        self._cond2_cached = t.unsqueeze(0).to(self.device).to(self.cfg.weight_dtype)
        self._cond2_latent_cached = None
        self._cond2_candidates_paths = []
        self._cond2_candidates_tensor = None
        self._cond2_candidates_latent = None
        self._last_cond2_tensor = self._cond2_cached

    @torch.inference_mode()
    def set_cond2_from_image(self, img: Image.Image) -> None:
        """In-memory variant of set_cond2_from_path (no disk I/O)."""
        img = img.convert("RGB")
        t = _preprocess_pil_to_tensor(img, target_size_hw=self.target_hw)
        self._cond2_cached = t.unsqueeze(0).to(self.device).to(self.cfg.weight_dtype)
        self._cond2_latent_cached = None
        self._cond2_candidates_paths = []
        self._cond2_candidates_tensor = None
        self._cond2_candidates_latent = None
        self._last_cond2_tensor = self._cond2_cached

    @torch.inference_mode()
    def set_cond2_from_array(self, rgb_u8: np.ndarray) -> None:
        """Set cond2 from (H,W,3) uint8 numpy array (no disk I/O)."""
        self.set_cond2_from_image(Image.fromarray(rgb_u8, mode="RGB"))

    @torch.inference_mode()
    def set_cond2_candidates_from_paths(self, cond2_paths: list, *, chunk: int = 8) -> None:
        paths = [str(p) for p in cond2_paths]
        if not paths:
            raise ValueError("cond2_paths is empty")
        self._cond2_candidates_paths = paths
        self._cond2_cached = None
        self._cond2_latent_cached = None

        tensors_cpu = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            t = _preprocess_pil_to_tensor(img, target_size_hw=self.target_hw)
            tensors_cpu.append(t)
        cond2 = torch.stack(tensors_cpu).to(self.device).to(self.cfg.weight_dtype)
        self._cond2_candidates_tensor = cond2

        vae_scale = getattr(self.vae.config, "scaling_factor", 0.13025)
        latents = []
        for i in range(0, cond2.shape[0], int(chunk)):
            x = cond2[i:i + int(chunk)]
            z = self.vae.encode(x).latent_dist.sample() * vae_scale
            latents.append(z)
        self._cond2_candidates_latent = torch.cat(latents)
        self._last_cond2_tensor = self._cond2_candidates_tensor[:1]

    # ---- inference ----

    @torch.inference_mode()
    def infer_from_render_u8(
        self,
        render_rgb_u8: torch.Tensor,
        *,
        cond2_index: Optional[int] = None,
        profile: bool = False,
        profile_tag: str = "",
    ) -> torch.Tensor:
        """(H,W,3) uint8 GPU tensor -> decoded (1,3,H,W) in [-1,1]."""
        if self._cond2_cached is None and self._cond2_candidates_tensor is None:
            raise RuntimeError("cond2 not set.")

        self._last_profile = {}
        use_profile = bool(profile or self.cfg.profile)

        def _sync():
            if torch.cuda.is_available() and self.device.type == "cuda":
                torch.cuda.synchronize(device=self.device)

        t0 = time.perf_counter() if use_profile else 0.0
        cond1 = _preprocess_u8_tensor(render_rgb_u8, target_size_hw=self.target_hw).unsqueeze(0)
        cond1 = cond1.to(self.cfg.weight_dtype)
        self._last_cond1_tensor = cond1

        if self._cond2_cached is not None:
            cond2 = self._cond2_cached
            z_c2: Optional[torch.Tensor] = None
        else:
            ci = int(cond2_index or 0)
            cond2 = self._cond2_candidates_tensor[ci:ci + 1]  # type: ignore
            z_c2 = self._cond2_candidates_latent[ci:ci + 1]  # type: ignore
        self._last_cond2_tensor = cond2

        if use_profile:
            _sync()
            self._last_profile["cond1_pre_ms"] = (time.perf_counter() - t0) * 1000

        vae_scale = getattr(self.vae.config, "scaling_factor", 0.13025)
        if use_profile:
            t1 = time.perf_counter()
        z_c1 = self.vae.encode(cond1).latent_dist.sample() * vae_scale
        if use_profile:
            _sync()
            self._last_profile["cond1_vae_ms"] = (time.perf_counter() - t1) * 1000

        if z_c2 is None:
            if self._cond2_latent_cached is None:
                if use_profile:
                    t2 = time.perf_counter()
                self._cond2_latent_cached = self.vae.encode(cond2).latent_dist.sample() * vae_scale
                if use_profile:
                    _sync()
                    self._last_profile["cond2_vae_ms"] = (time.perf_counter() - t2) * 1000
            z_c2 = self._cond2_latent_cached

        latent_h, latent_w = z_c1.shape[2], z_c1.shape[3]
        z = torch.randn(1, 4, latent_h, latent_w, device=self.device, dtype=self.cfg.weight_dtype)

        hw = torch.tensor([[float(self.target_hw[0]), float(self.target_hw[1])]], device=self.device, dtype=self.cfg.weight_dtype)
        ar = torch.tensor([[float(self.target_hw[0]) / float(self.target_hw[1])]], device=self.device, dtype=self.cfg.weight_dtype)
        caption_embs = torch.zeros(1, 1, self.max_sequence_length, 4096, device=self.device, dtype=self.cfg.weight_dtype)

        def model_fn_wrapper(x, timestep, cond=None, **kwargs):
            kw = kwargs.copy()
            kw["tri_condition"] = True
            return self.model.forward_with_dpmsolver(x, timestep, y=cond, **kw)

        model_kwargs = dict(
            data_info={"img_hw": hw, "aspect_ratio": ar},
            mask=None, tri_condition=True,
            cond1=z_c1, cond2=z_c2,
            debug_mask_log=False, use_cond2_cross_attn=False,
        )

        if int(self.cfg.step) == 1:
            diffusion = IDDPM("1000", learn_sigma=True, pred_sigma=True)
            alphas = torch.from_numpy(diffusion.alphas_cumprod).to(device=self.device, dtype=torch.float32)
            ts = torch.tensor([999], device=self.device, dtype=torch.long)
            out = model_fn_wrapper(z, ts, cond=caption_embs, **model_kwargs)
            eps = out.chunk(2, dim=1)[0] if out.shape[1] == 8 else out
            a = (alphas[ts] ** 0.5).view(-1, 1, 1, 1)
            s = ((1 - alphas[ts]) ** 0.5).view(-1, 1, 1, 1)
            samples = (a * z.float() - s * eps.float()).to(self.cfg.weight_dtype)
        else:
            diffusion = IDDPM("1000", learn_sigma=True, pred_sigma=True)
            alphas = torch.from_numpy(diffusion.alphas_cumprod).to(device=self.device, dtype=torch.float32)
            mid_t = int(self.cfg.mid_t)

            ts1 = torch.tensor([999], device=self.device, dtype=torch.long)
            if use_profile:
                t3 = time.perf_counter()
            out1 = model_fn_wrapper(z, ts1, cond=caption_embs, **model_kwargs)
            eps1 = out1.chunk(2, dim=1)[0] if out1.shape[1] == 8 else out1
            a1 = (alphas[ts1] ** 0.5).view(-1, 1, 1, 1)
            s1 = ((1 - alphas[ts1]) ** 0.5).view(-1, 1, 1, 1)
            pred_x0 = a1 * z.float() - s1 * eps1.float()
            if use_profile:
                _sync()
                self._last_profile["dmd_step1_ms"] = (time.perf_counter() - t3) * 1000

            ts_mid = torch.tensor([mid_t], device=self.device, dtype=torch.long)
            am = (alphas[ts_mid] ** 0.5).view(-1, 1, 1, 1)
            sm = ((1 - alphas[ts_mid]) ** 0.5).view(-1, 1, 1, 1)
            noisy = (am * pred_x0 + sm * torch.randn_like(pred_x0)).to(self.cfg.weight_dtype)

            if use_profile:
                t4 = time.perf_counter()
            out2 = model_fn_wrapper(noisy, ts_mid, cond=caption_embs, **model_kwargs)
            eps2 = out2.chunk(2, dim=1)[0] if out2.shape[1] == 8 else out2
            samples = (am * noisy.float() - sm * eps2.float()).to(self.cfg.weight_dtype)
            if use_profile:
                _sync()
                self._last_profile["dmd_step2_ms"] = (time.perf_counter() - t4) * 1000

        if use_profile:
            t5 = time.perf_counter()
        decoded = self.vae.decode(samples / vae_scale).sample
        if use_profile:
            _sync()
            self._last_profile["vae_decode_ms"] = (time.perf_counter() - t5) * 1000
            self._last_profile["total_ms"] = sum(v for k, v in self._last_profile.items() if k.endswith("_ms"))
        return decoded

    @torch.inference_mode()
    def infer_from_render_u8_multistep(
        self,
        render_rgb_u8: torch.Tensor,
        *,
        sample_steps: int,
        cfg_scale: float = 4.5,
        cond2_index: Optional[int] = None,
    ) -> torch.Tensor:
        """Multi-step sampling (e.g. 14 steps) via DPM solver."""
        if int(sample_steps) <= 0:
            raise ValueError(f"sample_steps must be > 0, got {sample_steps}")
        if self._cond2_cached is None and self._cond2_candidates_tensor is None:
            raise RuntimeError("cond2 not set.")

        cond1 = _preprocess_u8_tensor(render_rgb_u8, target_size_hw=self.target_hw).unsqueeze(0)
        cond1 = cond1.to(self.cfg.weight_dtype)
        self._last_cond1_tensor = cond1

        if self._cond2_cached is not None:
            cond2 = self._cond2_cached
            z_c2: Optional[torch.Tensor] = None
        else:
            ci = int(cond2_index or 0)
            cond2 = self._cond2_candidates_tensor[ci:ci + 1]  # type: ignore
            z_c2 = self._cond2_candidates_latent[ci:ci + 1]  # type: ignore
        self._last_cond2_tensor = cond2

        vae_scale = getattr(self.vae.config, "scaling_factor", 0.13025)
        z_c1 = self.vae.encode(cond1).latent_dist.sample() * vae_scale
        if z_c2 is None:
            if self._cond2_latent_cached is None:
                self._cond2_latent_cached = self.vae.encode(cond2).latent_dist.sample() * vae_scale
            z_c2 = self._cond2_latent_cached

        latent_h, latent_w = z_c1.shape[2], z_c1.shape[3]
        z = torch.randn(1, 4, latent_h, latent_w, device=self.device, dtype=self.cfg.weight_dtype)
        hw = torch.tensor([[float(self.target_hw[0]), float(self.target_hw[1])]], device=self.device, dtype=self.cfg.weight_dtype)
        ar = torch.tensor([[float(self.target_hw[0]) / float(self.target_hw[1])]], device=self.device, dtype=self.cfg.weight_dtype)
        caption_embs = torch.zeros(1, 1, self.max_sequence_length, 4096, device=self.device, dtype=self.cfg.weight_dtype)
        null_y = caption_embs.clone()

        def model_fn_wrapper(x, timestep, cond=None, **kwargs):
            kw = kwargs.copy()
            c1 = kw.get("cond1")
            c2 = kw.get("cond2")
            if c1 is not None and c2 is not None and x.shape[0] != c1.shape[0]:
                r = x.shape[0] // c1.shape[0]
                kw["cond1"] = c1.repeat(r, 1, 1, 1)
                kw["cond2"] = c2.repeat(r, 1, 1, 1)
            kw["tri_condition"] = True
            kw["use_cond2_cross_attn"] = False
            kw["debug_mask_log"] = False
            return self.model.forward_with_dpmsolver(x, timestep, y=cond, **kw)

        model_kwargs = dict(
            data_info={"img_hw": hw, "aspect_ratio": ar},
            mask=None, tri_condition=True,
            cond1=z_c1, cond2=z_c2,
            debug_mask_log=False, use_cond2_cross_attn=False,
        )

        dpm_solver = DPMS(
            model_fn_wrapper,
            condition=caption_embs,
            uncondition=null_y,
            cfg_scale=float(cfg_scale),
            model_kwargs=model_kwargs,
        )
        samples = dpm_solver.sample(z, steps=int(sample_steps), order=2,
                                    skip_type="time_uniform", method="multistep")
        samples = samples.to(self.cfg.weight_dtype)
        return self.vae.decode(samples / vae_scale).sample

    # ---- debug helpers ----

    def debug_get_cond2_tensor(self) -> torch.Tensor:
        if self._last_cond2_tensor is None:
            raise RuntimeError("cond2 not set")
        return self._last_cond2_tensor

    def debug_get_last_cond1_tensor(self) -> torch.Tensor:
        if self._last_cond1_tensor is None:
            raise RuntimeError("no cond1 cached; run infer_from_render_u8 first")
        return self._last_cond1_tensor
