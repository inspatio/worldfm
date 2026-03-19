"""
Image-to-Panorama generation via HunyuanWorld.

Modified from HunyuanWorld-1.0/demo_panogen.py

If hy3dworld is not pip-installed, call ``ensure_hy3dworld(path)`` with the
HunyuanWorld-1.0 repo root **before** instantiating Image2PanoramaDemo.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Try importing hy3dworld; fall back gracefully so the module itself can
# always be imported (e.g. for testing / inspection).
# ---------------------------------------------------------------------------
try:
    from hy3dworld import Image2PanoramaPipelines, Perspective
    from hy3dworld.AngelSlim.gemm_quantization_processor import FluxFp8GeMMProcessor
    from hy3dworld.AngelSlim.attention_quantization_processor import FluxFp8AttnProcessor2_0
    from hy3dworld.AngelSlim.cache_helper import DeepCacheHelper
except ImportError:
    Image2PanoramaPipelines = None  # type: ignore[assignment,misc]
    Perspective = None  # type: ignore[assignment,misc]
    FluxFp8GeMMProcessor = None  # type: ignore[assignment,misc]
    FluxFp8AttnProcessor2_0 = None  # type: ignore[assignment,misc]
    DeepCacheHelper = None  # type: ignore[assignment,misc]


def ensure_hy3dworld(path: str) -> None:
    """Add *path* (HunyuanWorld-1.0 repo root) to ``sys.path`` and import
    ``hy3dworld``.  No-op if hy3dworld is already importable."""
    global Image2PanoramaPipelines, Perspective
    global FluxFp8GeMMProcessor, FluxFp8AttnProcessor2_0, DeepCacheHelper

    if Image2PanoramaPipelines is not None:
        return

    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

    from hy3dworld import (  # noqa: F811
        Image2PanoramaPipelines as _Pipelines,
        Perspective as _Perspective,
    )
    from hy3dworld.AngelSlim.gemm_quantization_processor import (  # noqa: F811
        FluxFp8GeMMProcessor as _GeMMProc,
    )
    from hy3dworld.AngelSlim.attention_quantization_processor import (  # noqa: F811
        FluxFp8AttnProcessor2_0 as _AttnProc,
    )
    from hy3dworld.AngelSlim.cache_helper import (  # noqa: F811
        DeepCacheHelper as _CacheHelper,
    )

    Image2PanoramaPipelines = _Pipelines
    Perspective = _Perspective
    FluxFp8GeMMProcessor = _GeMMProc
    FluxFp8AttnProcessor2_0 = _AttnProc
    DeepCacheHelper = _CacheHelper


class Image2PanoramaDemo:
    def __init__(self, args):
        if Image2PanoramaPipelines is None:
            raise ImportError(
                "hy3dworld is not available. Either pip-install it, or call\n"
                "  modules.panogen.ensure_hy3dworld('/path/to/HunyuanWorld-1.0')\n"
                "before creating Image2PanoramaDemo."
            )

        self.args = args
        self.height, self.width = 960, 1920

        self.THETA = 0
        self.PHI = 0
        self.FOV = 80
        self.guidance_scale = 30
        self.num_inference_steps = 50
        self.true_cfg_scale = 2.0
        self.shifting_extend = 0
        self.blend_extend = 6

        self.lora_path = "tencent/HunyuanWorld-1"
        self.model_path = "black-forest-labs/FLUX.1-Fill-dev"

        self.pipe = Image2PanoramaPipelines.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.load_lora_weights(
            self.lora_path,
            subfolder="HunyuanWorld-PanoDiT-Image",
            weight_name="lora.safetensors",
            torch_dtype=torch.bfloat16,
        )
        self.pipe.fuse_lora()
        self.pipe.unload_lora_weights()
        try:
            self.pipe.enable_model_cpu_offload()
        except RuntimeError:
            self.pipe.to("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe.enable_vae_tiling()

        self.general_negative_prompt = (
            "human, person, people, messy,"
            "low-quality, blur, noise, low-resolution"
        )
        self.general_positive_prompt = "high-quality,  high-resolution, sharp, clear, 8k"

        if self.args.fp8_attention:
            self.pipe.transformer.set_attn_processor(FluxFp8AttnProcessor2_0())
        if self.args.fp8_gemm:
            FluxFp8GeMMProcessor(self.pipe.transformer)

    def run(
        self,
        prompt: str,
        negative_prompt: str,
        image_path: str,
        seed: int = 42,
        output_path: str = "output_panorama",
        *,
        save_to_disk: bool = True,
    ) -> Image.Image:
        """Generate a panorama from a perspective image.

        Returns the panorama as a PIL Image.  When *save_to_disk* is False the
        image is **not** written to ``output_path/panorama.png``.
        """
        prompt = prompt + ", " + self.general_positive_prompt
        negative_prompt = self.general_negative_prompt + ", " + negative_prompt

        perspective_img = cv2.imread(image_path)
        height_fov, width_fov = perspective_img.shape[:2]
        if width_fov > height_fov:
            ratio = width_fov / height_fov
            w = int((self.FOV / 360) * self.width)
            h = int(w / ratio)
            perspective_img = cv2.resize(
                perspective_img, (w, h), interpolation=cv2.INTER_AREA)
        else:
            ratio = height_fov / width_fov
            h = int((self.FOV / 180) * self.height)
            w = int(h / ratio)
            perspective_img = cv2.resize(
                perspective_img, (w, h), interpolation=cv2.INTER_AREA)

        equ = Perspective(perspective_img, self.FOV,
                          self.THETA, self.PHI, crop_bound=False)
        img, mask = equ.GetEquirec(self.height, self.width)
        mask = cv2.erode(mask.astype(np.uint8), np.ones(
            (3, 3), np.uint8), iterations=5)

        img = img * mask

        mask = mask.astype(np.uint8) * 255
        mask = 255 - mask

        mask = Image.fromarray(mask[:, :, 0])
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        helper = None
        if self.args.cache:
            helper = DeepCacheHelper(
                self.pipe.transformer,
                no_cache_steps=list(range(0, 10)) + list(range(10, 40, 3)) + list(range(40, 50)),
                no_cache_block_id={"single": [38]},
            )
            helper.start_timestep = 0
            helper.enable()

        image = self.pipe(
            prompt=prompt,
            image=img,
            mask_image=mask,
            height=self.height,
            width=self.width,
            negative_prompt=negative_prompt,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            blend_extend=self.blend_extend,
            shifting_extend=self.shifting_extend,
            true_cfg_scale=self.true_cfg_scale,
            helper=helper,
        ).images[0]

        if save_to_disk:
            os.makedirs(output_path, exist_ok=True)
            image.save(os.path.join(output_path, "panorama.png"))

        return image
