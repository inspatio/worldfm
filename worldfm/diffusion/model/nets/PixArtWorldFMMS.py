# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# Modified from PixArt-Sigma's repos:
#   https://github.com/PixArt-alpha/PixArt-sigma/blob/master/diffusion/model/nets/PixArtMS.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import os
import traceback
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from worldfm.diffusion.model.builder import MODELS
from worldfm.diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from worldfm.diffusion.model.nets.PixArtWorldFM_blocks import t2i_modulate, CaptionEmbedder, AttentionKVCompress, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, SizeEmbedder
from worldfm.diffusion.model.nets.PixArtWorldFM import PixArtWorldFM, get_2d_sincos_pos_embed
from worldfm.diffusion.model.nets.plucker import compute_plucker_rays
from worldfm.diffusion.model.nets.prope import get_rope_coeffs_2d, prepare_prope_apply_fns, reorder_tokens_to_camera_major, reorder_tokens_from_camera_major
from worldfm.diffusion.utils.logger import get_root_logger


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class PixArtWorldFMMSBlock(nn.Module):
    """
    A PixArtWorldFMMS block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., input_size=None,
                 sampling=None, sr_ratio=1, qk_norm=False, disable_cross_attn=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionKVCompress(
            hidden_size, num_heads=num_heads, qkv_bias=True, sampling=sampling, sr_ratio=sr_ratio,
            qk_norm=qk_norm, **block_kwargs
        )
        self.cross_attn = MultiHeadCrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # to be compatible with lower version pytorch
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size ** 0.5)
        self.disable_cross_attn = disable_cross_attn

    def forward(self, x, y, t, mask=None, HW=None, block_id=None, **kwargs):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
        x_in = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + self.drop_path(
            gate_msa
            * self.attn(
                x_in,
                HW=HW,
                block_id=block_id,
                use_prope=bool(kwargs.get("use_prope", False)),
                prope_viewmats=kwargs.get("prope_viewmats", None),
                prope_Ks=kwargs.get("prope_Ks", None),
                prope_image_hw=kwargs.get("prope_image_hw", None),
                prope_cache=kwargs.get("prope_cache", None),
            )
        )
        if not self.disable_cross_attn:
            cond2_tokens = kwargs.get("cond2_tokens", None)
            if cond2_tokens is not None:
                # In cond2-cross-attn mode, attend to cond2 tokens (reference image).
                # IMPORTANT: MultiHeadCrossAttention flattens batch into a single sequence internally;
                # we must pass a BlockDiagonalMask seqlen list to prevent cross-sample mixing.
                if cond2_tokens.shape[0] != B:
                    raise ValueError(f"cond2_tokens batch mismatch: x.B={B}, cond2_tokens.B={cond2_tokens.shape[0]}")
                cond2_seqlen = int(cond2_tokens.shape[1])
                mask_list = [cond2_seqlen] * B
                attn_out = self.cross_attn(x, cond2_tokens, mask_list)
                scale = float(kwargs.get("cond2_cross_attn_scale", 1.0))
                if scale != 1.0:
                    attn_out = attn_out * scale
                # Optional debug: sample stats (rank0/first steps controlled by PixArtWorldFMMS.forward)
                if bool(kwargs.get("debug_cond2_stats", False)) and (block_id == 0):
                    try:
                        # rank0 + step gating
                        _rank0 = True
                        try:
                            import torch.distributed as dist
                            if dist.is_available() and dist.is_initialized():
                                _rank0 = (dist.get_rank() == 0)
                        except Exception:
                            _rank0 = True
                        debug_step = kwargs.get("debug_step", None)
                        debug_steps = int(kwargs.get("debug_steps", 1) or 1)
                        if (not _rank0) or (debug_step is not None and int(debug_step) >= debug_steps):
                            raise RuntimeError("skip")
                        logger = get_root_logger()
                        with torch.no_grad():
                            td = attn_out.detach().reshape(-1)
                            n = int(td.numel())
                            k = min(4096, n) if n > 0 else 0
                            if k > 0:
                                if k < n:
                                    step = max(n // k, 1)
                                    idx = torch.arange(0, n, step, device=td.device, dtype=torch.long)
                                    if idx.numel() > k:
                                        idx = idx[:k]
                                    if idx.numel() > 0:
                                        idx = torch.clamp(idx, 0, n - 1)
                                    samp = td.index_select(0, idx)
                                else:
                                    samp = td
                                logger.info(
                                    f"[DebugCond2] cross_attn_out(sample): shape={tuple(attn_out.shape)} sample_n={k} "
                                    f"mean={samp.mean().item():.6f} std={samp.to(dtype=torch.float32).std().item():.6f} "
                                    f"min={samp.min().item():.6f} max={samp.max().item():.6f} "
                                    f"nan={bool(torch.isnan(samp).any().item())} inf={bool(torch.isinf(samp).any().item())}"
                                )
                    except Exception:
                        pass
                x = x + attn_out
            else:
                x = x + self.cross_attn(x, y, mask)

        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


#############################################################################
#                           Core PixArtWorldFMMS Model                             #
#################################################################################
@MODELS.register_module()
class PixArtWorldFMMS(PixArtWorldFM):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            learn_sigma=True,
            pred_sigma=True,
            drop_path: float = 0.,
            caption_channels=4096,
            pe_interpolation=1.,
            config=None,
            model_max_length=120,
            micro_condition=False,
            qk_norm=False,
            kv_compress_config=None,
            disable_cross_attn=False,
            **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            learn_sigma=learn_sigma,
            pred_sigma=pred_sigma,
            drop_path=drop_path,
            pe_interpolation=pe_interpolation,
            config=config,
            model_max_length=model_max_length,
            qk_norm=qk_norm,
            kv_compress_config=kv_compress_config,
            **kwargs,
        )
        self.h = self.w = 0
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        # Check if mask channel injection is enabled via kwargs
        self.use_mask_channel = kwargs.get('use_mask_channel', False)
        if self.use_mask_channel:
            in_channels = in_channels + 1
        self.x_embedder = PatchEmbed(patch_size, in_channels, hidden_size, bias=True)
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob, act_layer=approx_gelu, token_num=model_max_length)
        # Normalize cond2 tokens before cross-attn to reduce distribution shift / scale issues.
        # elementwise_affine=False -> no extra parameters, safe for checkpoint compatibility.
        self.cond2_token_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # NOTE: Cross-attention can be disabled via `disable_cross_attn`.
        # In tri-condition mode we can optionally inject cond2 through cross-attention (enabled by `use_cond2_cross_attn`).
        self.micro_conditioning = micro_condition
        if self.micro_conditioning:
            self.csize_embedder = SizeEmbedder(hidden_size//3)  # c_size embed
            self.ar_embedder = SizeEmbedder(hidden_size//3)     # aspect ratio embed
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        if kv_compress_config is None:
            kv_compress_config = {
                'sampling': None,
                'scale_factor': 1,
                'kv_compress_layer': [],
            }
        self.blocks = nn.ModuleList([
            PixArtWorldFMMSBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],
                input_size=(input_size // patch_size, input_size // patch_size),
                sampling=kv_compress_config['sampling'],
                sr_ratio=int(kv_compress_config['scale_factor']) if i in kv_compress_config['kv_compress_layer'] else 1,
                qk_norm=qk_norm,
                disable_cross_attn=disable_cross_attn,
            )
            for i in range(depth)
        ])
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)
        # Optional pose injection (Plücker) projector: 6-dim -> hidden_size
        self.plucker_proj = nn.Sequential(
            nn.Linear(6, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        # PRoPE caches (to avoid per-layer re-precompute)
        self._prope_coeff_cache: dict = {}
        self._prope_apply_cache_key = None
        self._prope_apply_cache_val = None
        self._prope_cache_debug_printed = False
        # Pos-embed caches (avoid numpy->torch generation every forward)
        self._pos_embed_cache: dict = {}
        self._cond2_pos_embed_cache: dict = {}

        self.initialize()

    def forward(self, x, timestep, y, mask=None, data_info=None, split_output=False, **kwargs):
        """
        Forward pass of PixArtWorldFMMS.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        split_output: If True, splits the output sequence in half and returns only the first half.
                      Used when input x contains [target, condition] concatenated in width.
        """
        # Automatic Condition Concatenation logic
        # This handles the case where x is the target (4ch) but conditions are passed in kwargs
        # This allows diffusion/model/gaussian_diffusion.py to remain generic
        use_cond2_cross_attn = bool(kwargs.get("use_cond2_cross_attn", False))
        debug_cond2_stats = bool(kwargs.get("debug_cond2_stats", False))
        debug_step = kwargs.get("debug_step", None)
        debug_steps = int(kwargs.get("debug_steps", 1) or 1)
        # rank0-only debug to avoid 4x log spam in DDP
        _rank0 = True
        if debug_cond2_stats:
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    _rank0 = (dist.get_rank() == 0)
            except Exception:
                _rank0 = True

        def _should_debug() -> bool:
            if not debug_cond2_stats or (not _rank0):
                return False
            if debug_step is None:
                return True
            try:
                return int(debug_step) < int(debug_steps)
            except Exception:
                return True

        def _sample_stats(name: str, t: torch.Tensor, sample_k: int = 4096) -> None:
            # No stateful "print once" flags here (checkpoint recompute must see same code path).
            if not _should_debug():
                return
            logger = get_root_logger()
            with torch.no_grad():
                td = t.detach()
                flat = td.reshape(-1)
                n = int(flat.numel())
                if n <= 0:
                    logger.info(f"[DebugCond2] {name}: empty")
                    return
                k_target = min(int(sample_k), n)
                if k_target < n:
                    # Use integer indexing only (avoid linspace float rounding -> out-of-range indices on CUDA).
                    step = max(n // k_target, 1)
                    idx = torch.arange(0, n, step, device=flat.device, dtype=torch.long)
                    if idx.numel() > k_target:
                        idx = idx[:k_target]
                    # Safety: ensure max index < n
                    if idx.numel() > 0:
                        idx = torch.clamp(idx, 0, n - 1)
                    samp = flat.index_select(0, idx)
                else:
                    samp = flat
                k = int(samp.numel())
                # stats on sample to avoid huge temporary allocations
                mean = samp.mean().item()
                # use float32 only on sample (small)
                std = samp.to(dtype=torch.float32).std().item() if samp.numel() > 1 else 0.0
                mn = samp.min().item()
                mx = samp.max().item()
                nan = bool(torch.isnan(samp).any().item())
                inf = bool(torch.isinf(samp).any().item())
                logger.info(f"[DebugCond2] {name}: shape={tuple(t.shape)} sample_n={k} mean={mean:.6f} std={std:.6f} min={mn:.6f} max={mx:.6f} nan={nan} inf={inf}")
        cond2_tokens = None
        if kwargs.get('tri_condition', False):
            cond1 = kwargs['cond1']
            cond2 = kwargs['cond2']
            cond1_mask_latent = kwargs.get('cond1_mask_latent', None)
            cond1_mask_inject_mode = kwargs.get('cond1_mask_inject_mode', 'add')
            # Only concat if not already concatenated (check width or explicit flag if needed)
            # Here we assume if x matches cond width, it's likely just the target
            if x.shape[-1] == cond1.shape[-1]:
                if use_cond2_cross_attn:
                    # Cond2 cross attention mode: cond2 is NOT concatenated into the self-attn stream.
                    # Only concat x and cond1 (and mask stripe if applicable).
                    cond2_to_embed = None
                    if cond1_mask_latent is not None and cond1_mask_inject_mode == 'channel':
                        # Channel injection: concat mask as an extra channel to cond1
                        cond1_with_mask = torch.cat([cond1, cond1_mask_latent], dim=1)  # [B, 5, H, W]
                        x_padded = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)  # [B, 5, H, W]
                        x = torch.cat([x_padded, cond1_with_mask], dim=3)
                        kwargs['split_num'] = 2
                        # Process cond2 separately for cross-attn (also need to pad to 5 channels)
                        cond2_to_embed = torch.cat([cond2, torch.zeros_like(cond2[:, :1])], dim=1)  # [B, 5, H, W]
                    elif cond1_mask_latent is not None:
                        # Width concat mode with mask
                        x = torch.cat([x, cond1, cond1_mask_latent], dim=3)
                        kwargs['split_num'] = 3
                        cond2_to_embed = cond2  # [B, 4, H, W]
                    else:
                        x = torch.cat([x, cond1], dim=3)
                        kwargs['split_num'] = 2
                        cond2_to_embed = cond2  # [B, 4, H, W]
                    split_output = True
                    # Embed cond2 for cross-attn tokens
                    if cond2_to_embed is not None:
                        cond2_to_embed = cond2_to_embed.to(self.dtype)
                        # Get spatial dimensions for cond2
                        cond2_h, cond2_w = cond2_to_embed.shape[2], cond2_to_embed.shape[3]
                        cond2_h_tokens = cond2_h // self.patch_size
                        cond2_w_tokens = cond2_w // self.patch_size
                        # Embed cond2: x_embedder returns (B, N, D) when flatten=True (default)
                        cond2_embed = self.x_embedder(cond2_to_embed)  # (B, N_cond2, D) where D=hidden_size
                        # Add positional embedding for cond2 (cached)
                        embed_dim = cond2_embed.shape[-1]
                        cache_key = (int(cond2_h_tokens), int(cond2_w_tokens), int(embed_dim), str(cond2_embed.device), str(cond2_embed.dtype), float(self.pe_interpolation), int(self.base_size))
                        cond2_pos_embed = self._cond2_pos_embed_cache.get(cache_key, None)
                        if cond2_pos_embed is None:
                            cond2_pos_embed = torch.from_numpy(
                                get_2d_sincos_pos_embed(
                                    embed_dim, (cond2_h_tokens, cond2_w_tokens),
                                    pe_interpolation=self.pe_interpolation, base_size=self.base_size
                                )
                            ).unsqueeze(0).to(cond2_embed.device).to(self.dtype)
                            self._cond2_pos_embed_cache[cache_key] = cond2_pos_embed
                        cond2_tokens = cond2_embed + cond2_pos_embed  # (B, N_cond2, D)
                        cond2_tokens = self.cond2_token_norm(cond2_tokens)
                        cond2_tokens = cond2_tokens.contiguous()
                        _sample_stats("cond2_tokens(after_norm)", cond2_tokens)
                elif cond1_mask_latent is not None and cond1_mask_inject_mode == 'channel':
                    # Channel injection: concat mask as an extra channel to cond1
                    # cond1_mask_latent should be [B, 1, H, W] (single channel mask)
                    # cond1 is [B, 4, H, W], after concat becomes [B, 5, H, W]
                    cond1_with_mask = torch.cat([cond1, cond1_mask_latent], dim=1)  # [B, 5, H, W]
                    # In channel mode, we need to pad x and cond2 to 5 channels to match cond1_with_mask
                    # Pad with zeros: [B, 4, H, W] -> [B, 5, H, W]
                    x_padded = torch.cat([x, torch.zeros_like(x[:, :1])], dim=1)  # [B, 5, H, W]
                    cond2_padded = torch.cat([cond2, torch.zeros_like(cond2[:, :1])], dim=1)  # [B, 5, H, W]
                    x = torch.cat([x_padded, cond1_with_mask, cond2_padded], dim=3)
                    kwargs['split_num'] = 3
                    split_output = True
                elif cond1_mask_latent is not None:
                    # Width concat mode (original concat behavior)
                    # Mask is provided as an extra "stripe" (same shape as cond1 latent: [B,4,H,W]).
                    # We concatenate in width to keep in_channels=4 and remain pretrained-compatible.
                    x = torch.cat([x, cond1, cond1_mask_latent, cond2], dim=3)
                    kwargs['split_num'] = 4
                    split_output = True
                else:
                    x = torch.cat([x, cond1, cond2], dim=3)
                    kwargs['split_num'] = 3
                    split_output = True
        elif kwargs.get('two_condition', False):
            cond2 = kwargs['cond2']
            if x.shape[-1] == cond2.shape[-1]:
                x = torch.cat([x, cond2], dim=3)
                kwargs['split_num'] = 2
                split_output = True

        # Check if split_num is in kwargs and extract it, removing it from kwargs to avoid passing to blocks
        split_num = kwargs.pop('split_num', 2)

        # Clean up kwargs for checkpointing to prevent "Unexpected keyword arguments" error
        # These arguments might be passed via model_kwargs but are not accepted by block/checkpoint
        keys_to_pop = [
            'split_output', 'split_num',
            'tri_condition', 'cond1', 'cond2', 'cond1_mask_latent',
            'two_condition',
            'condition_latent', 'cond1_latent', 'cond2_latent',
            'use_cond2_cross_attn',
        ]
        for k in keys_to_pop:
            if k in kwargs:
                kwargs.pop(k)

        # Pass cond2 cross-attn tokens to blocks if available
        if cond2_tokens is not None:
            kwargs["cond2_tokens"] = cond2_tokens

        # Extract pose/plücker flags early (we inject into tokens before transformer blocks).
        use_plucker = bool(kwargs.pop("use_plucker", False))
        plucker_viewmats = kwargs.pop("plucker_viewmats", None)
        plucker_Ks = kwargs.pop("plucker_Ks", None)
        plucker_image_hw = kwargs.pop("plucker_image_hw", None)

        # If PRoPE is enabled, default `prope_image_hw` from data_info so the caller
        # only needs to pass camera matrices (viewmats/Ks).
        if (
            kwargs.get("use_prope", False)
            and ("prope_viewmats" in kwargs)
            and ("prope_image_hw" not in kwargs)
            and (data_info is not None)
            and ("img_hw" in data_info)
        ):
            kwargs["prope_image_hw"] = data_info["img_hw"]

        bs = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        self.h, self.w = x.shape[-2]//self.patch_size, x.shape[-1]//self.patch_size
        # pos embed (cached; varies with concat width so cache key includes h/w)
        pos_cache_key = (int(self.h), int(self.w), int(self.pos_embed.shape[-1]), str(x.device), str(x.dtype), float(self.pe_interpolation), int(self.base_size))
        pos_embed = self._pos_embed_cache.get(pos_cache_key, None)
        if pos_embed is None:
            pos_embed = torch.from_numpy(
                get_2d_sincos_pos_embed(
                    self.pos_embed.shape[-1], (self.h, self.w), pe_interpolation=self.pe_interpolation,
                    base_size=self.base_size
                )
            ).unsqueeze(0).to(x.device).to(self.dtype)
            self._pos_embed_cache[pos_cache_key] = pos_embed

        x = self.x_embedder(x) + pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        _sample_stats("x_tokens(after_patch+pos)", x)

        # No extra HW needed for cond2 cross-attn mode (cond2 tokens are used only in cross-attn).

        # === PRoPE precompute/cache (shared across blocks within this forward) ===
        # We precompute apply_fn_q/kv/o and reorder fns once per forward to avoid
        # repeating projection-matrix work in each attention block.
        if (
            kwargs.get("use_prope", False)
            and ("prope_viewmats" in kwargs)
            and (kwargs.get("prope_viewmats", None) is not None)
            and bool(kwargs.get("use_prope_cache", True))
        ):
            try:
                vm_in = kwargs.get("prope_viewmats", None)
                Ks_in = kwargs.get("prope_Ks", None)
                ihw_in = kwargs.get("prope_image_hw", None)
                if ihw_in is None and data_info is not None and ("img_hw" in data_info):
                    ihw_in = data_info["img_hw"]
                cameras = int(vm_in.shape[1])
                patches_y = int(self.h)
                patches_x_total = int(self.w)
                prope_cache = None
                if cameras > 0 and (patches_x_total % cameras == 0):
                    patches_x = patches_x_total // cameras
                    # Attention dtype: match block attention dtype (fp32_attention => float32)
                    attn_dtype = self.dtype
                    try:
                        first_attn = self.blocks[0].attn
                        if getattr(first_attn, "fp32_attention", False):
                            attn_dtype = torch.float32
                    except Exception:
                        pass
                    # Head dim must match AttentionKVCompress internal reshape:
                    # q: (B,N,num_heads, head_dim) where head_dim = hidden_size // num_heads
                    try:
                        num_heads = int(self.blocks[0].attn.num_heads)
                    except Exception:
                        num_heads = int(getattr(self, "num_heads", 1))
                    head_dim = int(x.shape[-1] // max(num_heads, 1))
                    # Cache key: identity-based (fast). In inference, tensors are reused so this hits.
                    cache_key = (
                        id(vm_in),
                        id(Ks_in) if Ks_in is not None else None,
                        id(ihw_in) if ihw_in is not None else None,
                        patches_x,
                        patches_y,
                        head_dim,
                        str(x.device),
                        str(attn_dtype),
                    )
                    if cache_key == self._prope_apply_cache_key:
                        prope_cache = self._prope_apply_cache_val
                    else:
                        # RoPE coeff cache by (patches_x,patches_y,head_dim,device,dtype)
                        coeff_key = (patches_x, patches_y, head_dim, str(x.device), str(attn_dtype))
                        if coeff_key not in self._prope_coeff_cache:
                            self._prope_coeff_cache[coeff_key] = get_rope_coeffs_2d(
                                patches_x=patches_x,
                                patches_y=patches_y,
                                head_dim=head_dim,
                                device=x.device,
                                dtype=attn_dtype,
                            )
                        coeffs_x, coeffs_y = self._prope_coeff_cache[coeff_key]
                        if ihw_in is None:
                            ihw_in = torch.tensor([[patches_y, patches_x]], device=x.device, dtype=torch.float32).repeat(x.shape[0], 1)
                        apply_fn_q, apply_fn_kv, apply_fn_o = prepare_prope_apply_fns(
                            head_dim=head_dim,
                            viewmats=vm_in.to(device=x.device, dtype=attn_dtype),
                            Ks=Ks_in.to(device=x.device, dtype=attn_dtype) if Ks_in is not None else None,
                            patches_x=patches_x,
                            patches_y=patches_y,
                            image_hw=ihw_in.to(device=x.device, dtype=torch.float32),
                            coeffs_x=coeffs_x,
                            coeffs_y=coeffs_y,
                        )
                        # reorder fns for merged-width <-> camera-major
                        def _reorder_to(t: torch.Tensor) -> torch.Tensor:
                            return reorder_tokens_to_camera_major(
                                t,
                                cameras=cameras,
                                patches_y=patches_y,
                                patches_x_total=patches_x_total,
                                is_bnhd=True,
                            )

                        def _reorder_from(t: torch.Tensor) -> torch.Tensor:
                            return reorder_tokens_from_camera_major(
                                t,
                                cameras=cameras,
                                patches_y=patches_y,
                                patches_x_total=patches_x_total,
                                is_bnhd=True,
                            )

                        prope_cache = {
                            "apply_fn_q": apply_fn_q,
                            "apply_fn_kv": apply_fn_kv,
                            "apply_fn_o": apply_fn_o,
                            "reorder_to": _reorder_to,
                            "reorder_from": _reorder_from,
                        }
                        self._prope_apply_cache_key = cache_key
                        self._prope_apply_cache_val = prope_cache
                kwargs["prope_cache"] = prope_cache
            except Exception:
                # If anything goes wrong, just disable cache and fall back to per-layer path.
                kwargs["prope_cache"] = None
                if os.environ.get("PROPE_CACHE_DEBUG", "0") == "1":
                    try:
                        import torch.distributed as dist
                        rank0 = (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)
                    except Exception:
                        rank0 = True
                    if rank0:
                        logger = get_root_logger()
                        logger.info("[PRoPE][cache][error] PixArtWorldFMMS.forward failed to prepare shared prope_cache; will fall back to per-layer compute.\n" + traceback.format_exc())
        else:
            # Explicitly disable shared cache (for A/B testing) or missing camera params.
            kwargs["prope_cache"] = None

        # === Plücker pose token injection (LVSM-style) ===
        # We add a learned projection of per-token Plücker features (o×d, d) to x.
        # Token ordering is row-major over the *merged width* (target|cond1|cond2).
        if use_plucker:
            if plucker_viewmats is None or plucker_Ks is None:
                raise ValueError("use_plucker=True but plucker_viewmats/plucker_Ks not provided.")
            if plucker_image_hw is None:
                # default from data_info if available
                if data_info is None or ("img_hw" not in data_info):
                    raise ValueError("use_plucker=True but plucker_image_hw not provided and data_info['img_hw'] missing.")
                plucker_image_hw = data_info["img_hw"]
            # Shapes: viewmats (B,V,4,4), Ks (B,V,3,3), image_hw (B,2)
            B = x.shape[0]
            V = int(plucker_viewmats.shape[1])
            h_tokens = int(self.h)
            w_tokens_total = int(self.w)
            if w_tokens_total % V != 0:
                raise ValueError(f"Plücker expects merged-width tokens divisible by views. w_tokens_total={w_tokens_total}, V={V}")
            w_tokens_per_view = w_tokens_total // V
            plucker = compute_plucker_rays(
                w2c=plucker_viewmats,
                K=plucker_Ks,
                image_hw=plucker_image_hw,
                token_hw=(h_tokens, w_tokens_per_view),
                device=x.device,
                dtype=x.dtype,
            )  # (B,V,h,w,6)
            plucker = plucker.reshape(B, V, h_tokens, w_tokens_per_view, 6)
            plucker = plucker.permute(0, 2, 1, 3, 4).contiguous()  # (B,h,V,w,6) -> matches merged width
            plucker = plucker.view(B, h_tokens, w_tokens_total, 6).view(B, h_tokens * w_tokens_total, 6)
            x = x + self.plucker_proj(plucker)
        t = self.t_embedder(timestep)  # (N, D)

        if self.micro_conditioning:
            c_size, ar = data_info['img_hw'].to(self.dtype), data_info['aspect_ratio'].to(self.dtype)
            csize = self.csize_embedder(c_size, bs)  # (N, D)
            ar = self.ar_embedder(ar, bs)  # (N, D)
            t = t + torch.cat([csize, ar], dim=1)

        t0 = self.t_block(t)

        debug_mask_log = bool(kwargs.get("debug_mask_log", False))
        y = self.y_embedder(y, self.training)  # (N, D)

        if mask is not None:
            # Debug logs are OFF by default. Enable by passing `debug_mask_log=True` into model kwargs.
            if debug_mask_log:
                logger = get_root_logger()
                logger.info(f"[MaskDebug] Original mask shape: {mask.shape}, y shape: {y.shape}, mask dtype: {mask.dtype}")
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
            if debug_mask_log and (not hasattr(self, '_y_lens_debug_logged')):
                self._y_lens_debug_logged = True
                logger = get_root_logger()
                logger.info(f"[YLenDebug] y_lens: {y_lens[:5]}... (total {len(y_lens)} samples)")
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        for i, block in enumerate(self.blocks):
            # Keep grad-checkpointing ON even in debug mode; debug functions above do not mutate state
            # and only operate on detached samples, so recomputation graph stays consistent.
            x = auto_grad_checkpoint(block, x, y, t0, y_lens, (self.h, self.w), block_id=i, **kwargs)  # (N, T, D)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        # print(f"[DEBUG] After final_layer x shape: {x.shape}")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        _sample_stats("out_latent(before_split)", x)

        if split_output:
            # Ensure valid split_num
            split_num = max(1, split_num)
            chunk_width = x.shape[-1] // split_num
            if chunk_width == 0:
                raise ValueError(f"Invalid chunk_width computed: split_num={split_num}, width={x.shape[-1]}")
            x = x[..., :chunk_width]

        return x

    def forward_with_dpmsolver(self, x, timestep, y, data_info, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, data_info=data_info, **kwargs)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, data_info, mask=None, **kwargs):
        """
        Forward pass of PixArtWorldFMMS, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask, data_info=data_info, **kwargs)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        assert self.h * self.w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], self.h, self.w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, self.h * p, self.w * p))
        return imgs

    def initialize(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        if self.micro_conditioning:
            nn.init.normal_(self.csize_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.csize_embedder.mlp[2].weight, std=0.02)
            nn.init.normal_(self.ar_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.ar_embedder.mlp[2].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArtWorldFMMS blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


#################################################################################
#                            PixArtWorldFMMS Configs                                #
#################################################################################
@MODELS.register_module()
def PixArtWorldFMMS_XL_2(**kwargs):
    return PixArtWorldFMMS(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
