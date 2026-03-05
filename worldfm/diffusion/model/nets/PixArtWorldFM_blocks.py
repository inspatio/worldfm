# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# Modified from PixArt-Sigma's repos:
#   https://github.com/PixArt-alpha/PixArt-sigma/blob/master/diffusion/model/nets/PixArt_blocks.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import xformers.ops
import os
from einops import rearrange
from timm.models.vision_transformer import Mlp, Attention as Attention_
from worldfm.diffusion.utils.logger import get_root_logger


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0., proj_drop=0., **block_kwargs):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model*2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query: img tokens; key/value: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        attn_bias = None
        if mask is not None:
            # Debug: check types (OFF by default; enable with env CROSS_ATTN_MASK_DEBUG=1)
            if (os.environ.get("CROSS_ATTN_MASK_DEBUG", "0") == "1") and (not hasattr(self, '_cross_attn_mask_debug_logged')):
                self._cross_attn_mask_debug_logged = True
                logger = get_root_logger()
                logger.info(f"[CrossAttnMask] N={N}, type(N)={type(N)}, B={B}, type(B)={type(B)}")
                if hasattr(mask, 'shape'):
                    logger.info(f"[CrossAttnMask] mask type: {type(mask)}, mask shape: {mask.shape}, mask dtype: {mask.dtype}")
                    if mask.numel() <= 10:
                        logger.info(f"[CrossAttnMask] mask value: {mask.tolist()}")
                    else:
                        logger.info(f"[CrossAttnMask] mask value (first 5): {mask[:5].tolist()}")
                else:
                    logger.info(f"[CrossAttnMask] mask type: {type(mask)}, mask value: {mask}")
                seqlens_list = [N] * B
                logger.info(f"[CrossAttnMask] [N] * B = {seqlens_list[:5]}... (total {len(seqlens_list)}), type of first element: {type(seqlens_list[0])}")
            # Ensure N is int, not tensor
            N_int = int(N) if isinstance(N, torch.Tensor) else N
            B_int = int(B) if isinstance(B, torch.Tensor) else B
            # BlockDiagonalMask.from_seqlens expects kv_seqlen to be a list, not a tensor
            if isinstance(mask, torch.Tensor):
                # Convert tensor to list
                mask_list = mask.tolist()
            elif isinstance(mask, list):
                mask_list = mask
            else:
                mask_list = None
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N_int] * B_int, mask_list)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentionKVCompress(Attention_):
    """Multi-head Attention block with KV token compression and qk norm."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        sampling='conv',
        sr_ratio=1,
        qk_norm=False,
        **block_kwargs,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, **block_kwargs)

        self.sampling=sampling    # ['conv', 'ave', 'uniform', 'uniform_every']
        self.sr_ratio = sr_ratio
        if sr_ratio > 1 and sampling == 'conv':
            # Avg Conv Init.
            self.sr = nn.Conv2d(dim, dim, groups=dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr.weight.data.fill_(1/sr_ratio**2)
            self.sr.bias.data.zero_()
            self.norm = nn.LayerNorm(dim)
        if qk_norm:
            self.q_norm = nn.LayerNorm(dim)
            self.k_norm = nn.LayerNorm(dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def downsample_2d(self, tensor, H, W, scale_factor, sampling=None):
        if sampling is None or scale_factor == 1:
            return tensor
        B, N, C = tensor.shape

        if sampling == 'uniform_every':
            return tensor[:, ::scale_factor], int(N // scale_factor)

        tensor = tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
        new_H, new_W = int(H / scale_factor), int(W / scale_factor)
        new_N = new_H * new_W

        if sampling == 'ave':
            tensor = F.interpolate(
                tensor, scale_factor=1 / scale_factor, mode='nearest'
            ).permute(0, 2, 3, 1)
        elif sampling == 'uniform':
            tensor = tensor[:, :, ::scale_factor, ::scale_factor].permute(0, 2, 3, 1)
        elif sampling == 'conv':
            tensor = self.sr(tensor).reshape(B, C, -1).permute(0, 2, 1)
            tensor = self.norm(tensor)
        else:
            raise ValueError

        return tensor.reshape(B, new_N, C).contiguous(), new_N

    def forward(
        self,
        x,
        mask=None,
        HW=None,
        block_id=None,
        *,
        kv_tokens=None,  # optional: asymmetric self-attn KV stream (Q comes from `x`)
        HW_kv=None,      # optional: spatial HW for kv_tokens (only used for KV compression)
        # PRoPE (Projective Relative Positional Encoding) - optional
        use_prope: bool = False,
        prope_viewmats=None,  # (B, cameras, 4, 4)
        prope_Ks=None,  # (B, cameras, 3, 3) or None
        prope_image_hw=None,  # (B, 2) with [H, W] in pixels
        prope_cache=None,  # dict from PixArtWorldFMMS.forward with apply fns + reorder fns
    ):
        B, Nq, C = x.shape
        x_kv = x if kv_tokens is None else kv_tokens
        Nk = int(x_kv.shape[1])
        new_N = Nk

        if HW is None:
            Hq = Wq = int(Nq ** 0.5)
        else:
            Hq, Wq = HW
        if HW_kv is None:
            Hk, Wk = Hq, Wq
        else:
            Hk, Wk = HW_kv

        # Compute QKV from KV stream (prefix contains the Q tokens). When kv_tokens is provided:
        # - Q uses the first Nq tokens
        # - K,V use all Nk tokens
        qkv = self.qkv(x_kv).reshape(B, Nk, 3, C)
        q_all, k, v = qkv.unbind(2)
        # Slice for queries may produce non-contiguous view; make it contiguous to keep attention kernels happy.
        q = q_all[:, :Nq].contiguous()
        k = k.contiguous()
        v = v.contiguous()
        dtype = q.dtype
        q = self.q_norm(q)
        k = self.k_norm(k)

        # KV compression
        if self.sr_ratio > 1:
            k, new_N = self.downsample_2d(k, Hk, Wk, self.sr_ratio, sampling=self.sampling)
            v, new_N = self.downsample_2d(v, Hk, Wk, self.sr_ratio, sampling=self.sampling)

        q = q.reshape(B, Nq, self.num_heads, C // self.num_heads).to(dtype)
        k = k.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.reshape(B, new_N, self.num_heads, C // self.num_heads).to(dtype)

        use_fp32_attention = getattr(self, 'fp32_attention', False)     # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # -----------------------
        # PRoPE: only for self-attn without KV compression (new_N == N)
        # Token ordering in this repo's tri-condition is "concat on width", so we
        # must reorder tokens to camera-major before applying PRoPE and invert it
        # afterwards.
        # -----------------------
        _prope_apply_fn_o = None
        _prope_reorder_back = None
        if (
            use_prope
            and (prope_viewmats is not None)
            and (new_N == Nq)
            and (Nk == Nq)
            and (HW is not None)
        ):
            try:
                # Debug logger: only rank0 prints; gated by env to avoid overhead.
                _debug = os.environ.get("PROPE_CACHE_DEBUG", "0") == "1"
                _rank0 = True
                if _debug:
                    try:
                        import torch.distributed as dist
                        if dist.is_available() and dist.is_initialized():
                            _rank0 = (dist.get_rank() == 0)
                    except Exception:
                        _rank0 = True
                # Print at most once per module to avoid slowing training with log I/O.
                _logged_key = "_prope_cache_debug_logged"
                _already_logged = bool(getattr(self, _logged_key, False))
                _log_this = bool(_debug and _rank0 and (block_id == 0) and (not _already_logged))

                cameras = int(prope_viewmats.shape[1])
                if cameras > 0 and int(W) % cameras == 0:
                    patches_y = int(H)
                    patches_x_total = int(W)
                    patches_x = patches_x_total // cameras
                    # Fast path: use precomputed apply_fns and reorder fns if provided.
                    apply_fn_q = apply_fn_kv = apply_fn_o = None
                    reorder_to = reorder_from = None
                    if isinstance(prope_cache, dict):
                        apply_fn_q = prope_cache.get("apply_fn_q", None)
                        apply_fn_kv = prope_cache.get("apply_fn_kv", None)
                        apply_fn_o = prope_cache.get("apply_fn_o", None)
                        reorder_to = prope_cache.get("reorder_to", None)
                        reorder_from = prope_cache.get("reorder_from", None)

                    if apply_fn_q is None or apply_fn_kv is None or apply_fn_o is None or reorder_to is None or reorder_from is None:
                        if _log_this:
                            logger = get_root_logger()
                            logger.info(
                                f"[PRoPE][cache][fallback] block0 did NOT receive valid prope_cache -> per-layer prepare_prope_apply_fns (slow). "
                                f"HW=({patches_y},{patches_x_total}) cameras={cameras} head_dim={q.shape[-1]} dtype={q.dtype} device={q.device}"
                            )
                            setattr(self, _logged_key, True)
                        # Fallback: per-layer compute (old behavior).
                        from functools import partial as _partial
                        from worldfm.diffusion.model.nets.prope import (
                            get_rope_coeffs_2d,
                            prepare_prope_apply_fns,
                            reorder_tokens_to_camera_major,
                            reorder_tokens_from_camera_major,
                        )

                        cache_key = (patches_x, patches_y, q.shape[-1], str(q.device), str(q.dtype))
                        coeff_cache = getattr(self, "_prope_coeff_cache", None)
                        if coeff_cache is None:
                            coeff_cache = {}
                            setattr(self, "_prope_coeff_cache", coeff_cache)
                        if cache_key not in coeff_cache:
                            coeff_cache[cache_key] = get_rope_coeffs_2d(
                                patches_x=patches_x,
                                patches_y=patches_y,
                                head_dim=q.shape[-1],
                                device=q.device,
                                dtype=q.dtype,
                            )
                        (coeffs_x, coeffs_y) = coeff_cache[cache_key]

                        if prope_image_hw is None:
                            prope_image_hw = torch.tensor(
                                [[patches_y, patches_x]], device=q.device, dtype=torch.float32
                            ).repeat(B, 1)

                        apply_fn_q, apply_fn_kv, apply_fn_o = prepare_prope_apply_fns(
                            head_dim=q.shape[-1],
                            viewmats=prope_viewmats.to(device=q.device, dtype=q.dtype),
                            Ks=prope_Ks.to(device=q.device, dtype=q.dtype) if prope_Ks is not None else None,
                            patches_x=patches_x,
                            patches_y=patches_y,
                            image_hw=prope_image_hw.to(device=q.device, dtype=torch.float32),
                            coeffs_x=coeffs_x,
                            coeffs_y=coeffs_y,
                        )
                        reorder_to = _partial(
                            reorder_tokens_to_camera_major,
                            cameras=cameras,
                            patches_y=patches_y,
                            patches_x_total=patches_x_total,
                            is_bnhd=True,
                        )
                        reorder_from = _partial(
                            reorder_tokens_from_camera_major,
                            cameras=cameras,
                            patches_y=patches_y,
                            patches_x_total=patches_x_total,
                            is_bnhd=True,
                        )
                    else:
                        if _log_this:
                            logger = get_root_logger()
                            logger.info(
                                f"[PRoPE][cache][fastpath] block0 received shared prope_cache -> reuse apply_fns across ALL self-attn blocks this step. "
                                f"HW=({patches_y},{patches_x_total}) cameras={cameras} head_dim={q.shape[-1]} dtype={q.dtype} device={q.device}"
                            )
                            setattr(self, _logged_key, True)

                    # Convert to (B, heads, seqlen, head_dim) for PRoPE transforms
                    q_bnhd = q.permute(0, 2, 1, 3).contiguous()
                    k_bnhd = k.permute(0, 2, 1, 3).contiguous()
                    v_bnhd = v.permute(0, 2, 1, 3).contiguous()

                    # Reorder merged-width row-major -> camera-major row-major
                    q_bnhd = reorder_to(q_bnhd)
                    k_bnhd = reorder_to(k_bnhd)
                    v_bnhd = reorder_to(v_bnhd)

                    q = apply_fn_q(q_bnhd).permute(0, 2, 1, 3).contiguous()
                    k = apply_fn_kv(k_bnhd).permute(0, 2, 1, 3).contiguous()
                    v = apply_fn_kv(v_bnhd).permute(0, 2, 1, 3).contiguous()

                    _prope_apply_fn_o = apply_fn_o
                    _prope_reorder_back = reorder_from
            except Exception:
                _prope_apply_fn_o = None
                _prope_reorder_back = None

        attn_bias = None
        if mask is not None:
            attn_bias = torch.zeros([B * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float('-inf'))
        # Prefer xFormers on CUDA (matches original repo behavior). Fallback to PyTorch SDPA when unsupported.
        # This is important for stability when `fp32_attention=True` (q/k/v cast to fp32).
        try:
            # xFormers FlashAttention has known issues when seqlen_q != seqlen_k on some versions.
            # For asymmetric self-attn (kv_tokens provided), force SDPA fallback for correctness.
            if q.is_cuda and (kv_tokens is None):
                x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
            else:
                raise NotImplementedError("xFormers attention requires CUDA; fallback to SDPA.")
        except Exception as e:
            # Log once to avoid spamming.
            if not hasattr(self, "_sdpa_fallback_logged"):
                self._sdpa_fallback_logged = True
                print(f"[AttentionKVCompress] Falling back to PyTorch SDPA due to: {type(e).__name__}: {e}")

            # SDPA expects (B, heads, seqlen, head_dim)
            q_ = q.permute(0, 2, 1, 3).contiguous()
            k_ = k.permute(0, 2, 1, 3).contiguous()
            v_ = v.permute(0, 2, 1, 3).contiguous()
            # We don't support self-attn masks here; current callers use mask=None for self-attn.
            o_ = F.scaled_dot_product_attention(
                q_, k_, v_, attn_mask=None, dropout_p=float(self.attn_drop.p) if self.training else 0.0, is_causal=False
            )
            x = o_.permute(0, 2, 1, 3).contiguous()

        if _prope_apply_fn_o is not None and _prope_reorder_back is not None:
            x_bnhd = x.permute(0, 2, 1, 3).contiguous()
            x_bnhd = _prope_apply_fn_o(x_bnhd)
            x_bnhd = _prope_reorder_back(x_bnhd)
            x = x_bnhd.permute(0, 2, 1, 3).contiguous()

        x = x.view(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


#################################################################################
#   AMP attention with fp32 softmax to fix loss NaN problem during training     #
#################################################################################
class Attention(Attention_):
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        use_fp32_attention = getattr(self, 'fp32_attention', False)
        if use_fp32_attention:
            q, k = q.float(), k.float()
        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FinalLayer(nn.Module):
    """
    The final layer of PixArtWorldFM.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArtWorldFM.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size ** 0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MaskFinalLayer(nn.Module):
    """
    The final layer of PixArtWorldFM.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(final_hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(c_emb_size, 2 * final_hidden_size, bias=True)
        )
    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DecoderLayer(nn.Module):
    """
    The final layer of PixArtWorldFM.
    """

    def __init__(self, hidden_size, decoder_hidden_size):
        super().__init__()
        self.norm_decoder = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, decoder_hidden_size, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    def forward(self, x, t):
        shift, scale = self.adaLN_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_decoder(x), shift, scale)
        x = self.linear(x)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype


class SizeEmbedder(TimestepEmbedder):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__(hidden_size=hidden_size, frequency_embedding_size=frequency_embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.outdim = hidden_size

    def forward(self, s, bs):
        if s.ndim == 1:
            s = s[:, None]
        assert s.ndim == 2
        if s.shape[0] != bs:
            s = s.repeat(bs//s.shape[0], 1)
            assert s.shape[0] == bs
        b, dims = s.shape[0], s.shape[1]
        s = rearrange(s, "b d -> (b d)")
        s_freq = self.timestep_embedding(s, self.frequency_embedding_size).to(self.dtype)
        s_emb = self.mlp(s_freq)
        s_emb = rearrange(s_emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        return s_emb

    @property
    def dtype(self):
        # 返回模型参数的数据类型
        return next(self.parameters()).dtype


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0]).cuda() < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate='tanh'), token_num=120):
        super().__init__()
        self.y_proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0)
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class CaptionEmbedderDoubleBr(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate='tanh'), token_num=120):
        super().__init__()
        self.proj = Mlp(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0)
        self.embedding = nn.Parameter(torch.randn(1, in_channels) / 10 ** 0.5)
        self.y_embedding = nn.Parameter(torch.randn(token_num, in_channels) / 10 ** 0.5)
        self.uncond_prob = uncond_prob

    def token_drop(self, global_caption, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(global_caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        global_caption = torch.where(drop_ids[:, None], self.embedding, global_caption)
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return global_caption, caption

    def forward(self, caption, train, force_drop_ids=None):
        assert caption.shape[2: ] == self.y_embedding.shape
        global_caption = caption.mean(dim=2).squeeze()
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            global_caption, caption = self.token_drop(global_caption, caption, force_drop_ids)
        y_embed = self.proj(global_caption)
        return y_embed, caption