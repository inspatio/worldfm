from mmengine.registry import Registry

from worldfm.diffusion.model.utils import set_grad_checkpoint

MODELS = Registry('models')

_NETS_IMPORTED = False


def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    # Ensure model modules are imported and registered before building.
    # Use lazy import to avoid circular-import issues during module init.
    global _NETS_IMPORTED
    if not _NETS_IMPORTED:
        from . import nets  # noqa: F401
        _NETS_IMPORTED = True
    model = MODELS.build(cfg, default_args=kwargs)
    if use_grad_checkpoint:
        set_grad_checkpoint(model, use_fp32_attention=use_fp32_attention, gc_step=gc_step)
    return model
