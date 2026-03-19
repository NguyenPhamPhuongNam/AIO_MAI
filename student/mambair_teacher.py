import os
import sys
from typing import Dict, Any
import torch


def _pick_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for k in ["params", "state_dict", "model", "net", "params_ema", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
    raise RuntimeError("Unknown checkpoint format")


def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str = "module.") -> Dict[str, torch.Tensor]:
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix):]: v for k, v in sd.items()}
    return sd


def build_mambairv2_lightsr_x3(mambair_repo: str):
    sys.path.insert(0, os.path.abspath(mambair_repo))
    from basicsr.archs.mambairv2light_arch import MambaIRv2Light

    net = MambaIRv2Light(
        upscale=3,
        img_size=64,
        in_chans=3,
        img_range=1.0,
        embed_dim=48,
        d_state=8,
        depths=[5, 5, 5, 5],
        num_heads=[4, 4, 4, 4],
        window_size=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        mlp_ratio=1.0,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
    )
    return net


@torch.inference_mode()
def load_teacher(mambair_repo: str, ckpt_path: str, device="cuda", strict: bool = False):
    """
    Robust loader:
      - picks sd from common keys
      - strips DDP prefix 'module.'
      - strict=False by default with warning summary
    """
    net = build_mambairv2_lightsr_x3(mambair_repo)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _pick_state_dict(ckpt)
    sd = _strip_prefix(sd, "module.")

    missing, unexpected = net.load_state_dict(sd, strict=strict)
    if (not strict) and (missing or unexpected):
        print(
            f"[WARN] load_teacher({os.path.basename(ckpt_path)}): "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
        if len(missing) and len(missing) <= 10:
            print("  missing keys:", missing)
        if len(unexpected) and len(unexpected) <= 10:
            print("  unexpected keys:", unexpected)

    net.eval().to(device)
    for p in net.parameters():
        p.requires_grad_(False)

    # Convention for the training pipeline
    net._expects_01 = True
    return net