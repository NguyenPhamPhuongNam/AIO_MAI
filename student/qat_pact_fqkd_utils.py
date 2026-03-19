from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Iterable, Tuple


# =========================================================
# 1) Hook-based feature extractor (non-invasive)
# =========================================================
class HookBasedFeatureExtractor:
    """
    Attach forward hooks to chosen module names and cache outputs.
    MUST call clear() every iteration to avoid OOM.
    Also records which requested hooks were actually found.
    """
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.features: Dict[str, torch.Tensor] = {}
        self.hooks = []

        # keep order, remove duplicates
        self.requested_layer_names = list(dict.fromkeys(layer_names))
        wanted = set(self.requested_layer_names)

        self.found_layer_names: List[str] = []
        found = set()

        for name, module in model.named_modules():
            if name in wanted:
                self.hooks.append(module.register_forward_hook(self._mk_hook(name)))
                self.found_layer_names.append(name)
                found.add(name)

        self.missing_layer_names = [n for n in self.requested_layer_names if n not in found]

    def _mk_hook(self, name: str):
        def hook(_module, _inp, out):
            self.features[name] = out
        return hook

    def clear(self):
        self.features.clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# =========================================================
# 2) F-QKD: spatial attention transfer loss
# =========================================================
def spatial_attention_loss(f_student: torch.Tensor, f_teacher: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compare normalized spatial attention maps:
      att = sum_c |f|^2  => shape (B,1,H,W) -> flatten -> L2 normalize -> L1 distance
    """
    att_s = torch.sum(torch.abs(f_student) ** 2, dim=1, keepdim=True)
    att_t = torch.sum(torch.abs(f_teacher) ** 2, dim=1, keepdim=True)
    att_s = att_s.view(att_s.size(0), -1)
    att_t = att_t.view(att_t.size(0), -1)
    att_s = F.normalize(att_s, p=2, dim=1, eps=eps)
    att_t = F.normalize(att_t, p=2, dim=1, eps=eps)
    return F.l1_loss(att_s, att_t)


# =========================================================
# 3) PACT activation (NO internal FakeQuant)
# =========================================================
class PACTActivation(nn.Module):
    """
    Learnable clamp (PACT).
    IMPORTANT: Do NOT embed FakeQuant here.
    FX-QAT will place fakequant/observers around this module.
    """
    def __init__(self, init_min: float = -2.0, init_max: float = 2.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(init_max), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(float(init_min), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Safety: ensure min<=max (avoid runtime error if beta>alpha)
        lo = torch.minimum(self.beta, self.alpha)
        hi = torch.maximum(self.beta, self.alpha)
        return torch.clamp(x, lo, hi)


def inject_pact_activations(
    model: nn.Module,
    rep_block_classes: tuple[type, ...],
    replace_relu_only: bool = True,
    init_min: float = -2.0,
    init_max: float = 2.0,
    pact_on_output: bool = False,
    out_init_min: float = 0.0,
    out_init_max: float = 255.0,
) -> Tuple[List[str], bool]:
    """
    Replace block.act with PACT.
    Returns:
      - replaced act module names
      - whether output clamp was replaced
    """
    replaced_act_names: List[str] = []

    for name, m in model.named_modules():
        if isinstance(m, rep_block_classes):
            if hasattr(m, "act") and isinstance(getattr(m, "act"), nn.Module):
                old = getattr(m, "act")
                should_replace = (not replace_relu_only) or isinstance(old, nn.ReLU)
                if should_replace:
                    new_act = PACTActivation(init_min=init_min, init_max=init_max)

                    # Put new PACT module on the same device/dtype as the block
                    ref_param = next(m.parameters(), None)
                    if ref_param is not None:
                        new_act = new_act.to(device=ref_param.device, dtype=ref_param.dtype)

                    m.act = new_act
                    replaced_act_names.append(f"{name}.act" if name else "act")

    replaced_output = False
    if pact_on_output and hasattr(model, "out_clamp") and isinstance(model.out_clamp, nn.Module):
        new_out = PACTActivation(init_min=out_init_min, init_max=out_init_max)

        # Put output PACT on the same device/dtype as the model
        ref_param = next(model.parameters(), None)
        if ref_param is not None:
            new_out = new_out.to(device=ref_param.device, dtype=ref_param.dtype)

        model.out_clamp = new_out
        replaced_output = True

    return replaced_act_names, replaced_output


def count_pact_modules(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, PACTActivation))


def build_fqkd_layer_list(n_rep: int, picks: Iterable[int]) -> List[str]:
    """
    Convert indices -> ['rep.1','rep.3',...], ignoring out-of-range.
    """
    out: List[str] = []
    for i in picks:
        if 0 <= int(i) < int(n_rep):
            out.append(f"rep.{int(i)}")
    return out