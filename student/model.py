# # import torch
# # import torch.nn as nn
# # from typing import Tuple, Literal

# # # -------------------------
# # # Helpers
# # # -------------------------
# # def conv3x3(in_ch: int, out_ch: int, bias: bool = True, groups: int = 1) -> nn.Conv2d:
# #     return nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias, groups=groups)

# # def conv1x1(in_ch: int, out_ch: int, bias: bool = True, groups: int = 1) -> nn.Conv2d:
# #     return nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias, groups=groups)

# # def pad_1x1_to_3x3_center(w_1x1: torch.Tensor) -> torch.Tensor:
# #     w_3x3 = w_1x1.new_zeros((w_1x1.size(0), w_1x1.size(1), 3, 3))
# #     w_3x3[:, :, 1, 1] = w_1x1[:, :, 0, 0]
# #     return w_3x3

# # @torch.no_grad()
# # def fuse_batchnorm_into_conv(W: torch.Tensor, b: torch.Tensor, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
# #     gamma = bn.weight
# #     beta = bn.bias
# #     mean = bn.running_mean
# #     var = bn.running_var
# #     eps = bn.eps
# #     inv_std = torch.rsqrt(var + eps)
# #     scale = gamma * inv_std
# #     W_fused = W * scale.view(-1, 1, 1, 1)
# #     b_fused = (b - mean) * scale + beta
# #     return W_fused, b_fused


# # def identity_depthwise_3x3_kernel(C: int, device, dtype) -> torch.Tensor:
# #     W = torch.zeros((C, 1, 3, 3), device=device, dtype=dtype)
# #     idx = torch.arange(C, device=device)
# #     W[idx, 0, 1, 1] = 1.0
# #     return W

# # def identity_3x3_kernel(C: int, device, dtype) -> torch.Tensor:
# #     W = torch.zeros((C, C, 3, 3), device=device, dtype=dtype)
# #     idx = torch.arange(C, device=device)
# #     W[idx, idx, 1, 1] = 1.0
# #     return W

# # # -------------------------
# # # Activations
# # # -------------------------
# # class Identity(nn.Module):
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         return x

# # def make_activation(mode: Literal["none", "relu"]) -> nn.Module:
# #     if mode == "none":
# #         return Identity()
# #     if mode == "relu":
# #         return nn.ReLU(inplace=False)
# #     raise ValueError(f"Unknown activation mode: {mode}")

# # # -------------------------
# # # Output clamp
# # # -------------------------
# # class Min255(nn.Module):
# #     """Clamp max to 255 only; negative values are preserved."""
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         mx = x.new_tensor(255.0)
# #         return torch.minimum(x, mx)

# # class MinClip(nn.Module):
# #     """min(ReLU(x), 255)"""
# #     def __init__(self):
# #         super().__init__()
# #         self.relu = nn.ReLU(inplace=False)

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         x = self.relu(x)
# #         mx = x.new_tensor(255.0)
# #         return torch.minimum(x, mx)

# # class Clamp0_255(nn.Module):
# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         lo = x.new_tensor(0.0)
# #         hi = x.new_tensor(255.0)
# #         x = torch.maximum(x, lo)
# #         x = torch.minimum(x, hi)
# #         return x

# # def make_output_clamp(mode: Literal["min255", "minclip", "clamp_0_255", "none"]) -> nn.Module:
# #     if mode == "min255":
# #         return Min255()
# #     if mode == "minclip":
# #         return MinClip()
# #     if mode == "clamp_0_255":
# #         return Clamp0_255()
# #     if mode == "none":
# #         return Identity()
# #     raise ValueError(f"Unknown clamp mode: {mode}")

# # # -------------------------
# # # RepConv
# # # -------------------------
# # class RepConv(nn.Module):
# #     """
# #     train: 1x1->3x3->1x1 + skip 1x1 (+ optional BN) -> act
# #     deploy: single 3x3 (+ act)
# #     """
# #     def __init__(
# #         self,
# #         channels: int,
# #         deploy: bool = False,
# #         rep_use_bn: bool = False,
# #         act_mode: Literal["none", "relu"] = "none",
# #     ):
# #         super().__init__()
# #         self.c = channels
# #         self.deploy = deploy
# #         self.act = make_activation(act_mode)

# #         if deploy:
# #             self.reparam = conv3x3(channels, channels, bias=True)
# #             self.bn = Identity()
# #         else:
# #             self.conv1_a = conv1x1(channels, channels, bias=False)
# #             self.conv3 = conv3x3(channels, channels, bias=True)
# #             self.conv1_b = conv1x1(channels, channels, bias=True)
# #             self.conv1_s = conv1x1(channels, channels, bias=True)
# #             self.bn = nn.BatchNorm2d(channels) if rep_use_bn else Identity()

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         if self.deploy:
# #             return self.act(self.reparam(x))
# #         y_main = self.conv1_b(self.conv3(self.conv1_a(x)))
# #         y_skip = self.conv1_s(x)
# #         y = y_main + y_skip
# #         y = self.bn(y)
# #         return self.act(y)

# #     @torch.no_grad()
# #     def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
# #         W1 = self.conv1_a.weight.squeeze(-1).squeeze(-1)  # (C,C)
# #         W3 = self.conv3.weight  # (C,C,3,3)
# #         b3 = self.conv3.bias
# #         W2 = self.conv1_b.weight.squeeze(-1).squeeze(-1)  # (C,C)
# #         b2 = self.conv1_b.bias

# #         tmp = torch.einsum("mnxy,ni->mixy", W3, W1)          # (C,C,3,3)
# #         W_main = torch.einsum("om,mixy->oixy", W2, tmp)      # (C,C,3,3)
# #         b_main = b2 + torch.matmul(W2, b3)                   # (C,)

# #         W_skip = pad_1x1_to_3x3_center(self.conv1_s.weight)
# #         b_skip = self.conv1_s.bias

# #         W_eq = W_main + W_skip
# #         b_eq = b_main + b_skip

# #         if isinstance(self.bn, nn.BatchNorm2d):
# #             W_eq, b_eq = fuse_batchnorm_into_conv(W_eq, b_eq, self.bn)
# #         return W_eq, b_eq

# #     @torch.no_grad()
# #     def switch_to_deploy(self) -> None:
# #         if self.deploy:
# #             return
# #         W, b = self.get_equivalent_kernel_bias()
# #         self.reparam = conv3x3(self.c, self.c, bias=True).to(device=W.device, dtype=W.dtype)
# #         self.reparam.weight.copy_(W)
# #         self.reparam.bias.copy_(b.to(W.dtype))

# #         del self.conv1_a, self.conv3, self.conv1_b, self.conv1_s
# #         self.bn = Identity()
# #         self.deploy = True

# # # -------------------------
# # # MobileOne-style rep block
# # # -------------------------
# # class _ConvBN(nn.Module):
# #     def __init__(self, conv: nn.Conv2d, use_bn: bool):
# #         super().__init__()
# #         self.conv = conv
# #         self.bn = nn.BatchNorm2d(conv.out_channels) if use_bn else Identity()

# #     def forward(self, x):
# #         return self.bn(self.conv(x))

# #     @torch.no_grad()
# #     def fuse(self) -> Tuple[torch.Tensor, torch.Tensor]:
# #         W = self.conv.weight
# #         if self.conv.bias is None:
# #             b = torch.zeros((W.size(0),), device=W.device, dtype=W.dtype)
# #         else:
# #             b = self.conv.bias
# #         if isinstance(self.bn, nn.BatchNorm2d):
# #             W, b = fuse_batchnorm_into_conv(W, b, self.bn)
# #         return W, b

# # class MobileOneRepBlock(nn.Module):
# #     def __init__(
# #         self,
# #         channels: int,
# #         deploy: bool = False,
# #         num_3x3_branches: int = 2,
# #         use_1x1: bool = True,
# #         use_identity: bool = True,
# #         rep_use_bn: bool = True,
# #         act_mode: Literal["none", "relu"] = "relu",
# #     ):
# #         super().__init__()
# #         assert int(num_3x3_branches) >= 1, "num_3x3_branches must be >= 1"
# #         self.c = channels
# #         self.deploy = deploy
# #         self.act = make_activation(act_mode)
# #         self.num_3x3 = int(num_3x3_branches)
# #         self.use_1x1 = bool(use_1x1)
# #         self.use_id = bool(use_identity)
# #         self.use_bn = bool(rep_use_bn)

# #         if deploy:
# #             self.reparam = conv3x3(channels, channels, bias=True)
# #         else:
# #             bias = not self.use_bn
# #             self.branches_3x3 = nn.ModuleList([
# #                 _ConvBN(conv3x3(channels, channels, bias=bias), use_bn=self.use_bn)
# #                 for _ in range(self.num_3x3)
# #             ])
# #             self.branch_1x1 = _ConvBN(conv1x1(channels, channels, bias=bias), use_bn=self.use_bn) if self.use_1x1 else None
# #             self.id_bn = nn.BatchNorm2d(channels) if (self.use_id and self.use_bn) else (Identity() if self.use_id else None)

# #     def forward(self, x):
# #         if self.deploy:
# #             return self.act(self.reparam(x))

# #         y = self.branches_3x3[0](x)
# #         for br in self.branches_3x3[1:]:
# #             y = y + br(x)
# #         if self.branch_1x1 is not None:
# #             y = y + self.branch_1x1(x)
# #         if self.id_bn is not None:
# #             y = y + self.id_bn(x)
# #         return self.act(y)

# #     @torch.no_grad()
# #     def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
# #         device = next(self.parameters()).device
# #         dtype = next(self.parameters()).dtype
# #         W_sum = torch.zeros((self.c, self.c, 3, 3), device=device, dtype=dtype)
# #         b_sum = torch.zeros((self.c,), device=device, dtype=dtype)

# #         for br in self.branches_3x3:
# #             W, b = br.fuse()
# #             W_sum += W
# #             b_sum += b

# #         if self.branch_1x1 is not None:
# #             W1, b1 = self.branch_1x1.fuse()
# #             W_sum += pad_1x1_to_3x3_center(W1)
# #             b_sum += b1

# #         if self.id_bn is not None:
# #             W_id = identity_3x3_kernel(self.c, device=device, dtype=dtype)
# #             b_id = torch.zeros((self.c,), device=device, dtype=dtype)
# #             if isinstance(self.id_bn, nn.BatchNorm2d):
# #                 W_id, b_id = fuse_batchnorm_into_conv(W_id, b_id, self.id_bn)
# #             W_sum += W_id
# #             b_sum += b_id

# #         return W_sum, b_sum

# #     @torch.no_grad()
# #     def switch_to_deploy(self) -> None:
# #         if self.deploy:
# #             return
# #         W, b = self.get_equivalent_kernel_bias()
# #         self.reparam = conv3x3(self.c, self.c, bias=True).to(device=W.device, dtype=W.dtype)
# #         self.reparam.weight.copy_(W)
# #         self.reparam.bias.copy_(b.to(W.dtype))

# #         del self.branches_3x3
# #         if hasattr(self, "branch_1x1"):
# #             del self.branch_1x1
# #         if hasattr(self, "id_bn"):
# #             del self.id_bn
# #         self.deploy = True

# # # -------------------------
# # # RepDW block
# # # -------------------------
# # class RepDWBlock(nn.Module):
# #     """
# #     Re-parameterizable depthwise block.

# #     Train-time:
# #       (dw3x3 + dw1x1(+pad) + identity_bn) -> pw1x1 -> +residual -> act

# #     Deploy-time:
# #       fused single dw3x3 -> pw1x1 -> +residual -> act
# #     """
# #     def __init__(
# #         self,
# #         channels: int,
# #         deploy: bool = False,
# #         rep_use_bn: bool = True,
# #         act_mode: Literal["none", "relu"] = "relu",
# #         use_dw_1x1: bool = True,
# #         use_identity: bool = True,
# #     ):
# #         super().__init__()
# #         self.c = channels
# #         self.deploy = deploy
# #         self.use_dw_1x1 = bool(use_dw_1x1)
# #         self.use_identity = bool(use_identity)
# #         self.use_bn = bool(rep_use_bn)
# #         self.act = make_activation(act_mode)

# #         if deploy:
# #             self.dw_reparam = conv3x3(channels, channels, bias=True, groups=channels)
# #             self.pw = conv1x1(channels, channels, bias=True)
# #         else:
# #             bias = not self.use_bn

# #             self.dw3 = _ConvBN(
# #                 conv3x3(channels, channels, bias=bias, groups=channels),
# #                 use_bn=self.use_bn,
# #             )

# #             self.dw1 = _ConvBN(
# #                 conv1x1(channels, channels, bias=bias, groups=channels),
# #                 use_bn=self.use_bn,
# #             ) if self.use_dw_1x1 else None

# #             self.id_bn = nn.BatchNorm2d(channels) if (self.use_identity and self.use_bn) else (Identity() if self.use_identity else None)

# #             self.pw = conv1x1(channels, channels, bias=True)

# #     def forward(self, x):
# #         if self.deploy:
# #             y = self.pw(self.dw_reparam(x))
# #             y = y + x
# #             return self.act(y)

# #         y = self.dw3(x)
# #         if self.dw1 is not None:
# #             y = y + self.dw1(x)
# #         if self.id_bn is not None:
# #             y = y + self.id_bn(x)

# #         y = self.pw(y)
# #         y = y + x
# #         return self.act(y)

# #     @torch.no_grad()
# #     def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
# #         device = next(self.parameters()).device
# #         dtype = next(self.parameters()).dtype

# #         # depthwise kernel shape: (C,1,3,3)
# #         W_sum = torch.zeros((self.c, 1, 3, 3), device=device, dtype=dtype)
# #         b_sum = torch.zeros((self.c,), device=device, dtype=dtype)

# #         W3, b3 = self.dw3.fuse()
# #         W_sum += W3
# #         b_sum += b3

# #         if self.dw1 is not None:
# #             W1, b1 = self.dw1.fuse()
# #             W_sum += pad_1x1_to_3x3_center(W1)
# #             b_sum += b1

# #         if self.id_bn is not None:
# #             W_id = identity_depthwise_3x3_kernel(self.c, device=device, dtype=dtype)
# #             b_id = torch.zeros((self.c,), device=device, dtype=dtype)
# #             if isinstance(self.id_bn, nn.BatchNorm2d):
# #                 W_id, b_id = fuse_batchnorm_into_conv(W_id, b_id, self.id_bn)
# #             W_sum += W_id
# #             b_sum += b_id

# #         return W_sum, b_sum

# #     @torch.no_grad()
# #     def switch_to_deploy(self) -> None:
# #         if self.deploy:
# #             return

# #         W, b = self.get_equivalent_kernel_bias()
# #         self.dw_reparam = conv3x3(self.c, self.c, bias=True, groups=self.c).to(device=W.device, dtype=W.dtype)
# #         self.dw_reparam.weight.copy_(W)
# #         self.dw_reparam.bias.copy_(b.to(W.dtype))

# #         del self.dw3
# #         if hasattr(self, "dw1"):
# #             del self.dw1
# #         if hasattr(self, "id_bn"):
# #             del self.id_bn

# #         self.deploy = True

# # # -------------------------
# # # AntSR
# # # -------------------------
# # ConcatHTR = Literal["3x3_3x3", "1x1_3x3", "1x1_1x1"]
# # SkipMode = Literal["add", "add1x1", "concat_lr", "concat_raw"]
# # RepType = Literal["repconv", "mobileone", "repdw"]

# # def _make_trans_layers(in_ch: int, mid_ch: int, mode: ConcatHTR) -> Tuple[nn.Module, nn.Module]:
# #     if mode == "3x3_3x3":
# #         return conv3x3(in_ch, mid_ch, True), conv3x3(mid_ch, mid_ch, True)
# #     if mode == "1x1_3x3":
# #         return conv1x1(in_ch, mid_ch, True), conv3x3(mid_ch, mid_ch, True)
# #     if mode == "1x1_1x1":
# #         return conv1x1(in_ch, mid_ch, True), conv1x1(mid_ch, mid_ch, True)
# #     raise ValueError(f"Unknown concat_htr: {mode}")

# # class AntSR(nn.Module):
# #     def __init__(
# #         self,
# #         scale: int = 3,
# #         channels: int = 32,
# #         n_rep: int = 4,
# #         deploy: bool = False,
# #         rep_type: RepType = "repconv",
# #         rep_use_bn: bool = False,
# #         rep_act_mode: Literal["none", "relu"] = "none",
# #         mo_branches: int = 2,
# #         mo_use_1x1: bool = True,
# #         mo_use_identity: bool = True,
# #         out_clamp_mode: Literal["min255", "minclip", "clamp_0_255", "none"] = "minclip",
# #         skip_mode: SkipMode = "add",
# #         concat_htr: ConcatHTR = "3x3_3x3",
# #         use_global_add: bool = True,
# #     ):
# #         super().__init__()
# #         assert scale == 3, "This implementation targets x3 SR"
# #         self.scale = scale
# #         self.skip_mode = skip_mode
# #         self.concat_htr = concat_htr
# #         self.use_global_add = bool(use_global_add)
# #         self.rep_type = rep_type
# #         self.n_rep = int(n_rep)

# #         self.conv_in = conv3x3(3, channels, bias=True)

# #         if rep_type == "repconv":
# #             blocks = [RepConv(channels, deploy=deploy, rep_use_bn=rep_use_bn, act_mode=rep_act_mode) for _ in range(n_rep)]
# #         elif rep_type == "mobileone":
# #             blocks = [MobileOneRepBlock(
# #                 channels,
# #                 deploy=deploy,
# #                 num_3x3_branches=mo_branches,
# #                 use_1x1=mo_use_1x1,
# #                 use_identity=mo_use_identity,
# #                 rep_use_bn=rep_use_bn,
# #                 act_mode=rep_act_mode,
# #             ) for _ in range(n_rep)]
# #         elif rep_type == "repdw":
# #             act = "relu" if rep_act_mode == "none" else rep_act_mode
# #             blocks = [RepDWBlock(
# #                 channels,
# #                 deploy=deploy,
# #                 rep_use_bn=rep_use_bn,
# #                 act_mode=act,
# #                 use_dw_1x1=True,
# #                 use_identity=True,
# #             ) for _ in range(n_rep)]
# #         else:
# #             raise ValueError(f"Unknown rep_type: {rep_type}")

# #         self.rep = nn.ModuleList(blocks)

# #         if skip_mode in ("add1x1", "concat_lr"):
# #             self.lr_proj = conv1x1(3, channels, bias=True)
# #         else:
# #             self.lr_proj = Identity()

# #         if skip_mode == "concat_raw":
# #             self.htr1, self.htr2 = _make_trans_layers(channels + 3, channels, concat_htr)
# #         elif skip_mode == "concat_lr":
# #             self.htr1, self.htr2 = _make_trans_layers(channels * 2, channels, concat_htr)
# #         else:
# #             self.htr1 = Identity()
# #             self.htr2 = Identity()

# #         self.conv_out = conv3x3(channels, 3 * (scale * scale), bias=True)  # 27
# #         self.out_clamp = make_output_clamp(out_clamp_mode)
# #         self.ps = nn.PixelShuffle(scale)

# #     def set_out_clamp_mode(self, mode: Literal["min255", "minclip", "clamp_0_255", "none"]):
# #         self.out_clamp = make_output_clamp(mode)

# #     def forward(self, lr: torch.Tensor) -> torch.Tensor:
# #         feat = self.conv_in(lr)
# #         x = feat
# #         for blk in self.rep:
# #             x = blk(x)

# #         if self.use_global_add:
# #             x = x + feat

# #         if self.skip_mode == "add1x1":
# #             x = x + self.lr_proj(lr)
# #         elif self.skip_mode == "concat_lr":
# #             lr_f = self.lr_proj(lr)
# #             x = torch.cat([x, lr_f], dim=1)
# #             x = self.htr2(self.htr1(x))
# #         elif self.skip_mode == "concat_raw":
# #             x = torch.cat([x, lr], dim=1)
# #             x = self.htr2(self.htr1(x))

# #         x = self.conv_out(x)
# #         x = self.out_clamp(x)
# #         return self.ps(x)

#     @torch.no_grad()
#     def switch_to_deploy(self) -> None:
#         self.eval()
#         for blk in self.rep:
#             if hasattr(blk, "switch_to_deploy"):
#                 blk.switch_to_deploy()









import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Literal

# -------------------------
# Helpers
# -------------------------
def conv3x3(in_ch: int, out_ch: int, bias: bool = True, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=bias, groups=groups)

def conv1x1(in_ch: int, out_ch: int, bias: bool = True, groups: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias, groups=groups)

def pad_1x1_to_3x3_center(w_1x1: torch.Tensor) -> torch.Tensor:
    w_3x3 = w_1x1.new_zeros((w_1x1.size(0), w_1x1.size(1), 3, 3))
    w_3x3[:, :, 1, 1] = w_1x1[:, :, 0, 0]
    return w_3x3

@torch.no_grad()
def fuse_batchnorm_into_conv(W: torch.Tensor, b: torch.Tensor, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    inv_std = torch.rsqrt(var + eps)
    scale = gamma * inv_std
    W_fused = W * scale.view(-1, 1, 1, 1)
    b_fused = (b - mean) * scale + beta
    return W_fused, b_fused


def identity_depthwise_3x3_kernel(C: int, device, dtype) -> torch.Tensor:
    W = torch.zeros((C, 1, 3, 3), device=device, dtype=dtype)
    idx = torch.arange(C, device=device)
    W[idx, 0, 1, 1] = 1.0
    return W

def identity_3x3_kernel(C: int, device, dtype) -> torch.Tensor:
    W = torch.zeros((C, C, 3, 3), device=device, dtype=dtype)
    idx = torch.arange(C, device=device)
    W[idx, idx, 1, 1] = 1.0
    return W

# -------------------------
# Activations
# -------------------------
class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

def make_activation(mode: Literal["none", "relu"]) -> nn.Module:
    if mode == "none":
        return Identity()
    if mode == "relu":
        return nn.ReLU(inplace=False)
    raise ValueError(f"Unknown activation mode: {mode}")

# -------------------------
# Output clamp
# -------------------------
class Min255(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mx = x.new_tensor(255.0)
        return torch.minimum(x, mx)

class MinClip(nn.Module):
    """min(ReLU(x), 255)"""
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(x)
        mx = x.new_tensor(255.0)
        return torch.minimum(x, mx)

class Clamp0_255(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo = x.new_tensor(0.0)
        hi = x.new_tensor(255.0)
        x = torch.maximum(x, lo)
        x = torch.minimum(x, hi)
        return x

def make_output_clamp(mode: Literal["min255", "minclip", "clamp_0_255", "none"]) -> nn.Module:
    if mode == "min255":
        return Min255()
    if mode == "minclip":
        return MinClip()
    if mode == "clamp_0_255":
        return Clamp0_255()
    if mode == "none":
        return Identity()
    raise ValueError(f"Unknown clamp mode: {mode}")

# -------------------------
# RepConv
# -------------------------
class RepConv(nn.Module):
    """
    train: 1x1->3x3->1x1 + skip 1x1 (+ optional BN) -> act
    deploy: single 3x3 (+ act)
    """
    def __init__(
        self,
        channels: int,
        deploy: bool = False,
        rep_use_bn: bool = False,
        act_mode: Literal["none", "relu"] = "none",
    ):
        super().__init__()
        self.c = channels
        self.deploy = deploy
        self.act = make_activation(act_mode)

        if deploy:
            self.reparam = conv3x3(channels, channels, bias=True)
            self.bn = Identity()
        else:
            self.conv1_a = conv1x1(channels, channels, bias=False)
            self.conv3 = conv3x3(channels, channels, bias=True)
            self.conv1_b = conv1x1(channels, channels, bias=True)
            self.conv1_s = conv1x1(channels, channels, bias=True)
            self.bn = nn.BatchNorm2d(channels) if rep_use_bn else Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.reparam(x))
        y_main = self.conv1_b(self.conv3(self.conv1_a(x)))
        y_skip = self.conv1_s(x)
        y = y_main + y_skip
        y = self.bn(y)
        return self.act(y)

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        W1 = self.conv1_a.weight.squeeze(-1).squeeze(-1)  # (C,C)
        W3 = self.conv3.weight  # (C,C,3,3)
        b3 = self.conv3.bias
        W2 = self.conv1_b.weight.squeeze(-1).squeeze(-1)  # (C,C)
        b2 = self.conv1_b.bias

        tmp = torch.einsum("mnxy,ni->mixy", W3, W1)          # (C,C,3,3)
        W_main = torch.einsum("om,mixy->oixy", W2, tmp)      # (C,C,3,3)
        b_main = b2 + torch.matmul(W2, b3)                   # (C,)

        W_skip = pad_1x1_to_3x3_center(self.conv1_s.weight)
        b_skip = self.conv1_s.bias

        W_eq = W_main + W_skip
        b_eq = b_main + b_skip

        if isinstance(self.bn, nn.BatchNorm2d):
            W_eq, b_eq = fuse_batchnorm_into_conv(W_eq, b_eq, self.bn)
        return W_eq, b_eq

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        W, b = self.get_equivalent_kernel_bias()
        self.reparam = conv3x3(self.c, self.c, bias=True).to(device=W.device, dtype=W.dtype)
        self.reparam.weight.copy_(W)
        self.reparam.bias.copy_(b.to(W.dtype))

        del self.conv1_a, self.conv3, self.conv1_b, self.conv1_s
        self.bn = Identity()
        self.deploy = True

# -------------------------
# MobileOne-style rep block
# -------------------------
class _ConvBN(nn.Module):
    def __init__(self, conv: nn.Conv2d, use_bn: bool):
        super().__init__()
        self.conv = conv
        self.bn = nn.BatchNorm2d(conv.out_channels) if use_bn else Identity()

    def forward(self, x):
        return self.bn(self.conv(x))

    @torch.no_grad()
    def fuse(self) -> Tuple[torch.Tensor, torch.Tensor]:
        W = self.conv.weight
        if self.conv.bias is None:
            b = torch.zeros((W.size(0),), device=W.device, dtype=W.dtype)
        else:
            b = self.conv.bias
        if isinstance(self.bn, nn.BatchNorm2d):
            W, b = fuse_batchnorm_into_conv(W, b, self.bn)
        return W, b

class MobileOneRepBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        deploy: bool = False,
        num_3x3_branches: int = 2,
        use_1x1: bool = True,
        use_identity: bool = True,
        rep_use_bn: bool = True,
        act_mode: Literal["none", "relu"] = "relu",
    ):
        super().__init__()
        self.c = channels
        self.deploy = deploy
        self.act = make_activation(act_mode)
        self.num_3x3 = int(num_3x3_branches)
        self.use_1x1 = bool(use_1x1)
        self.use_id = bool(use_identity)
        self.use_bn = bool(rep_use_bn)

        if deploy:
            self.reparam = conv3x3(channels, channels, bias=True)
        else:
            bias = not self.use_bn
            self.branches_3x3 = nn.ModuleList([
                _ConvBN(conv3x3(channels, channels, bias=bias), use_bn=self.use_bn)
                for _ in range(self.num_3x3)
            ])
            self.branch_1x1 = _ConvBN(conv1x1(channels, channels, bias=bias), use_bn=self.use_bn) if self.use_1x1 else None
            self.id_bn = nn.BatchNorm2d(channels) if (self.use_id and self.use_bn) else (Identity() if self.use_id else None)

    def forward(self, x):
        if self.deploy:
            return self.act(self.reparam(x))

        y = self.branches_3x3[0](x)
        for br in self.branches_3x3[1:]:
            y = y + br(x)
        if self.branch_1x1 is not None:
            y = y + self.branch_1x1(x)
        if self.id_bn is not None:
            y = y + self.id_bn(x)
        return self.act(y)

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        W_sum = torch.zeros((self.c, self.c, 3, 3), device=device, dtype=dtype)
        b_sum = torch.zeros((self.c,), device=device, dtype=dtype)

        for br in self.branches_3x3:
            W, b = br.fuse()
            W_sum += W
            b_sum += b

        if self.branch_1x1 is not None:
            W1, b1 = self.branch_1x1.fuse()
            W_sum += pad_1x1_to_3x3_center(W1)
            b_sum += b1

        if self.id_bn is not None:
            W_id = identity_3x3_kernel(self.c, device=device, dtype=dtype)
            b_id = torch.zeros((self.c,), device=device, dtype=dtype)
            if isinstance(self.id_bn, nn.BatchNorm2d):
                W_id, b_id = fuse_batchnorm_into_conv(W_id, b_id, self.id_bn)
            W_sum += W_id
            b_sum += b_id

        return W_sum, b_sum

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        W, b = self.get_equivalent_kernel_bias()
        self.reparam = conv3x3(self.c, self.c, bias=True).to(device=W.device, dtype=W.dtype)
        self.reparam.weight.copy_(W)
        self.reparam.bias.copy_(b.to(W.dtype))

        del self.branches_3x3
        if hasattr(self, "branch_1x1"):
            del self.branch_1x1
        if hasattr(self, "id_bn"):
            del self.id_bn
        self.deploy = True

# -------------------------
# RepDW block
# -------------------------
class RepDWBlock(nn.Module):
    """
    Re-parameterizable depthwise block.

    Train-time:
      (dw3x3 + dw1x1(+pad) + identity_bn) -> pw1x1 -> +residual -> act

    Deploy-time:
      fused single dw3x3 -> pw1x1 -> +residual -> act
    """
    def __init__(
        self,
        channels: int,
        deploy: bool = False,
        rep_use_bn: bool = True,
        act_mode: Literal["none", "relu"] = "relu",
        use_dw_1x1: bool = True,
        use_identity: bool = True,
    ):
        super().__init__()
        self.c = channels
        self.deploy = deploy
        self.use_dw_1x1 = bool(use_dw_1x1)
        self.use_identity = bool(use_identity)
        self.use_bn = bool(rep_use_bn)
        self.act = make_activation(act_mode)

        if deploy:
            self.dw_reparam = conv3x3(channels, channels, bias=True, groups=channels)
            self.pw = conv1x1(channels, channels, bias=True)
        else:
            bias = not self.use_bn

            self.dw3 = _ConvBN(
                conv3x3(channels, channels, bias=bias, groups=channels),
                use_bn=self.use_bn,
            )

            self.dw1 = _ConvBN(
                conv1x1(channels, channels, bias=bias, groups=channels),
                use_bn=self.use_bn,
            ) if self.use_dw_1x1 else None

            self.id_bn = nn.BatchNorm2d(channels) if (self.use_identity and self.use_bn) else (Identity() if self.use_identity else None)

            self.pw = conv1x1(channels, channels, bias=True)

    def forward(self, x):
        if self.deploy:
            y = self.pw(self.dw_reparam(x))
            y = y + x
            return self.act(y)

        y = self.dw3(x)
        if self.dw1 is not None:
            y = y + self.dw1(x)
        if self.id_bn is not None:
            y = y + self.id_bn(x)

        y = self.pw(y)
        y = y + x
        return self.act(y)

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # depthwise kernel shape: (C,1,3,3)
        W_sum = torch.zeros((self.c, 1, 3, 3), device=device, dtype=dtype)
        b_sum = torch.zeros((self.c,), device=device, dtype=dtype)

        W3, b3 = self.dw3.fuse()
        W_sum += W3
        b_sum += b3

        if self.dw1 is not None:
            W1, b1 = self.dw1.fuse()
            W_sum += pad_1x1_to_3x3_center(W1)
            b_sum += b1

        if self.id_bn is not None:
            W_id = identity_depthwise_3x3_kernel(self.c, device=device, dtype=dtype)
            b_id = torch.zeros((self.c,), device=device, dtype=dtype)
            if isinstance(self.id_bn, nn.BatchNorm2d):
                W_id, b_id = fuse_batchnorm_into_conv(W_id, b_id, self.id_bn)
            W_sum += W_id
            b_sum += b_id

        return W_sum, b_sum

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        if self.deploy:
            return

        W, b = self.get_equivalent_kernel_bias()
        self.dw_reparam = conv3x3(self.c, self.c, bias=True, groups=self.c).to(device=W.device, dtype=W.dtype)
        self.dw_reparam.weight.copy_(W)
        self.dw_reparam.bias.copy_(b.to(W.dtype))

        del self.dw3
        if hasattr(self, "dw1"):
            del self.dw1
        if hasattr(self, "id_bn"):
            del self.id_bn

        self.deploy = True

# -------------------------
# AntSR
# -------------------------
ConcatHTR = Literal["3x3_3x3", "1x1_3x3", "1x1_1x1"]
SkipMode = Literal["add", "add1x1", "concat_lr", "concat_raw"]
RepType = Literal["repconv", "mobileone", "repdw"]

def _make_trans_layers(in_ch: int, mid_ch: int, mode: ConcatHTR) -> Tuple[nn.Module, nn.Module]:
    if mode == "3x3_3x3":
        return conv3x3(in_ch, mid_ch, True), conv3x3(mid_ch, mid_ch, True)
    if mode == "1x1_3x3":
        return conv1x1(in_ch, mid_ch, True), conv3x3(mid_ch, mid_ch, True)
    if mode == "1x1_1x1":
        return conv1x1(in_ch, mid_ch, True), conv1x1(mid_ch, mid_ch, True)
    raise ValueError(f"Unknown concat_htr: {mode}")

class AntSR(nn.Module):
    def __init__(
        self,
        scale: int = 3,
        channels: int = 32,
        n_rep: int = 4,
        deploy: bool = False,
        rep_type: RepType = "repconv",
        rep_use_bn: bool = False,
        rep_act_mode: Literal["none", "relu"] = "none",
        mo_branches: int = 2,
        mo_use_1x1: bool = True,
        mo_use_identity: bool = True,
        out_clamp_mode: Literal["min255", "minclip", "clamp_0_255", "none"] = "minclip",
        skip_mode: SkipMode = "add",
        concat_htr: ConcatHTR = "3x3_3x3",
        use_global_add: bool = True,
        # NEW
        image_residual: bool = True,
        residual_base_mode: Literal["bilinear", "nearest", "bicubic"] = "bilinear",
        use_block_res_scale: bool = True,
        res_scale_init: float = 0.10,
    ):
        super().__init__()
        assert scale == 3, "This implementation targets x3 SR"

        self.scale = scale
        self.skip_mode = skip_mode
        self.concat_htr = concat_htr
        self.use_global_add = bool(use_global_add)
        self.rep_type = rep_type
        self.n_rep = int(n_rep)

        # NEW
        self.image_residual = bool(image_residual)
        self.residual_base_mode = residual_base_mode
        self.use_block_res_scale = bool(use_block_res_scale)

        self.conv_in = conv3x3(3, channels, bias=True)

        if rep_type == "repconv":
            blocks = [
                RepConv(
                    channels,
                    deploy=deploy,
                    rep_use_bn=rep_use_bn,
                    act_mode=rep_act_mode,
                )
                for _ in range(n_rep)
            ]
        elif rep_type == "mobileone":
            blocks = [
                MobileOneRepBlock(
                    channels,
                    deploy=deploy,
                    num_3x3_branches=mo_branches,
                    use_1x1=mo_use_1x1,
                    use_identity=mo_use_identity,
                    rep_use_bn=rep_use_bn,
                    act_mode=rep_act_mode,
                )
                for _ in range(n_rep)
            ]
        elif rep_type == "repdw":
            act = "relu" if rep_act_mode == "none" else rep_act_mode
            blocks = [
                RepDWBlock(
                    channels,
                    deploy=deploy,
                    rep_use_bn=rep_use_bn,
                    act_mode=act,
                    use_dw_1x1=True,
                    use_identity=True,
                )
                for _ in range(n_rep)
            ]
        else:
            raise ValueError(f"Unknown rep_type: {rep_type}")

        self.rep = nn.ModuleList(blocks)

        # NEW: learnable per-block residual scaling
        if self.use_block_res_scale:
            self.block_gammas = nn.Parameter(torch.full((n_rep,), float(res_scale_init), dtype=torch.float32))
        else:
            self.register_parameter("block_gammas", None)

        if skip_mode in ("add1x1", "concat_lr"):
            self.lr_proj = conv1x1(3, channels, bias=True)
        else:
            self.lr_proj = Identity()

        if skip_mode == "concat_raw":
            self.htr1, self.htr2 = _make_trans_layers(channels + 3, channels, concat_htr)
        elif skip_mode == "concat_lr":
            self.htr1, self.htr2 = _make_trans_layers(channels * 2, channels, concat_htr)
        else:
            self.htr1 = Identity()
            self.htr2 = Identity()

        self.conv_out = conv3x3(channels, 3 * (scale * scale), bias=True)
        self.out_clamp = make_output_clamp(out_clamp_mode)
        self.ps = nn.PixelShuffle(scale)

    def set_out_clamp_mode(self, mode: Literal["min255", "minclip", "clamp_0_255", "none"]):
        self.out_clamp = make_output_clamp(mode)

    def _upsample_base(self, lr: torch.Tensor) -> torch.Tensor:
        if self.residual_base_mode == "nearest":
            return F.interpolate(lr, scale_factor=self.scale, mode="nearest")
        elif self.residual_base_mode == "bilinear":
            return F.interpolate(lr, scale_factor=self.scale, mode="bilinear", align_corners=False)
        elif self.residual_base_mode == "bicubic":
            return F.interpolate(lr, scale_factor=self.scale, mode="bicubic", align_corners=False)
        else:
            raise ValueError(f"Unknown residual_base_mode: {self.residual_base_mode}")

    def forward(self, lr: torch.Tensor) -> torch.Tensor:
        feat = self.conv_in(lr)
        x = feat

        for i, blk in enumerate(self.rep):
            x_in = x
            x_blk = blk(x_in)

            if self.block_gammas is not None:
                gamma = torch.clamp(self.block_gammas[i], 0.0, 1.0).to(device=x_blk.device, dtype=x_blk.dtype)
                x = x_in + gamma * (x_blk - x_in)
            else:
                x = x_blk

        if self.use_global_add:
            x = x + feat

        if self.skip_mode == "add1x1":
            x = x + self.lr_proj(lr)
        elif self.skip_mode == "concat_lr":
            lr_f = self.lr_proj(lr)
            x = torch.cat([x, lr_f], dim=1)
            x = self.htr2(self.htr1(x))
        elif self.skip_mode == "concat_raw":
            x = torch.cat([x, lr], dim=1)
            x = self.htr2(self.htr1(x))

        # Main branch predicts image-space residual
        x = self.conv_out(x)
        x = self.ps(x)

        # NEW: global image residual
        if self.image_residual:
            base = self._upsample_base(lr)
            x = x + base

        x = self.out_clamp(x)
        return x

    @torch.no_grad()
    def switch_to_deploy(self) -> None:
        self.eval()
        for blk in self.rep:
            if hasattr(blk, "switch_to_deploy"):
                blk.switch_to_deploy()