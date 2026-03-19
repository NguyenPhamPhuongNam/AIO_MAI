import os
import sys
import time
import json
import csv
import math
import argparse
import random
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Tuple, List, Any
from collections import OrderedDict
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import AntSR
from data_div2k_pairs import DIV2KPairX3, resolve_div2k_paths

from qat_pact_fqkd_utils import (
    HookBasedFeatureExtractor,
    spatial_attention_loss,
    inject_pact_activations,
    PACTActivation,
    count_pact_modules,
)
# -------------------------
# Speed knobs
# -------------------------
torch.backends.cudnn.benchmark = True


# -------------------------
# Seed
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Logging utilities (tqdm-friendly)
# -------------------------
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


class FullMetricsOnlyFilter(logging.Filter):
    """File log writes ONLY records tagged with full_metrics=True."""
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, "full_metrics", False))


def setup_logger(out_dir: str, name: str = "train", file_full_metrics_only: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # avoid duplicate handlers
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    # console: keep everything
    h_console = TqdmLoggingHandler()
    h_console.setLevel(logging.INFO)
    h_console.setFormatter(fmt)
    logger.addHandler(h_console)

    # file: FULL metrics only (optional)
    log_path = os.path.join(out_dir, "train.log")
    h_file = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    h_file.setLevel(logging.INFO)
    h_file.setFormatter(fmt)
    if file_full_metrics_only:
        h_file.addFilter(FullMetricsOnlyFilter())
    logger.addHandler(h_file)

    return logger, log_path


def dump_run_config(out_dir: str, args, extra: dict):
    path = os.path.join(out_dir, "run_config.json")
    payload = {
        "cmd": " ".join(sys.argv),
        "args": vars(args),
        "extra": extra,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


# -------------------------
# Metrics CSV (dynamic header by val_shaves + report_ssim)
# -------------------------
def build_metrics_header(val_shaves: Tuple[int, ...], report_ssim: bool) -> List[str]:
    shaves = tuple(sorted(set(int(s) for s in val_shaves)))
    cols = ["time", "stage", "epoch", "lr", "train_loss", "best_key", "val_best"]
    for sh in shaves:
        cols += [
            f"psnr_sr_rgb_sh{sh}",
            f"psnr_sr_y_sh{sh}",
            f"psnr_bi_rgb_sh{sh}",
            f"psnr_bi_y_sh{sh}",
        ]
    if report_ssim:
        for sh in shaves:
            cols += [
                f"ssim_sr_rgb_sh{sh}",
                f"ssim_bi_rgb_sh{sh}",
            ]
    cols += ["note"]
    return cols


def ensure_metrics_csv(out_dir: str, header_cols: List[str]) -> str:
    """If metrics.csv exists but header mismatches -> create metrics_vN.csv."""
    path = os.path.join(out_dir, "metrics.csv")

    def _write_header(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header_cols)

    if not os.path.exists(path):
        _write_header(path)
        return path

    try:
        with open(path, "r", encoding="utf-8") as f:
            first = f.readline().strip().split(",")
        if first == header_cols:
            return path
    except Exception:
        pass

    for i in range(2, 100):
        vpath = os.path.join(out_dir, f"metrics_v{i}.csv")
        if not os.path.exists(vpath):
            _write_header(vpath)
            return vpath

    vpath = os.path.join(out_dir, f"metrics_v{int(time.time())}.csv")
    _write_header(vpath)
    return vpath


def append_metrics_csv(csv_path: str, header_cols: List[str], row: dict):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    out = []
    for c in header_cols:
        if c == "time":
            out.append(t)
        else:
            out.append(row.get(c, ""))
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(out)


def append_jsonl(path: str, obj: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# -------------------------
# Metrics helpers
# -------------------------
def _rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    # x: NCHW, 0..255
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _shave_border(a: torch.Tensor, shave: int) -> torch.Tensor:
    if shave <= 0:
        return a
    h, w = a.shape[-2], a.shape[-1]
    if h <= 2 * shave or w <= 2 * shave:
        return a
    return a[..., shave:-shave, shave:-shave]


def psnr_255(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def _required_metric_keys(shaves: Tuple[int, ...], report_ssim: bool) -> List[str]:
    shaves = tuple(sorted(set(int(s) for s in shaves)))
    keys: List[str] = []
    for sh in shaves:
        keys += [
            f"psnr_sr_rgb_sh{sh}",
            f"psnr_sr_y_sh{sh}",
            f"psnr_bi_rgb_sh{sh}",
            f"psnr_bi_y_sh{sh}",
        ]
    if report_ssim:
        for sh in shaves:
            keys += [
                f"ssim_sr_rgb_sh{sh}",
                f"ssim_bi_rgb_sh{sh}",
            ]
    return keys


def metrics_complete(m: Dict[str, float], shaves: Tuple[int, ...], report_ssim: bool) -> bool:
    if not isinstance(m, dict) or len(m) == 0:
        return False
    req = _required_metric_keys(shaves, report_ssim)
    for k in req:
        if k not in m:
            return False
        try:
            v = float(m[k])
        except Exception:
            return False
        if math.isnan(v) or math.isinf(v):
            return False
    return True


def format_full_metrics_line(
    stage: str,
    epoch: int,
    lr: float,
    train_loss: float,
    best_key: str,
    score: float,
    is_best: int,
    shaves: Tuple[int, ...],
    report_ssim: bool,
    m: Dict[str, float],
    note: str = "val",
) -> str:
    shaves = tuple(sorted(set(int(s) for s in shaves)))
    parts = [
        "[FULL_METRICS]",
        f"stage={stage}",
        f"epoch={epoch}",
        f"lr={lr:.6e}",
        f"train_loss={train_loss:.6f}",
        f"best_key={best_key}",
        f"score={score:.6f}",
        f"is_best={int(is_best)}",
        f"note={note}",
        f"val_shaves={list(shaves)}",
        f"report_ssim={int(bool(report_ssim))}",
    ]
    for sh in shaves:
        parts.append(f"psnr_sr_rgb_sh{sh}={float(m[f'psnr_sr_rgb_sh{sh}']):.6f}")
        parts.append(f"psnr_sr_y_sh{sh}={float(m[f'psnr_sr_y_sh{sh}']):.6f}")
        parts.append(f"psnr_bi_rgb_sh{sh}={float(m[f'psnr_bi_rgb_sh{sh}']):.6f}")
        parts.append(f"psnr_bi_y_sh{sh}={float(m[f'psnr_bi_y_sh{sh}']):.6f}")
        if report_ssim:
            parts.append(f"ssim_sr_rgb_sh{sh}={float(m[f'ssim_sr_rgb_sh{sh}']):.6f}")
            parts.append(f"ssim_bi_rgb_sh{sh}={float(m[f'ssim_bi_rgb_sh{sh}']):.6f}")
    return " ".join(parts)


# -------------------------
# Optional SSIM (simple, stable)
# -------------------------
def _gaussian_1d(win: int, sigma: float, device, dtype):
    coords = torch.arange(win, device=device, dtype=dtype) - (win - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g


def _ssim_per_channel(img1: torch.Tensor, img2: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    # img1,img2: NCHW in [0..255]
    device, dtype = img1.device, img1.dtype
    g1 = _gaussian_1d(win, sigma, device, dtype).view(1, 1, 1, win)
    g2 = _gaussian_1d(win, sigma, device, dtype).view(1, 1, win, 1)

    def blur(x):
        C = x.size(1)
        x = F.conv2d(x, g1.expand(C, 1, 1, win), padding=(0, win // 2), groups=C)
        x = F.conv2d(x, g2.expand(C, 1, win, 1), padding=(win // 2, 0), groups=C)
        return x

    mu1 = blur(img1)
    mu2 = blur(img2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2
    sigma1_sq = blur(img1 * img1) - mu1_sq
    sigma2_sq = blur(img2 * img2) - mu2_sq
    sigma12 = blur(img1 * img2) - mu12

    L = 255.0
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-12)
    return ssim_map.mean(dim=(-1, -2))


@torch.no_grad()
def validate_metrics_all(
    model,
    loader,
    device,
    scale: int,
    max_images: int = 0,
    shaves: Tuple[int, ...] = (0, 3),
    report_ssim: bool = False,
    ssim_win: int = 11,
    ssim_sigma: float = 1.5,
) -> Dict[str, float]:
    model.eval()
    shaves = tuple(sorted(set(int(s) for s in shaves)))
    acc: Dict[str, List[float]] = {}

    def _push(k: str, v: float):
        acc.setdefault(k, []).append(v)

    seen = 0
    for lr, hr in loader:
        lr = lr.to(device)
        hr = hr.to(device)

        bi = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        sr = model(lr)

        H, W = hr.shape[-2], hr.shape[-1]
        sr = sr[..., :H, :W]
        bi = bi[..., :H, :W]

        for sh in shaves:
            sr_s = _shave_border(sr, sh)
            bi_s = _shave_border(bi, sh)
            hr_s = _shave_border(hr, sh)

            sr_rgb = torch.clamp(sr_s, 0.0, 255.0)
            bi_rgb = torch.clamp(bi_s, 0.0, 255.0)
            hr_rgb = torch.clamp(hr_s, 0.0, 255.0)

            _push(f"psnr_sr_rgb_sh{sh}", psnr_255(sr_rgb, hr_rgb))
            _push(f"psnr_bi_rgb_sh{sh}", psnr_255(bi_rgb, hr_rgb))

            sr_y = torch.clamp(_rgb_to_y(sr_s), 0.0, 255.0)
            bi_y = torch.clamp(_rgb_to_y(bi_s), 0.0, 255.0)
            hr_y = torch.clamp(_rgb_to_y(hr_s), 0.0, 255.0)

            _push(f"psnr_sr_y_sh{sh}", psnr_255(sr_y, hr_y))
            _push(f"psnr_bi_y_sh{sh}", psnr_255(bi_y, hr_y))

            if report_ssim:
                ssim_sr = _ssim_per_channel(sr_rgb, hr_rgb, win=ssim_win, sigma=ssim_sigma).mean().item()
                ssim_bi = _ssim_per_channel(bi_rgb, hr_rgb, win=ssim_win, sigma=ssim_sigma).mean().item()
                _push(f"ssim_sr_rgb_sh{sh}", float(ssim_sr))
                _push(f"ssim_bi_rgb_sh{sh}", float(ssim_bi))

        seen += 1
        if max_images > 0 and seen >= max_images:
            break

    return {k: float(np.mean(v)) if len(v) else 0.0 for k, v in acc.items()}



# EMA (parameters + BN buffers)
# -------------------------
def _named_bn_buffers(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            prefix = f"{module_name}." if module_name else ""
            out[prefix + "running_mean"] = module.running_mean
            out[prefix + "running_var"] = module.running_var
            out[prefix + "num_batches_tracked"] = module.num_batches_tracked
    return out


class EMA:
    """
    EMA for:
      - trainable parameters
      - BN buffers (running_mean, running_var, num_batches_tracked)

    Backward-compatible:
      - old checkpoints may only contain {"shadow": ...} for parameters
      - new checkpoints contain {"shadow_params": ..., "shadow_bn": ...}
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.shadow_bn: Dict[str, torch.Tensor] = {}
        self.backup_params: Dict[str, torch.Tensor] = {}
        self.backup_bn: Dict[str, torch.Tensor] = {}
        self._init_from_model(model)

    def _init_from_model(self, model: torch.nn.Module):
        self.shadow_params = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow_params[n] = p.detach().clone()

        self.shadow_bn = {}
        for n, b in _named_bn_buffers(model).items():
            self.shadow_bn[n] = b.detach().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay

        # parameters
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n not in self.shadow_params:
                self.shadow_params[n] = p.detach().clone()
                continue
            self.shadow_params[n].mul_(d).add_(p.detach(), alpha=1.0 - d)

        # BN buffers
        cur_bn = _named_bn_buffers(model)
        for n, b in cur_bn.items():
            if n not in self.shadow_bn:
                self.shadow_bn[n] = b.detach().clone()
                continue

            if torch.is_floating_point(b):
                self.shadow_bn[n].mul_(d).add_(b.detach(), alpha=1.0 - d)
            else:
                # integer buffers like num_batches_tracked: keep latest exact value
                self.shadow_bn[n] = b.detach().clone()

    def apply(self, model: torch.nn.Module):
        self.backup_params = {}
        self.backup_bn = {}

        # parameters
        for n, p in model.named_parameters():
            if n in self.shadow_params:
                self.backup_params[n] = p.detach().clone()
                p.data.copy_(self.shadow_params[n].data)

        # BN buffers
        cur_bn = _named_bn_buffers(model)
        for n, b in cur_bn.items():
            if n in self.shadow_bn:
                self.backup_bn[n] = b.detach().clone()
                b.data.copy_(self.shadow_bn[n].data)

    def restore(self, model: torch.nn.Module):
        # parameters
        for n, p in model.named_parameters():
            if n in self.backup_params:
                p.data.copy_(self.backup_params[n].data)

        # BN buffers
        cur_bn = _named_bn_buffers(model)
        for n, b in cur_bn.items():
            if n in self.backup_bn:
                b.data.copy_(self.backup_bn[n].data)

        self.backup_params = {}
        self.backup_bn = {}

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow_params": {k: v.detach().cpu() for k, v in self.shadow_params.items()},
            "shadow_bn": {k: v.detach().cpu() for k, v in self.shadow_bn.items()},
        }

    def load_state_dict(self, sd: dict, model: torch.nn.Module):
        self.decay = float(sd.get("decay", self.decay))

        # backward compatibility
        old_shadow = sd.get("shadow", None)
        shadow_params = sd.get("shadow_params", old_shadow if old_shadow is not None else {})
        shadow_bn = sd.get("shadow_bn", {})

        self.shadow_params = {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if n in shadow_params:
                self.shadow_params[n] = shadow_params[n].to(p.device, dtype=p.dtype).clone()
            else:
                self.shadow_params[n] = p.detach().clone()

        self.shadow_bn = {}
        cur_bn = _named_bn_buffers(model)
        for n, b in cur_bn.items():
            if n in shadow_bn:
                self.shadow_bn[n] = shadow_bn[n].to(b.device, dtype=b.dtype).clone()
            else:
                self.shadow_bn[n] = b.detach().clone()


# -------------------------
# DCT loss (FFT-based DCT-II) with CACHE
# -------------------------
_DCT_CACHE = {}

def _get_dct_cache(N: int, device: torch.device, dtype: torch.dtype):
    key = (N, device.type, device.index if device.type == "cuda" else -1, dtype)
    if key in _DCT_CACHE:
        return _DCT_CACHE[key]
    even_idx = torch.arange(0, N, 2, device=device)
    odd_idx = torch.arange(N - 1, -1, -2, device=device)
    cplx_dtype = torch.complex64 if dtype in (torch.float16, torch.float32) else torch.complex128
    k = torch.arange(N, device=device, dtype=dtype)
    W = torch.exp(-1j * math.pi * k / (2.0 * N)).to(cplx_dtype)
    _DCT_CACHE[key] = {"even": even_idx, "odd": odd_idx, "W": W}
    return _DCT_CACHE[key]

def _dct_1d(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    N = x.size(dim)
    cache = _get_dct_cache(N, x.device, x.dtype)
    even = x.index_select(dim, cache["even"])
    odd = x.index_select(dim, cache["odd"])
    v = torch.cat([even, odd], dim=dim)
    V = torch.fft.fft(v, dim=dim)
    out = (V * cache["W"]).real * 2.0
    return out

def dct_2d(x: torch.Tensor) -> torch.Tensor:
    x = _dct_1d(x, dim=-1)
    x = _dct_1d(x, dim=-2)
    return x

def dct_l1_loss(sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
    sr_f = sr.float()
    hr_f = hr.float()
    return torch.mean(torch.abs(dct_2d(sr_f) - dct_2d(hr_f)))


# -------------------------
# Teacher Cache (KD from precomputed SR)
# -------------------------
class TeacherCache:
    """
    Supports:
      - cache_dir/{stem}.png (uint8 RGB)
      - cache_dir/{stem}.npy (HWC float16/float32)
      - cache_dir/{stem}.npz (arr=HWC float16/float32)
    Returns crops in BCHW float32, range [0..255]
    """
    def __init__(self, cache_dir: str, max_keep: int = 64, prefer: str = "png"):
        self.cache_dir = cache_dir
        self.max_keep = int(max_keep)
        self.prefer = prefer
        self.mem = OrderedDict()  # stem -> np array HWC

    def _load_any(self, stem: str) -> np.ndarray:
        if stem in self.mem:
            arr = self.mem.pop(stem)
            self.mem[stem] = arr
            return arr

        p_png = os.path.join(self.cache_dir, f"{stem}.png")
        p_npy = os.path.join(self.cache_dir, f"{stem}.npy")
        p_npz = os.path.join(self.cache_dir, f"{stem}.npz")

        if self.prefer == "png" and os.path.exists(p_png):
            im = Image.open(p_png).convert("RGB")
            arr = np.array(im)
        elif os.path.exists(p_npy):
            arr = np.load(p_npy, mmap_mode="r")
        elif os.path.exists(p_npz):
            z = np.load(p_npz)
            arr = z["arr"]
        elif os.path.exists(p_png):
            im = Image.open(p_png).convert("RGB")
            arr = np.array(im)
        else:
            raise FileNotFoundError(f"Teacher cache missing for stem={stem} in {self.cache_dir}")

        self.mem[stem] = arr
        if len(self.mem) > self.max_keep:
            self.mem.popitem(last=False)
        return arr

    def get_batch_crop_255(self, metas_list: List[Dict[str, Any]], device):
        crops = []
        for m in metas_list:
            arr = self._load_any(m["stem"])  # HWC
            x, y = int(m["x"]), int(m["y"])
            ps, scale = int(m["ps"]), int(m["scale"])
            xs, ys = x * scale, y * scale
            hs = ps * scale
            crop = arr[ys:ys + hs, xs:xs + hs, :]  # HWC
            if crop.dtype == np.uint8:
                t = torch.from_numpy(crop).to(torch.float32)
            else:
                t = torch.from_numpy(crop.astype(np.float32))
            t = t.permute(2, 0, 1)  # CHW
            crops.append(t)
        return torch.stack(crops, 0).to(device)  # BCHW 0..255


def _normalize_metas(metas: Any, batch_size: int) -> Optional[List[Dict[str, Any]]]:
    """DataLoader collate for dict => dict of lists/tensors. Convert to list[dict] length B."""
    if metas is None:
        return None
    if isinstance(metas, list) and (len(metas) == 0 or isinstance(metas[0], dict)):
        return metas
    if isinstance(metas, dict):
        out: List[Dict[str, Any]] = []
        for i in range(batch_size):
            mi: Dict[str, Any] = {}
            for k, v in metas.items():
                if isinstance(v, torch.Tensor):
                    mi[k] = int(v[i].item())
                elif isinstance(v, (list, tuple)):
                    mi[k] = v[i]
                else:
                    mi[k] = v
            if isinstance(mi.get("stem"), (list, tuple)):
                mi["stem"] = mi["stem"][0]
            out.append(mi)
        return out
    return None


# -------------------------
# Tricks
# -------------------------
def channel_shuffle_rgb(lr: torch.Tensor, hr: torch.Tensor):
    perm = torch.randperm(3, device=lr.device)
    return lr[:, perm], hr[:, perm]


@torch.no_grad()
def apply_weight_clipping(model: torch.nn.Module, clip_other: float = 2.0, clip_rep: float = 3.0):
    for name, p in model.named_parameters():
        if not name.endswith(".weight"):
            continue
        if p.ndim != 4:
            continue
        lim = clip_rep if "rep." in name else clip_other
        p.clamp_(-lim, lim)


def freeze_bn_(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eval()
            if m.weight is not None:
                m.weight.requires_grad_(False)
            if m.bias is not None:
                m.bias.requires_grad_(False)

@torch.no_grad()
def recalibrate_bn_stats(
    model: torch.nn.Module,
    loader,
    device,
    num_batches: int = 64,
    channels_last: bool = True,
    logger=None,
):
    """
    Recompute BN running stats using a few forward passes with no grad.
    Useful right before switch_to_deploy() so fused Conv+BN uses cleaner stats.
    """
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    if len(bn_layers) == 0:
        if logger is not None:
            logger.info("[bn-recalib] skipped (no BN layers)")
        return

    num_batches = int(num_batches)
    if num_batches <= 0:
        if logger is not None:
            logger.info("[bn-recalib] skipped (num_batches <= 0)")
        return

    was_training = model.training

    # reset BN stats
    for m in bn_layers:
        m.reset_running_stats()

    # only BN layers in train mode, others eval
    model.eval()
    for m in bn_layers:
        m.train()

    seen = 0
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            lr_img = batch[0]
        else:
            lr_img = batch

        lr_img = lr_img.to(device)
        if channels_last:
            try:
                lr_img = lr_img.to(memory_format=torch.channels_last)
            except Exception:
                pass

        _ = model(lr_img)
        seen += 1
        if seen >= num_batches:
            break

    model.train(was_training)

    if logger is not None:
        logger.info(f"[bn-recalib] updated BN stats with {seen} batches")


# -------------------------
# Schedulers
# -------------------------
def make_cosine_warmup_scheduler(opt, total_epochs: int, warmup_ratio: float = 0.1):
    warmup_epochs = max(1, int(total_epochs * warmup_ratio))
    def lr_lambda(ep: int):
        if ep < warmup_epochs:
            return float(ep + 1) / float(warmup_epochs)
        t = (ep - warmup_epochs) / max(1, (total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * t))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def make_step_halve_scheduler(opt, step_size: int = 40, gamma: float = 0.5):
    return torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)


# -------------------------
# QAT helpers
# -------------------------
def prepare_qat_model(
    model: torch.nn.Module,
    backend: str = "qnnpack",
    mode: str = "eager",
    *,
    enable_pact: bool = False,
    pact_cfg: Optional[dict] = None,
) -> torch.nn.Module:
    import torch.ao.quantization as tq
    torch.backends.quantized.engine = backend
    model.train()
    mode = (mode or "eager").lower()

    if mode == "eager":
        if enable_pact:
            print("[WARN] enable_pact_qat=True is intended for FX-QAT. Continuing without PACT injection in eager mode.")
        model.qconfig = tq.get_default_qat_qconfig(backend)
        tq.prepare_qat(model, inplace=True)
        return model

    if mode == "fx":
        from torch.ao.quantization.quantize_fx import QConfigMapping, prepare_qat_fx, PrepareCustomConfig
        from bbox_sr3.student.model import RepConv, RepDWBlock, MobileOneRepBlock

        qconfig = tq.get_default_qat_qconfig(backend)

        # Remember current model device/dtype before injecting any new modules
        ref_param = next(model.parameters())
        model_device = ref_param.device
        model_dtype = ref_param.dtype

        # Inject PACT BEFORE prepare_qat_fx.
        # IMPORTANT: replace_relu_only=False so PACT still works when block.act is Identity().
        if enable_pact:
            cfg = pact_cfg or {}
            replaced_acts, replaced_output = inject_pact_activations(
                model,
                rep_block_classes=(RepConv, RepDWBlock, MobileOneRepBlock),
                replace_relu_only=False,
                init_min=cfg.get("init_min", -2.0),
                init_max=cfg.get("init_max", 2.0),
                pact_on_output=cfg.get("pact_on_output", False),
                out_init_min=cfg.get("out_min", 0.0),
                out_init_max=cfg.get("out_max", 255.0),
            )

            # VERY IMPORTANT:
            # Newly injected modules may default to CPU unless explicitly moved.
            # Force the whole model back to the original device/dtype.
            model.to(device=model_device, dtype=model_dtype)

            n_pact = count_pact_modules(model)
            print(
                f"[pact] injected={n_pact} "
                f"block_acts={len(replaced_acts)} output={int(bool(replaced_output))}"
            )
            if n_pact == 0:
                raise RuntimeError(
                    "enable_pact_qat=True nhưng không inject được PACT nào. "
                    "Hãy kiểm tra cấu hình activation/output clamp."
                )

        qconfig_mapping = QConfigMapping().set_global(qconfig)

        # Keep PACT as leaf module (won't be overridden by FX tracing)
        prepare_custom_config = PrepareCustomConfig().set_non_traceable_module_classes([PACTActivation])

        example_inputs = getattr(model, "_qat_example_inputs", None)
        if example_inputs is None:
            example_inputs = (torch.randn(1, 3, 128, 128, device=model_device, dtype=model_dtype),)

        prepared = prepare_qat_fx(
            model,
            qconfig_mapping=qconfig_mapping,
            example_inputs=example_inputs,
            prepare_custom_config=prepare_custom_config,
        )
        return prepared

    raise ValueError(f"Unknown qat mode: {mode}")


def strip_qat_to_clean_state(
    clean_model: torch.nn.Module,
    qat_state: dict,
    logger=None,
    min_keep_ratio: float = 0.80,
) -> dict:
    clean_sd = clean_model.state_dict()

    kept = {}
    dropped_from_qat = []
    missing_for_clean = []

    for k, v in clean_sd.items():
        if k in qat_state and hasattr(qat_state[k], "shape") and qat_state[k].shape == v.shape:
            kept[k] = qat_state[k]
        else:
            missing_for_clean.append(k)

    for k, v in qat_state.items():
        if not (k in clean_sd and hasattr(v, "shape") and v.shape == clean_sd[k].shape):
            dropped_from_qat.append(k)

    keep_ratio = len(kept) / max(1, len(clean_sd))

    if logger is not None:
        logger.info(
            f"[qat-export] clean_keys={len(clean_sd)} kept={len(kept)} "
            f"missing_for_clean={len(missing_for_clean)} dropped_from_qat={len(dropped_from_qat)} "
            f"keep_ratio={keep_ratio:.3f}"
        )
        if missing_for_clean[:10]:
            logger.info(f"[qat-export] first missing clean keys: {missing_for_clean[:10]}")
        if dropped_from_qat[:10]:
            logger.info(f"[qat-export] first dropped QAT keys: {dropped_from_qat[:10]}")

    if keep_ratio < float(min_keep_ratio):
        raise RuntimeError(
            f"Too few QAT weights matched clean model: "
            f"kept={len(kept)} clean={len(clean_sd)} keep_ratio={keep_ratio:.3f}"
        )

    return kept


def qat_schedule_step(model: torch.nn.Module, local_ep: int, disable_observer_ep: int, freeze_fakequant_ep: int):
    import torch.ao.quantization as tq
    if disable_observer_ep >= 0 and local_ep == disable_observer_ep:
        tq.disable_observer(model)
    if freeze_fakequant_ep >= 0 and local_ep == freeze_fakequant_ep:
        if hasattr(tq, "freeze_fake_quant"):
            tq.freeze_fake_quant(model)
        else:
            tq.disable_observer(model)


# -------------------------
# CKPT helpers
# -------------------------
def load_ckpt_weights(model, ckpt_path, device, prefer_ema: bool = True) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Invalid ckpt: {ckpt_path}")
    if prefer_ema and ("model_ema" in ckpt):
        sd = ckpt["model_ema"]
    elif "model" in ckpt:
        sd = ckpt["model"]
    else:
        raise RuntimeError(f"Invalid ckpt keys: {ckpt.keys()}")
    model.load_state_dict(sd, strict=True)
    model.to(device)
    return ckpt


def load_teacher_antsr_export(teacher_ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """Teacher export ckpt with keys {'cfg','model'}. AntSR teacher expects LR in [0..255]."""
    tckpt = torch.load(teacher_ckpt_path, map_location="cpu")
    if not (isinstance(tckpt, dict) and "cfg" in tckpt and "model" in tckpt):
        raise RuntimeError("Teacher ckpt must be export ckpt with keys {'cfg','model'}.")
    tcfg = tckpt["cfg"]
    tsd = tckpt["model"]
    teacher = AntSR(deploy=True, **tcfg).eval().to(device)
    teacher.load_state_dict(tsd, strict=True)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher._expects_01 = False   # input convention
    teacher._returns_01 = False   # output convention
    return teacher


def load_teacher_mambair(mambair_repo: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load pretrained MambaIRv2 LightSR x3 as teacher.
    MambaIR teacher expects LR in [0..1].
    """
    from mambair_teacher import load_teacher as load_mambair_teacher
    teacher = load_mambair_teacher(
        mambair_repo=mambair_repo,
        ckpt_path=ckpt_path,
        device=str(device),
        strict=False,
    )
    teacher._expects_01 = True    # input convention
    teacher._returns_01 = True    # output convention
    return teacher


# -------------------------
# Extra losses
# -------------------------
def charbonnier_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.mean(torch.sqrt((x - y) * (x - y) + eps * eps))


def multiscale_aux_loss(sr01: torch.Tensor, hr01: torch.Tensor, aux_scale: float = 2/3) -> torch.Tensor:
    sr2 = F.interpolate(sr01, scale_factor=aux_scale, mode="bicubic", align_corners=False)
    hr2 = F.interpolate(hr01, scale_factor=aux_scale, mode="bicubic", align_corners=False)
    return F.l1_loss(sr2, hr2)


def haar_dwt2(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh


def haar_wavelet_loss(sr01: torch.Tensor, hr01: torch.Tensor, levels: int = 1) -> torch.Tensor:
    loss = 0.0
    x_s, x_h = sr01, hr01
    for _ in range(max(1, int(levels))):
        ll_s, lh_s, hl_s, hh_s = haar_dwt2(x_s)
        ll_h, lh_h, hl_h, hh_h = haar_dwt2(x_h)
        loss = loss + (F.l1_loss(lh_s, lh_h) + F.l1_loss(hl_s, hl_h) + F.l1_loss(hh_s, hh_h))
        x_s, x_h = ll_s, ll_h
    return loss


def haar_wavelet_hf_loss(sr01: torch.Tensor, ref01: torch.Tensor, levels: int = 1) -> torch.Tensor:
    """
    Distill only high-frequency wavelet subbands: LH, HL, HH.
    sr01, ref01: BCHW in [0..1]
    """
    loss = 0.0
    x_s, x_r = sr01, ref01
    for _ in range(max(1, int(levels))):
        ll_s, lh_s, hl_s, hh_s = haar_dwt2(x_s)
        ll_r, lh_r, hl_r, hh_r = haar_dwt2(x_r)
        loss = loss + (
            F.l1_loss(lh_s, lh_r) +
            F.l1_loss(hl_s, hl_r) +
            F.l1_loss(hh_s, hh_r)
        )
        x_s, x_r = ll_s, ll_r
    return loss


# -------------------------
# KD schedule helpers
# -------------------------
def kd_linear_warmup_decay(local_ep: int, total_eps: int, start: float, end: float, warmup_ratio: float = 0.15) -> float:
    total_eps = max(1, int(total_eps))
    warm = max(0, min(total_eps, int(total_eps * warmup_ratio)))
    if warm == 0:
        t = local_ep / max(1, total_eps - 1)
        return float(start + (end - start) * t)
    if local_ep < warm:
        return float(start)
    t = (local_ep - warm) / max(1, (total_eps - warm - 1))
    return float(start + (end - start) * t)



def kd_three_phase_schedule(
    local_ep: int,
    total_eps: int,
    high: float = 0.5,
    mid: float = 0.1,
    low: float = 0.05,
    p1: float = 0.3,
    p2: float = 0.7,
) -> float:
    total_eps = max(1, int(total_eps))
    t = (local_ep + 1) / total_eps

    if t <= p1:
        return float(high)
    if t <= p2:
        alpha = (t - p1) / max(1e-8, (p2 - p1))
        return float(high + (mid - high) * alpha)
    return float(low)


def confidence_weight_map(
    t01: torch.Tensor,
    hr01: torch.Tensor,
    gamma: float = 12.0,
    min_w: float = 0.05,
    max_w: float = 1.0,
) -> torch.Tensor:
    # pixel-wise / local confidence map
    err = torch.mean(torch.abs(t01 - hr01), dim=1, keepdim=True)  # (B,1,H,W)
    w = torch.exp(-gamma * err)
    return w.clamp(min_w, max_w)


def residual_kd_loss(
    sr01: torch.Tensor,
    t01: torch.Tensor,
    lr_img_255: torch.Tensor,
    hr_shape_hw: Tuple[int, int],
) -> torch.Tensor:
    H, W = hr_shape_hw
    bi01 = F.interpolate(lr_img_255, size=(H, W), mode="bicubic", align_corners=False) / 255.0
    sr_res = sr01 - bi01
    t_res = t01 - bi01
    return F.l1_loss(sr_res, t_res)

# -------------------------
# Loss builder (on 0..1)
# -------------------------
def compute_base_loss(mode: str, sr01: torch.Tensor, hr01: torch.Tensor, dct_w: float, charb_eps: float = 1e-3) -> torch.Tensor:
    mode = mode.lower()
    if mode == "dct":
        return dct_l1_loss(sr01, hr01)

    if mode.startswith("l2"):
        base = F.mse_loss(sr01, hr01)
    elif mode.startswith("charb"):
        base = charbonnier_loss(sr01, hr01, eps=charb_eps)
    else:
        base = F.l1_loss(sr01, hr01)

    if "dct" in mode:
        base = base + float(dct_w) * dct_l1_loss(sr01, hr01)
    return base


# -------------------------
# KD helpers (robust range + crop)
# -------------------------
@torch.no_grad()
def teacher_to_01_crop(
    teacher_out: torch.Tensor,
    hr_shape_hw: Tuple[int, int],
    returns_01: bool,
) -> torch.Tensor:
    """
    teacher_out: (B,3,H,W)
    returns_01:
      True  -> teacher output already in [0..1]
      False -> teacher output in [0..255]
    """
    if teacher_out.dtype not in (torch.float16, torch.float32):
        teacher_out = teacher_out.to(torch.float32)

    t01 = teacher_out if bool(returns_01) else (teacher_out / 255.0)

    H, W = hr_shape_hw
    t01 = t01[..., :H, :W].clamp(0.0, 1.0)
    return t01

# -------------------------
# FQKD hook name builder (FX-safe)
# -------------------------
def build_fqkd_hook_names(rep_type: str, indices: List[int], qat_on_deploy: bool) -> List[str]:
    """
    For FX-QAT, forward hooks fire on modules that are actually called as call_module.
    In deploy_before_qat path:
      - repconv/mobileone deploy => each block has .reparam Conv2d called
      - repdw => use .pw (pointwise Conv2d) called
    """
    rep_type = (rep_type or "").lower()
    names: List[str] = []
    for i in indices:
        if rep_type in ("repdw",):
            names.append(f"rep.{i}.pw")
        else:
            if qat_on_deploy:
                names.append(f"rep.{i}.reparam")
            else:
                # fallback (may not fire under FX if traced through)
                names.append(f"rep.{i}")
    return names


# -------------------------
# Train one stage
# -------------------------
def train_stage(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int,
    base_lr: float,
    out_dir: str,
    start_epoch: int,
    val_max: int,
    stage_tag: str,
    # full val
    val_shaves: Tuple[int, ...] = (0, 3),
    best_key: str = "psnr_sr_rgb_sh0",
    report_ssim: bool = False,
    ssim_win: int = 11,
    ssim_sigma: float = 1.5,
    grad_clip: float = 0.0,
    scheduler_type: str = "none",
    channel_shuffle: bool = False,
    loss_mode: str = "l1",
    dct_w: float = 0.05,
    # training-only extra losses
    ms_w: float = 0.0,
    ms_scale: float = 2/3,
    wav_w: float = 0.0,
    wav_levels: int = 1,
    charb_eps: float = 1e-3,
    weight_clip: bool = True,
    wc_other: float = 2.0,
    wc_rep: float = 3.0,
    val_every: int = 1,
    # EMA
    ema: Optional[EMA] = None,
    # KD teacher (output KD)
    teacher: Optional[torch.nn.Module] = None,
    kd_w: float = 0.0,
    kd_loss: str = "l1",
    kd_freq_w: float = 0.0,
    kd_res_w: float = 0.0,
    kd_wave_w: float = 0.0,
    kd_wave_levels: int = 1,
    # KD schedules (epoch-based; -1 disables)
    kd_w_start: float = -1.0,
    kd_w_end: float = -1.0,
    kd_w_warmup: float = 0.15,
    kd_freq_w_start: float = -1.0,
    kd_freq_w_end: float = -1.0,
    kd_freq_w_warmup: float = 0.15,
    # Teacher cache (for KD)
    teacher_cache_dir: Optional[str] = None,
    # BN freeze
    freeze_bn_epoch: int = -1,
    # QAT schedule (stage-local)
    qat_disable_observer_ep: int = -1,
    qat_freeze_fakequant_ep: int = -1,
    # RESUME
    resume_ckpt: Optional[dict] = None,
    # logging
    logger=None,
    metrics_csv: Optional[str] = None,
    metrics_header: Optional[List[str]] = None,
    val_jsonl: Optional[str] = None,
    log_every_steps: int = 200,
    # perf
    use_amp: bool = True,
    channels_last: bool = True,
    # debug
    debug_range_steps: int = 0,
    # write restrictions
    file_log_full_metrics_only: bool = True,
    csv_full_metrics_only: bool = True,
    # ---- FQKD (feature distill) ----
    fqkd_teacher: Optional[torch.nn.Module] = None,
    fqkd_layers: Optional[List[str]] = None,
    fqkd_w: float = 0.0,
    # ---- PACT regularization ----
    pact_l2: float = 0.0,

    kd_conf_gamma: float = 12.0,
    kd_conf_min: float = 0.05,
    kd_conf_max: float = 1.0,
    kd_sched_p1: float = 0.3,
    kd_sched_p2: float = 0.7,
    kd_low_floor: float = 0.05,
    kd_use_residual: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, f"ckpt_best_{stage_tag}.pt")
    last_path = os.path.join(out_dir, f"ckpt_last_{stage_tag}.pt")

    # Safety: if KD requested but no teacher/cache, disable to avoid crashing
    if (kd_w and kd_w > 0) and (teacher is None) and (teacher_cache_dir is None):
        if logger is not None:
            logger.info(f"[warn] KD requested (kd_w={kd_w}) but no teacher/teacher_cache_dir. Disabling KD for {stage_tag}.")
        kd_w = 0.0
        kd_freq_w = 0.0
        kd_w_start = kd_w_end = -1.0
        kd_freq_w_start = kd_freq_w_end = -1.0

    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    if channels_last:
        try:
            model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    if scheduler_type == "cos_warmup":
        sch = make_cosine_warmup_scheduler(opt, total_epochs=epochs, warmup_ratio=0.1)
    elif scheduler_type == "step_halve":
        sch = make_step_halve_scheduler(opt, step_size=40, gamma=0.5)
    else:
        sch = None

    tcache = TeacherCache(teacher_cache_dir, max_keep=256, prefer="npz") if teacher_cache_dir else None

    local_ep_start = 0
    best_score = -1e9
    bn_frozen = False

    # ---- FQKD hooks ----
    # ---- FQKD hooks ----
    ext_t = ext_s = None
    if (fqkd_w and fqkd_w > 0) and (fqkd_teacher is not None) and fqkd_layers:
        fqkd_teacher.eval()
        for p in fqkd_teacher.parameters():
            p.requires_grad_(False)

        ext_t = HookBasedFeatureExtractor(fqkd_teacher, fqkd_layers)
        ext_s = HookBasedFeatureExtractor(model, fqkd_layers)

        found_t = getattr(ext_t, "found_layer_names", [])
        found_s = getattr(ext_s, "found_layer_names", [])
        missing_t = getattr(ext_t, "missing_layer_names", [])
        missing_s = getattr(ext_s, "missing_layer_names", [])

        if len(found_t) == 0 or len(found_s) == 0:
            raise RuntimeError(
                f"FQKD enabled but no hooks matched requested layers. "
                f"requested={fqkd_layers} teacher_found={found_t} student_found={found_s}"
            )

        if logger is not None:
            logger.info(
                f"[fqkd] enabled fqkd_w={fqkd_w} "
                f"requested={fqkd_layers} teacher_found={found_t} student_found={found_s}"
            )
            if missing_t:
                logger.info(f"[fqkd] teacher missing hooks: {missing_t}")
            if missing_s:
                logger.info(f"[fqkd] student missing hooks: {missing_s}")

    if logger is not None:
        logger.info(
            f"[stage-start] {stage_tag} start_epoch={start_epoch} epochs={epochs} base_lr={base_lr} "
            f"loss_mode={loss_mode} dct_w={dct_w} ms_w={ms_w} wav_w={wav_w} "
            f"kd_w={kd_w} kd_freq_w={kd_freq_w} kd_res_w={kd_res_w} "
            f"kd_wave_w={kd_wave_w} kd_wave_levels={kd_wave_levels} "
            f"teacher_cache_dir={teacher_cache_dir} "
            f"fqkd_w={fqkd_w} pact_l2={pact_l2} "
            f"val_every={val_every} val_max={val_max} best_key={best_key} "
            f"use_amp={int(bool(use_amp))} channels_last={int(bool(channels_last))}"
        )

    if resume_ckpt is not None:
        if "model" in resume_ckpt:
            model.load_state_dict(resume_ckpt["model"], strict=True)
        elif "model_ema" in resume_ckpt:
            model.load_state_dict(resume_ckpt["model_ema"], strict=True)

        if "opt" in resume_ckpt:
            try:
                opt.load_state_dict(resume_ckpt["opt"])
            except Exception as e:
                if logger is not None:
                    logger.info(f"[resume] WARN: cannot load optimizer state: {e}")

        if sch is not None and "sch" in resume_ckpt and resume_ckpt["sch"] is not None:
            try:
                sch.load_state_dict(resume_ckpt["sch"])
            except Exception as e:
                if logger is not None:
                    logger.info(f"[resume] WARN: cannot load scheduler state: {e}")

        if ema is not None and "ema" in resume_ckpt and resume_ckpt["ema"] is not None:
            try:
                ema.load_state_dict(resume_ckpt["ema"], model)
            except Exception as e:
                if logger is not None:
                    logger.info(f"[resume] WARN: cannot load EMA shadow: {e}")

        best_score = float(resume_ckpt.get("best_score", best_score))
        last_epoch = int(resume_ckpt.get("epoch", start_epoch - 1))
        local_ep_start = max(0, (last_epoch - start_epoch + 1))
        if logger is not None:
            logger.info(f"[resume] stage={stage_tag} last_epoch={last_epoch} -> local_ep_start={local_ep_start}/{epochs}")

    for local_ep in range(local_ep_start, epochs):
        ep = start_epoch + local_ep

        model.train()

        # IMPORTANT:
        # freeze_bn_() must be called AFTER model.train(),
        # otherwise model.train() will turn BN layers back to train mode.
        if (freeze_bn_epoch >= 0) and (ep >= freeze_bn_epoch):
            freeze_bn_(model)
            if (not bn_frozen) and (logger is not None):
                logger.info(f"[bn] frozen BN at epoch {ep} (freeze_bn_epoch={freeze_bn_epoch})")
            bn_frozen = True

        if stage_tag.startswith("s3") and (qat_disable_observer_ep >= 0 or qat_freeze_fakequant_ep >= 0):
            qat_schedule_step(model, local_ep, qat_disable_observer_ep, qat_freeze_fakequant_ep)

        # KD schedules
        kd_w_cur = kd_w
        kd_freq_w_cur = kd_freq_w

        if kd_w_start >= 0 and kd_w_end >= 0:
            kd_w_cur = kd_three_phase_schedule(
                local_ep=local_ep,
                total_eps=epochs,
                high=kd_w_start,
                mid=kd_w_end,
                low=max(kd_low_floor, 0.5 * kd_w_end),
                p1=kd_sched_p1,
                p2=kd_sched_p2,
            )

        if kd_freq_w_start >= 0 and kd_freq_w_end >= 0:
            kd_freq_w_cur = kd_three_phase_schedule(
                local_ep=local_ep,
                total_eps=epochs,
                high=kd_freq_w_start,
                mid=kd_freq_w_end,
                low=max(0.0, min(kd_low_floor, kd_freq_w_end)),
                p1=kd_sched_p1,
                p2=kd_sched_p2,
            )

        pbar = tqdm(train_loader, desc=f"[{stage_tag}] epoch {ep} lr={opt.param_groups[0]['lr']:.2e}")
        losses = []

        for step_idx, batch in enumerate(pbar, start=1):
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                lr_img, hr_img, metas = batch
            else:
                lr_img, hr_img = batch
                metas = None

            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            # Expect [0..255]
            if channel_shuffle:
                lr_img, hr_img = channel_shuffle_rgb(lr_img, hr_img)

            if channels_last:
                try:
                    lr_img = lr_img.to(memory_format=torch.channels_last)
                    hr_img = hr_img.to(memory_format=torch.channels_last)
                except Exception:
                    pass

            if logger is not None and debug_range_steps > 0 and (local_ep == 0) and (step_idx <= debug_range_steps):
                logger.info(
                    f"[dbg] {stage_tag} ep={ep} step={step_idx} "
                    f"lr_range={lr_img.min().item():.4f}..{lr_img.max().item():.4f} "
                    f"hr_range={hr_img.min().item():.4f}..{hr_img.max().item():.4f}"
                )

            if ext_t is not None and ext_s is not None:
                ext_t.clear()
                ext_s.clear()

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                sr = model(lr_img)

                Hh, Wh = int(hr_img.shape[-2]), int(hr_img.shape[-1])
                sr = sr[..., :Hh, :Wh]

                if logger is not None and debug_range_steps > 0 and (local_ep == 0) and (step_idx <= debug_range_steps):
                    logger.info(f"[dbg] {stage_tag} ep={ep} step={step_idx} sr_range={sr.min().item():.4f}..{sr.max().item():.4f}")

                sr01 = sr / 255.0
                hr01 = hr_img / 255.0

                base = compute_base_loss(loss_mode, sr01, hr01, dct_w=dct_w, charb_eps=charb_eps)

                if ms_w and ms_w > 0:
                    base = base + float(ms_w) * multiscale_aux_loss(sr01, hr01, aux_scale=ms_scale)

                if wav_w and wav_w > 0:
                    base = base + float(wav_w) * haar_wavelet_loss(sr01, hr01, levels=wav_levels)

                loss = base

                # ---- Teacher KD: output + residual + wavelet ----
                need_teacher_signal = (
                    (kd_w_cur > 0) or
                    (kd_freq_w_cur > 0) or
                    (kd_res_w > 0) or
                    (kd_wave_w > 0)
                )

                if need_teacher_signal:
                    Hh, Wh = int(hr_img.shape[-2]), int(hr_img.shape[-1])

                    if (tcache is not None) and (metas is not None):
                        metas_list = _normalize_metas(metas, batch_size=lr_img.size(0))
                        if metas_list is None:
                            raise RuntimeError("Cannot normalize metas from DataLoader.")
                        t_sr255 = tcache.get_batch_crop_255(metas_list, device=device)
                        t01 = (t_sr255 / 255.0).clamp(0.0, 1.0)
                        t01 = t01[..., :Hh, :Wh]
                    else:
                        if teacher is None:
                            raise RuntimeError("KD requested but no teacher and no teacher_cache_dir.")
                        with torch.no_grad():
                            if getattr(teacher, "_expects_01", False):
                                t_out = teacher(lr_img / 255.0).detach()
                            else:
                                t_out = teacher(lr_img).detach()

                        t01 = teacher_to_01_crop(
                            t_out,
                            (Hh, Wh),
                            returns_01=bool(getattr(teacher, "_returns_01", False)),
                        )

                    # 1) Output KD
                    if kd_w_cur > 0:
                        w_conf = confidence_weight_map(
                            t01=t01,
                            hr01=hr01,
                            gamma=kd_conf_gamma,
                            min_w=kd_conf_min,
                            max_w=kd_conf_max,
                        )

                        if kd_loss == "l2":
                            kd_pix = torch.mean(w_conf * ((sr01 - t01) ** 2))
                        elif kd_loss == "charb":
                            kd_pix = torch.mean(
                                w_conf * torch.sqrt((sr01 - t01) ** 2 + charb_eps * charb_eps)
                            )
                        else:
                            kd_pix = torch.mean(w_conf * torch.abs(sr01 - t01))

                        loss = loss + float(kd_w_cur) * kd_pix

                    # 2) Explicit residual KD
                    if kd_res_w and kd_res_w > 0:
                        kd_res = residual_kd_loss(sr01, t01, lr_img, (Hh, Wh))
                        loss = loss + float(kd_res_w) * kd_res

                    # 3) Legacy frequency KD path (DCT or residual)
                    if kd_freq_w_cur and kd_freq_w_cur > 0:
                        if kd_use_residual:
                            kd_hf = residual_kd_loss(sr01, t01, lr_img, (Hh, Wh))
                        else:
                            kd_hf = dct_l1_loss(sr01, t01)
                        loss = loss + float(kd_freq_w_cur) * kd_hf

                    # 4) Wavelet KD on high-frequency bands only
                    if kd_wave_w and kd_wave_w > 0:
                        kd_wave = haar_wavelet_hf_loss(sr01, t01, levels=kd_wave_levels)
                        loss = loss + float(kd_wave_w) * kd_wave

            # ---- FQKD (feature distill) ----
                        # ---- FQKD (feature distill) ----
            if ext_t is not None and ext_s is not None and fqkd_w > 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=False):
                        _ = fqkd_teacher(lr_img)

                loss_f = 0.0
                ok = 0
                for ln in fqkd_layers or []:
                    if (ln in ext_s.features) and (ln in ext_t.features):
                        f_s = ext_s.features[ln].float()
                        f_t = ext_t.features[ln].float()
                        loss_f = loss_f + spatial_attention_loss(f_s, f_t)
                        ok += 1

                if ok == 0:
                    teacher_keys = sorted(list(ext_t.features.keys()))
                    student_keys = sorted(list(ext_s.features.keys()))
                    ext_t.clear()
                    ext_s.clear()
                    raise RuntimeError(
                        "[fqkd] Hooks were registered but no hooked features were produced during forward. "
                        f"requested={fqkd_layers} teacher_feature_keys={teacher_keys} "
                        f"student_feature_keys={student_keys}"
                    )

                loss = loss + float(fqkd_w) * (loss_f / ok)

                ext_t.clear()
                ext_s.clear()

            # ---- PACT regularization ----
            if pact_l2 and pact_l2 > 0:
                reg = 0.0
                for n, p in model.named_parameters():
                    if n.endswith(".alpha") or n.endswith(".beta"):
                        reg = reg + torch.sum(p * p)
                loss = loss + float(pact_l2) * reg

            scaler.scale(loss).backward()

            if grad_clip and grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(opt)
            scaler.update()

            # IMPORTANT ORDER: clip -> EMA
            if weight_clip:
                apply_weight_clipping(model, clip_other=wc_other, clip_rep=wc_rep)
            if ema is not None:
                ema.update(model)

            losses.append(loss.item())
            mean_loss = float(np.mean(losses)) if len(losses) else 0.0
            pbar.set_postfix(loss=mean_loss)

            if (logger is not None) and (log_every_steps > 0) and (step_idx % int(log_every_steps) == 0):
                lr_now = float(opt.param_groups[0]["lr"])
                logger.info(
                    f"[train] {stage_tag} epoch={ep} step={step_idx}/{len(train_loader)} "
                    f"loss_mean={mean_loss:.6f} lr={lr_now:.3e} kd_w={kd_w_cur:.4f} kd_freq_w={kd_freq_w_cur:.4f} "
                    f"fqkd_w={fqkd_w:.4f} pact_l2={pact_l2:.2e}"
                )

        if sch is not None:
            sch.step()

        train_loss_mean = float(np.mean(losses)) if len(losses) else 0.0
        lr_now = float(opt.param_groups[0]["lr"])
        if logger is not None:
            logger.info(f"[epoch-done] {stage_tag} epoch={ep} train_loss={train_loss_mean:.6f} lr={lr_now:.3e}")

        do_val = (val_every <= 1) or ((local_ep + 1) % val_every == 0) or (local_ep == epochs - 1)
        if do_val:
            if ema is not None:
                ema.apply(model)

            m_all = validate_metrics_all(
                model,
                val_loader,
                device,
                scale=3,
                max_images=val_max,
                shaves=val_shaves,
                report_ssim=report_ssim,
                ssim_win=ssim_win,
                ssim_sigma=ssim_sigma,
            )

            if ema is not None:
                ema.restore(model)

            cur_score = float(m_all.get(best_key, 0.0))

            # console-friendly val line
            def _g(k: str) -> float:
                return float(m_all.get(k, 0.0))

            msg = f"[val/{stage_tag}] epoch {ep} | "
            for sh in tuple(sorted(set(int(s) for s in val_shaves))):
                msg += (
                    f"SR RGB sh{sh}={_g(f'psnr_sr_rgb_sh{sh}'):.4f} "
                    f"SR Y sh{sh}={_g(f'psnr_sr_y_sh{sh}'):.4f} || "
                    f"BI RGB sh{sh}={_g(f'psnr_bi_rgb_sh{sh}'):.4f} "
                    f"BI Y sh{sh}={_g(f'psnr_bi_y_sh{sh}'):.4f} | "
                )
                if report_ssim:
                    msg += (
                        f"SSIM(SR) sh{sh}={_g(f'ssim_sr_rgb_sh{sh}'):.5f} "
                        f"SSIM(BI) sh{sh}={_g(f'ssim_bi_rgb_sh{sh}'):.5f} | "
                    )
            if logger is not None:
                logger.info(msg)
            else:
                print(msg)

            if val_jsonl is not None:
                append_jsonl(val_jsonl, {
                    "stage": stage_tag,
                    "epoch": ep,
                    "lr": lr_now,
                    "train_loss": train_loss_mean,
                    "best_key": best_key,
                    "val_metrics": m_all,
                })

            complete = metrics_complete(m_all, val_shaves, report_ssim)

            # ONLY write train.log / csv if FULL metrics exist
            if complete and (logger is not None):
                full_line = format_full_metrics_line(
                    stage=stage_tag, epoch=ep, lr=lr_now, train_loss=train_loss_mean,
                    best_key=best_key, score=cur_score, is_best=0,
                    shaves=val_shaves, report_ssim=report_ssim, m=m_all, note="val",
                )
                logger.info(full_line, extra={"full_metrics": True})

            if complete and (metrics_csv is not None) and (metrics_header is not None):
                row = {
                    "stage": stage_tag,
                    "epoch": ep,
                    "lr": lr_now,
                    "train_loss": train_loss_mean,
                    "best_key": best_key,
                    "val_best": cur_score,
                    "note": "val",
                }
                for k in _required_metric_keys(val_shaves, report_ssim):
                    row[k] = float(m_all[k])
                append_metrics_csv(metrics_csv, metrics_header, row)
            elif (not complete) and (metrics_csv is not None) and (metrics_header is not None) and (not csv_full_metrics_only):
                row = {
                    "stage": stage_tag,
                    "epoch": ep,
                    "lr": lr_now,
                    "train_loss": train_loss_mean,
                    "best_key": best_key,
                    "val_best": cur_score,
                    "note": "val_partial",
                }
                for k in _required_metric_keys(val_shaves, report_ssim):
                    row[k] = float(m_all.get(k, ""))
                append_metrics_csv(metrics_csv, metrics_header, row)

            ckpt = {
                "epoch": ep,
                "stage": stage_tag,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sch": (sch.state_dict() if sch is not None else None),
                "val_metrics": m_all,
                "best_key": best_key,
                "best_score": float(max(best_score, cur_score)),
                "loss_mode": loss_mode,
                "dct_w": float(dct_w),
                "ms_w": float(ms_w),
                "wav_w": float(wav_w),
                "kd_w": float(kd_w),
                "kd_freq_w": float(kd_freq_w),
                "kd_res_w": float(kd_res_w),
                "kd_wave_w": float(kd_wave_w),
                "kd_wave_levels": int(kd_wave_levels),
                "kd_loss": kd_loss,
                "fqkd_w": float(fqkd_w),
                "pact_l2": float(pact_l2),
                "has_ema": (ema is not None),
                "ema": (ema.state_dict() if ema is not None else None),
            }

            if ema is not None:
                ema.apply(model)
                ckpt["model_ema"] = model.state_dict()
                ema.restore(model)

            torch.save(ckpt, last_path)
            if logger is not None:
                logger.info(f"[save] {stage_tag} epoch={ep} -> {last_path}")

            if cur_score > best_score:
                best_score = cur_score
                ckpt["best_score"] = float(best_score)
                torch.save(ckpt, best_path)

                if complete and (logger is not None):
                    best_line = format_full_metrics_line(
                        stage=stage_tag, epoch=ep, lr=lr_now, train_loss=train_loss_mean,
                        best_key=best_key, score=best_score, is_best=1,
                        shaves=val_shaves, report_ssim=report_ssim, m=m_all, note="best",
                    )
                    logger.info(best_line, extra={"full_metrics": True})
                else:
                    if logger is not None:
                        logger.info(f"[best] saved best {stage_tag} by {best_key}={best_score:.4f} -> {best_path}")

    if ext_t is not None:
        ext_t.remove_hooks()
    if ext_s is not None:
        ext_s.remove_hooks()

    if logger is not None:
        logger.info(f"[stage-done] {stage_tag} best_score={best_score:.4f} best_path={best_path}")
    return best_path


# -------------------------
# Arg helpers
# -------------------------
def _parse_int_list(s: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    vals = [int(p) for p in parts]
    if len(vals) == 0:
        return (0, 3)
    return tuple(vals)


def _cli_flag_present(opt: str) -> bool:
    """Detect whether a BooleanOptionalAction flag was explicitly provided."""
    return (f"--{opt}" in sys.argv) or (f"--no-{opt}" in sys.argv)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs_antsr/antsr_sc")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=1234)

    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val_max", type=int, default=0)
    ap.add_argument("--val_every", type=int, default=1)

    # perf knobs
    ap.add_argument("--amp_fp32", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--val_shaves", type=str, default="0,3")
    ap.add_argument("--best_key", type=str, default="psnr_sr_rgb_sh0")
    ap.add_argument("--report_ssim", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--ssim_win", type=int, default=11)
    ap.add_argument("--ssim_sigma", type=float, default=1.5)
    ap.add_argument("--debug_range_steps", type=int, default=0)

    # Model core
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--n_rep", type=int, default=4)
    ap.add_argument("--skip_mode", type=str, default="concat_raw", choices=["add", "add1x1", "concat_lr", "concat_raw"])
    ap.add_argument("--concat_htr", type=str, default="3x3_3x3", choices=["3x3_3x3", "1x1_3x3", "1x1_1x1"])
    ap.add_argument("--use_global_add", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--image_residual", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--residual_base_mode",
        type=str,
        default="bilinear",
        choices=["bilinear", "nearest", "bicubic"],
    )
    ap.add_argument("--use_block_res_scale", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--res_scale_init", type=float, default=0.10)
    # rep blocks
    ap.add_argument("--rep_type", type=str, default="repconv", choices=["repconv", "mobileone", "repdw"])
    ap.add_argument("--mo_branches", type=int, default=2)
    ap.add_argument("--mo_use_1x1", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--mo_use_identity", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--rep_use_bn", action="store_true")
    ap.add_argument("--rep_act_mode", type=str, default="none", choices=["none", "relu"])

    # clamp per phase
    ap.add_argument("--out_clamp_fp32", type=str, default="none", choices=["min255", "minclip", "clamp_0_255", "none"])
    ap.add_argument("--out_clamp_export", type=str, default="minclip", choices=["min255", "minclip", "clamp_0_255", "none"])
    ap.add_argument("--out_clamp_qat", type=str, default="minclip", choices=["min255", "minclip", "clamp_0_255", "none"])

    # Stages
    ap.add_argument("--epochs1", type=int, default=800)
    ap.add_argument("--epochs2", type=int, default=200)
    ap.add_argument("--epochs3", type=int, default=300)
    ap.add_argument("--lr1", type=float, default=1e-3)
    ap.add_argument("--lr2", type=float, default=2e-5)
    ap.add_argument("--lr3", type=float, default=1e-5)
    ap.add_argument("--patch1", type=int, default=128)
    ap.add_argument("--patch2", type=int, default=128)
    ap.add_argument("--patch3", type=int, default=128)

    loss_choices = ["l1", "l2", "charb", "dct", "l1dct", "l2dct", "charbdct"]
    ap.add_argument("--s1_loss", type=str, default="l1", choices=loss_choices)
    ap.add_argument("--s2_loss", type=str, default="l2", choices=loss_choices)
    ap.add_argument("--s3_loss", type=str, default="l1dct", choices=loss_choices)
    ap.add_argument("--dct_w_s1", type=float, default=0.0)
    ap.add_argument("--dct_w_s2", type=float, default=0.0)
    ap.add_argument("--dct_w_s3", type=float, default=0.05)

    # Training-only extra losses
    ap.add_argument("--ms_w_s1", type=float, default=0.0)
    ap.add_argument("--ms_w_s2", type=float, default=0.10)
    ap.add_argument("--ms_w_s3", type=float, default=0.0)
    ap.add_argument("--ms_scale", type=float, default=2/3)
    ap.add_argument("--wav_w_s1", type=float, default=0.0)
    ap.add_argument("--wav_w_s2", type=float, default=0.02)
    ap.add_argument("--wav_w_s3", type=float, default=0.01)
    ap.add_argument("--wav_levels", type=int, default=1)
    ap.add_argument("--charb_eps", type=float, default=1e-3)

    ap.add_argument("--scheduler1", type=str, default="cos_warmup", choices=["none", "cos_warmup"])
    ap.add_argument("--scheduler2", type=str, default="step_halve", choices=["none", "step_halve"])
    ap.add_argument("--scheduler3", type=str, default="step_halve", choices=["none", "step_halve"])

    ap.add_argument("--channel_shuffle_s2s3", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--grad_clip", type=float, default=0.0)

    # EMA + KD
    ap.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ema_decay", type=float, default=0.999)

    ap.add_argument("--teacher_ckpt", type=str, default=None)
    ap.add_argument("--teacher_from_stage2", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--teacher_type", type=str, default="antsr", choices=["antsr", "mambair"])
    ap.add_argument("--mambair_repo", type=str, default=None)
    ap.add_argument("--mambair_teacher_ckpt", type=str, default=None)

    ap.add_argument("--kd_w_s1", type=float, default=0.0)
    ap.add_argument("--kd_w_s2", type=float, default=0.0)
    ap.add_argument("--kd_w_s3", type=float, default=0.0)
    ap.add_argument("--kd_freq_w_s3", type=float, default=0.0)
    ap.add_argument("--kd_res_w_s2", type=float, default=0.01)
    ap.add_argument("--kd_res_w_s3", type=float, default=0.01)
    ap.add_argument("--kd_wave_w_s2", type=float, default=0.0)
    ap.add_argument("--kd_wave_w_s3", type=float, default=0.005)
    ap.add_argument("--kd_wave_levels", type=int, default=1)
    ap.add_argument("--kd_loss", type=str, default="l1", choices=["l1", "l2", "charb"])

    ap.add_argument("--kd_w_s3_start", type=float, default=-1.0)
    ap.add_argument("--kd_w_s3_end", type=float, default=-1.0)
    ap.add_argument("--kd_w_s3_warmup", type=float, default=0.15)
    ap.add_argument("--kd_freq_w_s3_start", type=float, default=-1.0)
    ap.add_argument("--kd_freq_w_s3_end", type=float, default=-1.0)
    ap.add_argument("--kd_freq_w_s3_warmup", type=float, default=0.15)

    ap.add_argument("--freeze_bn_epoch", type=int, default=80)
    ap.add_argument("--bn_recalib_batches", type=int, default=64)

    # QAT
    ap.add_argument("--qat", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--qat_backend", type=str, default="qnnpack", choices=["qnnpack", "fbgemm"])
    ap.add_argument("--qat_mode", type=str, default="eager", choices=["eager", "fx"])
    ap.add_argument("--deploy_before_qat", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--qat_disable_observer_ep", type=int, default=10)
    ap.add_argument("--qat_freeze_fakequant_ep", type=int, default=60)

    # Weight clipping
    ap.add_argument("--weight_clipping", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--wc_other", type=float, default=2.0)
    ap.add_argument("--wc_rep", type=float, default=3.0)

    ap.add_argument("--init_ckpt", type=str, default=None)
    ap.add_argument("--preset", type=str, default="balanced", choices=["balanced", "speed"])

    # Stage3 subset + teacher cache
    ap.add_argument("--stage3_ids_txt", type=str, default=None)
    ap.add_argument("--teacher_cache_dir", type=str, default=None)
    ap.add_argument("--stage3_batch1", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--log_every", type=int, default=200)

    # log/csv restrictions
    ap.add_argument("--file_full_metrics_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--csv_full_metrics_only", action=argparse.BooleanOptionalAction, default=True)

    # ---- PACT + FQKD (Stage3) ----
    ap.add_argument("--enable_pact_qat", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--pact_init_min", type=float, default=-2.0)
    ap.add_argument("--pact_init_max", type=float, default=2.0)
    ap.add_argument("--pact_on_output", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--pact_out_min", type=float, default=0.0)
    ap.add_argument("--pact_out_max", type=float, default=255.0)
    ap.add_argument("--pact_l2", type=float, default=1e-4)

    ap.add_argument("--fqkd_w", type=float, default=0.0)
    ap.add_argument("--fqkd_layers", type=str, default="1,3,5")


    ap.add_argument("--kd_conf_gamma", type=float, default=12.0)
    ap.add_argument("--kd_conf_min", type=float, default=0.05)
    ap.add_argument("--kd_conf_max", type=float, default=1.0)

    ap.add_argument("--kd_sched_p1", type=float, default=0.3)
    ap.add_argument("--kd_sched_p2", type=float, default=0.7)
    ap.add_argument("--kd_low_floor", type=float, default=0.05)

    ap.add_argument("--kd_use_residual_s2", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--kd_use_residual_s3", action=argparse.BooleanOptionalAction, default=True)

    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # preset behavior
    if args.preset == "speed":
        args.channels = min(args.channels, 24)
        args.n_rep = max(args.n_rep, 4)
        if not _cli_flag_present("use_global_add"):
            args.use_global_add = False
        args.skip_mode = "concat_raw"
        args.concat_htr = "1x1_1x1"

    val_shaves = _parse_int_list(args.val_shaves)

    train_hr, train_lr, valid_hr, valid_lr = resolve_div2k_paths(args.data_root, scale=3)

    def make_train_loader(lr_patch: int, id_list_txt=None, return_meta=False, batch_override=None, augment=True):
        train_ds = DIV2KPairX3(
            train_hr, train_lr, train=True, lr_patch=lr_patch, augment=augment,
            repeat=1, id_list_txt=id_list_txt, return_meta=return_meta
        )
        bs = batch_override if batch_override is not None else args.batch
        dl_kwargs = dict(
            dataset=train_ds, batch_size=bs, shuffle=True, num_workers=args.workers,
            drop_last=True, pin_memory=(device.type == "cuda"),
        )
        if args.workers > 0:
            dl_kwargs["persistent_workers"] = bool(args.persistent_workers)
            dl_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
        return DataLoader(**dl_kwargs)

    val_ds = DIV2KPairX3(valid_hr, valid_lr, train=False, lr_patch=128, augment=False, repeat=1)
    val_kwargs = dict(dataset=val_ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=(device.type == "cuda"))
    if args.workers > 0:
        val_kwargs["persistent_workers"] = bool(args.persistent_workers)
        val_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
    val_loader = DataLoader(**val_kwargs)

    # SAFETY: keep QAT clamp consistent with EXPORT clamp
    clamp_mismatch_fixed = False
    if args.qat and args.epochs3 > 0:
        if args.out_clamp_qat != args.out_clamp_export:
            args.out_clamp_qat = args.out_clamp_export
            clamp_mismatch_fixed = True

    cfg = dict(
        scale=3,
        channels=args.channels,
        n_rep=args.n_rep,
        rep_type=args.rep_type,
        rep_use_bn=args.rep_use_bn,
        rep_act_mode=args.rep_act_mode,
        mo_branches=args.mo_branches,
        mo_use_1x1=args.mo_use_1x1,
        mo_use_identity=args.mo_use_identity,
        out_clamp_mode=args.out_clamp_fp32,
        skip_mode=args.skip_mode,
        concat_htr=args.concat_htr,
        use_global_add=args.use_global_add,
        image_residual=args.image_residual,
        residual_base_mode=args.residual_base_mode,
        use_block_res_scale=args.use_block_res_scale,
        res_scale_init=args.res_scale_init,
    )

    run_dir = os.path.join(args.out_dir, args.preset)
    os.makedirs(run_dir, exist_ok=True)
    logger, log_path = setup_logger(run_dir, name=f"train_{args.preset}", file_full_metrics_only=args.file_full_metrics_only)

    if clamp_mismatch_fixed:
        logger.info(f"[fix] out_clamp_qat forced to match out_clamp_export: {args.out_clamp_qat}")

    metrics_header = build_metrics_header(val_shaves, args.report_ssim)
    metrics_csv = ensure_metrics_csv(run_dir, metrics_header)
    val_jsonl = os.path.join(run_dir, "val_metrics.jsonl")

    extra = {
        "model_cfg": cfg,
        "val_shaves": list(val_shaves),
        "resolved_paths": {"train_hr": train_hr, "train_lr": train_lr, "valid_hr": valid_hr, "valid_lr": valid_lr},
        "device": str(device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "metrics_header": metrics_header,
        "file_full_metrics_only": bool(args.file_full_metrics_only),
        "csv_full_metrics_only": bool(args.csv_full_metrics_only),
        "amp_fp32": bool(args.amp_fp32),
        "channels_last": bool(args.channels_last),
        "prefetch_factor": int(args.prefetch_factor),
        "persistent_workers": bool(args.persistent_workers),
    }
    cfg_path = dump_run_config(run_dir, args, extra)

    logger.info(f"Log file: {log_path} (FULL metrics only={args.file_full_metrics_only})")
    logger.info(f"Run config: {cfg_path}")
    logger.info(f"Metrics CSV: {metrics_csv} (FULL metrics only={args.csv_full_metrics_only})")
    logger.info(f"Val JSONL: {val_jsonl}")
    # EMA now tracks both parameters and BN buffers,
    # so stage-transition/export can safely prefer EMA when enabled.
    prefer_ema_for_stage_transition = bool(args.ema)
    logger.info(
        f"[ckpt] prefer_ema_for_stage_transition={int(prefer_ema_for_stage_transition)} "
        f"(ema={int(bool(args.ema))} rep_use_bn={int(bool(args.rep_use_bn))})"
    )
    if args.rep_type == "repdw":
        logger.info("[info] rep_type=repdw -> switch_to_deploy() is intentionally a no-op for rep blocks.")
    if args.rep_type == "mobileone" and args.rep_act_mode == "none":
        logger.info("[info] rep_type=mobileone with rep_act_mode=none now truly uses no activation.")
    logger.info(f"VAL SHAVES: {val_shaves} | BEST KEY: {args.best_key} | REPORT_SSIM: {args.report_ssim}")
    logger.info(
        f"PERF: amp_fp32={int(bool(args.amp_fp32))} channels_last={int(bool(args.channels_last))} "
        f"prefetch_factor={int(args.prefetch_factor)} persistent_workers={int(bool(args.persistent_workers))}"
    )

    model = AntSR(deploy=False, **cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {n_params:,}")

    teacher = None
    if args.teacher_type == "antsr":
        if args.teacher_ckpt:
            teacher = load_teacher_antsr_export(args.teacher_ckpt, device)
            logger.info(f"Teacher(AntSR) loaded from: {args.teacher_ckpt}")
    elif args.teacher_type == "mambair":
        if (args.mambair_repo is None) or (args.mambair_teacher_ckpt is None):
            raise RuntimeError("teacher_type=mambair requires --mambair_repo and --mambair_teacher_ckpt")
        teacher = load_teacher_mambair(args.mambair_repo, args.mambair_teacher_ckpt, device)
        logger.info(f"Teacher(MambaIR) loaded from: {args.mambair_teacher_ckpt}")

    if args.init_ckpt is not None and args.resume is None:
        logger.info(f"== Init weights from ckpt == {args.init_ckpt}")
        load_ckpt_weights(model, args.init_ckpt, device, prefer_ema=prefer_ema_for_stage_transition)

    resume_ckpt = None
    resume_stage = None
    if args.resume is not None:
        resume_ckpt = torch.load(args.resume, map_location="cpu")
        resume_stage = resume_ckpt.get("stage", None)
        if resume_stage is None:
            raise RuntimeError("Resume ckpt missing key 'stage'.")
        logger.info(f"[resume] loaded {args.resume} stage={resume_stage} epoch={resume_ckpt.get('epoch')}")

    ema = EMA(model, decay=args.ema_decay) if args.ema else None
    if ema is not None and resume_ckpt is not None and resume_ckpt.get("ema") is not None:
        ema.load_state_dict(resume_ckpt["ema"], model)
        logger.info("[resume] EMA loaded from checkpoint.")

    # ---- Stage 1 ----
    if args.epochs1 > 0 and (resume_stage is None or resume_stage == "s1_fp32"):
        model.set_out_clamp_mode(args.out_clamp_fp32)
        train_loader = make_train_loader(args.patch1, augment=True)
        s1_best = train_stage(
            model, train_loader, val_loader, device,
            epochs=args.epochs1, base_lr=args.lr1, out_dir=run_dir, start_epoch=0,
            val_max=args.val_max, stage_tag="s1_fp32",
            val_shaves=val_shaves, best_key=args.best_key,
            report_ssim=args.report_ssim, ssim_win=args.ssim_win, ssim_sigma=args.ssim_sigma,
            grad_clip=args.grad_clip, scheduler_type=args.scheduler1,
            channel_shuffle=False,
            loss_mode=args.s1_loss, dct_w=args.dct_w_s1,
            ms_w=args.ms_w_s1, ms_scale=args.ms_scale,
            wav_w=args.wav_w_s1, wav_levels=args.wav_levels,
            charb_eps=args.charb_eps,
            weight_clip=False,
            val_every=args.val_every,
            ema=ema,
            teacher=teacher, kd_w=args.kd_w_s1, kd_loss=args.kd_loss, kd_freq_w=0.0,
            teacher_cache_dir=None,
            freeze_bn_epoch=args.freeze_bn_epoch,
            resume_ckpt=(resume_ckpt if resume_stage == "s1_fp32" else None),
            logger=logger, metrics_csv=metrics_csv, metrics_header=metrics_header, val_jsonl=val_jsonl,
            log_every_steps=args.log_every,
            use_amp=bool(args.amp_fp32),
            channels_last=bool(args.channels_last),
            debug_range_steps=args.debug_range_steps,
            file_log_full_metrics_only=args.file_full_metrics_only,
            csv_full_metrics_only=args.csv_full_metrics_only,
        )
        logger.info(f"Load best Stage1 -> {s1_best}")
        load_ckpt_weights(model, s1_best, device, prefer_ema=prefer_ema_for_stage_transition)
        ema = EMA(model, decay=args.ema_decay) if args.ema else None

    # ---- Stage 2 ----
    s2_export_path = os.path.join(run_dir, "ckpt_best_s2_deploy.pt")
    if args.epochs2 > 0 and (resume_stage is None or resume_stage == "s2_fp32"):
        model.set_out_clamp_mode(args.out_clamp_fp32)
        train_loader = make_train_loader(args.patch2, augment=True)
        start_ep = args.epochs1
        s2_best = train_stage(
            model, train_loader, val_loader, device,
            epochs=args.epochs2, base_lr=args.lr2, out_dir=run_dir, start_epoch=start_ep,
            val_max=args.val_max, stage_tag="s2_fp32",
            val_shaves=val_shaves, best_key=args.best_key,
            report_ssim=args.report_ssim, ssim_win=args.ssim_win, ssim_sigma=args.ssim_sigma,
            grad_clip=args.grad_clip, scheduler_type=args.scheduler2,
            channel_shuffle=False if (teacher is not None and args.kd_w_s2 > 0) else args.channel_shuffle_s2s3,
            loss_mode=args.s2_loss, dct_w=args.dct_w_s2,
            ms_w=args.ms_w_s2, ms_scale=args.ms_scale,
            wav_w=args.wav_w_s2, wav_levels=args.wav_levels,
            charb_eps=args.charb_eps,
            weight_clip=args.weight_clipping, wc_other=args.wc_other, wc_rep=args.wc_rep,
            val_every=args.val_every,
            ema=ema,
            teacher=teacher,
            kd_w=args.kd_w_s2,
            kd_loss=args.kd_loss,
            kd_freq_w=0.0,
            kd_res_w=args.kd_res_w_s2,
            kd_wave_w=args.kd_wave_w_s2,
            kd_wave_levels=args.kd_wave_levels,
            teacher_cache_dir=None,
            freeze_bn_epoch=args.freeze_bn_epoch,
            resume_ckpt=(resume_ckpt if resume_stage == "s2_fp32" else None),
            logger=logger, metrics_csv=metrics_csv, metrics_header=metrics_header, val_jsonl=val_jsonl,
            log_every_steps=args.log_every,
            use_amp=bool(args.amp_fp32),
            channels_last=bool(args.channels_last),
            debug_range_steps=args.debug_range_steps,
            file_log_full_metrics_only=args.file_full_metrics_only,
            csv_full_metrics_only=args.csv_full_metrics_only,
            kd_conf_gamma=args.kd_conf_gamma,
            kd_conf_min=args.kd_conf_min,
            kd_conf_max=args.kd_conf_max,
            kd_sched_p1=args.kd_sched_p1,
            kd_sched_p2=args.kd_sched_p2,
            kd_low_floor=args.kd_low_floor,
            kd_use_residual=bool(args.kd_use_residual_s2),
        )
        logger.info(f"Load best Stage2 -> {s2_best}")
        load_ckpt_weights(model, s2_best, device, prefer_ema=prefer_ema_for_stage_transition)

        # Recalibrate BN stats before fusing to deploy
        if args.bn_recalib_batches > 0:
            recalib_loader_s2 = make_train_loader(args.patch2, augment=False)
            recalibrate_bn_stats(
                model,
                recalib_loader_s2,
                device,
                num_batches=args.bn_recalib_batches,
                channels_last=bool(args.channels_last),
                logger=logger,
            )

        # Export deploy after stage2
        model.set_out_clamp_mode(args.out_clamp_export)
        model.eval()
        model.switch_to_deploy()
        export_cfg = dict(cfg)
        export_cfg["out_clamp_mode"] = args.out_clamp_export
        torch.save({"model": model.state_dict(), "cfg": export_cfg}, s2_export_path)
        logger.info(f"Saved: {s2_export_path}")

    # ---- Validate DEPLOY model (must be >30 before QAT) ----
    if os.path.isfile(s2_export_path):
        deploy = AntSR(deploy=True, **torch.load(s2_export_path, map_location="cpu")["cfg"]).eval().to(device)
        deploy.load_state_dict(torch.load(s2_export_path, map_location="cpu")["model"], strict=True)
        m_dep = validate_metrics_all(
            deploy, val_loader, device, scale=3,
            max_images=args.val_max, shaves=val_shaves,
            report_ssim=args.report_ssim, ssim_win=args.ssim_win, ssim_sigma=args.ssim_sigma
        )
        challenge_psnr_key = "psnr_sr_rgb_sh0"
        psnr_dep = float(m_dep.get(challenge_psnr_key, 0.0))
        logger.info(f"[deploy-val/s2] {challenge_psnr_key}={psnr_dep:.4f}")
        if psnr_dep < 30.0:
            raise RuntimeError(
                f"[FAIL] Stage2 DEPLOY PSNR < 30dB: "
                f"{challenge_psnr_key}={psnr_dep:.4f}. Fix before QAT."
            )
    else:
        logger.info("[warn] s2_export_path missing; if you plan to run QAT you must have stage2 deploy export.")

    # rebuild non-deploy for stage3
    if os.path.isfile(os.path.join(run_dir, "ckpt_best_s2_fp32.pt")):
        s2_best_guess = os.path.join(run_dir, "ckpt_best_s2_fp32.pt")
        model = AntSR(deploy=False, **cfg).to(device)
        load_ckpt_weights(model, s2_best_guess, device, prefer_ema=prefer_ema_for_stage_transition)

    # auto-teacher from stage2 export if user wants antsr teacher and missing teacher
    if teacher is None and args.teacher_from_stage2 and os.path.isfile(s2_export_path) and args.teacher_type == "antsr":
        teacher = load_teacher_antsr_export(s2_export_path, device)
        logger.info(f"Teacher auto-loaded from stage2 export: {s2_export_path}")

    # ---- Stage 3 (QAT) ----
    if args.qat and args.epochs3 > 0 and (resume_stage is None or resume_stage == "s3_qat"):
        model.set_out_clamp_mode(args.out_clamp_qat)

        use_meta = (args.teacher_cache_dir is not None)
        s3_bs = 1 if args.stage3_batch1 else args.batch
        train_loader = make_train_loader(
            args.patch3,
            id_list_txt=args.stage3_ids_txt,
            return_meta=use_meta,
            batch_override=s3_bs,
            augment=False if (args.teacher_cache_dir is not None) else True
        )

        start_ep = args.epochs1 + args.epochs2

        qat_on_deploy = False
        if args.deploy_before_qat:
            model.eval()
            model.switch_to_deploy()
            model.train()
            qat_on_deploy = True
            logger.info("[qat] switched to deploy BEFORE QAT.")

        # Enforce FX-QAT when enable_pact_qat (so PACT won't be overridden)
        if args.enable_pact_qat and args.qat_mode != "fx":
            logger.info(f"[pact] enable_pact_qat=True -> forcing qat_mode=fx (was {args.qat_mode})")
            args.qat_mode = "fx"
        if args.enable_pact_qat:
            logger.info("[pact] PACT will replace block.act modules even if they are Identity().")

        # ---- Build FQKD teacher from Stage2 deploy export (AntSR) ----
        fqkd_teacher = None
        fqkd_hook_names = None
        if args.fqkd_w and args.fqkd_w > 0:
            try:
                idxs = [int(x.strip()) for x in args.fqkd_layers.split(",") if x.strip() != ""]
            except Exception:
                idxs = []
            idxs = [i for i in idxs if 0 <= i < args.n_rep]

            # FX-safe hook names
            fqkd_hook_names = build_fqkd_hook_names(args.rep_type, idxs, qat_on_deploy=qat_on_deploy)

            if os.path.isfile(s2_export_path):
                tck = torch.load(s2_export_path, map_location="cpu")
                tcfg = tck["cfg"]
                fqkd_teacher = AntSR(deploy=True, **tcfg).eval().to(device)
                fqkd_teacher.load_state_dict(tck["model"], strict=True)
                for p in fqkd_teacher.parameters():
                    p.requires_grad_(False)
                logger.info(f"[fqkd] teacher loaded from {s2_export_path} hook_names={fqkd_hook_names}")
            else:
                logger.info(f"[fqkd] WARN: missing {s2_export_path} -> disabling fqkd")
                args.fqkd_w = 0.0
                fqkd_teacher = None
                fqkd_hook_names = None

        if args.qat_mode == "fx":
            model._qat_example_inputs = (torch.randn(1, 3, args.patch3, args.patch3, device=device),)

        model = prepare_qat_model(
            model,
            backend=args.qat_backend,
            mode=args.qat_mode,
            enable_pact=bool(args.enable_pact_qat),
            pact_cfg={
                "init_min": args.pact_init_min,
                "init_max": args.pact_init_max,
                "pact_on_output": bool(args.pact_on_output),
                "out_min": args.pact_out_min,
                "out_max": args.pact_out_max,
            },
        ).to(device)
        logger.info(f"[qat] prepared QAT backend={args.qat_backend} mode={args.qat_mode} pact={int(bool(args.enable_pact_qat))}")

        ema_qat = EMA(model, decay=args.ema_decay) if args.ema else None
        if ema_qat is not None and resume_ckpt is not None and resume_stage == "s3_qat" and resume_ckpt.get("ema") is not None:
            ema_qat.load_state_dict(resume_ckpt["ema"], model)
            logger.info("[resume] EMA(QAT) loaded from checkpoint.")

        s3_best = train_stage(
            model, train_loader, val_loader, device,
            epochs=args.epochs3, base_lr=args.lr3, out_dir=run_dir, start_epoch=start_ep,
            val_max=args.val_max, stage_tag="s3_qat",
            val_shaves=val_shaves, best_key=args.best_key,
            report_ssim=args.report_ssim, ssim_win=args.ssim_win, ssim_sigma=args.ssim_sigma,
            grad_clip=args.grad_clip, scheduler_type=args.scheduler3,
                        channel_shuffle=False if (args.teacher_cache_dir is not None or args.kd_w_s3 > 0 or args.fqkd_w > 0) else args.channel_shuffle_s2s3,
            loss_mode=args.s3_loss, dct_w=args.dct_w_s3,
            ms_w=args.ms_w_s3, ms_scale=args.ms_scale,
            wav_w=args.wav_w_s3, wav_levels=args.wav_levels,
            charb_eps=args.charb_eps,
            weight_clip=args.weight_clipping, wc_other=args.wc_other, wc_rep=args.wc_rep,
            val_every=args.val_every,
            ema=ema_qat,
            teacher=teacher,
            kd_w=args.kd_w_s3,
            kd_loss=args.kd_loss,
            kd_freq_w=args.kd_freq_w_s3,
            kd_res_w=args.kd_res_w_s3,
            kd_wave_w=args.kd_wave_w_s3,
            kd_wave_levels=args.kd_wave_levels,
            kd_w_start=args.kd_w_s3_start, kd_w_end=args.kd_w_s3_end, kd_w_warmup=args.kd_w_s3_warmup,
            kd_freq_w_start=args.kd_freq_w_s3_start, kd_freq_w_end=args.kd_freq_w_s3_end, kd_freq_w_warmup=args.kd_freq_w_s3_warmup,
            teacher_cache_dir=args.teacher_cache_dir,
            freeze_bn_epoch=args.freeze_bn_epoch,
            qat_disable_observer_ep=args.qat_disable_observer_ep,
            qat_freeze_fakequant_ep=args.qat_freeze_fakequant_ep,
            resume_ckpt=(resume_ckpt if resume_stage == "s3_qat" else None),
            logger=logger, metrics_csv=metrics_csv, metrics_header=metrics_header, val_jsonl=val_jsonl,
            log_every_steps=args.log_every,
            use_amp=False,  # QAT stable
            channels_last=bool(args.channels_last),
            debug_range_steps=args.debug_range_steps,
            file_log_full_metrics_only=args.file_full_metrics_only,
            csv_full_metrics_only=args.csv_full_metrics_only,
            fqkd_teacher=fqkd_teacher,
            fqkd_layers=fqkd_hook_names,
            fqkd_w=float(args.fqkd_w),
            pact_l2=(args.pact_l2 if args.enable_pact_qat else 0.0),
            kd_conf_gamma=args.kd_conf_gamma,
            kd_conf_min=args.kd_conf_min,
            kd_conf_max=args.kd_conf_max,
            kd_sched_p1=args.kd_sched_p1,
            kd_sched_p2=args.kd_sched_p2,
            kd_low_floor=args.kd_low_floor,
            kd_use_residual=bool(args.kd_use_residual_s3),
        )

        logger.info(f"Best QAT ckpt: {s3_best}")
        qat_ckpt = torch.load(s3_best, map_location="cpu")
        qat_state = qat_ckpt.get("model_ema", qat_ckpt["model"])

        clean_cfg = dict(cfg)
        clean_cfg["out_clamp_mode"] = args.out_clamp_export

        clean = AntSR(deploy=qat_on_deploy, **clean_cfg).eval()
        filtered = strip_qat_to_clean_state(clean, qat_state, logger=logger)
        clean.load_state_dict(filtered, strict=False)
        clean.set_out_clamp_mode(args.out_clamp_export)

        # Optional BN recalibration only if clean model is still non-deploy
        if (not qat_on_deploy) and args.bn_recalib_batches > 0:
            recalib_loader_s3 = make_train_loader(
                args.patch3,
                batch_override=(1 if args.stage3_batch1 else args.batch),
                augment=False,
            )
            clean = clean.to(device)
            recalibrate_bn_stats(
                clean,
                recalib_loader_s3,
                device,
                num_batches=args.bn_recalib_batches,
                channels_last=bool(args.channels_last),
                logger=logger,
            )
            clean = clean.cpu().eval()

        clean.switch_to_deploy()

        deploy_path = os.path.join(run_dir, "ckpt_best_s3_qat_deploy.pt")
        torch.save({"model": clean.state_dict(), "cfg": clean_cfg}, deploy_path)
        logger.info(f"Saved: {deploy_path}")

        # Validate final DEPLOY model after QAT (must be >30)
        deploy2 = AntSR(deploy=True, **clean_cfg).eval().to(device)
        deploy2.load_state_dict(torch.load(deploy_path, map_location="cpu")["model"], strict=True)
        m_dep2 = validate_metrics_all(
            deploy2, val_loader, device, scale=3,
            max_images=args.val_max, shaves=val_shaves,
            report_ssim=args.report_ssim, ssim_win=args.ssim_win, ssim_sigma=args.ssim_sigma
        )
        challenge_psnr_key = "psnr_sr_rgb_sh0"
        psnr_dep2 = float(m_dep2.get(challenge_psnr_key, 0.0))
        logger.info(f"[deploy-val/s3] {challenge_psnr_key}={psnr_dep2:.4f}")
        if psnr_dep2 < 30.0:
            raise RuntimeError(
                f"[FAIL] Stage3 DEPLOY PSNR < 30dB: "
                f"{challenge_psnr_key}={psnr_dep2:.4f}."
            )

    logger.info(f"Done. run_dir: {run_dir}")


if __name__ == "__main__":
    main()