import os
import math
import time
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bbox_sr3.student.model import AntSR
from data_div2k_pairs import DIV2KPairX3, resolve_div2k_paths


# -------------------------
# Metrics helpers
# -------------------------
def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    # x: NCHW, range [0..255]
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def shave_border(x: torch.Tensor, shave: int) -> torch.Tensor:
    if shave <= 0:
        return x
    h, w = x.shape[-2], x.shape[-1]
    if h <= 2 * shave or w <= 2 * shave:
        return x
    return x[..., shave:-shave, shave:-shave]


def psnr_255(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-12) -> float:
    mse = torch.mean((pred - gt) ** 2).item()
    if mse < eps:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def gaussian_1d(win: int, sigma: float, device, dtype):
    coords = torch.arange(win, device=device, dtype=dtype) - (win - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    return g


def ssim_per_channel(img1: torch.Tensor, img2: torch.Tensor, win: int = 11, sigma: float = 1.5) -> torch.Tensor:
    # img1, img2: NCHW in [0..255]
    device, dtype = img1.device, img1.dtype
    g1 = gaussian_1d(win, sigma, device, dtype).view(1, 1, 1, win)
    g2 = gaussian_1d(win, sigma, device, dtype).view(1, 1, win, 1)

    def blur(x):
        c = x.size(1)
        x = F.conv2d(x, g1.expand(c, 1, 1, win), padding=(0, win // 2), groups=c)
        x = F.conv2d(x, g2.expand(c, 1, win, 1), padding=(win // 2, 0), groups=c)
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
    scale: int = 3,
    max_images: int = 0,
    shaves: Tuple[int, ...] = (0,),
    report_ssim: bool = False,
    ssim_win: int = 11,
    ssim_sigma: float = 1.5,
    channels_last: bool = True,
) -> Dict[str, float]:
    model.eval()
    shaves = tuple(sorted(set(int(s) for s in shaves)))

    acc: Dict[str, List[float]] = {}
    lat_ms: List[float] = []

    def push(k: str, v: float):
        acc.setdefault(k, []).append(v)

    seen = 0
    for lr, hr in loader:
        lr = lr.to(device, non_blocking=True)
        hr = hr.to(device, non_blocking=True)

        if channels_last:
            try:
                lr = lr.to(memory_format=torch.channels_last)
                hr = hr.to(memory_format=torch.channels_last)
            except Exception:
                pass

        bi = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        sr = model(lr)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        lat_ms.append((t1 - t0) * 1000.0)

        H, W = hr.shape[-2], hr.shape[-1]
        sr = sr[..., :H, :W]
        bi = bi[..., :H, :W]

        for sh in shaves:
            sr_s = shave_border(sr, sh)
            bi_s = shave_border(bi, sh)
            hr_s = shave_border(hr, sh)

            sr_rgb = torch.clamp(sr_s, 0.0, 255.0)
            bi_rgb = torch.clamp(bi_s, 0.0, 255.0)
            hr_rgb = torch.clamp(hr_s, 0.0, 255.0)

            push(f"psnr_sr_rgb_sh{sh}", psnr_255(sr_rgb, hr_rgb))
            push(f"psnr_bi_rgb_sh{sh}", psnr_255(bi_rgb, hr_rgb))

            sr_y = torch.clamp(rgb_to_y(sr_s), 0.0, 255.0)
            bi_y = torch.clamp(rgb_to_y(bi_s), 0.0, 255.0)
            hr_y = torch.clamp(rgb_to_y(hr_s), 0.0, 255.0)

            push(f"psnr_sr_y_sh{sh}", psnr_255(sr_y, hr_y))
            push(f"psnr_bi_y_sh{sh}", psnr_255(bi_y, hr_y))

            if report_ssim:
                ssim_sr = ssim_per_channel(sr_rgb, hr_rgb, win=ssim_win, sigma=ssim_sigma).mean().item()
                ssim_bi = ssim_per_channel(bi_rgb, hr_rgb, win=ssim_win, sigma=ssim_sigma).mean().item()
                push(f"ssim_sr_rgb_sh{sh}", float(ssim_sr))
                push(f"ssim_bi_rgb_sh{sh}", float(ssim_bi))

        seen += 1
        if max_images > 0 and seen >= max_images:
            break

    out = {k: float(np.mean(v)) if len(v) else 0.0 for k, v in acc.items()}
    out["runtime_val_avg_ms"] = float(np.mean(lat_ms)) if len(lat_ms) else 0.0
    out["runtime_val_min_ms"] = float(np.min(lat_ms)) if len(lat_ms) else 0.0
    out["runtime_val_max_ms"] = float(np.max(lat_ms)) if len(lat_ms) else 0.0
    out["num_images"] = int(seen)
    return out


# -------------------------
# Benchmark single input
# -------------------------
@torch.no_grad()
def benchmark_single_input(
    model,
    device,
    h: int,
    w: int,
    warmup: int = 20,
    runs: int = 50,
    channels_last: bool = True,
):
    model.eval()

    x = torch.randn(1, 3, h, w, device=device)

    if channels_last:
        try:
            x = x.to(memory_format=torch.channels_last)
        except Exception:
            pass

    # Warmup
    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    return {
        "bench_input_h": int(h),
        "bench_input_w": int(w),
        "bench_avg_ms": float(np.mean(times)),
        "bench_min_ms": float(np.min(times)),
        "bench_max_ms": float(np.max(times)),
        "bench_fps": float(1000.0 / np.mean(times)),
        "bench_runs": int(runs),
        "bench_warmup": int(warmup),
    }


# -------------------------
# Utils
# -------------------------
def parse_int_list(s: str) -> Tuple[int, ...]:
    vals = [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    if len(vals) == 0:
        return (0,)
    return tuple(vals)


def print_dict(title: str, d: Dict[str, float]):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val_max", type=int, default=0, help="0 = full val set")
    ap.add_argument("--shaves", type=str, default="0")
    ap.add_argument("--report_ssim", action="store_true")
    ap.add_argument("--ssim_win", type=int, default=11)
    ap.add_argument("--ssim_sigma", type=float, default=1.5)
    ap.add_argument("--channels_last", action="store_true")

    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--bench_h", type=int, default=720)
    ap.add_argument("--bench_w", type=int, default=1280)
    ap.add_argument("--bench_warmup", type=int, default=20)
    ap.add_argument("--bench_runs", type=int, default=50)

    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    shaves = parse_int_list(args.shaves)

    print(f"Using device: {device}")
    print(f"Checkpoint: {args.ckpt}")

    # Resolve paths
    train_hr, train_lr, valid_hr, valid_lr = resolve_div2k_paths(args.data_root, scale=3)
    print(f"Valid HR: {valid_hr}")
    print(f"Valid LR: {valid_lr}")

    # Val loader
    val_ds = DIV2KPairX3(
        hr_dir=valid_hr,
        lr_x3_dir=valid_lr,
        train=False,
        lr_patch=128,
        augment=False,
        repeat=1,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
        prefetch_factor=2 if args.workers > 0 else None,
    )

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if not (isinstance(ckpt, dict) and "cfg" in ckpt and "model" in ckpt):
        raise RuntimeError("Checkpoint must contain keys: {'cfg', 'model'}")

    cfg = ckpt["cfg"]
    sd = ckpt["model"]

    model = AntSR(deploy=True, **cfg).eval().to(device)
    model.load_state_dict(sd, strict=True)

    if args.channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    # Validate metrics on val set
    metrics = validate_metrics_all(
        model=model,
        loader=val_loader,
        device=device,
        scale=3,
        max_images=args.val_max,
        shaves=shaves,
        report_ssim=args.report_ssim,
        ssim_win=args.ssim_win,
        ssim_sigma=args.ssim_sigma,
        channels_last=args.channels_last,
    )
    print_dict("VAL METRICS", metrics)

    # Optional benchmark
    if args.benchmark:
        bench = benchmark_single_input(
            model=model,
            device=device,
            h=args.bench_h,
            w=args.bench_w,
            warmup=args.bench_warmup,
            runs=args.bench_runs,
            channels_last=args.channels_last,
        )
        print_dict("SINGLE-INPUT BENCHMARK", bench)


if __name__ == "__main__":
    main()