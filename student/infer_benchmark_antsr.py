import os
import time
import math
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import torch
import numpy as np
from PIL import Image

from bbox_sr3.student.model import AntSR


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def bytes_to_mb(x: Optional[int]) -> Optional[float]:
    if x is None:
        return None
    return float(x) / (1024.0 * 1024.0)


def safe_get_file_size(path: str) -> Optional[int]:
    try:
        return int(os.path.getsize(path))
    except Exception:
        return None


def state_dict_num_bytes(sd: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    total_bytes = 0
    total_params = 0
    for _, v in sd.items():
        if torch.is_tensor(v):
            total_bytes += v.numel() * v.element_size()
            total_params += v.numel()
    return int(total_bytes), int(total_params)


def read_pt_model_info(path: str) -> Dict:
    file_size_b = safe_get_file_size(path)
    info = {
        "path": path,
        "file_size_bytes": file_size_b,
        "file_size_mb": bytes_to_mb(file_size_b),
        "raw_weight_bytes": None,
        "raw_weight_mb": None,
        "param_count": None,
        "format": None,
    }

    if not os.path.exists(path):
        return info

    try:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "cfg" in obj and "model" in obj and isinstance(obj["model"], dict):
            raw_bytes, n_params = state_dict_num_bytes(obj["model"])
            info["raw_weight_bytes"] = raw_bytes
            info["raw_weight_mb"] = bytes_to_mb(raw_bytes)
            info["param_count"] = n_params
            info["format"] = "export_pt"
            return info

        if isinstance(obj, dict) and "model_ema" in obj and isinstance(obj["model_ema"], dict):
            raw_bytes, n_params = state_dict_num_bytes(obj["model_ema"])
            info["raw_weight_bytes"] = raw_bytes
            info["raw_weight_mb"] = bytes_to_mb(raw_bytes)
            info["param_count"] = n_params
            info["format"] = "train_ckpt_model_ema"
            return info

        if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            raw_bytes, n_params = state_dict_num_bytes(obj["model"])
            info["raw_weight_bytes"] = raw_bytes
            info["raw_weight_mb"] = bytes_to_mb(raw_bytes)
            info["param_count"] = n_params
            info["format"] = "train_ckpt_model"
            return info

        if isinstance(obj, dict):
            raw_bytes, n_params = state_dict_num_bytes(obj)
            if n_params > 0:
                info["raw_weight_bytes"] = raw_bytes
                info["raw_weight_mb"] = bytes_to_mb(raw_bytes)
                info["param_count"] = n_params
                info["format"] = "state_dict"

    except Exception as e:
        info["error"] = str(e)

    return info


def load_export_model(ckpt_path: str, device: torch.device, channels_last: bool = True) -> Tuple[torch.nn.Module, Dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not (isinstance(ckpt, dict) and "cfg" in ckpt and "model" in ckpt):
        raise RuntimeError(
            "Checkpoint phải là export/deploy ckpt có dạng {'cfg': ..., 'model': ...}. "
            "Hãy dùng ckpt_best_s2_deploy.pt hoặc ckpt_best_s3_qat_deploy.pt"
        )

    cfg = dict(ckpt["cfg"])
    model = AntSR(deploy=True, **cfg).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)

    if channels_last:
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass

    return model, cfg


def pil_to_tensor_255(img: Image.Image) -> torch.Tensor:
    arr = np.array(img.convert("RGB"), dtype=np.float32)  # HWC, 0..255
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    return t


def tensor_255_to_pil(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu().clamp(0.0, 255.0)
    x = x.permute(1, 2, 0).numpy().astype(np.uint8)
    return Image.fromarray(x)


def list_images(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        if p.suffix.lower() not in IMG_EXTS:
            raise RuntimeError(f"File không phải ảnh hỗ trợ: {input_path}")
        return [str(p)]

    if not p.is_dir():
        raise RuntimeError(f"Input không tồn tại: {input_path}")

    files = []
    for x in sorted(p.iterdir()):
        if x.is_file() and x.suffix.lower() in IMG_EXTS:
            files.append(str(x))
    if not files:
        raise RuntimeError(f"Không tìm thấy ảnh trong thư mục: {input_path}")
    return files


@torch.no_grad()
def run_single_image(
    model: torch.nn.Module,
    img_path: str,
    device: torch.device,
    out_dir: Optional[str] = None,
    channels_last: bool = True,
) -> Dict:
    img = Image.open(img_path).convert("RGB")
    lr = pil_to_tensor_255(img).unsqueeze(0).to(device)  # BCHW, 0..255

    if channels_last:
        try:
            lr = lr.to(memory_format=torch.channels_last)
        except Exception:
            pass

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    sr = model(lr)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    sr = sr[0]
    out_h, out_w = int(sr.shape[-2]), int(sr.shape[-1])

    save_path = None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        stem = Path(img_path).stem
        save_path = os.path.join(out_dir, f"{stem}_x3.png")
        tensor_255_to_pil(sr).save(save_path)

    return {
        "input": img_path,
        "output": save_path,
        "lr_hw": f"{img.height}x{img.width}",
        "sr_hw": f"{out_h}x{out_w}",
        "time_ms": (t1 - t0) * 1000.0,
    }


@torch.no_grad()
def benchmark_model(
    model: torch.nn.Module,
    device: torch.device,
    h: int,
    w: int,
    warmup: int,
    runs: int,
    channels_last: bool = True,
) -> Dict:
    x = torch.rand(1, 3, h, w, device=device, dtype=torch.float32) * 255.0

    if channels_last:
        try:
            x = x.to(memory_format=torch.channels_last)
        except Exception:
            pass

    for _ in range(warmup):
        _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    times_ms = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_ms = np.array(times_ms, dtype=np.float64)
    mean_ms = float(times_ms.mean())
    median_ms = float(np.median(times_ms))
    min_ms = float(times_ms.min())
    max_ms = float(times_ms.max())
    fps = 1000.0 / mean_ms if mean_ms > 0 else math.inf

    out_h = h * 3
    out_w = w * 3
    mpix = (out_h * out_w) / 1e6
    mpix_s = mpix / (mean_ms / 1000.0) if mean_ms > 0 else math.inf

    return {
        "bench_input_hw": f"{h}x{w}",
        "bench_output_hw": f"{out_h}x{out_w}",
        "warmup": warmup,
        "runs": runs,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "fps": fps,
        "mpix_per_sec": mpix_s,
    }


def print_model_info(ckpt_path: str, cfg: Dict):
    info = read_pt_model_info(ckpt_path)
    print("=" * 80)
    print("MODEL INFO")
    print(f"ckpt_path           : {ckpt_path}")
    print(f"format              : {info.get('format')}")
    print(f"file_size_bytes     : {info.get('file_size_bytes')}")
    print(f"file_size_mb        : {info.get('file_size_mb')}")
    print(f"raw_weight_bytes    : {info.get('raw_weight_bytes')}")
    print(f"raw_weight_mb       : {info.get('raw_weight_mb')}")
    print(f"param_count         : {info.get('param_count')}")
    print("cfg:")
    for k, v in cfg.items():
        print(f"  - {k}: {v}")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Export/deploy ckpt .pt")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no_channels_last", action="store_true")
    ap.add_argument("--input", type=str, default=None, help="1 ảnh hoặc thư mục ảnh LR")
    ap.add_argument("--out_dir", type=str, default=None, help="thư mục lưu ảnh SR")
    ap.add_argument("--benchmark", action="store_true", help="benchmark runtime")
    ap.add_argument("--bench_h", type=int, default=128)
    ap.add_argument("--bench_w", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    channels_last = not args.no_channels_last

    model, cfg = load_export_model(args.ckpt, device=device, channels_last=channels_last)
    print_model_info(args.ckpt, cfg)

    if args.input is not None:
        files = list_images(args.input)
        results = []

        print(f"[INFO] Found {len(files)} image(s)")
        for i, path in enumerate(files, start=1):
            res = run_single_image(
                model=model,
                img_path=path,
                device=device,
                out_dir=args.out_dir,
                channels_last=channels_last,
            )
            results.append(res)
            print(
                f"[{i}/{len(files)}] "
                f"LR={res['lr_hw']} -> SR={res['sr_hw']} | "
                f"{res['time_ms']:.3f} ms | "
                f"{Path(path).name}"
            )

        if results:
            arr = np.array([x["time_ms"] for x in results], dtype=np.float64)
            print("-" * 80)
            print("INFERENCE SUMMARY")
            print(f"num_images          : {len(results)}")
            print(f"mean_ms             : {arr.mean():.3f}")
            print(f"median_ms           : {np.median(arr):.3f}")
            print(f"min_ms              : {arr.min():.3f}")
            print(f"max_ms              : {arr.max():.3f}")
            if args.out_dir is not None:
                print(f"saved_to            : {args.out_dir}")
            print("-" * 80)

    if args.benchmark:
        bench = benchmark_model(
            model=model,
            device=device,
            h=args.bench_h,
            w=args.bench_w,
            warmup=args.warmup,
            runs=args.runs,
            channels_last=channels_last,
        )
        print("-" * 80)
        print("BENCHMARK")
        for k, v in bench.items():
            if isinstance(v, float):
                print(f"{k:20s}: {v:.6f}")
            else:
                print(f"{k:20s}: {v}")
        print("-" * 80)

    if args.input is None and not args.benchmark:
        print("[INFO] Bạn chưa chọn --input hoặc --benchmark, nên script chỉ load model và in model info.")


if __name__ == "__main__":
    main()