

## we use psnr_mode_fixed_int8_rgb_sh0 to check the final output quality without shave, which is the most relevant metric for real use cases for serving evaluation of contest submission
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import math
import glob
import json
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
from PIL import Image

import torch
import tensorflow as tf

from model import AntSR
from data_div2k_pairs import resolve_div2k_paths
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


# -------------------------
# Metrics
# -------------------------
def load_rgb_float255(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.float32)


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def shave_border(img: np.ndarray, shave: int) -> np.ndarray:
    if shave <= 0:
        return img
    h, w = img.shape[:2]
    if h <= 2 * shave or w <= 2 * shave:
        return img
    return img[shave:-shave, shave:-shave, ...]


def psnr_255(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    mse = float(np.mean((pred - gt) ** 2))
    if mse < eps:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)

def ssim_rgb_255(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.clip(pred.astype(np.float32), 0.0, 255.0)
    gt = np.clip(gt.astype(np.float32), 0.0, 255.0)

    pred_tf = tf.convert_to_tensor(pred[None, ...], dtype=tf.float32)  # NHWC
    gt_tf = tf.convert_to_tensor(gt[None, ...], dtype=tf.float32)      # NHWC

    # returns shape [1]
    val = tf.image.ssim(pred_tf, gt_tf, max_val=255.0)
    return float(val.numpy()[0])


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def metric_push(bucket: Dict[str, List[float]], key: str, value: float) -> None:
    bucket.setdefault(key, []).append(float(value))


def metric_mean(bucket: Dict[str, List[float]]) -> Dict[str, float]:
    return {k: float(np.mean(v)) for k, v in bucket.items()}


def lr_to_hr_path(lr_path: str, hr_dir: str, scale: int = 3) -> str:
    base = os.path.basename(lr_path)
    stem = os.path.splitext(base)[0]
    suffix = f"x{scale}"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    hr_path = os.path.join(hr_dir, f"{stem}.png")
    if not os.path.exists(hr_path):
        raise FileNotFoundError(f"Missing HR pair for {lr_path} -> {hr_path}")
    return hr_path


# -------------------------
# PyTorch runner
# -------------------------
def load_pytorch_model(deploy_ckpt: str, device: str = "cpu") -> torch.nn.Module:
    ckpt = torch.load(deploy_ckpt, map_location="cpu")
    if not isinstance(ckpt, dict) or "cfg" not in ckpt or "model" not in ckpt:
        raise RuntimeError("Expected deploy checkpoint with keys {'cfg', 'model'}")
    model = AntSR(deploy=True, **ckpt["cfg"]).eval().to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.requires_grad_(False)
    model = model.to(memory_format=torch.channels_last)
    return model


class PyTorchRunner:
    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.device = torch.device(device)

    @torch.inference_mode()
    def infer_full(self, lr_hwc_255: np.ndarray) -> np.ndarray:
        x = (
            torch.from_numpy(lr_hwc_255)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .contiguous()
            .float()
            .to(self.device)
        )
        x = x.to(memory_format=torch.channels_last)
        y = self.model(x).cpu().float().squeeze(0).permute(1, 2, 0).contiguous().numpy()
        return np.clip(y, 0.0, 255.0)


# -------------------------
# TFLite helpers
# -------------------------
def quantize_to_int8(x_float: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"Invalid quantization scale: {scale}")
    x_q = np.round(x_float / scale + zero_point)
    x_q = np.clip(x_q, -128, 127).astype(np.int8)
    return x_q


def dequantize_int8(x_q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (x_q.astype(np.float32) - float(zero_point)) * float(scale)


class TFLiteDynamicRunner:
    def __init__(self, tflite_path: str):
        self.tflite_path = tflite_path
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

    def inspect(self) -> Dict[str, Any]:
        return {
            "input_shape": tuple(int(x) for x in self.inp["shape"]),
            "input_shape_signature": tuple(int(x) for x in self.inp.get("shape_signature", self.inp["shape"])),
            "input_dtype": np.dtype(self.inp["dtype"]).name,
            "input_quantization": (float(self.inp["quantization"][0]), int(self.inp["quantization"][1])),
            "output_shape": tuple(int(x) for x in self.out["shape"]),
            "output_shape_signature": tuple(int(x) for x in self.out.get("shape_signature", self.out["shape"])),
            "output_dtype": np.dtype(self.out["dtype"]).name,
            "output_quantization": (float(self.out["quantization"][0]), int(self.out["quantization"][1])),
        }

    def infer(self, lr_hwc_255: np.ndarray) -> np.ndarray:
        x = lr_hwc_255[None, ...].astype(np.float32)

        self.interpreter.resize_tensor_input(self.inp["index"], list(x.shape), strict=False)
        self.interpreter.allocate_tensors()

        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

        in_dtype = np.dtype(self.inp["dtype"]).name
        out_dtype = np.dtype(self.out["dtype"]).name

        if in_dtype == "int8":
            in_scale, in_zp = self.inp["quantization"]
            x = quantize_to_int8(x, float(in_scale), int(in_zp))
        elif in_dtype != "float32":
            raise RuntimeError(f"Unsupported input dtype: {in_dtype}")

        self.interpreter.set_tensor(self.inp["index"], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])

        if out_dtype == "int8":
            out_scale, out_zp = self.out["quantization"]
            y = dequantize_int8(y, float(out_scale), int(out_zp))
        elif out_dtype != "float32":
            raise RuntimeError(f"Unsupported output dtype: {out_dtype}")

        return np.clip(y[0], 0.0, 255.0)


def safe_pad_reflect(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w, _ = img.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return img
    try:
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    except Exception:
        return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def positions(full_len: int, tile_len: int, overlap: int) -> List[int]:
    if full_len <= tile_len:
        return [0]
    stride = tile_len - 2 * overlap
    if stride <= 0:
        raise ValueError(f"Bad overlap={overlap} for tile_len={tile_len}")
    pos = list(range(0, full_len - tile_len + 1, stride))
    if pos[-1] != full_len - tile_len:
        pos.append(full_len - tile_len)
    return pos


class TFLiteFixedRunner:
    def __init__(self, tflite_path: str):
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

        self.input_shape = tuple(int(x) for x in self.inp["shape"])
        self.output_shape = tuple(int(x) for x in self.out["shape"])
        self.input_dtype = np.dtype(self.inp["dtype"]).name
        self.output_dtype = np.dtype(self.out["dtype"]).name

        if self.input_shape[0] != 1 or self.input_shape[3] != 3:
            raise RuntimeError(f"Unexpected input shape: {self.input_shape}")
        if self.output_shape[0] != 1 or self.output_shape[3] != 3:
            raise RuntimeError(f"Unexpected output shape: {self.output_shape}")

        self.tile_h = self.input_shape[1]
        self.tile_w = self.input_shape[2]
        self.scale = self.output_shape[1] // self.tile_h

    def infer_one_tile(self, tile_nhwc_255: np.ndarray) -> np.ndarray:
        x = tile_nhwc_255.astype(np.float32)
        if self.input_dtype == "int8":
            in_scale, in_zp = self.inp["quantization"]
            x = quantize_to_int8(x, float(in_scale), int(in_zp))
        self.interpreter.set_tensor(self.inp["index"], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])
        if self.output_dtype == "int8":
            out_scale, out_zp = self.out["quantization"]
            y = dequantize_int8(y, float(out_scale), int(out_zp))
        return np.clip(y, 0.0, 255.0)

    def infer_tiled(self, lr_hwc_255: np.ndarray, overlap: int = 16) -> np.ndarray:
        h, w, _ = lr_hwc_255.shape
        pad_h = max(h, self.tile_h)
        pad_w = max(w, self.tile_w)
        lr_pad = safe_pad_reflect(lr_hwc_255, pad_h, pad_w)
        ph, pw, _ = lr_pad.shape

        ys = positions(ph, self.tile_h, overlap)
        xs = positions(pw, self.tile_w, overlap)

        out_pad = np.zeros((ph * self.scale, pw * self.scale, 3), dtype=np.float32)

        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                tile = lr_pad[y : y + self.tile_h, x : x + self.tile_w, :]
                tile_out = self.infer_one_tile(tile[None, ...])[0]

                top_crop = 0 if yi == 0 else overlap * self.scale
                left_crop = 0 if xi == 0 else overlap * self.scale
                bottom_crop = 0 if yi == len(ys) - 1 else overlap * self.scale
                right_crop = 0 if xi == len(xs) - 1 else overlap * self.scale

                valid = tile_out[
                    top_crop : tile_out.shape[0] - bottom_crop,
                    left_crop : tile_out.shape[1] - right_crop,
                    :
                ]

                oy0 = y * self.scale + top_crop
                ox0 = x * self.scale + left_crop
                oy1 = oy0 + valid.shape[0]
                ox1 = ox0 + valid.shape[1]
                out_pad[oy0:oy1, ox0:ox1, :] = valid

        out = out_pad[: h * self.scale, : w * self.scale, :]
        return np.clip(out, 0.0, 255.0)


# -------------------------
# Evaluation
# -------------------------
def evaluate_all(
    deploy_ckpt: str,
    model_none_int8_path: str,
    data_root: str,
    device: str,
    shaves: List[int],
    max_images: int = 0,
    model_none_float_path: Optional[str] = None,
    model_fixed_path: Optional[str] = None,
    fixed_overlap: int = 16,
) -> Dict[str, Any]:
    pt_model = load_pytorch_model(deploy_ckpt, device=device)
    pt_runner = PyTorchRunner(pt_model, device=device)

    none_int8 = TFLiteDynamicRunner(model_none_int8_path)
    none_float = TFLiteDynamicRunner(model_none_float_path) if model_none_float_path else None
    fixed_runner = TFLiteFixedRunner(model_fixed_path) if model_fixed_path else None

    _, _, valid_hr, valid_lr = resolve_div2k_paths(data_root, scale=3)
    lr_files = sorted(glob.glob(os.path.join(valid_lr, "*.png")))
    if not lr_files:
        raise FileNotFoundError(f"No LR png found in {valid_lr}")

    results: Dict[str, List[float]] = {}
    per_image: List[Dict[str, Any]] = []
    seen = 0

    for idx, lr_path in enumerate(lr_files, start=1):
        hr_path = lr_to_hr_path(lr_path, valid_hr, scale=3)
        lr = load_rgb_float255(lr_path)
        hr = load_rgb_float255(hr_path)

        sr_pt = pt_runner.infer_full(lr)
        sr_none_int8 = none_int8.infer(lr)
        sr_none_float = none_float.infer(lr) if none_float else None
        sr_fixed = fixed_runner.infer_tiled(lr, overlap=fixed_overlap) if fixed_runner else None

        H, W = hr.shape[:2]
        sr_pt = sr_pt[:H, :W, :]
        sr_none_int8 = sr_none_int8[:H, :W, :]
        if sr_none_float is not None:
            sr_none_float = sr_none_float[:H, :W, :]
        if sr_fixed is not None:
            sr_fixed = sr_fixed[:H, :W, :]

        item = {
            "lr_file": os.path.basename(lr_path),
            "hr_file": os.path.basename(hr_path),
            "metrics": {},
        }

        print(f"[{idx}/{len(lr_files)}] {os.path.basename(lr_path)}", flush=True)

        for sh in shaves:
            hr_s = shave_border(hr, sh)
            pt_s = shave_border(sr_pt, sh)
            none_int8_s = shave_border(sr_none_int8, sh)

            metrics = {
                f"psnr_pt_rgb_sh{sh}": psnr_255(pt_s, hr_s),
                f"psnr_model_none_int8_rgb_sh{sh}": psnr_255(none_int8_s, hr_s),
                f"psnr_model_none_int8_vs_pt_rgb_sh{sh}": psnr_255(none_int8_s, pt_s),

                f"psnr_pt_y_sh{sh}": psnr_255(
                    shave_border(rgb_to_y(sr_pt), sh),
                    shave_border(rgb_to_y(hr), sh)
                ),
                f"psnr_model_none_int8_y_sh{sh}": psnr_255(
                    shave_border(rgb_to_y(sr_none_int8), sh),
                    shave_border(rgb_to_y(hr), sh)
                ),
                f"psnr_model_none_int8_vs_pt_y_sh{sh}": psnr_255(
                    shave_border(rgb_to_y(sr_none_int8), sh),
                    shave_border(rgb_to_y(sr_pt), sh)
                ),

                f"ssim_pt_rgb_sh{sh}": ssim_rgb_255(pt_s, hr_s),
                f"ssim_model_none_int8_rgb_sh{sh}": ssim_rgb_255(none_int8_s, hr_s),
                f"ssim_model_none_int8_vs_pt_rgb_sh{sh}": ssim_rgb_255(none_int8_s, pt_s),
            }

            if sr_none_float is not None:
                none_float_s = shave_border(sr_none_float, sh)
                metrics.update({
                    f"psnr_model_none_float_rgb_sh{sh}": psnr_255(none_float_s, hr_s),
                    f"psnr_model_none_float_vs_pt_rgb_sh{sh}": psnr_255(none_float_s, pt_s),
                    f"psnr_model_none_int8_vs_model_none_float_rgb_sh{sh}": psnr_255(none_int8_s, none_float_s),

                    f"psnr_model_none_float_y_sh{sh}": psnr_255(
                        shave_border(rgb_to_y(sr_none_float), sh),
                        shave_border(rgb_to_y(hr), sh)
                    ),
                    f"psnr_model_none_float_vs_pt_y_sh{sh}": psnr_255(
                        shave_border(rgb_to_y(sr_none_float), sh),
                        shave_border(rgb_to_y(sr_pt), sh)
                    ),
                    f"psnr_model_none_int8_vs_model_none_float_y_sh{sh}": psnr_255(
                        shave_border(rgb_to_y(sr_none_int8), sh),
                        shave_border(rgb_to_y(sr_none_float), sh)
                    ),

                    f"ssim_model_none_float_rgb_sh{sh}": ssim_rgb_255(none_float_s, hr_s),
                    f"ssim_model_none_float_vs_pt_rgb_sh{sh}": ssim_rgb_255(none_float_s, pt_s),
                    f"ssim_model_none_int8_vs_model_none_float_rgb_sh{sh}": ssim_rgb_255(none_int8_s, none_float_s),
                })

            if sr_fixed is not None:
                fixed_s = shave_border(sr_fixed, sh)
                metrics.update({
                    f"psnr_model_fixed_int8_rgb_sh{sh}": psnr_255(fixed_s, hr_s),
                    f"psnr_model_fixed_int8_vs_pt_rgb_sh{sh}": psnr_255(fixed_s, pt_s),

                    f"psnr_model_fixed_int8_y_sh{sh}": psnr_255(
                        shave_border(rgb_to_y(sr_fixed), sh),
                        shave_border(rgb_to_y(hr), sh)
                    ),
                    f"psnr_model_fixed_int8_vs_pt_y_sh{sh}": psnr_255(
                        shave_border(rgb_to_y(sr_fixed), sh),
                        shave_border(rgb_to_y(sr_pt), sh)
                    ),

                    f"ssim_model_fixed_int8_rgb_sh{sh}": ssim_rgb_255(fixed_s, hr_s),
                    f"ssim_model_fixed_int8_vs_pt_rgb_sh{sh}": ssim_rgb_255(fixed_s, pt_s),
                })

            for k, v in metrics.items():
                metric_push(results, k, v)
                item["metrics"][k] = float(v)

        per_image.append(item)
        seen += 1
        if max_images > 0 and seen >= max_images:
            break

    return {
        "summary": metric_mean(results),
        "per_image": per_image,
        "model_none_int8_info": none_int8.inspect(),
        "model_none_float_info": none_float.inspect() if none_float else None,
        "model_fixed_info": {
            "input_shape": fixed_runner.input_shape,
            "output_shape": fixed_runner.output_shape,
            "input_dtype": fixed_runner.input_dtype,
            "output_dtype": fixed_runner.output_dtype,
            "tile_h": fixed_runner.tile_h,
            "tile_w": fixed_runner.tile_w,
            "scale": fixed_runner.scale,
        } if fixed_runner else None,
        "num_images": int(seen),
        "shaves": list(shaves),
        "fixed_overlap": int(fixed_overlap),
    }


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy_ckpt", type=str, required=True)
    ap.add_argument("--model_none_int8", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--shaves", type=str, default="0,3")
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--model_none_float", type=str, default=None)
    ap.add_argument("--model_fixed", type=str, default=None)
    ap.add_argument("--fixed_overlap", type=int, default=16)
    ap.add_argument("--out_json", type=str, default="eval_div2k_full_report.json")
    args = ap.parse_args()

    shaves = parse_int_list(args.shaves)
    report = evaluate_all(
        deploy_ckpt=args.deploy_ckpt,
        model_none_int8_path=args.model_none_int8,
        model_none_float_path=args.model_none_float,
        model_fixed_path=args.model_fixed,
        data_root=args.data_root,
        device=args.device,
        shaves=shaves,
        max_images=args.max_images,
        fixed_overlap=args.fixed_overlap,
    )

    print("\n=== FINAL METRICS ===")
    for k in sorted(report["summary"].keys()):
        print(f"{k}: {report['summary'][k]:.6f}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved report to: {args.out_json}")


if __name__ == "__main__":
    main()




