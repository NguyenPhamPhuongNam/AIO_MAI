import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import math
import glob
import shutil
import random
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

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
# Misc
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(str(x) + "\n")


def psnr_255(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    mse = float(np.mean((pred - gt) ** 2))
    if mse < eps:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


# -------------------------
# Model
# -------------------------

def load_model(ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "cfg" not in ckpt or "model" not in ckpt:
        raise RuntimeError(
            f"Checkpoint must be deploy export with keys ['cfg', 'model'], got: {type(ckpt)}"
        )
    model = AntSR(deploy=True, **ckpt["cfg"]).eval()
    model.load_state_dict(ckpt["model"], strict=True)
    model.requires_grad_(False)
    model = model.to(memory_format=torch.channels_last)
    return model


# -------------------------
# DIV2K IO
# -------------------------

def _list_pngs(d: str) -> List[str]:
    return sorted(glob.glob(os.path.join(d, "*.png")))


def resolve_div2k_io(data_root: str) -> Dict[str, str]:
    train_hr, train_lr, valid_hr, valid_lr = resolve_div2k_paths(data_root, scale=3)
    return {
        "train_hr": train_hr,
        "train_lr": train_lr,
        "valid_hr": valid_hr,
        "valid_lr": valid_lr,
    }


def choose_lr_dir(paths: Dict[str, str], split: str) -> str:
    split = split.lower()
    if split == "train":
        return paths["train_lr"]
    if split == "valid":
        return paths["valid_lr"]
    raise ValueError(f"Unknown split: {split}")


def choose_hr_dir(paths: Dict[str, str], split: str) -> str:
    split = split.lower()
    if split == "train":
        return paths["train_hr"]
    if split == "valid":
        return paths["valid_hr"]
    raise ValueError(f"Unknown split: {split}")


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


def list_lr_files(data_root: str, split: str) -> Tuple[Dict[str, str], List[str]]:
    paths = resolve_div2k_io(data_root)
    lr_dir = choose_lr_dir(paths, split)
    files = _list_pngs(lr_dir)
    if not files:
        raise FileNotFoundError(f"No LR PNG files found in: {lr_dir}")
    return paths, files


def load_lr_hr_pair(lr_path: str, hr_dir: str, scale: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    hr_path = lr_to_hr_path(lr_path, hr_dir=hr_dir, scale=scale)
    lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32)
    hr = np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32)
    return lr, hr


# -------------------------
# Calibration file selection
# -------------------------

def _luma_variance(img: np.ndarray) -> float:
    y = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return float(np.var(y))


def rank_lr_files_by_variance(lr_files: List[str]) -> List[Tuple[str, float]]:
    scored: List[Tuple[str, float]] = []
    for p in lr_files:
        img = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
        scored.append((p, _luma_variance(img)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def pick_calibration_files(
    lr_files: List[str],
    num_samples: int,
    selection_mode: str = "stratified_var",
) -> List[str]:
    total = min(int(num_samples), len(lr_files))
    if total <= 0:
        return []
    mode = selection_mode.lower()
    if mode == "random":
        files = list(lr_files)
        random.shuffle(files)
        return files[:total]

    scored = rank_lr_files_by_variance(lr_files)
    only_files = [p for p, _ in scored]

    if mode == "topvar":
        return only_files[:total]

    if mode == "mixed":
        top_k = max(1, total // 2)
        rest_k = total - top_k
        top = only_files[: min(top_k, len(only_files))]
        remain = only_files[top_k:]
        random.shuffle(remain)
        return top + remain[:rest_k]

    if mode == "stratified_var":
        if total == 1:
            return [only_files[0]]
        n = len(only_files)
        idxs = np.linspace(0, n - 1, num=total, dtype=int).tolist()
        return [only_files[i] for i in idxs]

    raise ValueError(f"Unknown selection_mode: {selection_mode}")


# -------------------------
# Calibration builders
# -------------------------

def random_crop_pair(
    lr: np.ndarray,
    hr: np.ndarray,
    crop_h: int,
    crop_w: int,
    scale: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = lr.shape
    if h < crop_h or w < crop_w:
        raise ValueError(f"LR too small for crop: got {(h, w)} need {(crop_h, crop_w)}")
    top = random.randint(0, h - crop_h)
    left = random.randint(0, w - crop_w)
    lr_crop = lr[top : top + crop_h, left : left + crop_w, :]
    hr_crop = hr[top * scale : (top + crop_h) * scale, left * scale : (left + crop_w) * scale, :]
    return lr_crop, hr_crop


def center_crop_pair(
    lr: np.ndarray,
    hr: np.ndarray,
    crop_h: int,
    crop_w: int,
    scale: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = lr.shape
    if h < crop_h or w < crop_w:
        raise ValueError(f"LR too small for crop: got {(h, w)} need {(crop_h, crop_w)}")
    top = max(0, (h - crop_h) // 2)
    left = max(0, (w - crop_w) // 2)
    lr_crop = lr[top : top + crop_h, left : left + crop_w, :]
    hr_crop = hr[top * scale : (top + crop_h) * scale, left * scale : (left + crop_w) * scale, :]
    return lr_crop, hr_crop


def high_variance_crop_pair(
    lr: np.ndarray,
    hr: np.ndarray,
    crop_h: int,
    crop_w: int,
    scale: int = 3,
    num_candidates: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    h, w, _ = lr.shape
    if h < crop_h or w < crop_w:
        raise ValueError(f"LR too small for crop: got {(h, w)} need {(crop_h, crop_w)}")
    best_lr = None
    best_hr = None
    max_var = -1.0
    for _ in range(max(1, int(num_candidates))):
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        lr_patch = lr[top : top + crop_h, left : left + crop_w, :]
        hr_patch = hr[top * scale : (top + crop_h) * scale, left * scale : (left + crop_w) * scale, :]
        var = _luma_variance(lr_patch)
        if var > max_var:
            max_var = var
            best_lr = lr_patch
            best_hr = hr_patch
    if best_lr is None or best_hr is None:
        raise RuntimeError("Failed to sample high-variance paired crop.")
    return best_lr, best_hr


def choose_crop_pair(
    lr: np.ndarray,
    hr: np.ndarray,
    crop_h: int,
    crop_w: int,
    scale: int = 3,
    crop_mode: str = "mixed",
    highvar_candidates: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    mode = crop_mode.lower()
    if mode == "random":
        return random_crop_pair(lr, hr, crop_h, crop_w, scale=scale)
    if mode == "center":
        return center_crop_pair(lr, hr, crop_h, crop_w, scale=scale)
    if mode == "highvar":
        return high_variance_crop_pair(
            lr, hr, crop_h, crop_w, scale=scale, num_candidates=highvar_candidates
        )
    if mode == "mixed":
        r = random.random()
        if r < 0.4:
            return random_crop_pair(lr, hr, crop_h, crop_w, scale=scale)
        if r < 0.8:
            return high_variance_crop_pair(
                lr, hr, crop_h, crop_w, scale=scale, num_candidates=highvar_candidates
            )
        return center_crop_pair(lr, hr, crop_h, crop_w, scale=scale)
    raise ValueError(f"Unknown crop_mode: {crop_mode}")


def build_paired_canvas_sample(
    lr_files: List[str],
    hr_dir: str,
    out_h: int,
    out_w: int,
    scale: int = 3,
    tile: int = 128,
    crop_mode: str = "mixed",
    highvar_candidates: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    lr_canvas = np.zeros((out_h, out_w, 3), dtype=np.float32)
    hr_canvas = np.zeros((out_h * scale, out_w * scale, 3), dtype=np.float32)

    for y in range(0, out_h, tile):
        for x in range(0, out_w, tile):
            ph = min(tile, out_h - y)
            pw = min(tile, out_w - x)
            chosen = None
            for _ in range(32):
                lr_path = random.choice(lr_files)
                lr, hr = load_lr_hr_pair(lr_path, hr_dir=hr_dir, scale=scale)
                if lr.shape[0] >= ph and lr.shape[1] >= pw:
                    chosen = (lr, hr)
                    break
            if chosen is None:
                raise RuntimeError(
                    f"Could not find any LR image large enough for tile {(ph, pw)}. Try reducing tile size."
                )
            lr, hr = chosen
            lr_patch, hr_patch = choose_crop_pair(
                lr=lr,
                hr=hr,
                crop_h=ph,
                crop_w=pw,
                scale=scale,
                crop_mode=crop_mode,
                highvar_candidates=highvar_candidates,
            )
            lr_canvas[y : y + ph, x : x + pw, :] = lr_patch
            hr_canvas[y * scale : (y + ph) * scale, x * scale : (x + pw) * scale, :] = hr_patch
    return lr_canvas, hr_canvas


def representative_dataset_fixed_canvas(
    lr_files: List[str],
    hr_dir: str,
    out_h: int,
    out_w: int,
    num_samples: int,
    scale: int = 3,
    tile: int = 128,
    crop_mode: str = "mixed",
    highvar_candidates: int = 10,
):
    def generator():
        total = int(num_samples)
        for i in range(total):
            if i == 0 or (i + 1) % 10 == 0 or (i + 1) == total:
                print(f"[calib-fixed] {i + 1}/{total}", flush=True)
            lr_canvas, _ = build_paired_canvas_sample(
                lr_files=lr_files,
                hr_dir=hr_dir,
                out_h=out_h,
                out_w=out_w,
                scale=scale,
                tile=tile,
                crop_mode=crop_mode,
                highvar_candidates=highvar_candidates,
            )
            yield [lr_canvas[None, ...].astype(np.float32)]

    return generator


def representative_dataset_dynamic_lr(
    lr_files: List[str],
    hr_dir: str,
    num_samples: int,
    dynamic_mode: str = "full",
    dynamic_crop_h: int = 256,
    dynamic_crop_w: int = 256,
    highvar_candidates: int = 10,
):
    mode = dynamic_mode.lower()

    def generator():
        total = min(int(num_samples), len(lr_files))
        for i, p in enumerate(lr_files[:total], start=1):
            if i == 1 or i % 10 == 0 or i == total:
                print(f"[calib-dynamic:{mode}] {i}/{total}", flush=True)
            if mode == "full":
                lr = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
                yield [lr[None, ...].astype(np.float32)]
                continue

            lr, hr = load_lr_hr_pair(p, hr_dir=hr_dir, scale=3)
            if mode == "center":
                lr_patch, _ = center_crop_pair(lr, hr, dynamic_crop_h, dynamic_crop_w, scale=3)
            elif mode == "random":
                lr_patch, _ = random_crop_pair(lr, hr, dynamic_crop_h, dynamic_crop_w, scale=3)
            elif mode == "highvar":
                lr_patch, _ = high_variance_crop_pair(
                    lr, hr, dynamic_crop_h, dynamic_crop_w, scale=3, num_candidates=highvar_candidates
                )
            elif mode == "mixed":
                lr_patch, _ = choose_crop_pair(
                    lr,
                    hr,
                    dynamic_crop_h,
                    dynamic_crop_w,
                    scale=3,
                    crop_mode="mixed",
                    highvar_candidates=highvar_candidates,
                )
            else:
                raise ValueError(f"Unknown dynamic_mode: {dynamic_mode}")
            yield [lr_patch[None, ...].astype(np.float32)]

    return generator


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


def inspect_tflite_model(tflite_path: str) -> Dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    return {
        "input_shape": tuple(int(x) for x in inp["shape"]),
        "input_shape_signature": tuple(int(x) for x in inp.get("shape_signature", inp["shape"])),
        "input_dtype": np.dtype(inp["dtype"]).name,
        "input_quantization": (float(inp["quantization"][0]), int(inp["quantization"][1])),
        "output_shape": tuple(int(x) for x in out["shape"]),
        "output_shape_signature": tuple(int(x) for x in out.get("shape_signature", out["shape"])),
        "output_dtype": np.dtype(out["dtype"]).name,
        "output_quantization": (float(out["quantization"][0]), int(out["quantization"][1])),
    }


class TFLiteDynamicRunner:
    def __init__(self, tflite_path: str):
        self.tflite_path = tflite_path
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

    def inspect(self) -> Dict[str, Any]:
        return inspect_tflite_model(self.tflite_path)

    def infer(self, lr_nhwc_255: np.ndarray) -> np.ndarray:
        assert lr_nhwc_255.ndim == 4 and lr_nhwc_255.shape[0] == 1 and lr_nhwc_255.shape[-1] == 3
        self.interpreter.resize_tensor_input(self.inp["index"], list(lr_nhwc_255.shape), strict=False)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

        x = lr_nhwc_255.astype(np.float32)
        in_dtype = np.dtype(self.inp["dtype"]).name
        out_dtype = np.dtype(self.out["dtype"]).name

        if in_dtype == "int8":
            in_scale, in_zp = self.inp["quantization"]
            x = quantize_to_int8(x, float(in_scale), int(in_zp))
        elif in_dtype != "float32":
            raise RuntimeError(f"Unsupported TFLite input dtype: {in_dtype}")

        self.interpreter.set_tensor(self.inp["index"], x)
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(self.out["index"])

        if out_dtype == "int8":
            out_scale, out_zp = self.out["quantization"]
            y = dequantize_int8(y, float(out_scale), int(out_zp))
        elif out_dtype == "float32":
            y = y.astype(np.float32)
        else:
            raise RuntimeError(f"Unsupported TFLite output dtype: {out_dtype}")
        return np.clip(y, 0.0, 255.0)


def run_pytorch_sr(model: torch.nn.Module, lr_nhwc_255: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(lr_nhwc_255).permute(0, 3, 1, 2).contiguous().float()
    x = x.to(memory_format=torch.channels_last)
    with torch.inference_mode():
        y = model(x).cpu().float().permute(0, 2, 3, 1).contiguous().numpy()
    return np.clip(y, 0.0, 255.0)


# -------------------------
# Export
# -------------------------

def export_onnx(model: torch.nn.Module, onnx_path: str, dummy_input: torch.Tensor, dynamic: bool) -> None:
    kwargs = dict(
        model=model,
        args=(dummy_input,),
        f=onnx_path,
        opset_version=18,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=True,
    )
    if dynamic:
        kwargs["dynamic_axes"] = {
            "input": {2: "height", 3: "width"},
            "output": {2: "height_out", 3: "width_out"},
        }
    torch.onnx.export(**kwargs)


def run_onnx2tf(onnx_path: str, saved_model_dir: str) -> None:
    if shutil.which("onnx2tf") is None:
        raise RuntimeError("onnx2tf not found in PATH. Please install it first.")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    cmd = ["onnx2tf", "-i", onnx_path, "-o", saved_model_dir, "-v", "error"]
    subprocess.run(cmd, check=True, env=env)


def convert_saved_model_to_tflite_float(saved_model_dir: str, tflite_path: str) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    model_bytes = converter.convert()
    Path(tflite_path).write_bytes(model_bytes)


def convert_saved_model_to_tflite_int8(saved_model_dir: str, tflite_path: str, rep_dataset) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    Path(tflite_path).write_bytes(model_bytes)


# -------------------------
# Validation / checks
# -------------------------

def validate_artifact_triplet(
    model_tflite: str,
    model_none_tflite: str,
    model_none_float_tflite: str,
    fixed_h: int,
    fixed_w: int,
) -> Dict[str, Any]:
    info_fixed = inspect_tflite_model(model_tflite)
    info_dyn_int8 = inspect_tflite_model(model_none_tflite)
    info_dyn_fp32 = inspect_tflite_model(model_none_float_tflite)

    if tuple(info_fixed["input_shape"]) != (1, fixed_h, fixed_w, 3):
        raise RuntimeError(
            f"model.tflite input shape mismatch: got={info_fixed['input_shape']} expected={(1, fixed_h, fixed_w, 3)}"
        )
    if tuple(info_fixed["output_shape"]) != (1, fixed_h * 3, fixed_w * 3, 3):
        raise RuntimeError(
            f"model.tflite output shape mismatch: got={info_fixed['output_shape']} expected={(1, fixed_h * 3, fixed_w * 3, 3)}"
        )
    if info_fixed["input_dtype"] != "int8" or info_fixed["output_dtype"] != "int8":
        raise RuntimeError("model.tflite must be int8 -> int8")

    sig_in_dyn = tuple(info_dyn_int8["input_shape_signature"])
    sig_out_dyn = tuple(info_dyn_int8["output_shape_signature"])
    if info_dyn_int8["input_dtype"] != "int8" or info_dyn_int8["output_dtype"] != "int8":
        raise RuntimeError("model_none.tflite must be int8 -> int8")
    if len(sig_in_dyn) != 4 or sig_in_dyn[0] != 1 or sig_in_dyn[3] != 3:
        raise RuntimeError(f"Bad model_none.tflite input signature: {sig_in_dyn}")
    if len(sig_out_dyn) != 4 or sig_out_dyn[0] != 1 or sig_out_dyn[3] != 3:
        raise RuntimeError(f"Bad model_none.tflite output signature: {sig_out_dyn}")
    if not (sig_in_dyn[1] == -1 and sig_in_dyn[2] == -1):
        raise RuntimeError(f"model_none.tflite is not dynamic on H/W: {sig_in_dyn}")
    if not (sig_out_dyn[1] == -1 and sig_out_dyn[2] == -1):
        raise RuntimeError(f"model_none.tflite output is not dynamic on H/W: {sig_out_dyn}")

    sig_in_fp32 = tuple(info_dyn_fp32["input_shape_signature"])
    sig_out_fp32 = tuple(info_dyn_fp32["output_shape_signature"])
    if info_dyn_fp32["input_dtype"] != "float32" or info_dyn_fp32["output_dtype"] != "float32":
        raise RuntimeError("model_none_float.tflite must be float32 -> float32")
    if not (sig_in_fp32[1] == -1 and sig_in_fp32[2] == -1):
        raise RuntimeError(f"model_none_float.tflite is not dynamic on H/W: {sig_in_fp32}")
    if not (sig_out_fp32[1] == -1 and sig_out_fp32[2] == -1):
        raise RuntimeError(f"model_none_float.tflite output is not dynamic on H/W: {sig_out_fp32}")

    return {
        "model.tflite": info_fixed,
        "model_none.tflite": info_dyn_int8,
        "model_none_float.tflite": info_dyn_fp32,
    }


def smoke_verify_dynamic_models(
    model: torch.nn.Module,
    model_none_tflite: str,
    model_none_float_tflite: str,
    data_root: str,
    num_images: int,
) -> Dict[str, Any]:
    _, _, valid_hr, valid_lr = resolve_div2k_paths(data_root, scale=3)
    lr_files = sorted(glob.glob(os.path.join(valid_lr, "*.png")))
    if not lr_files:
        raise FileNotFoundError(f"No LR png found in {valid_lr}")

    dyn_int8 = TFLiteDynamicRunner(model_none_tflite)
    dyn_fp32 = TFLiteDynamicRunner(model_none_float_tflite)

    psnr_pt_hr = []
    psnr_int8_hr = []
    psnr_fp32_hr = []
    psnr_int8_pt = []
    psnr_fp32_pt = []
    per_image = []

    total = min(int(num_images), len(lr_files))
    for i, lr_path in enumerate(lr_files[:total], start=1):
        hr_path = lr_to_hr_path(lr_path, valid_hr, scale=3)
        lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32)
        hr = np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32)
        lr_in = lr[None, ...].astype(np.float32)

        sr_pt = run_pytorch_sr(model, lr_in)[0]
        sr_int8 = dyn_int8.infer(lr_in)[0]
        sr_fp32 = dyn_fp32.infer(lr_in)[0]

        H, W = hr.shape[:2]
        sr_pt = np.clip(sr_pt[:H, :W, :], 0.0, 255.0)
        sr_int8 = np.clip(sr_int8[:H, :W, :], 0.0, 255.0)
        sr_fp32 = np.clip(sr_fp32[:H, :W, :], 0.0, 255.0)
        hr = np.clip(hr, 0.0, 255.0)

        m_pt_hr = psnr_255(sr_pt, hr)
        m_int8_hr = psnr_255(sr_int8, hr)
        m_fp32_hr = psnr_255(sr_fp32, hr)
        m_int8_pt = psnr_255(sr_int8, sr_pt)
        m_fp32_pt = psnr_255(sr_fp32, sr_pt)

        psnr_pt_hr.append(m_pt_hr)
        psnr_int8_hr.append(m_int8_hr)
        psnr_fp32_hr.append(m_fp32_hr)
        psnr_int8_pt.append(m_int8_pt)
        psnr_fp32_pt.append(m_fp32_pt)
        per_image.append(
            {
                "lr_file": os.path.basename(lr_path),
                "hr_file": os.path.basename(hr_path),
                "psnr_pytorch_vs_hr": float(m_pt_hr),
                "psnr_model_none_int8_vs_hr": float(m_int8_hr),
                "psnr_model_none_float_vs_hr": float(m_fp32_hr),
                "psnr_model_none_int8_vs_pytorch": float(m_int8_pt),
                "psnr_model_none_float_vs_pytorch": float(m_fp32_pt),
            }
        )
        print(
            f"[smoke {i}/{total}] {os.path.basename(lr_path)} | PT-HR={m_pt_hr:.4f} | INT8-HR={m_int8_hr:.4f} | FP32-HR={m_fp32_hr:.4f}",
            flush=True,
        )

    return {
        "num_images": total,
        "avg_psnr_pytorch_vs_hr": float(np.mean(psnr_pt_hr)),
        "avg_psnr_model_none_int8_vs_hr": float(np.mean(psnr_int8_hr)),
        "avg_psnr_model_none_float_vs_hr": float(np.mean(psnr_fp32_hr)),
        "avg_psnr_model_none_int8_vs_pytorch": float(np.mean(psnr_int8_pt)),
        "avg_psnr_model_none_float_vs_pytorch": float(np.mean(psnr_fp32_pt)),
        "per_image": per_image,
    }


def score_dynamic_candidate(
    model: torch.nn.Module,
    model_none_tflite: str,
    data_root: str,
    num_images: int,
) -> Dict[str, float]:
    _, _, valid_hr, valid_lr = resolve_div2k_paths(data_root, scale=3)
    lr_files = sorted(glob.glob(os.path.join(valid_lr, "*.png")))
    runner = TFLiteDynamicRunner(model_none_tflite)

    psnr_int8_hr: List[float] = []
    psnr_int8_pt: List[float] = []
    total = min(int(num_images), len(lr_files))
    for lr_path in lr_files[:total]:
        hr_path = lr_to_hr_path(lr_path, valid_hr, scale=3)
        lr = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32)
        hr = np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32)
        lr_in = lr[None, ...].astype(np.float32)
        sr_pt = run_pytorch_sr(model, lr_in)[0]
        sr_int8 = runner.infer(lr_in)[0]
        H, W = hr.shape[:2]
        sr_pt = np.clip(sr_pt[:H, :W, :], 0.0, 255.0)
        sr_int8 = np.clip(sr_int8[:H, :W, :], 0.0, 255.0)
        hr = np.clip(hr[:H, :W, :], 0.0, 255.0)
        psnr_int8_hr.append(psnr_255(sr_int8, hr))
        psnr_int8_pt.append(psnr_255(sr_int8, sr_pt))
    return {
        "avg_psnr_int8_vs_hr": float(np.mean(psnr_int8_hr)),
        "avg_psnr_int8_vs_pytorch": float(np.mean(psnr_int8_pt)),
        "num_images": total,
    }


# -------------------------
# Search helpers
# -------------------------

def parse_search_presets(spec: str) -> List[Dict[str, Any]]:
    presets: List[Dict[str, Any]] = []
    chunks = [x.strip() for x in spec.split(";") if x.strip()]
    for chunk in chunks:
        parts = [x.strip() for x in chunk.split(",")]
        if len(parts) not in (5, 6):
            raise ValueError(
                "Each preset must be 'split,num,selection_mode,fixed_mode,dynamic_mode[,label]'"
            )
        split, num_s, sel_mode, fixed_mode, dynamic_mode = parts[:5]
        label = parts[5] if len(parts) == 6 else f"{split}_{num_s}_{sel_mode}_{fixed_mode}_{dynamic_mode}"
        presets.append(
            {
                "split": split,
                "num": int(num_s),
                "selection_mode": sel_mode,
                "fixed_mode": fixed_mode,
                "dynamic_mode": dynamic_mode,
                "label": label,
            }
        )
    return presets


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy_ckpt", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--fixed_h", type=int, default=720)
    ap.add_argument("--fixed_w", type=int, default=1280)
    ap.add_argument("--calib_split_fixed", type=str, default="valid", choices=["train", "valid"])
    ap.add_argument("--calib_split_dynamic", type=str, default="valid", choices=["train", "valid"])
    ap.add_argument("--num_calib_fixed", type=int, default=32)
    ap.add_argument("--num_calib_dynamic", type=int, default=32)
    ap.add_argument("--calib_mode_fixed", type=str, default="random", choices=["random", "center", "highvar", "mixed"])
    ap.add_argument("--dynamic_calib_mode", type=str, default="full", choices=["full", "center", "random", "highvar", "mixed"])
    ap.add_argument("--selection_mode_fixed", type=str, default="stratified_var", choices=["random", "topvar", "stratified_var", "mixed"])
    ap.add_argument("--selection_mode_dynamic", type=str, default="stratified_var", choices=["random", "topvar", "stratified_var", "mixed"])
    ap.add_argument("--dynamic_crop_h", type=int, default=256)
    ap.add_argument("--dynamic_crop_w", type=int, default=256)
    ap.add_argument("--highvar_candidates", type=int, default=10)
    ap.add_argument("--smoke_verify_images", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--auto_search", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--search_presets",
        type=str,
        default="valid,32,random,random,full,valid32_full;valid,64,random,random,full,valid64_full;valid,32,stratified_var,mixed,mixed,valid32_mixed;train,128,stratified_var,mixed,full,train128_full;train,256,stratified_var,mixed,full,train256_full",
    )
    ap.add_argument("--search_eval_images", type=int, default=20)

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    tflite_dir = os.path.join(args.out_dir, "TFLite")
    os.makedirs(tflite_dir, exist_ok=True)

    model = load_model(args.deploy_ckpt)

    # Always export ONNX / SavedModel once.
    with tempfile.TemporaryDirectory(prefix="mai_export_") as tmpdir:
        print("\n[1/6] Exporting fixed-size ONNX/SavedModel ...")
        onnx_fixed = os.path.join(tmpdir, "fixed.onnx")
        sm_fixed = os.path.join(tmpdir, "saved_model_fixed")
        dummy_input_fixed = torch.randn(1, 3, args.fixed_h, args.fixed_w, dtype=torch.float32).to(memory_format=torch.channels_last)
        export_onnx(model, onnx_fixed, dummy_input_fixed, dynamic=False)
        run_onnx2tf(onnx_fixed, sm_fixed)

        print("\n[2/6] Exporting dynamic-size ONNX/SavedModel ...")
        onnx_dyn = os.path.join(tmpdir, "dynamic.onnx")
        sm_dyn = os.path.join(tmpdir, "saved_model_dynamic")
        dummy_input_dyn = torch.randn(1, 3, args.dynamic_crop_h, args.dynamic_crop_w, dtype=torch.float32).to(memory_format=torch.channels_last)
        export_onnx(model, onnx_dyn, dummy_input_dyn, dynamic=True)
        run_onnx2tf(onnx_dyn, sm_dyn)

        best_cfg = {
            "fixed_split": args.calib_split_fixed,
            "dynamic_split": args.calib_split_dynamic,
            "num_fixed": int(args.num_calib_fixed),
            "num_dynamic": int(args.num_calib_dynamic),
            "fixed_mode": args.calib_mode_fixed,
            "dynamic_mode": args.dynamic_calib_mode,
            "selection_mode_fixed": args.selection_mode_fixed,
            "selection_mode_dynamic": args.selection_mode_dynamic,
            "search_score": None,
            "label": "manual",
        }
        search_results: List[Dict[str, Any]] = []

        if args.auto_search:
            print("\n[search] Searching calibration presets for best dynamic INT8 artifact ...")
            for preset in parse_search_presets(args.search_presets):
                split = preset["split"]
                paths, lr_all = list_lr_files(args.data_root, split=split)
                lr_sel = pick_calibration_files(
                    lr_all,
                    num_samples=preset["num"],
                    selection_mode=preset["selection_mode"],
                )
                hr_dir = choose_hr_dir(paths, split)
                rep_dynamic = representative_dataset_dynamic_lr(
                    lr_files=lr_sel,
                    hr_dir=hr_dir,
                    num_samples=len(lr_sel),
                    dynamic_mode=preset["dynamic_mode"],
                    dynamic_crop_h=args.dynamic_crop_h,
                    dynamic_crop_w=args.dynamic_crop_w,
                    highvar_candidates=args.highvar_candidates,
                )
                cand_path = os.path.join(tmpdir, f"candidate_{preset['label']}.tflite")
                convert_saved_model_to_tflite_int8(sm_dyn, cand_path, rep_dynamic)
                score = score_dynamic_candidate(
                    model=model,
                    model_none_tflite=cand_path,
                    data_root=args.data_root,
                    num_images=args.search_eval_images,
                )
                entry = {
                    "preset": preset,
                    "score": score,
                    "candidate_tflite": cand_path,
                }
                search_results.append(entry)
                print(
                    f"[search] {preset['label']} -> INT8-vs-HR={score['avg_psnr_int8_vs_hr']:.6f} INT8-vs-PT={score['avg_psnr_int8_vs_pytorch']:.6f}",
                    flush=True,
                )
            search_results.sort(key=lambda x: x["score"]["avg_psnr_int8_vs_hr"], reverse=True)
            winner = search_results[0]
            p = winner["preset"]
            best_cfg = {
                "fixed_split": p["split"],
                "dynamic_split": p["split"],
                "num_fixed": int(p["num"]),
                "num_dynamic": int(p["num"]),
                "fixed_mode": p["fixed_mode"],
                "dynamic_mode": p["dynamic_mode"],
                "selection_mode_fixed": p["selection_mode"],
                "selection_mode_dynamic": p["selection_mode"],
                "search_score": winner["score"],
                "label": p["label"],
            }
            print(f"[search] winner = {best_cfg['label']}", flush=True)

        # final calibration sets from chosen config
        div2k_fixed, fixed_lr_all = list_lr_files(args.data_root, split=best_cfg["fixed_split"])
        fixed_lr_files = pick_calibration_files(
            fixed_lr_all,
            num_samples=best_cfg["num_fixed"],
            selection_mode=best_cfg["selection_mode_fixed"],
        )
        fixed_hr_dir = choose_hr_dir(div2k_fixed, best_cfg["fixed_split"])

        div2k_dynamic, dynamic_lr_all = list_lr_files(args.data_root, split=best_cfg["dynamic_split"])
        dynamic_lr_files = pick_calibration_files(
            dynamic_lr_all,
            num_samples=best_cfg["num_dynamic"],
            selection_mode=best_cfg["selection_mode_dynamic"],
        )
        dynamic_hr_dir = choose_hr_dir(div2k_dynamic, best_cfg["dynamic_split"])

        save_json(os.path.join(args.out_dir, "resolved_div2k_paths_fixed.json"), div2k_fixed)
        save_json(os.path.join(args.out_dir, "resolved_div2k_paths_dynamic.json"), div2k_dynamic)
        save_lines(os.path.join(args.out_dir, "calibration_files_fixed.txt"), fixed_lr_files)
        save_lines(os.path.join(args.out_dir, "calibration_files_dynamic.txt"), dynamic_lr_files)

        rep_fixed = representative_dataset_fixed_canvas(
            lr_files=fixed_lr_files,
            hr_dir=fixed_hr_dir,
            out_h=args.fixed_h,
            out_w=args.fixed_w,
            num_samples=len(fixed_lr_files),
            scale=3,
            tile=128,
            crop_mode=best_cfg["fixed_mode"],
            highvar_candidates=args.highvar_candidates,
        )
        rep_dynamic = representative_dataset_dynamic_lr(
            lr_files=dynamic_lr_files,
            hr_dir=dynamic_hr_dir,
            num_samples=len(dynamic_lr_files),
            dynamic_mode=best_cfg["dynamic_mode"],
            dynamic_crop_h=args.dynamic_crop_h,
            dynamic_crop_w=args.dynamic_crop_w,
            highvar_candidates=args.highvar_candidates,
        )

        export_report: Dict[str, Any] = {
            "deploy_ckpt": args.deploy_ckpt,
            "data_root": args.data_root,
            "fixed_input_hw": [int(args.fixed_h), int(args.fixed_w)],
            "best_export_config": best_cfg,
            "search_results": search_results,
            "seed": int(args.seed),
            "artifacts": {},
        }

        model_fixed_path = os.path.join(tflite_dir, "model.tflite")
        model_none_int8_path = os.path.join(tflite_dir, "model_none.tflite")
        model_none_float_path = os.path.join(tflite_dir, "model_none_float.tflite")

        print("\n[3/6] Converting fixed-size INT8 model.tflite ...")
        convert_saved_model_to_tflite_int8(sm_fixed, model_fixed_path, rep_fixed)

        print("\n[4/6] Converting dynamic-size INT8 model_none.tflite ...")
        convert_saved_model_to_tflite_int8(sm_dyn, model_none_int8_path, rep_dynamic)

        print("\n[5/6] Converting dynamic-size FP32 model_none_float.tflite ...")
        convert_saved_model_to_tflite_float(sm_dyn, model_none_float_path)

        print("\n[6/6] Validating artifact triplet ...")
        artifact_info = validate_artifact_triplet(
            model_tflite=model_fixed_path,
            model_none_tflite=model_none_int8_path,
            model_none_float_tflite=model_none_float_path,
            fixed_h=args.fixed_h,
            fixed_w=args.fixed_w,
        )
        export_report["artifacts"]["tflite_info"] = artifact_info

        if args.smoke_verify_images > 0:
            print("\n[smoke] Verifying dynamic models on real DIV2K valid images ...")
            smoke = smoke_verify_dynamic_models(
                model=model,
                model_none_tflite=model_none_int8_path,
                model_none_float_tflite=model_none_float_path,
                data_root=args.data_root,
                num_images=args.smoke_verify_images,
            )
            export_report["smoke_verify_dynamic"] = smoke
            print(
                f"[smoke-summary] PT-HR={smoke['avg_psnr_pytorch_vs_hr']:.4f} | INT8-HR={smoke['avg_psnr_model_none_int8_vs_hr']:.4f} | FP32-HR={smoke['avg_psnr_model_none_float_vs_hr']:.4f}",
                flush=True,
            )

        export_report["artifacts"]["files"] = {
            "model.tflite": {"path": model_fixed_path, "bytes": int(os.path.getsize(model_fixed_path))},
            "model_none.tflite": {"path": model_none_int8_path, "bytes": int(os.path.getsize(model_none_int8_path))},
            "model_none_float.tflite": {"path": model_none_float_path, "bytes": int(os.path.getsize(model_none_float_path))},
        }
        save_json(os.path.join(args.out_dir, "export_report.json"), export_report)

    print("\nDone. Official artifacts:")
    print(f"  {model_fixed_path}")
    print(f"  {model_none_int8_path}")
    print(f"  {model_none_float_path}")
    print(f"\nSaved report: {os.path.join(args.out_dir, 'export_report.json')}")


if __name__ == "__main__":
    main()
