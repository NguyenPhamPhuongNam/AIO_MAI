import os
import json
import argparse
from typing import Dict, Any

import numpy as np
import tensorflow as tf


def inspect_tflite_model(tflite_path: str) -> Dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    return {
        "path": tflite_path,
        "input_shape": tuple(int(x) for x in inp["shape"]),
        "input_shape_signature": tuple(int(x) for x in inp.get("shape_signature", inp["shape"])),
        "input_dtype": np.dtype(inp["dtype"]).name,
        "input_quantization": (float(inp["quantization"][0]), int(inp["quantization"][1])),
        "output_shape": tuple(int(x) for x in out["shape"]),
        "output_shape_signature": tuple(int(x) for x in out.get("shape_signature", out["shape"])),
        "output_dtype": np.dtype(out["dtype"]).name,
        "output_quantization": (float(out["quantization"][0]), int(out["quantization"][1])),
    }


def quantize_to_int8(x_float: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if scale <= 0:
        raise ValueError(f"Invalid quantization scale: {scale}")
    x_q = np.round(x_float / scale + zero_point)
    x_q = np.clip(x_q, -128, 127).astype(np.int8)
    return x_q


def dequantize_int8(x_q: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (x_q.astype(np.float32) - float(zero_point)) * float(scale)


def run_fixed_smoke(model_path: str, fixed_h: int, fixed_w: int) -> Dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    x = np.random.uniform(0.0, 255.0, size=(1, fixed_h, fixed_w, 3)).astype(np.float32)

    if np.dtype(inp["dtype"]).name == "int8":
        in_scale, in_zp = inp["quantization"]
        x = quantize_to_int8(x, float(in_scale), int(in_zp))

    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])

    if np.dtype(out["dtype"]).name == "int8":
        out_scale, out_zp = out["quantization"]
        y = dequantize_int8(y, float(out_scale), int(out_zp))

    return {
        "output_shape_after_invoke": tuple(int(v) for v in y.shape),
        "output_min": float(np.min(y)),
        "output_max": float(np.max(y)),
    }


def run_dynamic_smoke(model_path: str, input_h: int, input_w: int) -> Dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    target_shape = [1, input_h, input_w, 3]
    interpreter.resize_tensor_input(inp["index"], target_shape, strict=False)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    x = np.random.uniform(0.0, 255.0, size=target_shape).astype(np.float32)
    in_dtype = np.dtype(inp["dtype"]).name
    out_dtype = np.dtype(out["dtype"]).name

    if in_dtype == "int8":
        in_scale, in_zp = inp["quantization"]
        x = quantize_to_int8(x, float(in_scale), int(in_zp))
    elif in_dtype != "float32":
        raise RuntimeError(f"Unsupported input dtype: {in_dtype}")

    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])

    if out_dtype == "int8":
        out_scale, out_zp = out["quantization"]
        y = dequantize_int8(y, float(out_scale), int(out_zp))
    elif out_dtype != "float32":
        raise RuntimeError(f"Unsupported output dtype: {out_dtype}")

    return {
        "input_shape_after_resize": tuple(int(v) for v in inp["shape"]),
        "output_shape_after_invoke": tuple(int(v) for v in y.shape),
        "expected_output_shape": (1, input_h * 3, input_w * 3, 3),
        "output_min": float(np.min(y)),
        "output_max": float(np.max(y)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission_dir", type=str, required=True)
    ap.add_argument("--fixed_h", type=int, default=720)
    ap.add_argument("--fixed_w", type=int, default=1280)
    ap.add_argument("--dynamic_test_h", type=int, default=64)
    ap.add_argument("--dynamic_test_w", type=int, default=96)
    ap.add_argument("--out_json", type=str, default="submission_artifact_check.json")
    args = ap.parse_args()

    # accept either root dir containing TFLite/ or the TFLite dir itself
    if os.path.isdir(os.path.join(args.submission_dir, "TFLite")):
        tflite_dir = os.path.join(args.submission_dir, "TFLite")
        root_dir = args.submission_dir
    else:
        tflite_dir = args.submission_dir
        root_dir = os.path.dirname(args.submission_dir)

    model_fixed = os.path.join(tflite_dir, "model.tflite")
    model_none = os.path.join(tflite_dir, "model_none.tflite")
    model_none_float = os.path.join(tflite_dir, "model_none_float.tflite")

    for p in [model_fixed, model_none, model_none_float]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing required artifact: {p}")

    fixed_info = inspect_tflite_model(model_fixed)
    dyn_info = inspect_tflite_model(model_none)
    dyn_float_info = inspect_tflite_model(model_none_float)

    print("=== Fixed model ===")
    print(json.dumps(fixed_info, ensure_ascii=False, indent=2))
    print("\n=== Dynamic INT8 model ===")
    print(json.dumps(dyn_info, ensure_ascii=False, indent=2))
    print("\n=== Dynamic FP32 model ===")
    print(json.dumps(dyn_float_info, ensure_ascii=False, indent=2))

    # fixed checks
    assert tuple(fixed_info["input_shape"]) == (1, args.fixed_h, args.fixed_w, 3), \
        f"Unexpected model.tflite input shape: {fixed_info['input_shape']}"
    assert tuple(fixed_info["output_shape"]) == (1, args.fixed_h * 3, args.fixed_w * 3, 3), \
        f"Unexpected model.tflite output shape: {fixed_info['output_shape']}"
    assert fixed_info["input_dtype"] == "int8", f"model.tflite input must be int8, got {fixed_info['input_dtype']}"
    assert fixed_info["output_dtype"] == "int8", f"model.tflite output must be int8, got {fixed_info['output_dtype']}"
    assert fixed_info["input_quantization"][0] > 0.0, f"Bad fixed input quantization: {fixed_info['input_quantization']}"
    assert fixed_info["output_quantization"][0] > 0.0, f"Bad fixed output quantization: {fixed_info['output_quantization']}"

    # dynamic int8 checks
    sig_in_dyn = tuple(dyn_info["input_shape_signature"])
    sig_out_dyn = tuple(dyn_info["output_shape_signature"])
    assert dyn_info["input_dtype"] == "int8", f"model_none.tflite input must be int8, got {dyn_info['input_dtype']}"
    assert dyn_info["output_dtype"] == "int8", f"model_none.tflite output must be int8, got {dyn_info['output_dtype']}"
    assert sig_in_dyn[0] == 1 and sig_in_dyn[3] == 3 and sig_in_dyn[1] == -1 and sig_in_dyn[2] == -1, \
        f"model_none.tflite input is not dynamic NHWC RGB: {sig_in_dyn}"
    assert sig_out_dyn[0] == 1 and sig_out_dyn[3] == 3 and sig_out_dyn[1] == -1 and sig_out_dyn[2] == -1, \
        f"model_none.tflite output is not dynamic NHWC RGB: {sig_out_dyn}"

    # dynamic float checks
    sig_in_dyn_f = tuple(dyn_float_info["input_shape_signature"])
    sig_out_dyn_f = tuple(dyn_float_info["output_shape_signature"])
    assert dyn_float_info["input_dtype"] == "float32", \
        f"model_none_float.tflite input must be float32, got {dyn_float_info['input_dtype']}"
    assert dyn_float_info["output_dtype"] == "float32", \
        f"model_none_float.tflite output must be float32, got {dyn_float_info['output_dtype']}"
    assert sig_in_dyn_f[0] == 1 and sig_in_dyn_f[3] == 3 and sig_in_dyn_f[1] == -1 and sig_in_dyn_f[2] == -1, \
        f"model_none_float.tflite input is not dynamic NHWC RGB: {sig_in_dyn_f}"
    assert sig_out_dyn_f[0] == 1 and sig_out_dyn_f[3] == 3 and sig_out_dyn_f[1] == -1 and sig_out_dyn_f[2] == -1, \
        f"model_none_float.tflite output is not dynamic NHWC RGB: {sig_out_dyn_f}"

    fixed_smoke = run_fixed_smoke(model_fixed, args.fixed_h, args.fixed_w)
    dyn_smoke = run_dynamic_smoke(model_none, args.dynamic_test_h, args.dynamic_test_w)
    dyn_float_smoke = run_dynamic_smoke(model_none_float, args.dynamic_test_h, args.dynamic_test_w)

    assert tuple(dyn_smoke["output_shape_after_invoke"]) == tuple(dyn_smoke["expected_output_shape"]), \
        f"Dynamic INT8 smoke output mismatch: {dyn_smoke}"
    assert tuple(dyn_float_smoke["output_shape_after_invoke"]) == tuple(dyn_float_smoke["expected_output_shape"]), \
        f"Dynamic FP32 smoke output mismatch: {dyn_float_smoke}"

    export_report_path = os.path.join(root_dir, "export_report.json")
    export_report = None
    if os.path.isfile(export_report_path):
        with open(export_report_path, "r", encoding="utf-8") as f:
            export_report = json.load(f)

    report = {
        "fixed_info": fixed_info,
        "dynamic_int8_info": dyn_info,
        "dynamic_float_info": dyn_float_info,
        "fixed_smoke": fixed_smoke,
        "dynamic_int8_smoke": dyn_smoke,
        "dynamic_float_smoke": dyn_float_smoke,
        "export_report_found": bool(export_report is not None),
        "export_report_preview": export_report.get("smoke_verify_dynamic", {}) if export_report else None,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nArtifact checks passed.")
    print(f"Saved check report to: {args.out_json}")


if __name__ == "__main__":
    main()