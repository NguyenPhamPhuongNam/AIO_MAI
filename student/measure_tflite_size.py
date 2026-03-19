import os
import json
import argparse
from typing import Dict, List


def file_size_info(path: str) -> Dict[str, float]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    nbytes = os.path.getsize(path)
    return {
        "path": os.path.abspath(path),
        "bytes": int(nbytes),
        "kb": float(nbytes / 1024.0),
        "mb": float(nbytes / (1024.0 * 1024.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        type=str,
        nargs="+",
        required=True,
        help="One or more .tflite files",
    )
    ap.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional path to save size report as JSON",
    )
    args = ap.parse_args()

    report: List[Dict[str, float]] = []

    print("=== TFLITE MODEL SIZES ===")
    for path in args.model:
        info = file_size_info(path)
        report.append(info)
        print(
            f"{info['path']}\n"
            f"  bytes: {info['bytes']}\n"
            f"  kb   : {info['kb']:.2f}\n"
            f"  mb   : {info['mb']:.4f}\n"
        )

    if args.out_json is not None:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved JSON report to: {args.out_json}")


if __name__ == "__main__":
    main()