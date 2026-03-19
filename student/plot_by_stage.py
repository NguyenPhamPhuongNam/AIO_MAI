import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import torch


def read_metrics(run_dir: str) -> pd.DataFrame:
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing metrics.csv: {path}")

    df = pd.read_csv(path)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

    for c in df.columns:
        if c in ("time", "stage", "best_key", "note"):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    sort_cols = []
    if "stage" in df.columns:
        sort_cols.append("stage")
    if "time" in df.columns:
        sort_cols.append("time")
    elif "epoch" in df.columns:
        sort_cols.append("epoch")

    if sort_cols:
        df = df.sort_values(sort_cols, kind="mergesort")

    return df.reset_index(drop=True)


def get_stage_order(df: pd.DataFrame):
    preferred = ["s1_fp32", "s2_fp32", "s3_qat"]
    existing = [s for s in preferred if s in df["stage"].dropna().unique().tolist()]
    others = sorted([s for s in df["stage"].dropna().unique().tolist() if s not in preferred])
    return existing + others


def choose_x(g: pd.DataFrame, prefer: str = "epoch"):
    prefer = prefer.lower()

    if prefer == "seq":
        return range(len(g)), "seq"

    if prefer == "time":
        if "time" in g.columns and g["time"].notna().any():
            return g["time"], "time"
        prefer = "epoch"

    if prefer == "epoch":
        if "epoch" in g.columns and g["epoch"].notna().any():
            e = g["epoch"].ffill()
            if (e.diff().fillna(0) >= 0).all():
                return g["epoch"], "epoch"
        return range(len(g)), "seq"

    return range(len(g)), "seq"


def finish_plot(out_path, title, xlabel, ylabel="value", has_line=True):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linewidth=0.3)
    if has_line:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_cols(out_path, x, xlabel, title, g: pd.DataFrame, cols, ylabel="value",
              deploy_value=None, deploy_label=None):
    plt.figure()
    has_line = False

    for c in cols:
        if c in g.columns and g[c].notna().any():
            plt.plot(x, g[c], marker="o", linewidth=1, markersize=3, label=c)
            has_line = True

    if deploy_value is not None and pd.notna(deploy_value):
        label = deploy_label if deploy_label else f"deploy={float(deploy_value):.4f}"
        plt.axhline(float(deploy_value), linestyle="--", linewidth=1.2, label=label)
        has_line = True

    finish_plot(out_path, title, xlabel, ylabel=ylabel, has_line=has_line)


def _parse_keyvals(text: str):
    out = {}
    for k, v in re.findall(r'([A-Za-z0-9_]+)=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)', text):
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def read_deploy_info(run_dir: str):
    """
    Parse deploy-related info from train.log.

    Expected patterns:
      [deploy-val/s2] psnr_sr_rgb_sh0=30.1234
      [deploy-val/s3] psnr_sr_rgb_sh0=30.1628
    """
    path = os.path.join(run_dir, "train.log")
    if not os.path.exists(path):
        return {}

    stage_alias = {
        "s1": "s1_fp32",
        "s2": "s2_fp32",
        "s3": "s3_qat",
    }

    deploy_rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r'^(.*?)\s+\|\s+INFO\s+\|\s+\[deploy-val/([^\]]+)\]\s+(.*)$', line.strip())
            if not m:
                continue

            t_str, short_stage, tail = m.groups()
            stage = stage_alias.get(short_stage.strip(), short_stage.strip())
            kv = _parse_keyvals(tail)

            row = {"stage": stage, "time": t_str}
            row.update(kv)
            deploy_rows.append(row)

    if not deploy_rows:
        return {}

    deploy_df = pd.DataFrame(deploy_rows)
    deploy_info = {}

    for stage, g in deploy_df.groupby("stage", sort=False):
        g = g.reset_index(drop=True)
        info = {
            "deploy_entries": len(g),
            "deploy_last_time": g.iloc[-1].get("time", None),
        }

        metric_cols = [c for c in g.columns if c not in ("stage", "time")]
        for c in metric_cols:
            s = pd.to_numeric(g[c], errors="coerce")
            if not s.notna().any():
                continue
            info[f"deploy_last_{c}"] = float(s.dropna().iloc[-1])
            info[f"deploy_best_{c}"] = float(s.max())

        deploy_info[stage] = info

    return deploy_info


def apply_manual_deploy_overrides(deploy_info: dict, deploy_s1=None, deploy_s2=None, deploy_s3=None):
    def _apply(stage_name: str, value):
        if value is None:
            return
        deploy_info.setdefault(stage_name, {})
        deploy_info[stage_name]["deploy_best_psnr_sr_rgb_sh0"] = float(value)
        deploy_info[stage_name]["deploy_last_psnr_sr_rgb_sh0"] = float(value)
        deploy_info[stage_name]["deploy_entries"] = 1
        deploy_info[stage_name]["deploy_last_time"] = "manual"

    _apply("s1_fp32", deploy_s1)
    _apply("s2_fp32", deploy_s2)
    _apply("s3_qat", deploy_s3)
    return deploy_info


def _bytes_to_mb(x):
    if x is None:
        return None
    return float(x) / (1024.0 * 1024.0)


def _safe_get_file_size(path):
    try:
        if path and os.path.exists(path):
            return int(os.path.getsize(path))
    except Exception:
        pass
    return None


def _state_dict_num_bytes(sd: dict):
    total = 0
    total_params = 0
    if not isinstance(sd, dict):
        return 0, 0
    for _, v in sd.items():
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
            total_params += v.numel()
    return int(total), int(total_params)


def _read_pt_model_info(path):
    """
    Read .pt checkpoint/export and estimate:
    - file size on disk
    - raw tensor bytes in state_dict
    - total parameter count
    """
    file_size_bytes = _safe_get_file_size(path)
    info = {
        "path": path,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": _bytes_to_mb(file_size_bytes),
        "raw_weight_bytes": None,
        "raw_weight_mb": None,
        "param_count": None,
        "format": None,
    }

    if not path or not os.path.exists(path):
        return info

    try:
        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            raw_bytes, n_params = _state_dict_num_bytes(obj["model"])
            info["raw_weight_bytes"] = raw_bytes
            info["raw_weight_mb"] = _bytes_to_mb(raw_bytes)
            info["param_count"] = n_params
            info["format"] = "export_pt"
            return info

        if isinstance(obj, dict):
            if "model_ema" in obj and isinstance(obj["model_ema"], dict):
                raw_bytes, n_params = _state_dict_num_bytes(obj["model_ema"])
                info["raw_weight_bytes"] = raw_bytes
                info["raw_weight_mb"] = _bytes_to_mb(raw_bytes)
                info["param_count"] = n_params
                info["format"] = "train_ckpt_model_ema"
                return info

            if "model" in obj and isinstance(obj["model"], dict):
                raw_bytes, n_params = _state_dict_num_bytes(obj["model"])
                info["raw_weight_bytes"] = raw_bytes
                info["raw_weight_mb"] = _bytes_to_mb(raw_bytes)
                info["param_count"] = n_params
                info["format"] = "train_ckpt_model"
                return info

        if isinstance(obj, dict):
            raw_bytes, n_params = _state_dict_num_bytes(obj)
            if n_params > 0:
                info["raw_weight_bytes"] = raw_bytes
                info["raw_weight_mb"] = _bytes_to_mb(raw_bytes)
                info["param_count"] = n_params
                info["format"] = "state_dict"
                return info

    except Exception as e:
        info["error"] = str(e)

    return info


def read_model_sizes(run_dir: str):
    """
    Return per-stage model size info by scanning common filenames inside run_dir.
    """
    candidates = {
        "s1_fp32": [
            ("best_ckpt_pt", os.path.join(run_dir, "ckpt_best_s1_fp32.pt")),
            ("last_ckpt_pt", os.path.join(run_dir, "ckpt_last_s1_fp32.pt")),
        ],
        "s2_fp32": [
            ("best_ckpt_pt", os.path.join(run_dir, "ckpt_best_s2_fp32.pt")),
            ("last_ckpt_pt", os.path.join(run_dir, "ckpt_last_s2_fp32.pt")),
            ("best_deploy_pt", os.path.join(run_dir, "ckpt_best_s2_deploy.pt")),
        ],
        "s3_qat": [
            ("best_ckpt_pt", os.path.join(run_dir, "ckpt_best_s3_qat.pt")),
            ("last_ckpt_pt", os.path.join(run_dir, "ckpt_last_s3_qat.pt")),
            ("best_deploy_pt", os.path.join(run_dir, "ckpt_best_s3_qat_deploy.pt")),
        ],
    }

    extra_tflites = [
        os.path.join(run_dir, "model.tflite"),
        os.path.join(run_dir, "best.tflite"),
        os.path.join(run_dir, "ckpt_best_s2_deploy.tflite"),
        os.path.join(run_dir, "ckpt_best_s3_qat_deploy.tflite"),
    ]

    model_sizes = {}

    for stage, items in candidates.items():
        st_info = {}
        for key, path in items:
            if path.endswith(".pt") and os.path.exists(path):
                info = _read_pt_model_info(path)
                for k, v in info.items():
                    st_info[f"{key}_{k}"] = v
        model_sizes[stage] = st_info

    tflite_rows = []
    for path in extra_tflites:
        if os.path.exists(path):
            size_b = _safe_get_file_size(path)
            tflite_rows.append({
                "path": path,
                "file_size_bytes": size_b,
                "file_size_mb": _bytes_to_mb(size_b),
            })

    if tflite_rows:
        model_sizes["_tflite_files"] = tflite_rows

    return model_sizes


def build_summary(df: pd.DataFrame, deploy_info: dict, model_sizes: dict) -> pd.DataFrame:
    rows = []

    stage_order = get_stage_order(df)
    for st in stage_order:
        g = df[df["stage"] == st].copy()
        row = {"stage": st}

        if not g.empty and "psnr_sr_rgb_sh0" in g.columns and g["psnr_sr_rgb_sh0"].notna().any():
            idx = g["psnr_sr_rgb_sh0"].idxmax()
            best = g.loc[idx]
            row.update({
                "best_epoch": best.get("epoch", None),
                "best_psnr_sr_rgb_sh0": best.get("psnr_sr_rgb_sh0", None),
                "best_psnr_sr_rgb_sh3": best.get("psnr_sr_rgb_sh3", None),
                "best_psnr_sr_y_sh0": best.get("psnr_sr_y_sh0", None),
                "best_psnr_sr_y_sh3": best.get("psnr_sr_y_sh3", None),
                "best_ssim_sr_rgb_sh0": best.get("ssim_sr_rgb_sh0", None),
                "best_ssim_sr_rgb_sh3": best.get("ssim_sr_rgb_sh3", None),
                "train_loss_at_best": best.get("train_loss", None),
                "lr_at_best": best.get("lr", None),
            })

        row.update(deploy_info.get(st, {}))
        row.update(model_sizes.get(st, {}))
        rows.append(row)

    existing_stages = {r["stage"] for r in rows}
    all_extra_stages = (set(deploy_info.keys()) | set(model_sizes.keys())) - {"_tflite_files"}

    for st in all_extra_stages:
        if st in existing_stages:
            continue
        row = {"stage": st}
        row.update(deploy_info.get(st, {}))
        row.update(model_sizes.get(st, {}))
        rows.append(row)

    return pd.DataFrame(rows)


def plot_stage_compare(df: pd.DataFrame, out_dir: str, deploy_info: dict):
    metrics = [
        ("psnr_sr_rgb_sh0", "compare_psnr_sr_rgb_sh0.png", "Compare stages | PSNR SR RGB sh0", "PSNR (dB)"),
        ("psnr_sr_rgb_sh3", "compare_psnr_sr_rgb_sh3.png", "Compare stages | PSNR SR RGB sh3", "PSNR (dB)"),
        ("ssim_sr_rgb_sh0", "compare_ssim_sr_rgb_sh0.png", "Compare stages | SSIM SR RGB sh0", "SSIM"),
        ("train_loss", "compare_train_loss.png", "Compare stages | Train loss", "loss"),
    ]

    stages = get_stage_order(df)

    for metric, filename, title, ylabel in metrics:
        plt.figure()
        has_line = False

        for st in stages:
            g = df[df["stage"] == st].copy()
            if metric in g.columns and g[metric].notna().any():
                x, _ = choose_x(g, prefer="epoch")
                plt.plot(x, g[metric], marker="o", linewidth=1, markersize=3, label=st)
                has_line = True

            if metric == "psnr_sr_rgb_sh0":
                dep = deploy_info.get(st, {})
                val = dep.get("deploy_best_psnr_sr_rgb_sh0", None)
                if val is not None:
                    plt.axhline(float(val), linestyle="--", linewidth=1.0, label=f"{st} deploy")
                    has_line = True

        finish_plot(os.path.join(out_dir, filename), title, "epoch/seq", ylabel=ylabel, has_line=has_line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--x", type=str, default="epoch", choices=["epoch", "time", "seq"])
    ap.add_argument("--deploy_s1", type=float, default=None)
    ap.add_argument("--deploy_s2", type=float, default=None)
    ap.add_argument("--deploy_s3", type=float, default=None)
    args = ap.parse_args()

    df = read_metrics(args.run_dir)

    if "stage" not in df.columns:
        raise RuntimeError("metrics.csv must contain column 'stage'")

    stages = get_stage_order(df)

    deploy_info = read_deploy_info(args.run_dir)
    deploy_info = apply_manual_deploy_overrides(
        deploy_info,
        deploy_s1=args.deploy_s1,
        deploy_s2=args.deploy_s2,
        deploy_s3=args.deploy_s3,
    )

    model_sizes = read_model_sizes(args.run_dir)

    base_out = os.path.join(args.run_dir, "plots_key")
    os.makedirs(base_out, exist_ok=True)

    summary = build_summary(df, deploy_info, model_sizes)
    summary_path = os.path.join(base_out, "summary_best.csv")
    summary.to_csv(summary_path, index=False)

    deploy_summary_path = os.path.join(base_out, "deploy_summary.csv")
    if deploy_info:
        pd.DataFrame(
            [{"stage": st, **info} for st, info in deploy_info.items()]
        ).to_csv(deploy_summary_path, index=False)
    else:
        pd.DataFrame(columns=["stage"]).to_csv(deploy_summary_path, index=False)

    model_sizes_path = os.path.join(base_out, "model_sizes.csv")
    ms_rows = []
    for st, info in model_sizes.items():
        if st == "_tflite_files":
            continue
        row = {"stage": st}
        row.update(info)
        ms_rows.append(row)

    if ms_rows:
        pd.DataFrame(ms_rows).to_csv(model_sizes_path, index=False)
    else:
        pd.DataFrame(columns=["stage"]).to_csv(model_sizes_path, index=False)

    for st in stages:
        g = df[df["stage"] == st].copy()
        if g.empty:
            continue

        out_dir = os.path.join(base_out, st)
        os.makedirs(out_dir, exist_ok=True)
        x, xlabel = choose_x(g, prefer=args.x)

        dep = deploy_info.get(st, {})
        deploy_psnr = dep.get("deploy_best_psnr_sr_rgb_sh0", None)
        ms = model_sizes.get(st, {})

        plot_cols(
            os.path.join(out_dir, "psnr_main.png"),
            x, xlabel, f"PSNR main | {st}", g,
            ["psnr_sr_rgb_sh0", "psnr_sr_rgb_sh3"],
            ylabel="PSNR (dB)",
            deploy_value=deploy_psnr,
            deploy_label=(f"deploy_psnr_sr_rgb_sh0={deploy_psnr:.4f}" if deploy_psnr is not None else None),
        )

        plt.figure()
        has_line = False
        for sr_col, bi_col, label in [
            ("psnr_sr_rgb_sh0", "psnr_bi_rgb_sh0", "delta_rgb_sh0"),
            ("psnr_sr_rgb_sh3", "psnr_bi_rgb_sh3", "delta_rgb_sh3"),
        ]:
            if sr_col in g.columns and bi_col in g.columns:
                d = pd.to_numeric(g[sr_col], errors="coerce") - pd.to_numeric(g[bi_col], errors="coerce")
                if d.notna().any():
                    plt.plot(x, d, marker="o", linewidth=1, markersize=3, label=label)
                    has_line = True
        finish_plot(
            os.path.join(out_dir, "delta_vs_bicubic.png"),
            f"Delta PSNR vs Bicubic | {st}",
            xlabel,
            ylabel="delta (dB)",
            has_line=has_line
        )

        if "ssim_sr_rgb_sh0" in g.columns:
            plot_cols(
                os.path.join(out_dir, "ssim_main.png"),
                x, xlabel, f"SSIM main | {st}", g,
                ["ssim_sr_rgb_sh0", "ssim_sr_rgb_sh3"],
                ylabel="SSIM"
            )

        plot_cols(
            os.path.join(out_dir, "loss.png"),
            x, xlabel, f"Train loss | {st}", g,
            ["train_loss"],
            ylabel="loss"
        )

        txt_path = os.path.join(out_dir, "stage_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"stage={st}\n")

            if "psnr_sr_rgb_sh0" in g.columns and g["psnr_sr_rgb_sh0"].notna().any():
                idx = g["psnr_sr_rgb_sh0"].idxmax()
                best = g.loc[idx]
                f.write(f"best_epoch={best.get('epoch', '')}\n")
                f.write(f"best_psnr_sr_rgb_sh0={best.get('psnr_sr_rgb_sh0', '')}\n")
                f.write(f"best_psnr_sr_rgb_sh3={best.get('psnr_sr_rgb_sh3', '')}\n")
                f.write(f"best_psnr_sr_y_sh0={best.get('psnr_sr_y_sh0', '')}\n")
                f.write(f"best_psnr_sr_y_sh3={best.get('psnr_sr_y_sh3', '')}\n")
                f.write(f"best_ssim_sr_rgb_sh0={best.get('ssim_sr_rgb_sh0', '')}\n")
                f.write(f"best_ssim_sr_rgb_sh3={best.get('ssim_sr_rgb_sh3', '')}\n")
                f.write(f"train_loss_at_best={best.get('train_loss', '')}\n")
                f.write(f"lr_at_best={best.get('lr', '')}\n")

            if dep:
                f.write("\n[deploy]\n")
                for k, v in dep.items():
                    f.write(f"{k}={v}\n")

            if ms:
                f.write("\n[model_size]\n")
                for k, v in ms.items():
                    f.write(f"{k}={v}\n")

        extra_parts = []
        if dep:
            extra_parts.append(f"deploy={dep}")

        if ms:
            size_str = []
            if ms.get("best_deploy_pt_file_size_mb") is not None:
                size_str.append(f"deploy_pt_file={ms['best_deploy_pt_file_size_mb']:.3f}MB")
            if ms.get("best_deploy_pt_raw_weight_mb") is not None:
                size_str.append(f"deploy_pt_raw={ms['best_deploy_pt_raw_weight_mb']:.3f}MB")
            if ms.get("best_deploy_pt_param_count") is not None:
                size_str.append(f"deploy_params={int(ms['best_deploy_pt_param_count'])}")
            if size_str:
                extra_parts.append(", ".join(size_str))

        if extra_parts:
            print(f"[OK] {st} -> {out_dir} | " + " | ".join(extra_parts))
        else:
            print(f"[OK] {st} -> {out_dir}")

    compare_dir = os.path.join(base_out, "_compare")
    os.makedirs(compare_dir, exist_ok=True)
    plot_stage_compare(df, compare_dir, deploy_info)

    print(f"[OK] Summary -> {summary_path}")
    print(f"[OK] Deploy summary -> {deploy_summary_path}")
    print(f"[OK] Model sizes -> {model_sizes_path}")

    if "_tflite_files" in model_sizes:
        print("[OK] TFLite files found:")
        for x in model_sizes["_tflite_files"]:
            mb = x.get("file_size_mb", None)
            p = x.get("path", "")
            if mb is not None:
                print(f"    - {p}: {mb:.3f} MB")
            else:
                print(f"    - {p}")

    print(f"Done -> {base_out}")


if __name__ == "__main__":
    main()