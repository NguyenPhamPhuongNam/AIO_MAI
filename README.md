<h1 align="center">🚀 Efficient INT8 Single-Image Super-Resolution</h1>

<h3 align="center">
Deployment-Aware Quantization and Teacher-Guided Training
</h3>

<div align="center">
  <p>
    <strong>Official repository for our MAI 2026 Quantized 4K Image Super-Resolution Challenge submission</strong>
  </p>

  <p>
    <strong>Pham Phuong Nam Nguyen</strong>&nbsp;&nbsp;
    <strong>Nam Tien Le</strong>&nbsp;&nbsp;
    <strong>Thi Kim Trang Vo</strong>&nbsp;&nbsp;
    <strong>Nhu Tinh Anh Nguyen</strong>
  </p>

  <p>
    <em>Accepted at the Mobile AI (MAI) 2026 Workshop at CVPR 2026</em>
  </p>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2604.20291" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv%3A2604.20291-b31b1b?style=plastic&logo=arxiv&logoColor=white" alt="Paper">
  </a>
  <img src="https://img.shields.io/badge/CVPRW-2026-4c6ef5?style=plastic" alt="CVPRW 2026">
  <img src="https://img.shields.io/badge/Task-4K%20Image%20Super--Resolution-00AEEF?style=plastic" alt="4K Image Super-Resolution">
  <img src="https://img.shields.io/badge/Deployment-INT8%20TFLite-2ea44f?style=plastic" alt="INT8 TFLite">
</p>

<div align="center">
  <img src="assets/visual_comparison_12panels.png" alt="Qualitative comparison of LR, HR, and AIO_MAI results" width="100%">
  <p>
    <em>
      Qualitative comparison of ×3 quantized image super-resolution on representative DIV2K validation images.
      From left to right: Full LR image, zoomed LR patch, HR ground truth, and our AIO_MAI result.
    </em>
  </p>
</div>

---

## ✨ Overview

This repository contains our full training, export, and evaluation pipeline for a compact **×3 single-image super-resolution** model designed for **mobile INT8 deployment**.

Our method is built around a lightweight **LR-space extract–refine–upsample** architecture with a **MobileOne-style re-parameterizable** student backbone, **teacher-guided training**, and **deploy-before-QAT** optimization for stable INT8 inference.

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| **LR-space super-resolution** | Lightweight design that performs most computation in the low-resolution space. |
| **MobileOne-style student backbone** | Re-parameterizable blocks for an efficient train-to-deploy workflow. |
| **Three-stage training pipeline** | Progressive training from FP32 reconstruction to INT8 robustness. |
| **Teacher-guided knowledge distillation** | Uses teacher supervision to improve reconstruction fidelity. |
| **Deploy-before-QAT** | Reduces train–deploy mismatch by applying QAT on the fused deploy graph. |
| **INT8 TFLite export** | Produces mobile-friendly deployable artifacts. |

---

## 🧠 Method Summary

Our student model follows an **extract–refine–upsample** design:

- A shallow input stem projects the LR image into feature space
- A stack of lightweight **MobileOne-style re-parameterizable blocks** refines features in LR space
- A global feature skip preserves coarse structures
- A final **PixelShuffle** head reconstructs the ×3 HR output

---

## 📚 Training Pipeline

The model is trained in **three stages**:

### Stage 1: Reconstruction Warm-up

Basic reconstruction training using **L1 loss**.

### Stage 2: Fidelity-Oriented Refinement

Fidelity-oriented refinement using:

- **Charbonnier loss**
- **DCT-domain supervision**
- **Confidence-weighted output-level knowledge distillation**
- **MambaIRv2Light** as teacher

### Stage 3: INT8 Quantization-Aware Training

**Quantization-aware training (QAT)** on the **fused deploy graph** to improve INT8 robustness and reduce quantization mismatch.

---

## 🏆 Main Results

| Setting | PSNR (dB) | SSIM | Notes |
|---|---:|---:|---|
| **Ours (Stage 2) FP32** | **30.28** | **0.863** | Best floating-point student |
| **Ours (Deploy) INT8** | **30.13** | **0.858** | Fixed-shape deployable TFLite artifact |
| **AIO MAI Submission** | **29.79** | **0.8634** | Final MAI 2026 test-set result, score = 1.8, top 5 |

### 🔬 Additional Ablation Highlights

- **MobileOne** provides the best FP32/INT8 trade-off among tested re-parameterizable blocks
- Stage 3 teacher-guided supervision improves INT8 reconstruction from **29.9114 / 0.853** to **30.0003 / 0.856**

---

## 📁 Repository Structure

```bash
AIO_MAI/
├── .env_stamps/
├── MambaIR/
├── student/
│   ├── train.py
│   ├── main.py
│   ├── eval_div2k_full_compare.py
│   ├── model.py
│   ├── README.md
│   └── ...
├── assets/
│   └── visual_comparison_12panels.png
├── README.md
├── mambairv2_lightSR_x3.pth
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/NguyenPhamPhuongNam/AIO_MAI.git
cd AIO_MAI
```

### 2. Install Ninja

```bash
pip install ninja
```

### 3. Install project dependencies

```bash
pip install -r requirements.txt
```

---

## 🗂️ Dataset Preparation

Download the **DIV2K** dataset and place it in a folder named `DIV2K` inside a `data` directory located **outside** the `AIO_MAI` repository.

Expected directory layout:

```bash
workspace/
├── AIO_MAI/
└── data/
    └── DIV2K/
```

> From inside `AIO_MAI/student`, the dataset path will be:
>
> ```bash
> ../../data/DIV2K
> ```

---

## 🏋️ Training

Move to the student folder first:

```bash
cd student
```

### Train Stage 1 + Stage 2

```bash
python train.py \
  --data_root ../../data/DIV2K \
  --out_dir ./runs_mobileone_32_8 \
  --preset balanced \
  --device cuda --seed 1234 \
  --rep_type mobileone --channels 32 --n_rep 8 --rep_act_mode relu \
  --mo_branches 5 --mo_use_1x1 --mo_use_identity --rep_use_bn \
  --skip_mode add --use_global_add \
  --no-image_residual \
  --no-use_block_res_scale \
  --out_clamp_fp32 none --out_clamp_export minclip --out_clamp_qat minclip \
  --epochs1 600 --epochs2 200 --epochs3 0 \
  --lr1 1e-3 --lr2 1.5e-5 --lr3 2e-6 \
  --patch1 128 --patch2 160 --patch3 144 \
  --s1_loss l1 \
  --s2_loss charbdct --dct_w_s2 0.02 \
  --s3_loss charbdct --dct_w_s3 0.015 \
  --scheduler1 cos_warmup --scheduler2 step_halve --scheduler3 step_halve \
  --grad_clip 1.0 \
  --ema --ema_decay 0.999 \
  --qat --deploy_before_qat --qat_mode fx --qat_backend qnnpack \
  --qat_disable_observer_ep 30 --qat_freeze_fakequant_ep 90 \
  --teacher_type mambair \
  --mambair_repo ../MambaIR \
  --mambair_teacher_ckpt ../mambairv2_lightSR_x3.pth \
  --kd_loss l1 \
  --kd_w_s2 0.03 \
  --kd_res_w_s2 0.0 \
  --kd_wave_w_s2 0.0 \
  --kd_w_s3 0.01 \
  --kd_res_w_s3 0.0 \
  --kd_wave_w_s3 0.0 \
  --kd_freq_w_s3 0.0 \
  --kd_w_s3_start 0.010 --kd_w_s3_end 0.005 \
  --kd_conf_gamma 10.0 --kd_conf_min 0.10 --kd_conf_max 0.75 \
  --kd_sched_p1 0.25 --kd_sched_p2 0.60 --kd_low_floor 0.01 \
  --fqkd_w 0.0 \
  --amp_fp32 --channels_last \
  --val_shaves 0 --best_key psnr_sr_rgb_sh0 \
  --report_ssim --ssim_win 11 --ssim_sigma 1.5 \
  --workers 2 --batch 16 --val_every 1 \
  --log_every 200 \
  --freeze_bn_epoch -1 \
  --bn_recalib_batches 64 \
  --no-channel_shuffle_s2s3
```

### Train Stage 3

```bash
python train.py \
  --data_root ../../data/DIV2K \
  --out_dir ./runs_mobileone_32_8 \
  --preset balanced \
  --device cuda \
  --seed 1234 \
  --rep_type mobileone \
  --channels 32 \
  --n_rep 8 \
  --rep_act_mode relu \
  --mo_branches 5 \
  --mo_use_1x1 \
  --mo_use_identity \
  --rep_use_bn \
  --skip_mode add \
  --use_global_add \
  --no-image_residual \
  --no-use_block_res_scale \
  --out_clamp_fp32 none \
  --out_clamp_export minclip \
  --out_clamp_qat minclip \
  --epochs1 0 \
  --epochs2 0 \
  --epochs3 150 \
  --lr3 2e-6 \
  --patch3 144 \
  --s3_loss charbdct \
  --dct_w_s3 0.015 \
  --scheduler3 step_halve \
  --grad_clip 1.0 \
  --ema \
  --ema_decay 0.999 \
  --qat \
  --deploy_before_qat \
  --qat_mode fx \
  --qat_backend qnnpack \
  --qat_disable_observer_ep 30 \
  --qat_freeze_fakequant_ep 90 \
  --init_ckpt ./runs_mobileone_32_8/balanced/ckpt_best_s2_fp32.pt \
  --teacher_type antsr \
  --teacher_ckpt ./runs_mobileone_32_8/balanced/ckpt_best_s2_deploy.pt \
  --kd_loss l1 \
  --kd_w_s3 0.02 \
  --kd_res_w_s3 0.01 \
  --kd_wave_w_s3 0.0 \
  --kd_freq_w_s3 0.0 \
  --kd_w_s3_start 0.03 \
  --kd_w_s3_end 0.01 \
  --kd_conf_gamma 10.0 \
  --kd_conf_min 0.10 \
  --kd_conf_max 0.75 \
  --kd_sched_p1 0.25 \
  --kd_sched_p2 0.70 \
  --kd_low_floor 0.01 \
  --fqkd_w 0.005 \
  --amp_fp32 \
  --channels_last \
  --val_shaves 0 \
  --best_key psnr_sr_rgb_sh0 \
  --report_ssim \
  --workers 2 \
  --batch 16 \
  --no-stage3_batch1 \
  --val_every 1 \
  --log_every 200 \
  --freeze_bn_epoch -1 \
  --bn_recalib_batches 64 \
  --no-channel_shuffle_s2s3
```

---

## 📦 Export

```bash
python main.py \
  --deploy_ckpt ./runs_mobileone_32_8/balanced/ckpt_best_s3_qat_deploy.pt \
  --data_root ../../data/DIV2K \
  --out_dir ./exports_moi/s3_auto_train \
  --fixed_h 720 \
  --fixed_w 1280 \
  --auto_search \
  --search_eval_images 20 \
  --smoke_verify_images 20 \
  --seed 42
```

---

## 📊 Evaluation

```bash
python eval_div2k_full_compare.py \
  --deploy_ckpt ./runs_mobileone_32_8/balanced/ckpt_best_s3_qat_deploy.pt \
  --model_none_int8 ./exports_moi/s3_auto_train/TFLite/model_none.tflite \
  --model_none_float ./exports_moi/s3_auto_train/TFLite/model_none_float.tflite \
  --model_fixed ./exports_moi/s3_auto_train/TFLite/model.tflite \
  --data_root ../../data/DIV2K \
  --device cpu \
  --shaves 0,3 \
  --fixed_overlap 16 \
  --out_json ./exports_moi/s3_auto_train/eval_div2k_full_report.json
```

---

## 📜 Citation

If you find our work useful, please consider citing:

```bibtex
@article{nguyen2026efficientint8sr,
  title={Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training},
  author={Nguyen, Pham Phuong Nam and Le, Nam Tien and Vo, Thi Kim Trang and Nguyen, Nhu Tinh Anh},
  journal={arXiv preprint arXiv:2604.20291},
  year={2026}
}
```
