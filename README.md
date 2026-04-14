# AIO_MAI

Official repository for our MAI 2026 Quantized 4K Image Super-Resolution Challenge submission:

**Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training**

---

## Qualitative Results

<p align="center">
  <img src="assets/visual_comparison_12panels.png" alt="Qualitative comparison of LR, HR, and AIO_MAI results" width="100%">
</p>

<p align="center">
  <em>
    Qualitative comparison of ×3 quantized image super-resolution on representative DIV2K validation images.
    From left to right: Full LR image, zoomed LR patch, HR ground truth, and our AIO_MAI result.
  </em>
</p>

---

## Overview

This repository contains our full training, export, and evaluation pipeline for a compact **×3 single-image super-resolution** model designed for **mobile INT8 deployment**.

Our method is built around a lightweight **LR-space extract–refine–upsample** architecture with a **MobileOne-style re-parameterizable** student backbone, **teacher-guided training**, and **deploy-before-QAT** optimization for stable INT8 inference.

### Key Features

- Lightweight **LR-space** super-resolution design
- **MobileOne-style** re-parameterizable student backbone
- **Three-stage training** pipeline
- **Teacher-guided knowledge distillation**
- **Deploy-before-QAT** for reduced train–deploy mismatch
- **INT8 TFLite export** for mobile-friendly deployment

---

## Method Summary

Our student model follows an **extract–refine–upsample** design:

- A shallow input stem projects the LR image into feature space
- A stack of lightweight **MobileOne-style re-parameterizable blocks** refines features in LR space
- A global feature skip preserves coarse structures
- A final **PixelShuffle** head reconstructs the ×3 HR output

### Training Pipeline

The model is trained in **three stages**:

#### Stage 1
Basic reconstruction training using **L1 loss**.

#### Stage 2
Fidelity-oriented refinement using:

- **Charbonnier loss**
- **DCT-domain supervision**
- **Confidence-weighted output-level knowledge distillation**
- **MambaIRv2Light** as teacher

#### Stage 3
**Quantization-aware training (QAT)** on the **fused deploy graph** to improve INT8 robustness and reduce quantization mismatch.

---

## Main Results

| Setting | PSNR (dB) | SSIM | Notes |
|---|---:|---:|---|
| Ours (Stage 2) FP32 | 30.28 | 0.863 | Best floating-point student |
| Ours (Deploy) INT8 | 30.13 | 0.858 | Fixed-shape deployable TFLite artifact |
| AIO MAI Submission | 29.79 | 0.8634 | Final MAI 2026 test-set result, score = 1.8, top 5 |

### Additional Ablation Highlights

- **MobileOne** provides the best FP32/INT8 trade-off among tested re-parameterizable blocks
- Stage 3 teacher-guided supervision improves INT8 reconstruction from **29.9114 / 0.853** to **30.0003 / 0.856**

---

## Repository Structure

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

## Installation

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

## Dataset Preparation

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

## Training

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

## Export

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

## Evaluation

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

## Citation

```bibtex
@misc{nguyen2026efficientint8sr,
  title={Efficient INT8 Single-Image Super-Resolution via Deployment-Aware Quantization and Teacher-Guided Training},
  author={Pham Phuong Nam Nguyen and Nam Tien Le and Thi Kim Trang Vo and Nhu Tinh Anh Nguyen},
  year={2026}
}
```
