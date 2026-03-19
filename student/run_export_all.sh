#!/bin/bash
set -euo pipefail

DATA_ROOT="/home/tinhanh/MobileAI/data/DIV2K"
OUTPUT_DIR="mai_submissions"

echo "BẮT ĐẦU TẠO GÓI NỘP BÀI CODABENCH..."
mkdir -p "$OUTPUT_DIR"

for DIR in runs_mobileone_*; do
  [ -d "$DIR" ] || continue

  echo ""
  echo "========================================================="
  echo "🔥 Đang quét thư mục: $DIR"

  CKPT_PATH=$(find "$DIR" -path "*/ckpt_best_s3_qat_deploy.pt" | sort | head -n 1)

  if [ -z "$CKPT_PATH" ]; then
    echo "⚠️  [Bỏ qua] Không tìm thấy ckpt_best_s3_qat_deploy.pt"
    continue
  fi

  SUBMISSION_DIR="$OUTPUT_DIR/${DIR}_Submission/TFLite"
  mkdir -p "$SUBMISSION_DIR"

  echo "✅ Dùng checkpoint cuối: $CKPT_PATH"

  python export_mai_submission.py \
    --deploy_ckpt "$CKPT_PATH" \
    --data_root "$DATA_ROOT" \
    --out_dir "$SUBMISSION_DIR" \
    --fixed_h 720 \
    --fixed_w 1280 \
    --calib_split train \
    --num_calib_samples 250 \
    --calib_mode mixed \
    --highvar_candidates 10 \
    --verify_samples 4 \
    --verify_crop_mode mixed \
    --no-export_debug_models \
    --seed 42
done

echo ""
echo "🎉 HOÀN TẤT TẠO GÓI NỘP BÀI!"