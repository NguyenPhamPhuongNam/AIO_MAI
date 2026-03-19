import os
import glob
import argparse
import math
import numpy as np
import tensorflow as tf
from PIL import Image

def psnr_255_numpy(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-12) -> float:
    """Sao chép chuẩn xác hàm psnr_255 từ eval_stage3_detailed.py"""
    mse = np.mean((pred - gt) ** 2)
    if mse < eps:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite_model", type=str, required=True)
    ap.add_argument("--data_root", type=str, required=True)
    args = ap.parse_args()

    lr_dir = os.path.join(args.data_root, "DIV2K_valid_LR_bicubic_X3", "DIV2K_valid_LR_bicubic", "X3")
    hr_dir = os.path.join(args.data_root, "DIV2K_valid_HR", "DIV2K_valid_HR")
    
    if not os.path.exists(lr_dir):
        lr_dir = os.path.join(args.data_root, "DIV2K_valid_LR_bicubic", "X3")
    if not os.path.exists(hr_dir):
        hr_dir = os.path.join(args.data_root, "DIV2K_valid_HR")

    lr_files = sorted(glob.glob(os.path.join(lr_dir, "*.png")))
    hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.png")))

    print(f"🔥 Đang nạp mô hình: {args.tflite_model}")
    interpreter = tf.lite.Interpreter(model_path=args.tflite_model)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    in_scale, in_zp = input_details[0]['quantization']
    out_scale, out_zp = output_details[0]['quantization']

    total_psnr = 0.0
    num_images = len(lr_files)

    for i, (lr_path, hr_path) in enumerate(zip(lr_files, hr_files)):
        # 1. Đọc ảnh LR ở dải [0, 255]
        lr_img = np.array(Image.open(lr_path).convert("RGB"), dtype=np.float32)
        input_data = lr_img[None, ...]

        interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
        interpreter.allocate_tensors()

        # Ép về INT8 an toàn
        input_data_int8 = np.clip(np.round(input_data / in_scale) + in_zp, -128, 127).astype(np.int8)

        # Chạy suy luận
        interpreter.set_tensor(input_details[0]['index'], input_data_int8)
        interpreter.invoke()

        # 2. Lấy kết quả INT8 và Giải lượng tử về Float32
        output_data_int8 = interpreter.get_tensor(output_details[0]['index'])
        output_data_float = (output_data_int8.astype(np.float32) - out_zp) * out_scale
        
        # CHUẨN ĐẦU RA PYTORCH: Cắt kẹp (clamp) giữ nguyên là Float32, KHÔNG chuyển về uint8!
        sr_rgb = np.clip(output_data_float[0], 0.0, 255.0)

        # Đọc HR và giữ ở dạng Float32 [0, 255]
        hr_img = np.array(Image.open(hr_path).convert("RGB"), dtype=np.float32)

        # Cắt cho bằng kích thước (Giống code sr[..., :H, :W] của PyTorch)
        H, W, _ = hr_img.shape
        sr_rgb = sr_rgb[:H, :W, :]

        # Tính PSNR
        psnr = psnr_255_numpy(sr_rgb, hr_img)
        total_psnr += psnr
        
        if (i + 1) % 10 == 0:
            print(f"Đã xử lý {i + 1}/{num_images} ảnh... (PSNR tạm tính: {total_psnr / (i + 1):.2f} dB)")

    if num_images > 0:
        final_psnr = total_psnr / num_images
        print("-" * 50)
        print(f"🎉 HOÀN TẤT! ĐIỂM PSNR (CHUẨN PYTORCH): {final_psnr:.4f} dB")

if __name__ == "__main__":
    main()