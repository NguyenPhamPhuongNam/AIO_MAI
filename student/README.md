- You should use your path to replace our temporary path
- Run this command: cd ./AIO_MAI
- Download DIV2K dataset (setting name for folder dataset : DIV2K) and being contained in the folder "data" outside folder "AIO_MAI"
- Run this command: cd ./AIO_MAI/student
## To train model, we proceed step by step as follows:
# Train stage 1+2 
python train.py \
  --data_root ./data/DIV2K \
  --out_dir ./AIO_MAI/student/runs_mobileone_32_8 \
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
  --mambair_repo ./AIO_MAI/MambaIR \
  --mambair_teacher_ckpt ./AIO_MAI/mambairv2_lightSR_x3.pth \
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



# Train stage 3
python train.py \
  --data_root ./data/DIV2K \
  --out_dir ./AIO_MAI/student/runs_mobileone_32_8\
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
  --init_ckpt ./AIO_MAI/student/runs_mobileone_32_8/balanced/ckpt_best_s2_fp32.pt \
  --teacher_type antsr \
  --teacher_ckpt./AIO_MAI/student/runs_mobileone_32_8/balanced/ckpt_best_s2_deploy.pt \
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


# To export, we run:
python main.py   --deploy_ckpt ./AIO_MAI/student/runs_mobileone_32_8/balanced/ckpt_best_s3_qat_deploy.pt   --data_root ./AIO_MAI/data/DIV2K   --out_dir ./AIO_MAI/student/exports_moi/s3_auto_train   --fixed_h 720   --fixed_w 1280   --auto_search   --search_eval_images 20   --smoke_verify_images 20   --seed 42


# To evaluate, we run:
python eval_div2k_full_compare.py   --deploy_ckpt ./AIO_MAI/student/runs_mobileone_32_8/balanced/ckpt_best_s3_qat_deploy.pt   --model_none_int8 ./AIO_MAI/student/exports_moi/s3_auto_train/TFLite/model_none.tflite   --model_none_float./AIO_MAI/student/exports_moi/s3_auto_train/TFLite/model_none_float.tflite   --model_fixed ./AIO_MAI/student/exports_moi/s3_auto_train/TFLite/model.tflite   --data_root ./AIO_MAI/data/DIV2K   --device cpu   --shaves 0,3   --fixed_overlap 16   --out_json ./AIO_MAI/student/exports_moi/s3_auto_train/eval_div2k_full_report.json


