#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON:-python3}"
if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  if /opt/conda/bin/python3.12 -c "import torch" >/dev/null 2>&1; then
    PYTHON_BIN=/opt/conda/bin/python3.12
  fi
fi

"$PYTHON_BIN" train_errnet.py \
  --name errnet_baseline_openrr_identity_ft \
  --hyper \
  -r \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --reset_epoch_on_load \
  --lr 1e-5 \
  --nEpochs 70 \
  --lambda_gan 0 \
  --loss_profile structure \
  --lambda_mse 2.0 \
  --lambda_charbonnier 0.05 \
  --lambda_gradient 0.02 \
  --lambda_laplacian 0.005 \
  --lambda_ssim 0.01 \
  --lambda_vgg 0.005 \
  --reflection_noise_std 0 \
  --reflection_jpeg_prob 0 \
  --reflection_shift 3 \
  --reflection_color_jitter 0.03 \
  --reflection_alpha_low 0.45 \
  --reflection_alpha_high 1.0 \
  --transmission_alpha_low 0.95 \
  --transmission_alpha_high 1.0 \
  --openrr_train_dir datasets/processed_data/OpenRR-5k \
  --openrr_train_ratio 0.25 \
  --identity_train_dir datasets/processed_data/VOCdevkit/VOC2012/PNGImages \
  --identity_train_filelist VOC2012_224_train_png.txt \
  --identity_train_size 2000 \
  --identity_train_ratio 0.10 \
  --train_fusion_ratios 0.7,0.3 \
  --train_eval_epoch_schedule 5:50,1:70
