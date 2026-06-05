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
  --name errnet_gated_adapter_ft \
  --inet errnet_adapter \
  --hyper \
  -r \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --reset_epoch_on_load \
  --lr 5e-5 \
  --nEpochs 70 \
  --lambda_gan 0 \
  --loss_profile structure \
  --lambda_mse 1.5 \
  --lambda_charbonnier 0.2 \
  --lambda_gradient 0.05 \
  --lambda_laplacian 0.01 \
  --lambda_ssim 0.03 \
  --lambda_vgg 0.02 \
  --lambda_adapter_consistency 0.3 \
  --lambda_adapter_sparsity 0.05 \
  --adapter_freeze_backbone_epochs 8 \
  --adapter_backbone_lr_scale 0.05 \
  --adapter_lr_scale 1.0 \
  --reflection_noise_std 0 \
  --reflection_jpeg_prob 0 \
  --reflection_shift 3 \
  --reflection_color_jitter 0.03 \
  --reflection_alpha_low 0.45 \
  --reflection_alpha_high 1.0 \
  --transmission_alpha_low 0.95 \
  --transmission_alpha_high 1.0 \
  --train_eval_epoch_schedule 5:50,1:70
