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
  --name errnet_rdaunet_scratch \
  --inet rdaunet \
  --hyper \
  --lr 1e-4 \
  --nEpochs 70 \
  --lambda_gan 0 \
  --loss_profile structure \
  --lambda_mse 1.0 \
  --lambda_charbonnier 0.2 \
  --lambda_gradient 0.15 \
  --lambda_laplacian 0.02 \
  --lambda_ssim 0.05 \
  --reflection_noise_std 0 \
  --reflection_jpeg_prob 0 \
  --reflection_shift 3 \
  --reflection_color_jitter 0.03 \
  --reflection_alpha_low 0.5 \
  --reflection_alpha_high 1.0 \
  --train_eval_interval_epochs 1.0
