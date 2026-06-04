#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 train_errnet.py \
  --name errnet_mild_aug_legacy_ft \
  --hyper \
  -r \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --lr 2e-5 \
  --nEpochs 70 \
  --lambda_gan 0 \
  --loss_profile legacy \
  --reflection_noise_std 0 \
  --reflection_jpeg_prob 0 \
  --reflection_color_jitter 0.03 \
  --reflection_shift 3 \
  --reflection_alpha_low 0.5 \
  --reflection_alpha_high 1.0 \
  --train_eval_interval_epochs 1.0
