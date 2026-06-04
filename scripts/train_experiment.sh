#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 train_errnet.py \
  --name errnet_aug_only_ft \
  --hyper \
  -r \
  --icnn_path checkpoints/errnet/errnet_060_00463920.pt \
  --lr 2e-5 \
  --nEpochs 70 \
  --lambda_gan 0 \
  --loss_profile legacy \
  --train_eval_interval_epochs 1.0
