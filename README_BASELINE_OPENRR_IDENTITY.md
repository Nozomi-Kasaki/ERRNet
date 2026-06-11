# Baseline + OpenRR + Identity Fine-tuning

This branch adds a conservative fine-tuning recipe on top of the original ERRNet
checkpoint.

## Main idea

- Initialize from the official baseline checkpoint
- Add OpenRR-5k paired training data
- Add clean identity samples, where `input == target`
- Keep the reconstruction loss conservative so PSNR does not drift too much

## Training data

Default fusion:

```text
VOC synthetic + Berkeley real + OpenRR-5k + identity clean images
```

Identity samples are drawn from:

```text
datasets/processed_data/VOCdevkit/VOC2012/PNGImages
```

using the file list:

```text
VOC2012_224_train_png.txt
```

## Training

Run:

```bash
bash scripts/train_baseline_openrr_identity_ft.sh
```

The evaluation schedule is every 5 epochs through epoch 50, then every epoch
from epoch 51 to epoch 70.

The best checkpoint will be selected by the periodic full benchmark evaluation.
