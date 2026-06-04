# ERRNet Experiment: Augmentation Only

This branch is for the augmentation-only ablation:

- start from the official ERRNet baseline checkpoint
- keep the original aligned-data loss profile (`--loss_profile legacy`)
- enable the new reflection synthesis augmentation
- disable GAN during finetuning to preserve PSNR/SSIM

Large files are intentionally not tracked by Git. Put data and weights in:

```text
ERRNet/
  checkpoints/
    errnet/
      errnet_060_00463920.pt
  datasets/
    raw_data/
      VOCdevkit/
      CEILNet/
      real89/
      robustsirr_test_dataset/
      Dataset/
```

Prepare data:

```bash
cd /root/ERRNet
python3 datasets/prepare_test_data.py
python3 datasets/prepare_train_data.py
```

Train:

```bash
bash scripts/train_experiment.sh
```

The training script evaluates all benchmarks once per epoch and saves:

```text
checkpoints/errnet_aug_only_ft/errnet_latest.pt
checkpoints/errnet_aug_only_ft/errnet_best_eval.pt
```

The best checkpoint is selected by mean relative PSNR/SSIM against the baseline.
