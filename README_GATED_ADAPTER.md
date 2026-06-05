# ERRNet Gated Adapter Experiment

This branch adds a conservative architecture change for the DIP26 reflection-removal experiment.

## Idea

The model keeps the pretrained ERRNet as an internal backbone and adds a small reflection-aware gated residual adapter:

```text
input image + ERRNet hypercolumns
  -> pretrained ERRNet backbone -> T0
  -> gated adapter([input, T0, input - T0, edge(input), edge(T0)])
  -> final output = T0 + small gated correction
```

Compared with the previous RDA-UNet scratch experiment, this design preserves the strong baseline mapping at initialization and only lets the new module edit uncertain reflection regions.

## Training

Make sure the dataset and baseline checkpoint are already prepared:

```text
datasets/processed_data/
checkpoints/errnet/errnet_060_00463920.pt
```

Run:

```bash
bash scripts/train_gated_adapter_ft.sh
```

The script:

- loads `checkpoints/errnet/errnet_060_00463920.pt` into the adapter backbone;
- resets epoch counters, so it trains a full 70 epochs;
- freezes the ERRNet backbone for the first 8 epochs;
- then jointly trains the adapter and backbone with the backbone learning rate scaled by `0.05`;
- evaluates every 5 epochs before epoch 50 and every epoch from 51 to 70.

The best checkpoint is selected by the existing mean relative PSNR/SSIM score:

```text
checkpoints/errnet_gated_adapter_ft/errnet_best_eval.pt
```

Use `errnet_best_eval.pt` rather than `errnet_latest.pt` for final reporting unless the final evaluation log shows otherwise.

## Key Options

```bash
--inet errnet_adapter
--reset_epoch_on_load
--adapter_freeze_backbone_epochs 8
--adapter_backbone_lr_scale 0.05
--lambda_adapter_consistency 0.3
--lambda_adapter_sparsity 0.05
```

These options keep the new architecture close to the baseline while still allowing local correction.
