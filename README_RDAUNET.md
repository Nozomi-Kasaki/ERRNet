# RDA-UNet From-Scratch Experiment

This experiment adds a stronger reflection-removal architecture:

- residual dense blocks for local texture recovery
- CBAM channel/spatial attention for reflection-aware features
- U-Net skip fusion for detail preservation
- ASPP multi-scale bottleneck for global context
- Sobel edge guidance
- logit-space residual correction over the input RGB image to preserve PSNR

Run from scratch:

```bash
bash scripts/train_rdaunet_scratch.sh
```

Outputs:

```text
checkpoints/errnet_rdaunet_scratch/errnet_latest.pt
checkpoints/errnet_rdaunet_scratch/errnet_best_eval.pt
```

`errnet_best_eval.pt` is selected by mean relative PSNR/SSIM against the
provided baseline metrics. NCC and LMSE are still printed for analysis.
