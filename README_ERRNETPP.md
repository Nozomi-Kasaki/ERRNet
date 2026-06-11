# ERRNet++ OpenRR Training

This branch adds a stronger from-scratch model for single image reflection removal.

## Main Changes

- New architecture: `--inet errnet_pp`
- Residual logit-space output, so the model starts close to the input and learns only the reflection correction
- Dual-stream stem for RGB, hypercolumn, and edge features
- Deeper residual dense attention encoder-decoder
- ASPP bottleneck for multi-scale context
- Global color-affine branch and gated mask branch
- OpenRR-5k training support

The default `errnet_pp` width is 128 channels. With hypercolumn input, it has
about 23.4M parameters, compared with about 19.0M parameters for the original
ERRNet baseline.

## Training Data

The default training fusion is:

```text
VOC synthetic + Berkeley real + OpenRR-5k
```

with ratios:

```text
0.525 / 0.225 / 0.25
```

## Training

Run:

```bash
bash scripts/train_errnetpp_openrr.sh
```

The best checkpoint will be saved to:

```text
checkpoints/errnetpp_openrr_scratch/errnet_best_eval.pt
```

Use `errnet_best_eval.pt` for reporting.
