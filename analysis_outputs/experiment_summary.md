# Experiment Summary

This summary is based on the synced training logs under `checkpoints/*`.

## Best Overall Result

The strongest run is `errnet_baseline_openrr_identity_ft`.

- Best epoch: 64
- Selection score: +0.018381
- Mean dPSNR: +0.6550
- Mean dSSIM: +0.0095
- Mean dNCC: +0.0090
- Mean dLMSE: -0.0010

Best checkpoint name in the original training workspace:

```text
checkpoints/errnet_baseline_openrr_identity_ft/errnet_best_eval.pt
```

The checkpoint file itself is intentionally not tracked by Git.

## Experiment Ranking

| Run | Best Epoch | Selection Score | Mean dPSNR | Mean dSSIM | Mean dNCC | Mean dLMSE |
|---|---:|---:|---:|---:|---:|---:|
| `errnet_baseline_openrr_identity_ft` | 64 | +0.018381 | +0.6550 | +0.0095 | +0.0090 | -0.0010 |
| `errnet_gated_adapter_ft` | 25 | +0.005535 | +0.1213 | +0.0061 | +0.0026 | -0.0007 |
| `errnet_soft_struct_ft` | 64 | +0.001752 | -0.0459 | +0.0045 | +0.0026 | -0.0007 |
| `errnet_rdaunet_scratch` | 30 | -0.043354 | -1.8780 | -0.0163 | -0.0185 | +0.0008 |
| `errnetpp_openrr_scratch` | 5 | -0.079383 | -3.0433 | -0.0425 | -0.0496 | +0.0035 |

Negative `dLMSE` is better.

## Notes

- The best practical recipe is baseline checkpoint initialization, OpenRR-5k paired data, identity clean samples, and conservative MSE-dominant loss.
- Fully from-scratch larger architectures did not beat the baseline under the current data and training setup.
- The synced `loss_log.txt` files contain headers only, so the plot script falls back to benchmark evaluation trends. Future runs should log per-iteration `IPixel` and `VGG` loss values if true training-loss curves are needed.

