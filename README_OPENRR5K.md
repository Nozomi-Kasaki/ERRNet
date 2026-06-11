# OpenRR-5k Training Add-on

This experiment optionally adds OpenRR-5k paired real reflection data to the training fusion.

## Remote Data Layout

The dataset was downloaded from:

```text
https://huggingface.co/datasets/qiuzhangTiTi/OpenRR-5k/tree/main
```

The upstream file is named `trian_5k.zip`; this appears to be a typo for `train_5k.zip`.

On the remote machine, the large files are kept on the shared data disk:

```text
/DIP/PJ/ERRNet_data/OpenRR-5k/trian_5k.zip
/DIP/PJ/ERRNet_data/OpenRR-5k/trian_5k/
```

The project uses a symlink:

```text
datasets/processed_data/OpenRR-5k -> /DIP/PJ/ERRNet_data/OpenRR-5k/trian_5k
```

The extracted training split contains:

```text
blended: 5000 images
transmission_layer: 5000 images
```

## Training

Run:

```bash
bash scripts/train_gated_adapter_openrr_ft.sh
```

This script adds OpenRR-5k to the usual ERRNet training fusion:

```text
synthetic VOC pairs + Berkeley real pairs + OpenRR-5k pairs
```

The default OpenRR sampling ratio is conservative:

```text
VOC synthetic: 52.5%
Berkeley real: 22.5%
OpenRR-5k: 25.0%
```

The best checkpoint is selected by the same periodic full-benchmark evaluation:

```text
checkpoints/errnet_gated_adapter_openrr_ft/errnet_best_eval.pt
```
