# MSGRDBNet

## Introduction

MSGRDBNet is a PyTorch-based project for remote sensing image super-resolution. It is designed to recover spatial details and reconstruct high-resolution images from low-resolution remote sensing data. The repository provides complete workflows for model training, testing, and inference, and supports both paired LR-HR datasets and HR-only synthetic degradation training.

MSGRDBNet combines multi-scale feature modeling with global feature modeling to improve the reconstruction of textures, edges, and spatial structures in remote sensing imagery. The project provides unified configuration files, checkpoint management, PSNR/SSIM evaluation, single-image inference, and tiled inference for large remote sensing images, making it suitable for model reproduction, experimental evaluation, and super-resolution research.

## Features

- MSGRDBNet training and validation
- Paired LR-HR dataset support
- HR-only synthetic degradation training
- PSNR / SSIM evaluation
- Automatic best and latest checkpoint saving
- Single-image super-resolution inference
- Tiled inference for large remote sensing images
- Support for `.pth` and `.ckpt` checkpoints
- Configurable SSGM, window attention, or no global block

## Project Structure

```text
MSGRDBNet/
├── configs/
│   ├── synthetic_x4.yaml       # HR-only synthetic training
│   └── paired_x4.yaml          # Paired LR-HR training
├── msgrdbnet/
│   ├── __init__.py
│   ├── model.py                # MSGRDBNet architecture
│   ├── data.py                 # Dataset loading and augmentation
│   ├── degradation.py          # Synthetic degradation pipeline
│   ├── metrics.py              # PSNR / SSIM
│   └── utils.py                # Configuration and checkpoint utilities
├── weights/                    # Pretrained weights
├── outputs/                    # Training, testing, and inference outputs
├── train.py                    # Training entry
├── test.py                     # Testing entry
├── infer.py                    # Inference entry
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

Recommended environment:

- Python 3.10 / 3.11
- CUDA 12.1
- PyTorch 2.3.1
- Mamba-SSM 2.3.0
- einops 0.8.1

It is recommended to install PyTorch according to your local CUDA version before installing the remaining dependencies.

```bash
conda create -n msgrdbnet python=3.11 -y
conda activate msgrdbnet
```

Example for CUDA 12.1:

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The default configuration uses:

```yaml
global_block_type: ssgm
```

Therefore, `mamba-ssm` is required.

## Dataset Preparation

### 1. Paired LR-HR Dataset

This mode is suitable for paired super-resolution datasets such as SEN2NAIP.

LR and HR images should use the same filename or the same file stem, for example:

```text
0001.png  <->  0001.png
```

Recommended directory structure:

```text
datasets/SEN2NAIP/
├── train/
│   ├── LR/
│   └── HR/
├── val/
│   ├── LR/
│   └── HR/
└── test/
    ├── LR/
    └── HR/
```

Then configure `lr_dir`, `hr_dir`, crop size, and other parameters in:

```text
configs/paired_x4.yaml
```

### 2. HR-only Synthetic Dataset

If only HR images are available, LR images can be generated online during training.

Recommended directory structure:

```text
datasets/WHU-RS19/
├── train/
├── val/
└── test/
```

Place HR images directly in each directory.

The synthetic mode generates LR images using:

```text
Gaussian Blur
    ↓
Bicubic Downsampling
    ↓
Noise
    ↓
Low-Resolution Image
```

The degradation parameters are defined in:

```text
configs/synthetic_x4.yaml
```

Supported image formats:

```text
.png
.jpg
.jpeg
.bmp
.tif
.tiff
```

The model uses the first three image channels as RGB input by default.

## Training

### Synthetic Training

```bash
python train.py \
  --config configs/synthetic_x4.yaml \
  --devices 0 \
  --output outputs/synthetic_x4
```

### Paired Training

```bash
python train.py \
  --config configs/paired_x4.yaml \
  --devices 0 \
  --output outputs/paired_x4
```

### Multi-GPU Training

For example, to use GPU 0 and GPU 1:

```bash
python train.py \
  --config configs/paired_x4.yaml \
  --devices 0,1 \
  --output outputs/paired_x4
```

The default training setup uses:

- L1 Loss
- Adam Optimizer
- MultiStepLR
- Automatic Mixed Precision
- Validation PSNR for best-model selection

### Resume Training

```bash
python train.py \
  --config configs/paired_x4.yaml \
  --devices 0 \
  --output outputs/paired_x4 \
  --resume outputs/paired_x4/weights/last.pth
```

### Fine-tuning from Pretrained Weights

```bash
python train.py \
  --config configs/paired_x4.yaml \
  --pretrained weights/MSGRDBNet.pth \
  --output outputs/finetune
```

## Testing

```bash
python test.py \
  --weights outputs/paired_x4/weights/best.pth \
  --output outputs/test_paired \
  --save-images
```

Test outputs:

```text
outputs/test_paired/
├── metrics.csv
└── sr/
```

- `metrics.csv` contains per-image PSNR, SSIM, inference time, and averaged results.
- `sr/` contains reconstructed super-resolution images.

## Single-Image Inference

```bash
python infer.py \
  --input demo/LR.png \
  --weights outputs/paired_x4/weights/best.pth \
  --output outputs/infer/demo_SR.png
```

## Tiled Inference for Large Images

Large remote sensing images can be processed using tiled inference:

```bash
python infer.py \
  --input demo/large_LR.tif \
  --weights outputs/paired_x4/weights/best.pth \
  --output outputs/infer/large_SR.tif \
  --tile-size 256 \
  --overlap 32
```

Arguments:

- `tile-size`: tile size in LR pixels
- `overlap`: overlap between adjacent tiles
- `tile-size=0`: process the entire image directly

The current TIFF inference pipeline preserves image pixels but does not copy GeoTIFF CRS, affine transform, or other geospatial metadata.

## Model Weights

Manually downloaded or existing weights can be placed in:

```text
weights/
├── MSGRDBNet.pth
└── model.ckpt
```

Training checkpoints are saved in:

```text
outputs/<experiment>/weights/
├── best.pth
├── last.pth
└── epoch_XXXX.pth
```

- `best.pth`: checkpoint with the best validation PSNR
- `last.pth`: checkpoint from the latest epoch
- `epoch_XXXX.pth`: periodically saved checkpoint

For testing and inference, `best.pth` is recommended.

## Main Configuration Options

### global_block_type

```yaml
model:
  global_block_type: ssgm
```

Available options:

```text
ssgm
window_attn
none
```

Description:

- `ssgm`: Mamba-based state-space global feature modeling
- `window_attn`: window-based self-attention
- `none`: disables the global modeling block

### enable_scale_attn

```yaml
model:
  enable_scale_attn: false
```

This option controls the scale-attention mechanism in the multi-scale branches.

## Evaluation Metrics

The current testing pipeline reports:

- PSNR
- SSIM
- Inference Time

All evaluation results are saved to `metrics.csv`.

## License

See the `LICENSE` file for license information.
