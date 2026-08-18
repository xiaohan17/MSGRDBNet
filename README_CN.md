# MSGRDBNet

## 项目简介

MSGRDBNet 是一个基于 PyTorch 实现的遥感图像超分辨率重建项目，面向低分辨率遥感影像的空间细节恢复与高分辨率图像重建任务。项目提供完整的模型训练、测试和推理流程，并支持真实 LR-HR 配对数据与 HR-only 在线退化数据两种使用方式。

MSGRDBNet 采用多尺度特征建模与全局信息建模机制，用于增强遥感影像中的纹理、边缘和空间结构恢复能力。项目提供统一配置文件、训练权重管理、PSNR/SSIM 精度评估、单图推理以及大尺寸影像分块推理功能，适用于模型复现、实验验证和遥感超分辨率研究。

## 主要功能

- MSGRDBNet 模型训练与验证
- 支持真实 LR-HR 配对数据训练
- 支持 HR-only 在线退化训练
- PSNR / SSIM 精度评估
- 最优权重与最近权重自动保存
- 单幅图像超分辨率推理
- 大尺寸遥感影像分块推理
- 支持已有 `.pth` / `.ckpt` 权重加载
- 支持 SSGM、窗口注意力和无全局模块配置

## 项目结构

```text
MSGRDBNet/
├── configs/
│   ├── synthetic_x4.yaml       # HR-only 在线退化训练配置
│   └── paired_x4.yaml          # LR-HR 配对训练配置
├── msgrdbnet/
│   ├── __init__.py
│   ├── model.py                # MSGRDBNet 网络定义
│   ├── data.py                 # 数据读取、裁剪和增强
│   ├── degradation.py          # synthetic 模式退化模型
│   ├── metrics.py              # PSNR / SSIM
│   └── utils.py                # 配置、归一化和权重读写
├── weights/                    # 预训练权重
├── outputs/                    # 训练、测试和推理输出
├── train.py                    # 训练入口
├── test.py                     # 测试入口
├── infer.py                    # 推理入口
├── requirements.txt
├── LICENSE
└── README.md
```

## 环境依赖

推荐环境：

- Python 3.10 / 3.11
- CUDA 12.1
- PyTorch 2.3.1
- Mamba-SSM 2.3.0
- einops 0.8.1

建议先根据本机 CUDA 版本安装 PyTorch，再安装其他依赖。

```bash
conda create -n msgrdbnet python=3.11 -y
conda activate msgrdbnet
```

CUDA 12.1 示例：

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

默认配置使用：

```yaml
global_block_type: ssgm
```

因此需要安装 `mamba-ssm`。

## 数据集准备

### 1. LR-HR 配对数据

适用于 SEN2NAIP 等真实配对超分辨率数据集。

LR 与 HR 图像应使用相同文件名或相同文件 stem，例如：

```text
0001.png  <->  0001.png
```

推荐目录结构：

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

然后在：

```text
configs/paired_x4.yaml
```

中设置对应的 `lr_dir`、`hr_dir`、裁剪尺寸及其他训练参数。

### 2. HR-only Synthetic 数据

如果只有 HR 图像，可在训练过程中在线生成退化 LR 图像。

推荐目录结构：

```text
datasets/WHU-RS19/
├── train/
├── val/
└── test/
```

每个目录中直接存放 HR 图像。

Synthetic 模式通过以下退化过程在线生成 LR：

```text
Gaussian Blur
    ↓
Bicubic Downsampling
    ↓
Noise
    ↓
Low-Resolution Image
```

退化参数在：

```text
configs/synthetic_x4.yaml
```

中配置。

支持的图像格式：

```text
.png
.jpg
.jpeg
.bmp
.tif
.tiff
```

模型默认读取图像前三个通道作为 RGB 输入。

## 模型训练

### Synthetic 模式

```bash
python train.py \
  --config configs/synthetic_x4.yaml \
  --devices 0 \
  --output outputs/synthetic_x4
```

### Paired 模式

```bash
python train.py \
  --config configs/paired_x4.yaml \
  --devices 0 \
  --output outputs/paired_x4
```

### 多 GPU 训练

例如使用 GPU 0 和 GPU 1：

```bash
python train.py \
  --config configs/paired_x4.yaml \
  --devices 0,1 \
  --output outputs/paired_x4
```

默认训练流程使用：

- L1 Loss
- Adam Optimizer
- MultiStepLR
- Automatic Mixed Precision
- Validation PSNR 选择最优模型

### 继续训练

```bash
python train.py \
  --config configs/paired_x4.yaml \
  --devices 0 \
  --output outputs/paired_x4 \
  --resume outputs/paired_x4/weights/last.pth
```

### 加载预训练权重进行训练

```bash
python train.py \
  --config configs/paired_x4.yaml \
  --pretrained weights/MSGRDBNet.pth \
  --output outputs/finetune
```

## 模型测试

```bash
python test.py \
  --weights outputs/paired_x4/weights/best.pth \
  --output outputs/test_paired \
  --save-images
```

测试输出：

```text
outputs/test_paired/
├── metrics.csv
└── sr/
```

其中：

- `metrics.csv`：保存每张图像的 PSNR、SSIM、推理时间以及平均结果
- `sr/`：保存超分辨率重建结果

## 单图推理

```bash
python infer.py \
  --input demo/LR.png \
  --weights outputs/paired_x4/weights/best.pth \
  --output outputs/infer/demo_SR.png
```

## 大图分块推理

对于较大的遥感图像，可使用 tiled inference：

```bash
python infer.py \
  --input demo/large_LR.tif \
  --weights outputs/paired_x4/weights/best.pth \
  --output outputs/infer/large_SR.tif \
  --tile-size 256 \
  --overlap 32
```

参数说明：

- `tile-size`：LR 图像分块尺寸
- `overlap`：相邻分块重叠区域
- `tile-size=0`：整图直接推理

当前 TIFF 推理流程保留影像像素内容，但不会复制 GeoTIFF 的 CRS、仿射变换等地理参考信息。

## 权重存放

手动下载或已有权重建议统一放在：

```text
weights/
├── MSGRDBNet.pth
└── model.ckpt
```

训练过程中生成的权重默认保存在：

```text
outputs/<experiment>/weights/
├── best.pth
├── last.pth
└── epoch_XXXX.pth
```

其中：

- `best.pth`：验证集 PSNR 最优模型
- `last.pth`：最近一个 epoch 的模型
- `epoch_XXXX.pth`：按设定周期保存的阶段模型

测试和推理推荐优先使用：

```text
best.pth
```

## 主要配置项

### global_block_type

```yaml
model:
  global_block_type: ssgm
```

可选：

```text
ssgm
window_attn
none
```

说明：

- `ssgm`：使用 Mamba 状态空间模块进行全局特征建模
- `window_attn`：使用窗口自注意力进行特征建模
- `none`：关闭全局建模模块

### enable_scale_attn

```yaml
model:
  enable_scale_attn: false
```

用于控制多尺度分支中的尺度注意力机制。

## 评价指标

当前测试流程提供：

- PSNR
- SSIM
- Inference Time

结果统一保存至 `metrics.csv`。

## License

License information is provided in the `LICENSE` file.
