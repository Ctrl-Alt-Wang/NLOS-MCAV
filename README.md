# NLOS-MCAV: Passive NLOS Imaging Based on Multi-dimension Collaborative Attention Module

## 项目概述

NLOS-MCAV是一个基于深度学习的非视距(Non-Line-of-Sight, NLOS)图像重建项目。该项目专注于通过神经网络架构来解决图像重建问题，特别适用于非视距成像场景。

## 主要功能

- **图像去模糊**: 使用深度学习模型恢复模糊图像
- **图像去噪**: 去除图像中的噪声，提高图像质量
- **图像重建**: 从低质量输入重建高质量图像

## 项目结构

```
NLOS-MCAV/
├── model/                    # 神经网络模型定义
│   ├── r2unet.py            # R2U-Net模型
│   ├── network_swinir.py    # SwinIR Transformer模型
│   ├── network_swinir_fft.py # SwinIR + FFT模型
│   ├── network_swinir_fft_GLU.py # SwinIR + FFT + GLU模型
│   ├── MIMOUnet.py          # MIMO U-Net模型
├── train/                   # 训练脚本
│   ├── train_r2Unet_MY.py   # MY数据集训练
│   ├── train_r2Unet_MG.py   # MG数据集训练
│   ├── train_STL.py         # SuperModel数据集训练
│   └── train_loss.py        # 损失函数
├── test/                    # 测试脚本
│   ├── test.py              # 主要测试脚本
├── util/                    # 工具函数
│   ├── metrics.py           # 评估指标(SSIM, PSNR)
│   ├── layers.py            # 自定义网络层
│   └── create_model.py      # 模型创建工具
├── loss/                    # 损失函数
│   ├── model_TII.py         # TII损失模型
│   ├── ssim.py              # SSIM损失
│   └── resnet.py            # ResNet使用的相关损失
├── pytorch_msssim/          # MS-SSIM实现
```

## NLOS-MCAV基于以下模型架构修改调整

### 1. R2U-Net (Recurrent Residual U-Net)
- **文件**: `model/r2unet.py`
- **特点**: 结合了残差连接和循环结构的U-Net
- **应用**: 图像分割和重建任务

### 2. SwinIR (Swin Transformer for Image Restoration)
- **文件**: `model/network_swinir.py`
- **特点**: 基于Swin Transformer的图像恢复模型
- **变体**: 
  - SwinIR + FFT (`network_swinir_fft.py`)
  - SwinIR + FFT + GLU (`network_swinir_fft_GLU.py`)

## 数据集支持

项目支持多种数据集格式：

### 数据集类型
1. **MY1500**: 1500张图像的MY数据集
2. **MG1500**: 1500张图像的MG数据集  
3. **SuperModel**: 超模数据集
4. **Cartoon**: 卡通图像数据集
5. **STL**: STL格式数据集

### 数据格式
- **输入格式**: PNG, BMP图像文件
- **数据结构**: 
  - `train/BR/`: 训练集模糊图像
  - `train/GT/`: 训练集真值图像
  - `test/BR/`: 测试集模糊图像
  - `test/GT/`: 测试集真值图像

## 安装依赖

```bash
pip install torch torchvision
pip install numpy pillow
pip install scikit-image
pip install pytorch-msssim
```

## 使用方法

### 1. 训练模型

#### 训练 (MY数据集)
```bash
python train/train_r2Unet_MY.py \
    --datarootData "Data/MY1500/train/BR" \
    --datarootTarget "Data/MY1500/train/GT" \
    --epoches 200 \
    --learning_rate 1e-5
```

#### 训练 (SuperModel数据集)
```bash
python train/train_STL.py \
    --datarootData "Data/SuperModel/train/blur" \
    --datarootTarget "Data/SuperModel/train/sharp" \
    --epoches 300 \
    --learning_rate 1e-5
```

### 2. 测试模型

```bash
python test/test.py \
    --datarootTestData "Data/SuperModel/test/blur" \
    --datarootTestTarget "Data/SuperModel/test/sharp" \
    --test_checkpoints_load_dir "./checkpoints/NLOS-ST_SuperModel/" \
    --which_epoch "latest"
```

## 训练参数配置

### 通用参数
- `--epoches`: 训练轮数 (默认: 200)
- `--batchsize`: 批次大小 (默认: 1)
- `--learning_rate`: 学习率 (默认: 1e-5)
- `--sizeImage`: 图像尺寸 (默认: 256)
- `--gpu_id`: GPU设备ID (默认: '0')

### 数据路径参数
- `--datarootData`: 训练数据路径
- `--datarootTarget`: 训练标签路径
- `--datarootValData`: 验证数据路径
- `--datarootValTarget`: 验证标签路径

## 评估指标

项目使用以下指标评估模型性能：

### PSNR (Peak Signal-to-Noise Ratio)
- **实现**: `util/metrics.py`
- **用途**: 衡量图像重建质量

### SSIM (Structural Similarity Index)
- **实现**: `util/metrics.py`
- **用途**: 衡量图像结构相似性

## 注意事项

1. **数据预处理**: 确保输入图像格式正确
2. **路径配置**: 根据实际数据位置修改路径参数


## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 快速开始

### 环境准备

1. **克隆项目**
```bash
git clone https://github.com/Ctrl-Alt-Wang/NLOS-MCAV.git
cd NLOS-MCAV
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **准备数据集**
```
Data/
├── MY1500/
│   ├── train/
│   │   ├── BR/    # 模糊图像
│   │   └── GT/    # 真值图像
│   └── test/
│       ├── BR/
│       └── GT/
├── MG1500/
└── SuperModel/
```
## 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@misc{nlos-mcav,
  title={NLOS-MCAV: Non-Line-of-Sight Multi-Channel Adaptive Vision},
  author={Shaohui Jin},
  year={2024},
  url={https://github.com/Ctrl-Alt-Wang/NLOS-MCAV}
}
```

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
