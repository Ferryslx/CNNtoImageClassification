# CNN Image Classification - CIFAR-10

使用 PyTorch 搭建卷积神经网络对 CIFAR-10 数据集进行图像分类，包含从基础模型到优化的完整迭代过程。

## 项目结构

```
├── src/
│   ├── ImageClassification.py   # 基础 CNN 模型（2层卷积）
│   └── ModelOptimization.py     # 优化模型（6层卷积 + 正则化）
├── utils/
│   └── log.py                   # 日志工具类
├── model/                       # 训练好的模型权重
│   ├── CNNImageModel.pth
│   └── CNNImageModel_optimization.pth
├── log/                         # 训练日志
├── data/                        # CIFAR-10 数据集
└── README.md
```

## 模型架构

### 基础模型 (`ImageClassification.py`)

| 层 | 参数 |
|---|---|
| Conv1 | 3×32, kernel=3, stride=1 |
| Pool1 | MaxPool2d(2,2) |
| Conv2 | 32×128, kernel=3, stride=1 |
| Pool2 | MaxPool2d(2,2) |
| FC1 | 576 → 120 |
| FC2 | 120 → 84 |
| Output | 84 → 10 |

### 优化模型 (`ModelOptimization.py`)

6 层卷积 + 3 阶段下采样 + 全连接分类头，总参数量约 **1.47M**。

```
输入 (3×32×32)
  │
  ├─ Stage 1: Conv2d(3→32) → BN → ReLU → Conv2d(32→32) → BN → ReLU → MaxPool
  │   (32×32 → 16×16)
  ├─ Stage 2: Conv2d(32→64) → BN → ReLU → Conv2d(64→64) → BN → ReLU → MaxPool
  │   (16×16 → 8×8)
  ├─ Stage 3: Conv2d(64→128) → BN → ReLU → Conv2d(128→128) → BN → ReLU → MaxPool
  │   (8×8 → 4×4)
  │
  ├─ Flatten: 128 × 4 × 4 = 2048
  ├─ FC(2048 → 512) → BN → ReLU → Dropout(0.5)
  ├─ FC(512 → 256) → BN → ReLU → Dropout(0.5)
  └─ Output(256 → 10)
```

## 优化策略

| 方法 | 说明                                                                     |
|---|------------------------------------------------------------------------|
| **网络加深** | 2 层 → 6 层卷积，3 个下采样阶段                                                   |
| **Batch Normalization** | 每个卷积层和全连接层后添加 BN                                                       |
| **数据增强** | RandomCrop, HorizontalFlip, ColorJitter, RandomRotation, RandomErasing |
| **Dropout** | 全连接层后 p=0.5                                                            |
| **L2 正则化** | SGD weight_decay=1e-4                                                  |
| **早停策略** | patience=20, 监控验证集准确率                                                  |
| **学习率调整** | SGD (lr=0.1, momentum=0.9) + CosineAnnealingLR                         |

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- torchvision
- matplotlib
- torchsummary

## 使用方式

```bash
# 训练基础模型
python src/ImageClassification.py

# 训练优化模型
python src/ModelOptimization.py
```

数据集会自动下载到 `data/` 目录（第一次运行时下载 CIFAR-10，约 170MB）。

## 日志

训练日志自动保存到 `log/` 目录，包含每轮的损失、训练集准确率、验证集准确率和耗时。
