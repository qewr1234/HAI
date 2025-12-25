
# 🚗 Car Classification

**HAI(하이)! - Hecto AI Challenge 2025** 중고차 이미지 차종 분류 대회

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🏆 Result

| Metric | Score |
|--------|-------|
| **Final Rank** | **90th / 748 teams (Top 12%)** |

---

## 📋 Competition Overview

### 배경
최근 자동차 산업의 디지털 전환과 더불어, 다양한 차종을 빠르고 정확하게 인식하는 기술의 중요성이 커지고 있습니다. 특히 중고차 거래 플랫폼, 차량 관리 시스템, 자동 주차 및 보안 시스템 등 실생활에 밀접한 분야에서 정확한 차종 분류가 핵심 기술로 부상하고 있습니다.

### 주제
**중고차 이미지 차종 분류 AI 모델 개발**

다양한 중고차 차종 이미지 데이터를 분석하여 396개 클래스를 분류하는 AI 모델 개발

### 주최 / 주관
- **주최**: 헥토(Hecto)
- **주관**: 데이콘(Dacon)

### 평가 지표
**Log Loss (Cross Entropy)**

$$\text{LogLoss} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ij} \log(p_{ij})$$

- N: 전체 샘플 수
- C: 클래스 수 (396개)
- $y_{ij}$: i번째 샘플의 정답 클래스가 j이면 1, 아니면 0
- $p_{ij}$: i번째 샘플에 대해 모델이 클래스 j라고 예측한 확률

### 데이터셋

| 구분 | 설명 |
|------|------|
| Train | 396개 클래스, 총 33,137장 |
| Test | 8,258장 |
| Classes | 396개 차종 |

**동일 클래스 처리:**
- K5_3세대_하이브리드_2020_2022 = K5_하이브리드_3세대_2020_2023
- 디_올뉴니로_2022_2025 = 디_올_뉴_니로_2022_2025
- 718_박스터_2017_2024 = 박스터_718_2017_2024
- RAV4_2016_2018 = 라브4_4세대_2013_2018
- RAV4_5세대_2019_2024 = 라브4_5세대_2019_2024

---

## 🛠 Solution

### Model
- **Backbone**: ConvNeXt-Base (ImageNet-22k pretrained)
- **Input Size**: 384 × 384

### Training Techniques
| Technique | Description |
|-----------|-------------|
| **AMP** | Mixed Precision Training (FP16) |
| **EMA** | Exponential Moving Average (decay=0.9998) |
| **SWA** | Stochastic Weight Averaging |
| **CutMix** | 이미지 일부를 다른 이미지로 대체 |
| **MixUp** | 두 이미지를 선형 보간으로 혼합 |
| **R-Drop** | 동일 입력에 2번 forward → KL divergence 최소화 |
| **Label Smoothing** | 0.1 smoothing factor |

### Inference
- **TTA**: Test-Time Augmentation (5 transforms)
- **Ensemble**: Top-3 checkpoint averaging

---

## 📁 Project Structure

```
car_classification/
├── config.py        # Configuration dataclass
├── dataset.py       # Dataset & augmentation (CutMix, MixUp)
├── model.py         # Model, EMA, SWA, Loss functions
├── trainer.py       # Training logic
├── inference.py     # Inference & TTA
├── utils.py         # Utility functions
├── train.py         # Main training script
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Data Download

데이터셋은 대회 페이지에서 직접 다운로드해야 합니다:
- [데이콘 대회 페이지](https://dacon.io/competitions/official/236493/overview/description)

다운로드 후 아래 구조로 배치:

```
data/
├── train/
│   ├── 그랜저_6세대_하이브리드_2016_2019/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── 소나타_DN8_2019_2023/
│   │   └── ...
│   └── ... (396 classes)
├── test/
│   ├── TEST_00000.jpg
│   └── ...
├── test.csv
└── sample_submission.csv
```

### Training

```bash
# Basic training
python train.py --config base

# Light (faster, lower VRAM)
python train.py --config light

# Heavy (better performance)
python train.py --config heavy

# Custom
python train.py \
    --train_dir ./data/train \
    --test_dir ./data/test \
    --epochs 30 \
    --batch_size 32
```

### Inference Only

```bash
python train.py --eval_only --output_dir ./outputs
```

---

## ⚙️ Configuration

### Presets

| Preset | Model | Image Size | Batch | Epochs | VRAM |
|--------|-------|------------|-------|--------|------|
| `light` | ConvNeXt-Small | 224 | 64 | 20 | ~8GB |
| `base` | ConvNeXt-Base | 384 | 32 | 30 | ~16GB |
| `heavy` | ConvNeXt-Large | 384 | 16 | 40 | ~24GB |

### Command Line Options

```bash
--model_name convnext_base.fb_in22k_ft_in1k
--img_size 384
--batch_size 32
--epochs 30
--lr 1e-4
--no_amp      # Disable mixed precision
--no_ema      # Disable EMA
--no_rdrop    # Disable R-Drop
```

---

## 📚 References

- [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- [R-Drop: Regularized Dropout for Neural Networks](https://arxiv.org/abs/2106.14448)
- [CutMix: Regularization Strategy to Train Strong Classifiers](https://arxiv.org/abs/1905.04899)
- [MixUp: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

---

## 📝 License

MIT License
