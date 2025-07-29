# 3D Shape Reconstruction with 3D-R2N2 and GRU

This project implements a 3D Recurrent Reconstruction Neural Network (3D-R2N2) using PyTorch to reconstruct 3D voxelized shapes from single-view 2D images. The architecture is adapted from the original 3D-R2N2 paper, with modifications to reduce model complexity while maintaining reconstruction capability.

## 🧠 Overview

- **Task**: 3D shape reconstruction from single-view 2D images  
- **Dataset**: [ShapeNetCore](https://shapenet.org/) (subset of 25,000 samples used)  
- **Architecture**: Encoder → GRU → Decoder  
- **Output**: 3D voxel grids (32×32×32)  
- **ML Method**: Supervised Learning  
- **Framework**: PyTorch

## 🛠 Features

- Replaced LSTM with GRU for efficiency  
- Simplified decoder with single 3D convolution layer  
- Used Binary Cross-Entropy Loss (BCELoss)  
- Optimized with Adam, learning rate scheduler, and gradient clipping  
- Visualized outputs using TensorBoard and 3D scatter plots  
- Evaluated using Intersection over Union (IoU)

## 📊 Results

| Metric          | Value  |
|-----------------|--------|
| Training Loss   | 0.178  |
| Validation Loss | 0.204  |
| IoU Score       | 0.107  |

*Note: The simplified model trades off detail for reduced computational cost. Future improvements include training on the full dataset and testing alternative RNN layers.*

## 📁 Project Structure

```
├── data/
│   └── shapenet_subset/
├── model/
│   ├── encoder.py
│   ├── decoder.py
│   └── rnn_module.py
├── train.py
├── evaluate.py
├── utils/
│   └── dataloader.py
├── tensorboard_logs/
└── README.md
```

## 📦 Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- TensorBoard

```bash
pip install -r requirements.txt
```

## 🚀 Run

### Training

```bash
python train.py
```

### Evaluation

```bash
python evaluate.py
```

## 📌 References

- [3D-R2N2 Paper (Choy et al., 2016)](https://arxiv.org/abs/1604.00449)  
- [ShapeNet Dataset](https://shapenet.org/)
