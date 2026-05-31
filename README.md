# FashionMNIST Multi-Model Classifier
Deep learning benchmarking project comparing MLP, CNN, and ResNet architectures on the FashionMNIST dataset with data augmentation and learning rate optimization.

![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-7C3AED.svg)
![Status](https://img.shields.io/badge/Status-Completed-4B0082.svg)

<div align="center">

━━━━━━━━━━━━━━ ✦ ✧ ✦ ━━━━━━━━━━━━━━

</div>

## 🟣 Overview

This project implements a full deep learning pipeline for image classification using the FashionMNIST dataset. It benchmarks multiple neural network architectures including MLP, CNN variants, and a simplified ResNet model.

The system evaluates performance across models using accuracy, F1-score, confusion matrices, and per-class metrics. Optimization techniques such as data augmentation, batch normalization, and learning rate scheduling are used to improve generalization and training stability.

<br><br>

## 🟣 Key Features

- Multiple model comparison (MLP, CNN, Advanced CNN, ResNet)
- Data augmentation (rotation, flipping, affine transforms)
- Learning rate warmup + cosine annealing scheduler
- Batch normalization and dropout regularization
- Per-class evaluation metrics (precision, recall, F1-score)
- Confusion matrix and prediction visualizations
<br><br>

## 🟣 Results

| Model | Test Accuracy | F1-Score | Parameters |
|-------|---------------|----------|------------|
| Simplified ResNet | 93.2% | 93.1% | ~250K |
| Advanced CNN | 92.8% | 92.7% | ~450K |
| Basic CNN | 90.5% | 90.3% | ~200K |
| Simple MLP | 87.1% | 86.9% | ~650K |

<br><br>

## 🟣 Model Architectures

- Simple MLP — Fully connected baseline network
- Basic CNN — Standard convolutional feature extractor
- Advanced CNN — CNN with batch normalization and dropout
- Simplified ResNet — Residual learning-based CNN

<br><br>

## 🟣 System Pipeline

- Dataset loading and preprocessing using torchvision
- Train/validation/test split (90/10 + held-out test set)
- Data augmentation applied only to training set
- Model training with Adam optimizer and weight decay
- Learning rate scheduling (warmup + cosine annealing)
- Evaluation using accuracy, F1-score, and confusion matrix

<br><br>

## 🟣 Visual Results

<p align="center">
  <img src="results/all_models_training_history.png" width="700"/>
</p>

<br><br>

## 🟣 Key Metrics

- Accuracy comparison across all models
- Per-class classification performance
- Confusion matrix analysis for best model
- Learning rate schedule visualization
- Training vs validation convergence tracking

<br><br>

## 🟣 Technologies Used

- Python 3.9+
- PyTorch
- torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook

<br><br>

## 🟣 Future Improvements

- Hyperparameter optimization (grid/random search)
- Experiment tracking (MLflow or Weights & Biases)
- Ensemble methods for performance boosting
- Vision Transformer comparison
- Cross-validation for robustness evaluation
