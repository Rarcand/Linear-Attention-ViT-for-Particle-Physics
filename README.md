# Linear-Scale Attention ViT for CMS Particle Reconstruction

This repository contains an end-to-end implementation of a **Linear Attention Vision Transformer** designed to classify particle collision images and regress invariant mass from CMS detector data. By replacing standard softmax attention with a kernel-based approximation, this model scales linearly, $O(N)$, with image resolution, making it viable for high-granularity $125 \times 125$ detector hits.

## Core Features
* **Linear Attention Kernel**: Utilizes the $\phi(x) = \text{ELU}(x) + 1$ feature map to achieve $O(N)$ complexity while maintaining numerical stability on sparse physics data.
* **Self-Supervised Pre-training**: Includes a 50-epoch pre-training phase using a Masked Autoencoder (MAE) approach on unlabelled data to learn a "physics foundation" before classification.
* **Multi-Task Heads**: A dual-head architecture that simultaneously performs **Quark/Gluon classification** (Cross-Entropy) and **Invariant Mass regression** (MSE).
* **Physics-Informed Preprocessing**: Includes specialized noise-thresholding ($10^{-3}$) and $\log(1+x)$ scaling to handle the high dynamic range of detector energy deposits.

## Repository Structure
* **`notebooks/`**: Contains the primary Jupyter Notebook used for data loading, model definition, pre-training, and final benchmarking.
* **`weights/`**:
    * `pretrained_vit.pth`: Weights from the 50-epoch self-supervised reconstruction task.
    * `best_linear_vit_final.pth`: The final optimized weights used for classification and regression.

## Performance Benchmarks
The model was evaluated on the labelled CMS collision dataset, yielding the following results:

| Metric | Scratch Baseline | Finetuned (MAE) |
| :--- | :--- | :--- |
| **Final Acc** | **83.65%** | 79.45% |
| **Final MSE (Scaled)** | **0.1352** | 0.1774 |

> **Note**: The high accuracy of the scratch baseline demonstrates the strong capacity of the Linear Attention architecture to learn complex particle signatures even without pre-training initialization.

## Quick Start
1. **Environment**: Ensure you have `PyTorch`, `h5py`, and `tqdm` installed.
2. **Data**: Place the `.h5` collision datasets in the root directory.
3. **Run**: Open the main notebook in the `notebooks/` directory to reproduce the training or load the saved weights from `/weights` for inference.

---
**Author:** Ray Arcand  
**University:** University of Texas at Austin  
**Application:** GSoC 2026 — ML4SCI
