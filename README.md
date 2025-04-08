# 🥚 Alternative Non-Destructive Approach for Estimating Morphometric Measurements of Chicken Eggs from Tomographic Images with Computer Vision

This repository contains code and resources for estimating morphometric measurements (e.g., height, width, and shell thickness) of chicken eggs using **real CT images**. The system uses two approaches:

## 📌 Project Overview

Conventional methods for measuring internal structures of chicken eggs often require **destructive techniques**. In this phase, we propose an approach based on **3D real computed tomography (CT)** images.

The main goals of this part are:

- **Segmentation-based estimation** using 3D U-Net or Fully Convolutional Networks (FCN).
- **Direct estimation** using a head of fully connected layers for regression after feature extraction.
- Estimate internal measurements: **shell thickness**, **egg height**, **egg width**, **volumes**, and more.
- 
---

## 🧠 Methods

We use deep learning techniques, including:

- **3D U-Net / FCN** for semantic segmentation of CT volumes.
- **Fully Connected Regression Head** to directly estimate measurements from latent features.
- **PyTorch** framework for training and inference.

The system has been tested with real tomographic data of chicken eggs.

---

## 📁 Repository Structure

```bash
.
├── models/                 # 3D segmentation models (U-Net, FCN)
├── data/                   # Synthetic CT images and labels
├── utils/                  # Helper functions for preprocessing and postprocessing
├── train.py                # Training script
├── evaluate.py             # Evaluation and metric calculation
├── measure.py              # Morphometric measurement extraction
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```
