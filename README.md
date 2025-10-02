# 🥚 Alternative Non-Destructive Approach for Estimating Morphometric Measurements of Chicken Eggs from Tomographic Images with Computer Vision

[![DOI](https://img.shields.io/badge/DOI-10.3390%2Ffoods13244039-blue)](https://doi.org/10.3390/foods13244039)  


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

- **Data Acquisition**: CT scans of 150 chicken eggs under different storage conditions (0, 7, 14, 21, 28 days).  
- **Annotation**: Manual labeling with [CVAT](https://github.com/opencv/cvat), including **shell, yolk, albumen, and air chamber**.  
- **Preprocessing**: Normalization, cropping, and voxel volume estimation (1.269 mm³).  
- **Models**:
  - **3D U-Net** – encoder-decoder with skip connections.  
  - **3D FCN** – deep convolutional backbone with upsampling.  
  - **Regression Head** – fully connected layers to directly estimate shell thickness, egg height, and width.  
- **Framework**: Implemented in **PyTorch** with NVIDIA RTX 4090 training.

---

## 📁 Repository Structure

```bash
.
├── models/model_3D.py          # 3D segmentation models (U-Net, FCN)
├── data/data_backup.tar.xz     # 3D CT images and labels
├── Models/tests_train.py       # Training script
├── graph_metrics_real.py       # Metrics curves
├── 3D_Voxel_test.py            # Morphometric measurement extraction
└── README.md                   # Project documentation
```

---

## 🚀 Usage
The weights are available [HERE](https://drive.google.com/file/d/1GWIUzPglChFNVKUfSk7UN0tTI6wznKH8/view?usp=drive_link)
---

## 📜 Cite Our Work
### BibTeX
@article{article,
author = {Vargas, Jean and Abreu, Katariny and de Paula, Davi and Salvadeo, Denis and Souza, Lilian and Rabello, Carlos},
title = {Alternative Non-Destructive Approach for Estimating Morphometric Measurements of Chicken Eggs from Tomographic Images with Computer Vision},
journal = {Foods},
volume = {13},
year = {2024},
number = {24},
article-number = {4039},
month = {12},
issn = {2304-8158},
doi = {10.3390/foods13244039}
}

## 🙏 Acknowledgments
This research was funded in part by the São Paulo State Research Support Foundation (FAPESP) and the Pernambuco State Science and Technology Support Foundation (FACEPE).

