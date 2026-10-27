# Adaptive Deep Learning and Machine Learning Model Selection in Sketch-to-3D Face Reconstruction for Forensic Applications

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Three.js](https://img.shields.io/badge/Three.js-r128-000000?style=for-the-badge&logo=three.js&logoColor=white)](https://threejs.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.style=for-the-badge)](LICENSE)

---

## 📌 Project Overview

Forensic facial reconstruction translates hand-drawn eyewitness sketches into photorealistic facial images and 3D geometric models for law enforcement identification. Traditional systems rely on a single, statically chosen model regardless of variations in sketch stroke density, contrast, or line sharpness.

This repository implements an **Adaptive Multi-Model Forensic Sketch-to-2D & 3D Face System** featuring a **Zero Ground-Truth (Zero-GT) Dynamic Model Selection Engine** operating strictly within a **4.0 GB VRAM hardware constraint** (NVIDIA GeForce RTX 3050 Laptop GPU).

The system integrates **5 candidate models** across Deep Learning and Machine Learning paradigms:
1. **Deep U-Net (Perceptual Loss):** Residual Bottleneck U-Net trained with Multi-Scale SSIM and VGG-16 Perceptual Loss.
2. **Pix2Pix Conditional GAN:** 8-layer U-Net Generator + $70 \times 70$ PatchGAN Discriminator.
3. **ResNet Bottleneck LSGAN:** 6 Residual Blocks + LSGAN Least-Squares Discriminator with Feature Matching.
4. **Eigenface Dual-PCA + Ridge Regression:** Dual Principal Component Analysis (50 eigen-components) + Ridge Regression.
5. **Multi-Scale Patch K-NN Regressor:** Local $32 \times 32$ spatial patch retrieval ($k=3$) with inverse-distance blending.

---

## 🚀 Key Technical Features

- **Zero-GT Dynamic Selection Engine:** Evaluates candidate 2D outputs without requiring ground-truth target photos using a multi-criteria formula:
  - **Structural Identity Match (+0.70):** Sobel SSIM, Canny IoU, Hair/Beard Luminance Correlation, and Pearson Correlation.
  - **No-Reference Facial Realism (+0.20):** RGB skin color mask ratio and gradient norm smoothness.
  - **Landmark Edge Sharpness (+0.15):** Canny Laplacian contour sharpness.
  - **2D FFT Grid Artifact Penalty (-1.00):** Frequency spectrum corner-to-center ratio and 8px grid discontinuities.
  - **Geometric Distortion Penalty (-1.50):** Facial asymmetry and diagonal gradient energy imbalance.
- **Strict 4GB VRAM Memory Lifecycle:** Deletes temporary candidate tensors (`del sketch_tensor`, `torch.cuda.empty_cache()`, `gc.collect()`) BEFORE instantiating `GFPGANer` in freed VRAM, eliminating CUDA OOM crashes.
- **GFPGAN v1.3 AI Face Restoration:** Enhances facial features (pores, eyes, hair strands) with strict BGR $\leftrightarrow$ RGB channel transformations, preventing blue channel swap halos.
- **Parametric 3D Surface Reconstruction (3DMM):** Analytical depth map equation $Z(x,y) = Z_{\text{base}} + Z_{\text{nose}} + Z_{\text{eyes}} + Z_{\text{chin}}$ mapped onto a $48 \times 48$ spatial plane geometry (2,401 3D vertices, 4,608 triangular faces).
- **Interactive WebGL Dashboard:** Single-page FastAPI + Three.js application featuring candidate confidence progress bars, OrbitControls 3D viewing, camera presets (Front, Left, Right, Top), and 3D file exports (.OBJ, .PLY, .GLB).

---

## 🛠️ Tech Stack

| Layer | Framework / Library |
| :--- | :--- |
| **Deep Learning** | PyTorch 2.x, PyTorch AMP (`torch.cuda.amp`), GFPGAN v1.3, Facexlib, BasicSR |
| **Machine Learning** | Scikit-Learn (GradientBoostingRegressor, PCA, Ridge), Joblib |
| **Computer Vision** | OpenCV (`cv2`), Scikit-Image (`skimage.metrics.ssim`), Pillow (PIL) |
| **Backend API** | FastAPI, Uvicorn, PyYAML |
| **Frontend & 3D** | HTML5, CSS3, JavaScript (ES6), Three.js WebGL, OrbitControls |
| **Documentation & Evaluation** | Matplotlib, Pandas, NumPy, ReportLab |

---

## 📁 Repository Structure

```text
Adaptive-Model-Selection-for-Forensic-Sketch-to-3D-Reconstruction/
├── backend/
│   └── app.py                  # FastAPI REST Server Entrypoint
├── inference/
│   └── pipeline.py             # End-to-End Orchestrator & VRAM Sequence
├── model_selection/
│   ├── meta_selector.py        # Zero-GT Adaptive Selection Engine
│   └── feature_extractor.py    # Sketch Structural Feature Extractor
├── models/
│   ├── model_1_pix2pix.py      # Pix2Pix Conditional GAN
│   ├── model_2_unet.py         # Deep U-Net (Perceptual Loss)
│   ├── model_3_resnet_gan.py   # ResNet Bottleneck LSGAN
│   ├── model_4_pca_ridge.py    # Eigenface Dual-PCA + Ridge
│   └── model_5_knn_patch.py    # Multi-Scale Patch K-NN Regressor
├── face_3d/
│   ├── mesh_generator.py       # 3DMM Depth Map Z(x,y) & Surface Mesh
│   └── exporter.py             # Wavefront .OBJ, .PLY, .GLB Exporter
├── frontend/
│   ├── index.html              # Responsive Dark Dashboard UI
│   ├── app.js                  # Three.js WebGL OrbitControls Script
│   └── styles.css              # Custom Styling
├── configs/
│   └── config.yaml             # Master Configuration File
├── scripts/
│   └── run_paper_experiments.py # Camera-Ready Benchmark Generator
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation
```

---

## 💻 Installation & Setup

### 1. Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.1+ (Supported on 4.0 GB VRAM GPUs like RTX 3050)

### 2. Clone the Repository
```bash
git clone https://github.com/dhanushri432/Adaptive-Model-Selection-for-Forensic-Sketch-to-3D-Reconstruction.git
cd Adaptive-Model-Selection-for-Forensic-Sketch-to-3D-Reconstruction
```

### 3. Windows 11 / Python 3.12 setup

From the project root in PowerShell, use the project's virtual environment and pinned CPU-compatible dependency set:

```powershell
Set-Location C:\Users\Dell\Downloads\project\project
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `.venv` already exists, do not recreate it; install only if its dependencies are missing. The pinned set keeps PyTorch 2.5.1 CPU paired with torchvision 0.20.1.

### 4. Start the local demonstration

```powershell
.\start_demo.ps1 -Port 8001
```

The script checks that the selected local port can be bound and starts one Uvicorn process without a development reloader. Port 8001 is the default because port 8000 may be occupied. The frontend uses same-origin API paths, so its API URL follows the selected port automatically.

Open **`http://127.0.0.1:8001`**, **`http://127.0.0.1:8001/health`**, or **`http://127.0.0.1:8001/docs`**. The current frontend is static HTML/JavaScript with Three.js; it has no React/npm build step.

Live inference requires all six trained model/selector checkpoints. When those files are absent, the server remains available for the clearly labelled Saved Results Demo, while sketch inference returns HTTP 503. Existing saved images, charts, and meshes are not generated from a newly uploaded sketch.

The root-level `app_frontend.py` / `app_backend.py` files are a separate legacy prototype with a different API and training-stub pathways. They are not used by `start_demo.ps1`; use `backend/app.py` and `frontend/index.html` for the real pipeline and saved-results presentation above.

---

## 📊 Benchmark Results

Evaluated on unseen test partition sketches (15% split; 249 CUFS/CUFSF paired identities):

| Pipeline Architecture | SSIM $\uparrow$ | PSNR (dB) $\uparrow$ | LPIPS $\downarrow$ | Identity Match Score $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline A** (Static Deep U-Net) | $0.7624 \pm 0.0312$ | $23.15$ | $0.2104$ | $0.7812$ |
| **Baseline B** (Random Model Selection) | $0.6148 \pm 0.0845$ | $18.42$ | $0.3842$ | $0.5420$ |
| **Proposed Stage-2 Dynamic Selector** (Ours) | **$\mathbf{0.8415 \pm 0.0210}$** | **$\mathbf{26.85}$** | **$\mathbf{0.1142}$** | **$\mathbf{0.9145}$** |

---

## 📜 Citation & License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
