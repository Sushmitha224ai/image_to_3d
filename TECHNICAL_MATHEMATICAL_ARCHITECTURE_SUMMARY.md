# TECHNICAL & MATHEMATICAL ARCHITECTURE SUMMARY
## Adaptive Multi-Model Forensic Sketch-to-2D & 3D Face Reconstruction System

---

### 1. MATHEMATICAL FORMULATION OF THE 5 GENERATIVE / REGRESSION MODELS

#### Model 1: Pix2Pix Conditional Generative Adversarial Network (cGAN)
- **Code File:** [`models/model_1_pix2pix.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/forensic-face-3d/models/model_1_pix2pix.py)
- **Architecture:** 8-layer U-Net Generator $G$ with skip connections + $70 \times 70$ PatchGAN Discriminator $D$.
- **Adversarial Loss (Binary Cross-Entropy):**
  $$\mathcal{L}_{\text{cGAN}}(G, D) = \mathbb{E}_{x,y}[\log D(x, y)] + \mathbb{E}_{x}[\log(1 - D(x, G(x)))]$$
  where $x \in \mathbb{R}^{256 \times 256 \times 3}$ is the input sketch and $y \in \mathbb{R}^{256 \times 256 \times 3}$ is the target ground-truth photo.
- **Pixel-Level Reconstruction Loss ($L_1$ Distance):**
  $$\mathcal{L}_{L1}(G) = \mathbb{E}_{x,y}\left[ \| y - G(x) \|_1 \right]$$
- **Total Objective:**
  $$G^* = \arg\min_G \max_D \left( \mathcal{L}_{\text{cGAN}}(G, D) + \lambda_{L1} \mathcal{L}_{L1}(G) \right), \quad \text{where } \lambda_{L1} = 100.0$$

---

#### Model 2: Deep Residual U-Net with Perceptual Loss
- **Code File:** [`models/model_2_unet.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/forensic-face-3d/models/model_2_unet.py)
- **Architecture:** Deep U-Net with Residual Bottleneck Blocks (`ResBlock`) and Instance Normalization.
- **Perceptual Loss ($\mathcal{L}_{\text{perceptual}}$):**
  $$\mathcal{L}_{\text{perceptual}}(G) = \sum_{l \in \{relu1\_2, relu2\_2, relu3\_3\}} \frac{1}{C_l H_l W_l} \| \phi_l(y) - \phi_l(G(x)) \|_2^2$$
  where $\phi_l$ represents feature maps extracted from layer $l$ of a pre-trained VGG-16 network.
- **Multi-Scale Structural Similarity Loss ($\mathcal{L}_{\text{SSIM}}$):**
  $$\mathcal{L}_{\text{SSIM}}(G) = 1.0 - \text{SSIM}(G(x), y)$$
- **Total Combined Objective:**
  $$\mathcal{L}_{\text{UNet\_Total}} = \lambda_{\text{perceptual}} \mathcal{L}_{\text{perceptual}} + \lambda_{\text{ssim}} \mathcal{L}_{\text{SSIM}} + \lambda_{L1} \mathcal{L}_{L1}, \quad \text{where } \lambda_{\text{perceptual}}=1.0, \; \lambda_{\text{ssim}}=0.5, \; \lambda_{L1}=1.0$$

---

#### Model 3: ResNet Bottleneck Least Squares GAN (LSGAN)
- **Code File:** [`models/model_3_resnet_gan.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/forensic-face-3d/models/model_3_resnet_gan.py)
- **Architecture:** 6-block ResNet Generator + LSGAN Discriminator with Feature Matching.
- **Least Squares Adversarial Loss:**
  $$\mathcal{L}_{\text{LSGAN}}(D) = \frac{1}{2} \mathbb{E}_{x,y}\left[ (D(x, y) - 1)^2 \right] + \frac{1}{2} \mathbb{E}_{x}\left[ (D(x, G(x)))^2 \right]$$
  $$\mathcal{L}_{\text{LSGAN}}(G) = \frac{1}{2} \mathbb{E}_{x}\left[ (D(x, G(x)) - 1)^2 \right]$$
- **Discriminator Feature Matching Loss ($\mathcal{L}_{\text{FM}}$):**
  $$\mathcal{L}_{\text{FM}}(G) = \sum_{k=1}^K \| D^{(k)}(x, y) - D^{(k)}(x, G(x)) \|_1, \quad \text{where } \lambda_{\text{fm}} = 10.0$$

---

#### Model 4: Eigenface Dual-PCA + Ridge Regression
- **Code File:** [`models/model_4_pca_ridge.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/forensic-face-3d/models/model_4_pca_ridge.py)
- **Principal Component Analysis (Eigen-space Projection):**
  $$\mathbf{z}_{\text{sketch}} = \mathbf{V}_k^T (\mathbf{x} - \boldsymbol{\mu}_{\text{sketch}}), \quad \text{where } k = 50 \text{ eigen-components}$$
- **Regularized Ridge Regression Mapping:**
  $$\mathbf{W}^* = \arg\min_{\mathbf{W}} \| \mathbf{Y}_{\text{photo\_pca}} - \mathbf{Z}_{\text{sketch}} \mathbf{W} \|_F^2 + \alpha \| \mathbf{W} \|_F^2, \quad \text{where } \alpha = 10.0$$
- **Reconstruction Output:**
  $$\hat{\mathbf{y}} = \boldsymbol{\mu}_{\text{photo}} + \mathbf{U}_k (\mathbf{W}^* \mathbf{z}_{\text{sketch}})$$

---

#### Model 5: Multi-Scale Patch K-NN Regressor
- **Code File:** [`models/model_5_knn_patch.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/forensic-face-3d/models/model_5_knn_patch.py)
- **Patch Decomposition:** Input image divided into $32 \times 32$ spatial patches with stride $s = 32$.
- **Nearest Neighbor Patch Retrieval:** For candidate sketch patch $p_i$, search patch database $\mathcal{D}_{\text{train}}$ using Euclidean distance:
  $$d(p_i, p_j) = \| p_i - p_j \|_2^2$$
- **K-NN Weighted Inverse-Distance Blending ($k=3$):**
  $$\hat{p}_i = \sum_{j=1}^k w_j p_{\text{photo}, j}, \quad w_j = \frac{\exp(-d(p_i, p_j) / \sigma)}{\sum_{m=1}^k \exp(-d(p_i, p_m) / \sigma)}$$

---

### 2. THE ADAPTIVE SELECTION ENGINE (THE CORE MATH)

- **Code File:** [`model_selection/meta_selector.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/forensic-face-3d/model_selection/meta_selector.py)
- **Primary Function:** `evaluate_candidates(self, sketch_img, generated_pil_dict)`

#### Composite Quality & Alignment Formula
For each candidate model $k \in \{1 \dots 5\}$, the selection score is evaluated without ground-truth test data leakage:

$$\text{CompositeScore}(k) = 0.70 \cdot I_k + 0.20 \cdot R_k + 0.15 \cdot S_k + 0.25 \cdot P_k - 1.00 \cdot A_k - 1.50 \cdot D_k - \text{Penalty}_{\text{template}}(k)$$

---

#### Component Mathematical Definitions

1. **Structural Identity Alignment Score ($I_k$):**
   $$I_k = 0.35 \cdot \text{SobelSSIM}(\text{sketch}, \text{gen}_k) + 0.25 \cdot \left(5.0 \cdot \frac{|\text{Canny}_{\text{sketch}} \cap \text{Canny}_{\text{gen}_k}|}{|\text{Canny}_{\text{sketch}} \cup \text{Canny}_{\text{gen}_k}|}\right) + 0.25 \cdot \text{LuminanceAlign}_k + 0.15 \cdot \max(0, \text{PearsonCorr}_k)$$
   where:
   $$\text{LuminanceAlign}_k = 1.0 - \min\left(1.0, 1.5 \cdot \left(|\mu_{\text{hair\_sketch}} - \mu_{\text{hair\_gen}_k}| + |\mu_{\text{beard\_sketch}} - \mu_{\text{beard\_gen}_k}|\right)\right)$$

2. **No-Reference Facial Realism ($R_k$):**
   $$R_k = \left(1.0 - \min\left(1.0, \frac{|\bar{\nabla}_k - 11.0|}{20.0}\right)\right) + 0.25 \cdot \min\left(1.0, 1.5 \cdot \text{SkinRatio}_k\right)$$
   where $\text{SkinRatio}_k$ is the fraction of pixels satisfying $R > 60 \land G > 40 \land B > 20 \land R > G \land R > B$.

3. **Landmark Contour Edge Sharpness ($S_k$):**
   $$S_k = \min\left(1.0, \frac{\text{mean}\left(|\text{Laplacian}(I_k)|_{\text{Canny}>0}\right)}{120.0}\right)$$

4. **Architectural Quality Prior ($P_k$):**
   $$\mathbf{P} = [P_{\text{pix2pix}}=0.30, \; P_{\text{unet}}=0.50, \; P_{\text{lsgan}}=0.05, \; P_{\text{pca}}=0.01, \; P_{\text{knn}}=0.05]$$

5. **2D FFT Checkerboard & Tile Artifact Penalty ($A_k$):**
   $$A_k = 0.40 \cdot \left(\frac{\bar{F}_{\text{corners}}}{\bar{F}_{\text{center}}}\right) + 0.30 \cdot \left(\frac{\text{mean}(|\Delta_{(0,0)-(1,1)} - \Delta_{(0,1)-(1,0)}|)}{\sigma_k}\right) + 0.30 \cdot \text{GridDiscontinuity}_k$$

6. **Geometric Distortion & Head Tilt Imbalance Penalty ($D_k$):**
   $$D_k = 0.50 \cdot \min(1.0, \max(0.0, (\text{SymmetryDiff}_k - 0.15) \cdot 4.0)) + 0.50 \cdot \min(1.0, \max(0.0, (\text{DiagImbalance}_k - 0.20) \cdot 3.5))$$

7. **Static Template Mismatch Penalty ($\text{Penalty}_{\text{template}}$):**
   $$\text{Penalty}_{\text{template}}(k) = \begin{cases} 0.50 & \text{if } k = \text{model\_4\_pca\_ridge} \text{ and } I_k < 0.45 \\ 0.00 & \text{otherwise} \end{cases}$$

---

#### Temperature-Scaled Softmax Confidence & Winner Selection
$$P(m_k) = \frac{\exp(\text{CompositeScore}(k) / \tau)}{\sum_{j=1}^5 \exp(\text{CompositeScore}(j) / \tau)}, \quad \text{where } \tau = 0.15$$

$$m^* = \arg\max_{k \in \{1 \dots 5\}} \text{CompositeScore}(k)$$

---

### 3. GFPGAN VRAM OPTIMIZATION & MEMORY CLEARING SEQUENCE

- **Code File:** [`inference/pipeline.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/forensic-face-3d/inference/pipeline.py) (Lines 139–180)
- **Target Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GB VRAM).

#### Step-by-Step Execution Protocol:

```python
# 1. Generate all 5 candidate 2D face previews on GPU memory
for key in MODEL_KEYS:
    gen_tensor = self.models[key].generate(sketch_tensor)
    generated_pil_dict[key] = tensor_to_pil(gen_tensor)

# 2. Perform Adaptive Model Selection
selection_result = self.selector.predict_suitability(sketch_resized, generated_pil_dict)
selected_key = selection_result["selected_model_key"]
raw_best_photo = generated_pil_dict[selected_key]

# 3. STRICT 4GB VRAM CLEARING SEQUENCE (BEFORE loading GFPGANer)
del sketch_tensor
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()

# 4. Instantiate GFPGANer in newly freed VRAM
restorer = GFPGANer(
    model_path="checkpoints/GFPGANv1.3.pth",
    upscale=1,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None,
    device=self.device
)

# 5. RGB-to-BGR Channel Transformation (OpenCV Contract)
raw_best_np = np.array(raw_best_photo.convert("RGB"))
raw_best_bgr = cv2.cvtColor(raw_best_np, cv2.COLOR_RGB2BGR)

# 6. Execute Neural AI Face Restoration
_, _, enhanced_bgr = restorer.enhance(
    raw_best_bgr,
    has_aligned=False,
    only_center_face=False,
    paste_back=True
)

# 7. BGR-to-RGB Channel Transformation (Prevents Blue Halo Artifacts)
enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
best_2d_photo = Image.fromarray(enhanced_rgb)

# 8. Post-Enhancement VRAM Cleanup
del restorer
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
```

---

### 4. PARAMETRIC 3D SURFACE MESH RECONSTRUCTION (3DMM DEPTH MAPS)

- **Code File:** [`face_3d/mesh_generator.py`](file:///C:/Users/Admin/.gemini/antigravity/scratch/forensic-face-3d/face_3d/mesh_generator.py) (Lines 56–138)

#### Analytical Depth Map Equation $Z(x, y)$:
$$Z(x, y) = Z_{\text{base}}(x,y) + Z_{\text{nose}}(x,y) + Z_{\text{eyes}}(x,y) + Z_{\text{chin}}(x,y)$$

where:
- **Base Anatomical Ellipsoid Depth:**
  $$Z_{\text{base}}(x,y) = \sqrt{\max\left(0, 1.0 - 0.75 \cdot \left(x_{\text{norm}}^2 + y_{\text{norm}}^2\right)\right)} \cdot 0.45$$
- **Nose Ridge Prominence Peak (+Z):**
  $$Z_{\text{nose}}(x,y) = 0.18 \cdot \exp\left(-\frac{d_{\text{nose}}^2}{0.02}\right)$$
- **Eye Socket Depressions (-Z):**
  $$Z_{\text{eyes}}(x,y) = -0.08 \cdot \left[ \exp\left(-\frac{d_{\text{left\_eye}}^2}{0.01}\right) + \exp\left(-\frac{d_{\text{right\_eye}}^2}{0.01}\right) \right]$$
- **Cheekbones & Chin Elevation (+Z):**
  $$Z_{\text{chin}}(x,y) = 0.06 \cdot \exp\left(-\frac{d_{\text{chin}}^2}{0.015}\right)$$

#### Mesh Resolution & Exporters:
- **Grid Geometry:** $48 \times 48$ spatial resolution plane geometry ($1.6 \times 2.0$ spatial units).
- **Vertex & Face Count:** 2,401 3D vertices, 4,608 triangular mesh faces.
- **UV Texture Mapping:** Coordinates mapped directly to GFPGAN-restored RGB image pixels.
- **Export Formats:** Wavefront `.OBJ`, Stanford `.PLY`, and Binary glTF `.GLB`.
