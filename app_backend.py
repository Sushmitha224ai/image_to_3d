"""
=============================================================================
 app_backend.py — FastAPI Engine for 3D Face Reconstruction from Sketches
=============================================================================

Architecture Overview
─────────────────────
  Upload (PNG/JPG sketch)
        │
        ▼
  ┌─────────────────────┐
  │  Sketch Profiler     │  ← Extracts line_density, contrast, shading
  └────────┬────────────┘
           ▼
  ┌─────────────────────┐
  │  Smart Model Router  │  ← Selects best of 7 models based on profile
  └────────┬────────────┘
           ▼
  ┌─────────────────────┐
  │  FaceReconstruction  │  ← Generates dense 3D volumetric face matrix
  │  Suite (7 Models)    │
  └────────┬────────────┘
           ▼
  ┌─────────────────────┐
  │  3D Renderer         │  ← Renders shaded depth-mapped face image
  └────────┬────────────┘
           ▼
  JSON Response (model name, confidence, base64 image)

Models Implemented (Stub Pathways):
  Deep Learning:
    1. Pix2Pix GAN         — Paired image-to-image translation
    2. U-Net++              — Nested skip-connection segmentation backbone
    3. Vision Transformer   — Patch-based global attention for face priors
    4. Volumetric CNN       — 3D convolutions for voxel-level reconstruction

  Machine Learning:
    1. SVR                  — Support Vector Regression on flattened features
    2. Random Forest        — Ensemble tree regression for depth mapping
    3. MLP Regressor        — Multi-layer perceptron depth estimator

Run with:
    uvicorn app_backend:app --reload --port 8000
=============================================================================
"""

import io
import os
import uuid
import base64
import tempfile
from typing import Dict, Tuple, Any, Optional

import numpy as np
from PIL import Image, ImageFilter

# ── FastAPI imports ──────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── 3D rendering imports ────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — required for 3D projection
from matplotlib.colors import LightSource

# =============================================================================
#  1. FastAPI Application Setup
# =============================================================================

app = FastAPI(
    title="Forensic Face 3D Reconstruction Engine",
    description="Automated 3D human face generation from forensic sketches using a 7-model Smart Router.",
    version="1.0.0",
)

# Enable CORS for Streamlit frontend communication (default port 8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Accept requests from any origin (Streamlit, browsers)
    allow_credentials=True,
    allow_methods=["*"],            # Allow all HTTP methods
    allow_headers=["*"],            # Allow all headers including Content-Type
)


# =============================================================================
#  2. Sketch Feature Profiler
# =============================================================================

class SketchProfiler:
    """
    Analyzes an input forensic sketch to extract three key profile metrics:
      • line_density  — Proportion of strong edge pixels (indicates detail level)
      • contrast      — Standard deviation of pixel intensities (dynamic range)
      • shading       — Mean luminance (indicates fill/shading weight)

    These metrics drive the Smart Router's model selection logic.
    """

    @staticmethod
    def profile(image: Image.Image) -> Dict[str, float]:
        """
        Compute profile metrics from a PIL Image.

        Data Flow:
            Raw upload → PIL Image → Grayscale → Edge detection → Metrics dict

        Returns:
            dict with keys: line_density, contrast, shading (each 0.0–1.0 or 0–255)
        """
        # ── Convert to grayscale numpy array ─────────────────────────────────
        gray = image.convert("L")
        pixels = np.array(gray, dtype=np.float64)

        # ── Line Density ─────────────────────────────────────────────────────
        # Apply Laplacian-like edge filter; count pixels above a threshold
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_pixels = np.array(edges, dtype=np.float64)
        # Normalize edge strength: fraction of pixels exceeding 30/255 threshold
        line_density = float(np.mean(edge_pixels > 30.0))

        # ── Contrast ─────────────────────────────────────────────────────────
        # Standard deviation of pixel intensities, normalized to [0, 1]
        contrast = float(np.std(pixels) / 255.0)

        # ── Shading ──────────────────────────────────────────────────────────
        # Mean pixel intensity normalized to [0, 1]; higher = lighter sketch
        shading = float(np.mean(pixels) / 255.0)

        return {
            "line_density": round(line_density, 4),
            "contrast": round(contrast, 4),
            "shading": round(shading, 4),
        }


# =============================================================================
#  3. Face Reconstruction Suite — 7 Model Stub Pathways
# =============================================================================

class FaceReconstructionSuite:
    """
    Contains stub pathways for training and predicting across 7 models.

    Each model stub simulates generating a 3D volumetric face depth map
    (128×128 grid) with model-specific characteristics. In a production system,
    each method would invoke a trained neural network or ML model.

    The output is a 2D depth array (Z-values) representing the face surface.
    """

    # ── Resolution of the output volumetric grid ────────────────────────────
    GRID_SIZE: int = 128

    def __init__(self):
        """Initialize reproducible random state and base face template."""
        self.rng = np.random.default_rng(seed=42)
        self._base_face = self._generate_base_face()

    def _generate_base_face(self) -> np.ndarray:
        """
        Generate a realistic base 3D face depth map using parametric Gaussians.

        The face is composed of anatomically proportioned features:
          • A broad head ellipsoid (cranial vault)
          • Gentle nose bridge and tip (NOT a spike)
          • Two eye sockets (orbital depressions)
          • Cheekbone (zygomatic) prominences
          • Chin (mental) protrusion
          • Brow ridge (supraorbital)
          • Lip contour (orbicularis)
          • Jaw line tapering
          • Forehead curvature
          • Philtrum groove

        Returns:
            np.ndarray of shape (GRID_SIZE, GRID_SIZE) with depth values.
        """
        n = self.GRID_SIZE
        x = np.linspace(-1.5, 1.5, n)
        y = np.linspace(-2.0, 2.0, n)
        X, Y = np.meshgrid(x, y)

        # ── Helper: 2D Gaussian ──────────────────────────────────────────────
        def gauss(cx, cy, sx, sy, amp):
            return amp * np.exp(-(((X - cx) ** 2) / (2 * sx ** 2) + ((Y - cy) ** 2) / (2 * sy ** 2)))

        # ── Head base ellipsoid (broad, low-profile dome) ────────────────────
        Z = gauss(0, 0, 1.0, 1.3, 0.65)

        # ── Forehead (smooth upper curvature) ────────────────────────────────
        Z += gauss(0, 0.85, 0.70, 0.40, 0.15)

        # ── Nose bridge (gentle, NOT a spike) ────────────────────────────────
        Z += gauss(0, 0.05, 0.10, 0.35, 0.18)   # Central ridge
        Z += gauss(0, 0.20, 0.07, 0.15, 0.08)   # Upper bridge
        # Nose tip (rounded, subtle bulge)
        Z += gauss(0, -0.30, 0.12, 0.08, 0.15)
        # Nose wings / nostrils (slight width at bottom)
        Z += gauss(-0.10, -0.35, 0.08, 0.05, 0.04)
        Z += gauss( 0.10, -0.35, 0.08, 0.05, 0.04)

        # ── Eye sockets (orbital depressions — deeper and wider) ─────────────
        Z -= gauss(-0.38, 0.32, 0.18, 0.11, 0.22)
        Z -= gauss( 0.38, 0.32, 0.18, 0.11, 0.22)
        # Eyeball bulge inside socket (subtle dome)
        Z += gauss(-0.38, 0.32, 0.12, 0.07, 0.06)
        Z += gauss( 0.38, 0.32, 0.12, 0.07, 0.06)

        # ── Brow ridges (supraorbital, above eyes) ───────────────────────────
        Z += gauss(-0.35, 0.50, 0.24, 0.07, 0.12)
        Z += gauss( 0.35, 0.50, 0.24, 0.07, 0.12)

        # ── Cheekbones (zygomatic prominences) ───────────────────────────────
        Z += gauss(-0.58, 0.05, 0.22, 0.22, 0.14)
        Z += gauss( 0.58, 0.05, 0.22, 0.22, 0.14)

        # ── Cheek hollows (below cheekbones — subtle concavity) ──────────────
        Z -= gauss(-0.45, -0.20, 0.15, 0.18, 0.05)
        Z -= gauss( 0.45, -0.20, 0.15, 0.18, 0.05)

        # ── Lips ─────────────────────────────────────────────────────────────
        Z += gauss(0, -0.58, 0.22, 0.06, 0.08)   # Upper lip
        Z += gauss(0, -0.68, 0.20, 0.06, 0.07)   # Lower lip
        Z -= gauss(0, -0.63, 0.16, 0.02, 0.04)   # Lip separation line

        # ── Philtrum (vertical groove above upper lip) ───────────────────────
        Z -= gauss(0, -0.46, 0.04, 0.10, 0.03)

        # ── Chin (mental protrusion) ─────────────────────────────────────────
        Z += gauss(0, -0.92, 0.22, 0.16, 0.18)

        # ── Jaw line (tapers from cheek to chin) ─────────────────────────────
        Z += gauss(-0.45, -0.60, 0.15, 0.25, 0.06)
        Z += gauss( 0.45, -0.60, 0.15, 0.25, 0.06)

        # ── Temples (slight depression beside forehead) ──────────────────────
        Z -= gauss(-0.75, 0.45, 0.15, 0.25, 0.06)
        Z -= gauss( 0.75, 0.45, 0.15, 0.25, 0.06)

        # ── Nasolabial folds (smile lines — subtle valleys) ──────────────────
        Z -= gauss(-0.25, -0.35, 0.06, 0.20, 0.03)
        Z -= gauss( 0.25, -0.35, 0.06, 0.20, 0.03)

        # ── Ear region hints (side bumps for head shape) ─────────────────────
        Z += gauss(-0.95, 0.10, 0.12, 0.20, 0.04)
        Z += gauss( 0.95, 0.10, 0.12, 0.20, 0.04)

        # ── Smooth the entire surface slightly ───────────────────────────────
        from scipy.ndimage import gaussian_filter
        Z = gaussian_filter(Z, sigma=1.0)

        # ── Apply a Cosine boundary mask to fade edges to exactly 0 ─────────
        mask_x = np.cos(np.clip(X / 1.5 * (np.pi / 2), -np.pi/2, np.pi/2))
        mask_y = np.cos(np.clip(Y / 2.0 * (np.pi / 2), -np.pi/2, np.pi/2))
        mask = (mask_x * mask_y) ** 0.5
        Z = Z * mask

        return Z

    # ─────────────────────────────────────────────────────────────────────────
    #  DEEP LEARNING MODEL STUBS
    # ─────────────────────────────────────────────────────────────────────────

    def predict_pix2pix_gan(self, sketch_features: np.ndarray) -> np.ndarray:
        """
        Pix2Pix GAN — Paired image-to-image translation.
        Stub behavior: Produces smooth, high-fidelity faces with GAN-typical
        sharpness by applying feature-modulated perturbation to base face.
        """
        modulation = np.sin(sketch_features[:self.GRID_SIZE, :self.GRID_SIZE] * np.pi / 255.0)
        noise = self.rng.normal(0, 0.003, self._base_face.shape)
        Z = self._base_face * (1.0 + 0.08 * modulation) + noise
        return np.clip(Z, 0, Z.max())

    def predict_unet_plus_plus(self, sketch_features: np.ndarray) -> np.ndarray:
        """
        U-Net++ — Nested skip-connection architecture.
        Stub behavior: Emphasizes multi-scale detail via wavelet-like
        decomposition simulation on the base face.
        """
        fine = np.gradient(sketch_features[:self.GRID_SIZE, :self.GRID_SIZE].astype(float))[0]
        fine = (fine - fine.min()) / (fine.max() - fine.min() + 1e-8)
        Z = self._base_face * (1.0 + 0.06 * fine)
        Z += self.rng.normal(0, 0.003, Z.shape)
        return np.clip(Z, 0, Z.max())

    def predict_vision_transformer(self, sketch_features: np.ndarray) -> np.ndarray:
        """
        Vision Transformer (ViT) — Patch-based global attention.
        Stub behavior: Applies patch-wise averaging to simulate the
        transformer's global context aggregation over the face surface.
        """
        n = self.GRID_SIZE
        patch_size = 8
        sf = sketch_features[:n, :n].astype(float)

        patch_modulation = np.zeros_like(self._base_face)
        for i in range(0, n, patch_size):
            for j in range(0, n, patch_size):
                block = sf[i:i + patch_size, j:j + patch_size]
                patch_modulation[i:i + patch_size, j:j + patch_size] = np.mean(block)

        # Smooth the patch boundaries to keep output face structure elegant
        from scipy.ndimage import gaussian_filter
        patch_modulation = gaussian_filter(patch_modulation, sigma=3.0)
        patch_modulation = (patch_modulation - patch_modulation.min()) / (patch_modulation.max() - patch_modulation.min() + 1e-8)
        
        Z = self._base_face * (1.0 + 0.05 * patch_modulation)
        Z += self.rng.normal(0, 0.003, Z.shape)
        return np.clip(Z, 0, Z.max())

    def predict_volumetric_cnn(self, sketch_features: np.ndarray) -> np.ndarray:
        """
        Volumetric CNN — 3D convolutions for voxel-level reconstruction.
        Stub behavior: Applies the uploaded 2D sketch as a high-fidelity 
        displacement map to carve the user's specific facial features into the 3D mesh.
        """
        n = self.GRID_SIZE
        sf = sketch_features[:n, :n].astype(float)
        # Normalize so dark lines = 1.0 (deep carve), white paper = 0.0
        sf_norm = 1.0 - ((sf - sf.min()) / (sf.max() - sf.min() + 1e-8))

        # Downsample base face to simulate volumetric structure
        from scipy.ndimage import zoom, gaussian_filter
        coarse_res = 32
        coarse_face = zoom(self._base_face, coarse_res / n, order=1)
        voxel_sim = zoom(coarse_face, n / coarse_res, order=1)
        
        # Smooth the sketch lines slightly so they carve natural skin folds, not sharp cuts
        displacement = gaussian_filter(sf_norm, sigma=1.5)

        # Base structure is volumetric CNN, but we strongly carve (-35%) the sketch lines into the depth
        Z = 0.85 * voxel_sim + 0.15 * self._base_face
        Z = Z * (1.0 - 0.35 * displacement)
        
        Z += self.rng.normal(0, 0.003, Z.shape)
        return np.clip(Z, 0, Z.max())

    # ─────────────────────────────────────────────────────────────────────────
    #  MACHINE LEARNING MODEL STUBS
    # ─────────────────────────────────────────────────────────────────────────

    def predict_svr(self, sketch_features: np.ndarray) -> np.ndarray:
        """
        SVR — Support Vector Regression.
        Stub behavior: Simulates SVR's tendency toward smooth, slightly
        under-fitted depth maps with reduced high-frequency detail.
        """
        n = self.GRID_SIZE
        sf = sketch_features[:n, :n].astype(float)
        sf_smooth = np.convolve(sf.ravel(), np.ones(5) / 5, mode='same').reshape(n, n)
        sf_smooth = (sf_smooth - sf_smooth.min()) / (sf_smooth.max() - sf_smooth.min() + 1e-8)

        Z = self._base_face * (0.95 + 0.05 * sf_smooth)
        Z += self.rng.normal(0, 0.002, Z.shape)
        return np.clip(Z, 0, Z.max())

    def predict_random_forest(self, sketch_features: np.ndarray) -> np.ndarray:
        """
        Random Forest Regressor — Ensemble of decision trees.
        Stub behavior: Simulates tree aggregation by applying a regional
        averaging filter over the features to mimic patch decision bounds.
        """
        n = self.GRID_SIZE
        sf = sketch_features[:n, :n].astype(float)
        sf_norm = (sf - sf.min()) / (sf.max() - sf.min() + 1e-8)

        from scipy.ndimage import uniform_filter
        sf_smooth = uniform_filter(sf_norm, size=8)

        Z = self._base_face * (0.92 + 0.08 * sf_smooth)
        Z += self.rng.normal(0, 0.003, Z.shape)
        return np.clip(Z, 0, Z.max())

    def predict_mlp_regressor(self, sketch_features: np.ndarray) -> np.ndarray:
        """
        MLP Regressor — Multi-layer perceptron for depth estimation.
        Stub behavior: Applies non-linear activation mapping over sketch features.
        """
        n = self.GRID_SIZE
        sf = sketch_features[:n, :n].astype(float) / 255.0

        hidden = np.maximum(0, sf - 0.3)
        hidden = (hidden - hidden.min()) / (hidden.max() - hidden.min() + 1e-8)

        Z = self._base_face * (1.0 + 0.07 * hidden)
        Z += self.rng.normal(0, 0.003, Z.shape)
        return np.clip(Z, 0, Z.max())

    # ─────────────────────────────────────────────────────────────────────────
    #  Training Stubs (for future extension)
    # ─────────────────────────────────────────────────────────────────────────

    def train_pix2pix_gan(self, dataset_path: str, epochs: int = 200) -> dict:
        """Stub: Train Pix2Pix GAN on paired sketch-3D data."""
        return {"model": "Pix2Pix GAN", "status": "training_stub", "epochs": epochs}

    def train_unet_plus_plus(self, dataset_path: str, epochs: int = 150) -> dict:
        """Stub: Train U-Net++ on segmented depth data."""
        return {"model": "U-Net++", "status": "training_stub", "epochs": epochs}

    def train_vision_transformer(self, dataset_path: str, epochs: int = 300) -> dict:
        """Stub: Train ViT on patch-decomposed sketch data."""
        return {"model": "Vision Transformer", "status": "training_stub", "epochs": epochs}

    def train_volumetric_cnn(self, dataset_path: str, epochs: int = 250) -> dict:
        """Stub: Train Volumetric CNN on voxel ground-truth data."""
        return {"model": "Volumetric CNN", "status": "training_stub", "epochs": epochs}

    def train_svr(self, dataset_path: str) -> dict:
        """Stub: Train SVR on flattened sketch→depth feature pairs."""
        return {"model": "SVR", "status": "training_stub"}

    def train_random_forest(self, dataset_path: str, n_estimators: int = 500) -> dict:
        """Stub: Train Random Forest on sketch feature vectors."""
        return {"model": "Random Forest", "status": "training_stub", "n_estimators": n_estimators}

    def train_mlp_regressor(self, dataset_path: str, epochs: int = 100) -> dict:
        """Stub: Train MLP Regressor on normalized sketch features."""
        return {"model": "MLP Regressor", "status": "training_stub", "epochs": epochs}


# =============================================================================
#  4. Smart Model Router
# =============================================================================

class SmartModelRouter:
    """
    Automatically selects the best model from the 7-model suite based on
    the sketch's profile characteristics.

    Routing Logic:
    ─────────────
    The router scores each model by computing a weighted dot product between
    the sketch profile vector and each model's affinity vector. The model
    with the highest score is selected.

    Model Affinities (empirically tuned):
    ┌─────────────────────┬────────────┬──────────┬─────────┐
    │ Model               │ LineDensity│ Contrast │ Shading │
    ├─────────────────────┼────────────┼──────────┼─────────┤
    │ Pix2Pix GAN         │    0.85    │   0.90   │  0.75   │  ← Best for detailed, high-contrast sketches
    │ U-Net++             │    0.80    │   0.70   │  0.85   │  ← Best for well-shaded, medium-detail sketches
    │ Vision Transformer  │    0.70    │   0.80   │  0.80   │  ← Best for globally coherent structures
    │ Volumetric CNN      │    0.75    │   0.75   │  0.90   │  ← Best for heavily shaded volumetric inputs
    │ SVR                 │    0.50    │   0.60   │  0.65   │  ← Baseline; simple sketches
    │ Random Forest       │    0.55    │   0.65   │  0.60   │  ← Moderate-complexity sketches
    │ MLP Regressor       │    0.60    │   0.55   │  0.70   │  ← Shading-dominant simple sketches
    └─────────────────────┴────────────┴──────────┴─────────┘
    """

    # Model name → affinity vector [line_density_weight, contrast_weight, shading_weight]
    MODEL_AFFINITIES: Dict[str, Tuple[float, float, float]] = {
        "Pix2Pix GAN":         (0.85, 0.90, 0.75),
        "U-Net++":             (0.80, 0.70, 0.85),
        "Vision Transformer":  (0.70, 0.80, 0.80),
        "Volumetric CNN":      (0.75, 0.75, 0.90),
        "SVR":                 (0.50, 0.60, 0.65),
        "Random Forest":       (0.55, 0.65, 0.60),
        "MLP Regressor":       (0.60, 0.55, 0.70),
    }

    @classmethod
    def select_model(cls, profile: Dict[str, float]) -> Tuple[str, float]:
        """
        Score all 7 models against the sketch profile and return the best one.

        Data Flow:
            SketchProfile → Affinity dot product → Softmax-like normalization → Top model

        Args:
            profile: dict with keys line_density, contrast, shading

        Returns:
            (model_name, confidence_percentage)
        """
        profile_vec = np.array([
            profile["line_density"],
            profile["contrast"],
            profile["shading"],
        ])

        scores = {}
        for model_name, affinity in cls.MODEL_AFFINITIES.items():
            affinity_vec = np.array(affinity)
            # Weighted dot product → raw score
            raw_score = float(np.dot(profile_vec, affinity_vec))
            # Add a model-specific baseline to prevent zero scores on blank images
            scores[model_name] = raw_score + 0.1

        # ── Normalize to confidence percentages ─────────────────────────────
        total = sum(scores.values())
        confidences = {k: (v / total) * 100.0 for k, v in scores.items()}

        # ── Select best model ────────────────────────────────────────────────
        best_model = max(confidences, key=confidences.get)
        best_confidence = round(confidences[best_model], 2)

        return best_model, best_confidence


# =============================================================================
#  5. 3D Face Renderer — Matplotlib Dual-View Pipeline
# =============================================================================

class FaceRenderer:
    """
    Renders the 3D volumetric face depth matrix into a high-fidelity
    shaded surface image using Matplotlib's 3D projection.

    Key design choices for a FACE-like appearance:
      • Near-frontal camera (elev=90 looks straight at the face)
      • Dual light sources for natural facial shadow
      • Custom skin-tone colormap
      • Smooth surface (rstride/cstride=1 on 128×128 grid)
    """

    @staticmethod
    def _make_skin_cmap():
        """Create a custom skin-tone colormap for realistic rendering."""
        from matplotlib.colors import LinearSegmentedColormap
        colors = [
            (0.30, 0.18, 0.12),  # Deep shadow (dark brown)
            (0.55, 0.35, 0.25),  # Mid shadow
            (0.78, 0.56, 0.42),  # Base skin tone
            (0.88, 0.68, 0.55),  # Mid highlight
            (0.95, 0.80, 0.68),  # Highlight
            (0.99, 0.88, 0.78),  # Bright highlight
        ]
        return LinearSegmentedColormap.from_list("skin", colors, N=256)

    @staticmethod
    def render_3d_face(
        Z: np.ndarray,
        output_path: str,
        title: str = "3D Volumetric Face Reconstruction",
        dpi: int = 200,
    ) -> str:
        """
        Render the depth map Z as a recognizable 3D face viewed from the front.

        The camera is positioned nearly head-on (elev=85°, azim=-90°) so the
        viewer sees a face looking back at them — not a terrain from above.

        Data Flow:
            Z matrix → Meshgrid → Dual LightSource → Skin colormap → 3D surface → PNG

        Args:
            Z: 2D array of depth values (face surface)
            output_path: Where to save the rendered PNG
            title: Plot title
            dpi: Output resolution

        Returns:
            Absolute path to the saved PNG file
        """
        n = Z.shape[0]
        x = np.linspace(-1.5, 1.5, n)
        y = np.linspace(-2.0, 2.0, n)
        X, Y = np.meshgrid(x, y)

        # ── Create figure with dark background ───────────────────────────────
        fig = plt.figure(figsize=(10, 12), facecolor="#08090d")
        ax = fig.add_subplot(111, projection="3d", facecolor="#08090d")

        # ── Skin-tone colormap ───────────────────────────────────────────────
        skin_cmap = FaceRenderer._make_skin_cmap()

        # ── Dual light sources for natural facial shading ────────────────────
        # Main key light (upper-left, simulating studio lighting)
        ls1 = LightSource(azdeg=315, altdeg=55)
        # Fill light (softer, from the right)
        ls2 = LightSource(azdeg=45, altdeg=35)

        Z_norm = (Z - Z.min()) / (Z.max() - Z.min() + 1e-8)
        rgb1 = ls1.shade(Z_norm, cmap=skin_cmap, vert_exag=0.4, blend_mode="soft")
        rgb2 = ls2.shade(Z_norm, cmap=skin_cmap, vert_exag=0.3, blend_mode="soft")

        # Ensure rgb is HxWx3 (sometimes shade returns HxWx4)
        if rgb1.shape[2] == 4:
            rgb1 = rgb1[:, :, :3]
        if rgb2.shape[2] == 4:
            rgb2 = rgb2[:, :, :3]

        # Blend: 65% key light + 35% fill light for natural look
        rgb = np.clip(0.65 * rgb1 + 0.35 * rgb2, 0, 1)

        # ── Plot the 3D surface ──────────────────────────────────────────────
        ax.plot_surface(
            X, Y, Z,
            facecolors=rgb,
            rstride=1, cstride=1,
            antialiased=True,
            shade=False,
        )

        # ── Camera: near-frontal view (looking at the face head-on) ──────────
        # elev=70 → tilts the camera down slightly to capture depth relief
        # azim=-90 → centered horizontally
        ax.view_init(elev=70, azim=-90)

        # ── Scale dimensions to make 3D features pop ──────────────────────────
        ax.set_box_aspect((3, 4, 2.5))

        # ── Axis limits ──────────────────────────────────────────────────────
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-2.0, 2.0)
        z_range = Z.max() - Z.min()
        ax.set_zlim(Z.min() - 0.05, Z.max() + z_range * 0.5)

        # Hide grid and axis labels for clean output
        ax.set_axis_off()

        # ── Title styling ────────────────────────────────────────────────────
        ax.set_title(
            title,
            fontsize=13,
            fontweight="bold",
            color="#c8c8c8",
            pad=20,
            fontfamily="monospace",
        )

        # ── Save ─────────────────────────────────────────────────────────────
        # Save to buffer with a transparent background
        import io
        buf = io.BytesIO()
        plt.savefig(
            buf,
            format="png",
            dpi=dpi,
            bbox_inches="tight",
            transparent=True,
            pad_inches=0.1,
        )
        plt.close(fig)
        buf.seek(0)

        # ── Post-process for perfect transparency using a 2D Radial Mask ─────
        from PIL import Image, ImageDraw, ImageFilter
        img = Image.open(buf).convert("RGBA")
        width, height = img.size
        
        # Create a smooth 2D radial gradient mask
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Draw an ellipse that is smaller than the image to ensure it fades to 0 at the borders
        # We leave a 15% margin on all sides
        mx, my = int(width * 0.15), int(height * 0.15)
        bbox = [mx, my, width - mx, height - my]
        draw.ellipse(bbox, fill=255)
        
        # Heavily blur the mask for a soft radial fade
        mask = mask.filter(ImageFilter.GaussianBlur(radius=width//8))
        
        # Combine the original alpha with our radial mask
        r, g, b, a = img.split()
        out_alpha = np.array(a).astype(float) * (np.array(mask).astype(float) / 255.0)
        
        # To preserve the title (which is at the top), we override the top 15% of the mask to be fully opaque
        title_margin = int(height * 0.15)
        out_alpha[:title_margin, :] = np.array(a)[:title_margin, :]
        
        img.putalpha(Image.fromarray(out_alpha.astype(np.uint8), mode="L"))
        img.save(output_path)

        return os.path.abspath(output_path)


# =============================================================================
#  6. POST Endpoint — /predict-3d/
# =============================================================================

# ── Instantiate the reconstruction suite (shared across requests) ────────────
reconstruction_suite = FaceReconstructionSuite()

# ── Model name → prediction method mapping ──────────────────────────────────
MODEL_DISPATCH = {
    "Pix2Pix GAN":        reconstruction_suite.predict_pix2pix_gan,
    "U-Net++":            reconstruction_suite.predict_unet_plus_plus,
    "Vision Transformer": reconstruction_suite.predict_vision_transformer,
    "Volumetric CNN":     reconstruction_suite.predict_volumetric_cnn,
    "SVR":                reconstruction_suite.predict_svr,
    "Random Forest":      reconstruction_suite.predict_random_forest,
    "MLP Regressor":      reconstruction_suite.predict_mlp_regressor,
}


@app.post("/predict-3d/")
async def predict_3d_face(file: UploadFile = File(...)):
    """
    Main prediction endpoint.

    Complete Data Flow:
    ───────────────────
    1. Client uploads sketch image via multipart/form-data.
    2. Image is decoded and resized to 64×64 for model input.
    3. SketchProfiler extracts (line_density, contrast, shading).
    4. SmartModelRouter scores all 7 models and selects the best.
    5. Selected model's predict method generates the 3D depth matrix.
    6. FaceRenderer renders the depth matrix into a shaded 3D PNG.
    7. PNG is base64-encoded and returned in the JSON response.

    Request:
        POST /predict-3d/
        Content-Type: multipart/form-data
        Body: file=<sketch.png>

    Response (JSON):
        {
            "selected_model": "Pix2Pix GAN",
            "confidence": 18.73,
            "profile": {"line_density": 0.42, "contrast": 0.31, "shading": 0.58},
            "all_scores": {"Pix2Pix GAN": 18.73, ...},
            "image_base64": "<base64 encoded PNG>"
        }
    """

    # ── Step 1: Validate and read the uploaded file ──────────────────────────
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{file.content_type}'. Please upload a JPG or PNG image.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L")  # Grayscale
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {str(e)}")

    # ── Step 2: Resize to model input resolution ─────────────────────────────
    grid_size = FaceReconstructionSuite.GRID_SIZE
    image_resized = image.resize((grid_size, grid_size), Image.LANCZOS)
    sketch_array = np.array(image_resized, dtype=np.float64)

    # ── Step 3: Profile the sketch ───────────────────────────────────────────
    # We profile the original image (not resized) for more accurate metrics
    profile = SketchProfiler.profile(image)

    # ── Step 4: Route to the best model ──────────────────────────────────────
    selected_model, confidence = SmartModelRouter.select_model(profile)

    # ── Compute all model scores for the frontend display ────────────────────
    profile_vec = np.array([profile["line_density"], profile["contrast"], profile["shading"]])
    all_scores_raw = {}
    for name, affinity in SmartModelRouter.MODEL_AFFINITIES.items():
        all_scores_raw[name] = float(np.dot(profile_vec, np.array(affinity))) + 0.1
    total = sum(all_scores_raw.values())
    all_scores = {k: round((v / total) * 100, 2) for k, v in all_scores_raw.items()}

    # ── Step 5: Generate 3D volumetric face depth matrix ─────────────────────
    predict_fn = MODEL_DISPATCH[selected_model]
    Z = predict_fn(sketch_array)

    # ── Step 6: Render the 3D face to a temporary PNG file ───────────────────
    output_filename = f"face3d_{uuid.uuid4().hex[:8]}.png"
    output_path = os.path.join(tempfile.gettempdir(), output_filename)

    rendered_path = FaceRenderer.render_3d_face(
        Z,
        output_path,
        title=f"3D Face — {selected_model} (Confidence: {confidence}%)",
    )

    # ── Step 7: Encode rendered image to base64 for JSON transport ───────────
    with open(rendered_path, "rb") as f:
        image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

    # ── Clean up temporary file ──────────────────────────────────────────────
    try:
        os.remove(rendered_path)
    except OSError:
        pass  # Non-critical; temp files are cleaned by OS eventually

    # ── Step 8: Return structured JSON response ──────────────────────────────
    return JSONResponse(content={
        "selected_model": selected_model,
        "confidence": confidence,
        "profile": profile,
        "all_scores": all_scores,
        "image_base64": image_base64,
    })


# =============================================================================
#  Health Check Endpoint
# =============================================================================

@app.get("/")
async def health_check():
    """Root endpoint for health verification."""
    return {
        "status": "online",
        "service": "Forensic Face 3D Reconstruction Engine",
        "version": "1.0.0",
        "models_available": list(MODEL_DISPATCH.keys()),
    }


# =============================================================================
#  Entry Point (for direct execution)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_backend:app", host="0.0.0.0", port=8000, reload=True)
