import os
import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

class PCARidgeModel:
    """Traditional ML Model 1: Eigenface Dual-PCA + Ridge Regression Reconstruction."""
    def __init__(self, n_components=40, alpha=10.0, img_shape=(256, 256, 3)):
        self.n_components = n_components
        self.alpha = alpha
        self.img_shape = img_shape
        self.pca_sketch = PCA(n_components=n_components, random_state=42)
        self.pca_photo = PCA(n_components=n_components, random_state=42)
        self.regressor = Ridge(alpha=alpha, random_state=42)
        self.is_fitted = False

    def fit(self, sketch_images, photo_images):
        """
        sketch_images: numpy array of shape (N, H, W, 3) normalized to [0, 1] or [-1, 1]
        photo_images: numpy array of shape (N, H, W, 3)
        """
        N = sketch_images.shape[0]
        sketches_flat = sketch_images.reshape(N, -1)
        photos_flat = photo_images.reshape(N, -1)

        print(f"[*] Fitting PCA-Sketch ({sketches_flat.shape}) & PCA-Photo ({photos_flat.shape})...")

        # Adjust components if dataset size is smaller
        actual_comp = min(self.n_components, N - 1)
        self.pca_sketch.n_components = actual_comp
        self.pca_photo.n_components = actual_comp

        sketch_coeffs = self.pca_sketch.fit_transform(sketches_flat)
        photo_coeffs = self.pca_photo.fit_transform(photos_flat)

        print(f"[*] Fitting Ridge Regression (alpha={self.alpha})...")
        self.regressor.fit(sketch_coeffs, photo_coeffs)
        self.is_fitted = True
        return self

    def predict_np(self, sketch_img):
        """
        sketch_img: numpy array of shape (H, W, 3) or (B, H, W, 3)
        Returns reconstructed photo numpy array
        """
        if not self.is_fitted:
            raise RuntimeError("PCARidgeModel must be fitted before calling predict.")

        if sketch_img.ndim == 3:
            sketch_img = np.expand_dims(sketch_img, axis=0)

        B = sketch_img.shape[0]
        sketches_flat = sketch_img.reshape(B, -1)
        
        sketch_coeffs = self.pca_sketch.transform(sketches_flat)
        photo_coeffs_pred = self.regressor.predict(sketch_coeffs)
        photos_flat_pred = self.pca_photo.inverse_transform(photo_coeffs_pred)

        photos_pred = photos_flat_pred.reshape(B, *self.img_shape)
        return np.clip(photos_pred, -1.0, 1.0)

    def generate(self, sketch_tensor):
        """
        PyTorch tensor interface compatible with PyTorch models.
        sketch_tensor: (B, 3, 256, 256) in range [-1, 1]
        Returns: PyTorch tensor (B, 3, 256, 256) in range [-1, 1]
        """
        # Convert PyTorch tensor (B, C, H, W) -> numpy (B, H, W, C)
        sketch_np = sketch_tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
        pred_np = self.predict_np(sketch_np)
        # Convert numpy (B, H, W, C) -> PyTorch tensor (B, C, H, W)
        pred_tensor = torch.from_numpy(pred_np.transpose(0, 3, 1, 2)).float().to(sketch_tensor.device)
        return pred_tensor

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "pca_sketch": self.pca_sketch,
            "pca_photo": self.pca_photo,
            "regressor": self.regressor,
            "img_shape": self.img_shape,
            "is_fitted": self.is_fitted
        }, filepath)

    def load(self, filepath):
        data = joblib.load(filepath)
        self.pca_sketch = data["pca_sketch"]
        self.pca_photo = data["pca_photo"]
        self.regressor = data["regressor"]
        self.img_shape = data["img_shape"]
        self.is_fitted = data["is_fitted"]
