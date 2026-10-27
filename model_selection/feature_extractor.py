import numpy as np
import cv2
from PIL import Image

class SketchFeatureExtractor:
    """
    Extracts structural, textural, and quality features from input forensic sketches.
    CRITICAL: Operates strictly on the sketch image WITHOUT accessing ground-truth photos.
    """
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def extract_features(self, sketch_img):
        """
        sketch_img: PIL Image, numpy array (H,W,3) or (H,W), or PyTorch tensor.
        Returns: 1D feature vector (numpy array) containing normalized sketch descriptors.
        """
        if isinstance(sketch_img, Image.Image):
            gray = np.array(sketch_img.convert("L"))
        elif isinstance(sketch_img, np.ndarray):
            if sketch_img.ndim == 3:
                gray = cv2.cvtColor(sketch_img, cv2.COLOR_RGB2GRAY)
            else:
                gray = sketch_img
        else:
            # Handle PyTorch Tensor (C, H, W) or (B, C, H, W)
            if hasattr(sketch_img, "detach"):
                arr = sketch_img.detach().cpu().numpy()
                if arr.ndim == 4:
                    arr = arr[0]
                if arr.shape[0] == 3 or arr.shape[0] == 1:
                    arr = arr.transpose(1, 2, 0)
                # Rescale from [-1, 1] or [0, 1] to [0, 255]
                if arr.min() < 0:
                    arr = (arr + 1.0) * 127.5
                elif arr.max() <= 1.0:
                    arr = arr * 255.0
                gray = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                raise ValueError("Unsupported image type for feature extraction.")

        gray = cv2.resize(gray, self.target_size)

        # 1. Sharpness (Laplacian Variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = np.log1p(laplacian_var) / 10.0

        # 2. Edge Density (Sobel Gradient Magnitude)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_density = np.mean(edge_mag > 30.0)
        mean_edge_intensity = np.mean(edge_mag) / 255.0

        # 3. RMS Contrast & Brightness
        mean_intensity = np.mean(gray) / 255.0
        std_contrast = np.std(gray) / 255.0

        # 4. Local Texture Complexity (Multi-grid local variance)
        h, w = gray.shape
        grid_h, grid_w = h // 4, w // 4
        grid_vars = []
        for i in range(4):
            for j in range(4):
                patch = gray[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                grid_vars.append(np.var(patch))
        texture_var = np.mean(grid_vars) / 1000.0
        texture_max_ratio = np.max(grid_vars) / (np.min(grid_vars) + 1e-5)

        # 5. Facial Symmetry & Structural Centroid Ratio
        cy, cx = np.mean(np.where(gray < 200), axis=1) if np.any(gray < 200) else (h/2, w/2)
        center_dev_x = abs(cx - w/2) / (w/2)
        center_dev_y = abs(cy - h/2) / (h/2)

        # 6. Low-resolution spatial embedding (Sub-sampled 4x4 intensity matrix)
        sub_grid = cv2.resize(gray, (4, 4)).flatten() / 255.0

        features = np.array([
            sharpness,
            edge_density,
            mean_edge_intensity,
            mean_intensity,
            std_contrast,
            texture_var,
            np.log1p(texture_max_ratio),
            center_dev_x,
            center_dev_y,
            *sub_grid
        ], dtype=np.float32)

        return features

    def extract_batch(self, sketch_images):
        return np.array([self.extract_features(img) for img in sketch_images])
