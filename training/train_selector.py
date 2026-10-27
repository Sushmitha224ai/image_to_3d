import os
import sys
import glob
import yaml
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_1_pix2pix import Pix2PixModel
from models.model_2_unet import PerceptualUNetModel
from models.model_3_resnet_gan import ResNetLSGANModel
from models.model_4_pca_ridge import PCARidgeModel
from models.model_5_knn_patch import PatchKNNModel
from model_selection.meta_selector import AdaptiveModelSelector, MODEL_KEYS
from training.train_models import SketchPhotoDataset

def calculate_quality_score(gen_tensor, gt_tensor):
    """
    Computes metric quality score (combined SSIM + normalized PSNR) 
    between generated face and ground truth photo.
    """
    gen_np = ((gen_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    gt_np = ((gt_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

    score_ssim = ssim(gen_np, gt_np, channel_axis=2, data_range=255)
    score_psnr = psnr(gt_np, gen_np, data_range=255)
    
    # Combined score
    composite_score = 0.7 * score_ssim + 0.3 * (score_psnr / 40.0)
    return float(composite_score)

def train_adaptive_selector():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    checkpoints_dir = config["paths"]["checkpoints_dir"]
    val_dir = os.path.join(config["dataset"]["processed_dir"], "val")
    val_dataset = SketchPhotoDataset(val_dir)
    loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print("\n==========================================")
    print(" TRAINING ADAPTIVE META-MODEL SELECTOR")
    print("==========================================")

    # 1. Load Trained Candidate Models
    print("[*] Loading trained candidate models from checkpoints...")
    models = {}
    
    # Model 1
    m1 = Pix2PixModel(device="cuda")
    p1 = os.path.join(checkpoints_dir, "model_1_pix2pix_best.pth")
    if os.path.exists(p1): m1.load(p1)
    models["model_1_pix2pix"] = m1

    # Model 2
    m2 = PerceptualUNetModel(device="cuda")
    p2 = os.path.join(checkpoints_dir, "model_2_unet_best.pth")
    if os.path.exists(p2): m2.load(p2)
    models["model_2_unet"] = m2

    # Model 3
    m3 = ResNetLSGANModel(device="cuda")
    p3 = os.path.join(checkpoints_dir, "model_3_resnet_gan_best.pth")
    if os.path.exists(p3): m3.load(p3)
    models["model_3_resnet_gan"] = m3

    # Model 4
    m4 = PCARidgeModel()
    p4 = os.path.join(checkpoints_dir, "model_4_pca_ridge_best.pkl")
    if os.path.exists(p4): m4.load(p4)
    models["model_4_pca_ridge"] = m4

    # Model 5
    m5 = PatchKNNModel()
    p5 = os.path.join(checkpoints_dir, "model_5_knn_patch_best.pkl")
    if os.path.exists(p5): m5.load(p5)
    models["model_5_knn_patch"] = m5

    # 2. Extract Features & Evaluate Candidate Quality on Validation Set
    selector = AdaptiveModelSelector()
    X_features = []
    Y_scores = []

    print(f"[*] Evaluating candidate models across {len(val_dataset)} validation sketches...")

    for sketch_tensor, real_photo_tensor in loader:
        # Extract features strictly from sketch
        feat = selector.feature_extractor.extract_features(sketch_tensor)
        X_features.append(feat)

        sample_scores = []
        for key in MODEL_KEYS:
            model_obj = models[key]
            gen_tensor = model_obj.generate(sketch_tensor)
            score = calculate_quality_score(gen_tensor, real_photo_tensor)
            sample_scores.append(score)
        
        Y_scores.append(sample_scores)

    X_features = np.array(X_features)
    Y_scores = np.array(Y_scores)

    # 3. Fit Meta-Selector
    selector.fit(X_features, Y_scores)
    save_path = os.path.join(checkpoints_dir, "adaptive_selector_best.pkl")
    selector.save(save_path)
    print(f"[OK] Adaptive Meta-Selector successfully trained & saved to '{save_path}'.")

if __name__ == "__main__":
    train_adaptive_selector()
