import os
import sys
import glob
import time
import argparse
import yaml
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_1_pix2pix import Pix2PixModel
from models.model_2_unet import PerceptualUNetModel
from models.model_3_resnet_gan import ResNetLSGANModel
from models.model_4_pca_ridge import PCARidgeModel
from models.model_5_knn_patch import PatchKNNModel

class SketchPhotoDataset(Dataset):
    """PyTorch Dataset for paired sketch-photo images."""
    def __init__(self, split_dir="data/processed/train"):
        self.sketch_paths = sorted(glob.glob(os.path.join(split_dir, "sketches", "*.png")))
        self.photo_paths = sorted(glob.glob(os.path.join(split_dir, "photos", "*.png")))

    def __len__(self):
        return len(self.sketch_paths)

    def __getitem__(self, idx):
        s_img = Image.open(self.sketch_paths[idx]).convert("RGB")
        p_img = Image.open(self.photo_paths[idx]).convert("RGB")

        # Convert PIL to Tensor in range [-1, 1]
        s_arr = (np.array(s_img, dtype=np.float32) / 127.5) - 1.0
        p_arr = (np.array(p_img, dtype=np.float32) / 127.5) - 1.0

        s_tensor = torch.from_numpy(s_arr.transpose(2, 0, 1)).float()
        p_tensor = torch.from_numpy(p_arr.transpose(2, 0, 1)).float()

        return s_tensor, p_tensor

def get_numpy_data(split_dir="data/processed/train"):
    """Loads dataset into memory as numpy arrays for Scikit-Learn models."""
    dataset = SketchPhotoDataset(split_dir)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for s, p in loader:
        # Convert (B, C, H, W) -> (B, H, W, C)
        sketch_np = s.numpy().transpose(0, 2, 3, 1)
        photo_np = p.numpy().transpose(0, 2, 3, 1)
        return sketch_np, photo_np
    return None, None

def train_dl_model(model_key, model_obj, train_loader, epochs, checkpoints_dir):
    print(f"\n==========================================")
    print(f" TRAINING MODEL: {model_key.upper()}")
    print(f"==========================================")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None
    
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches = 0
        
        for sketch, photo in train_loader:
            metrics = model_obj.train_step(sketch, photo, scaler=scaler)
            loss_val = metrics.get("loss_G", metrics.get("loss_total", 0.0))
            epoch_loss += loss_val
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        elapsed = time.time() - start_time
        
        print(f"[{model_key.upper()}] Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} - Elapsed: {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(checkpoints_dir, f"{model_key}_best.pth")
            model_obj.save(save_path)

    print(f"[OK] Training completed for {model_key}. Best loss: {best_loss:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Unified Training Script for 5 Forensic Reconstruction Models")
    parser.add_argument("--model", type=str, default="all", choices=["all", "pix2pix", "unet", "resnet_gan", "pca_ridge", "knn_patch"])
    args = parser.parse_args()

    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    checkpoints_dir = config["paths"]["checkpoints_dir"]
    os.makedirs(checkpoints_dir, exist_ok=True)

    train_dataset = SketchPhotoDataset(os.path.join(config["dataset"]["processed_dir"], "train"))
    
    # ------------------
    # 1. Pix2Pix cGAN
    # ------------------
    if args.model in ["all", "pix2pix"]:
        loader = DataLoader(train_dataset, batch_size=config["models"]["model_1_pix2pix"]["batch_size"], shuffle=True)
        m1 = Pix2PixModel(device="cuda", lr=config["models"]["model_1_pix2pix"]["lr"])
        train_dl_model("model_1_pix2pix", m1, loader, config["models"]["model_1_pix2pix"]["epochs"], checkpoints_dir)

    # ------------------
    # 2. Deep U-Net
    # ------------------
    if args.model in ["all", "unet"]:
        loader = DataLoader(train_dataset, batch_size=config["models"]["model_2_unet"]["batch_size"], shuffle=True)
        m2 = PerceptualUNetModel(device="cuda", lr=config["models"]["model_2_unet"]["lr"])
        train_dl_model("model_2_unet", m2, loader, config["models"]["model_2_unet"]["epochs"], checkpoints_dir)

    # ------------------
    # 3. ResNet LSGAN
    # ------------------
    if args.model in ["all", "resnet_gan"]:
        loader = DataLoader(train_dataset, batch_size=config["models"]["model_3_resnet_gan"]["batch_size"], shuffle=True)
        m3 = ResNetLSGANModel(device="cuda", lr=config["models"]["model_3_resnet_gan"]["lr"])
        train_dl_model("model_3_resnet_gan", m3, loader, config["models"]["model_3_resnet_gan"]["epochs"], checkpoints_dir)

    # ------------------
    # 4. Eigenface PCA-Ridge
    # ------------------
    if args.model in ["all", "pca_ridge"]:
        print(f"\n==========================================")
        print(f" TRAINING MODEL: MODEL_4_PCA_RIDGE")
        print(f"==========================================")
        s_np, p_np = get_numpy_data(os.path.join(config["dataset"]["processed_dir"], "train"))
        m4 = PCARidgeModel(n_components=config["models"]["model_4_pca_ridge"]["n_components"], alpha=config["models"]["model_4_pca_ridge"]["alpha"])
        m4.fit(s_np, p_np)
        m4.save(os.path.join(checkpoints_dir, "model_4_pca_ridge_best.pkl"))
        print(f"[OK] Training completed for Model 4 PCA-Ridge.")

    # ------------------
    # 5. Patch K-NN Regressor
    # ------------------
    if args.model in ["all", "knn_patch"]:
        print(f"\n==========================================")
        print(f" TRAINING MODEL: MODEL_5_KNN_PATCH")
        print(f"==========================================")
        s_np, p_np = get_numpy_data(os.path.join(config["dataset"]["processed_dir"], "train"))
        m5 = PatchKNNModel(
            patch_size=config["models"]["model_5_knn_patch"]["patch_size"],
            stride=config["models"]["model_5_knn_patch"]["stride"],
            k_neighbors=config["models"]["model_5_knn_patch"]["k_neighbors"]
        )
        m5.fit(s_np, p_np)
        m5.save(os.path.join(checkpoints_dir, "model_5_knn_patch_best.pkl"))
        print(f"[OK] Training completed for Model 5 Patch K-NN.")

if __name__ == "__main__":
    main()
