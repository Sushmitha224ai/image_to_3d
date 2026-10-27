import os
import sys
import time
import base64
import io
import gc
import yaml
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as torchvision_functional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.modules.setdefault("torchvision.transforms.functional_tensor", torchvision_functional)

from models.model_1_pix2pix import Pix2PixModel
from models.model_2_unet import PerceptualUNetModel
from models.model_3_resnet_gan import ResNetLSGANModel
from models.model_4_pca_ridge import PCARidgeModel
from models.model_5_knn_patch import PatchKNNModel
from model_selection.meta_selector import AdaptiveModelSelector, MODEL_KEYS, MODEL_DISPLAY_NAMES
from face_3d.mesh_generator import Face3DMeshGenerator
from face_3d.exporter import Face3DExporter

from gfpgan import GFPGANer

MODEL_CHECKPOINT_FILES = {
    "model_1_pix2pix": "model_1_pix2pix_best.pth",
    "model_2_unet": "model_2_unet_best.pth",
    "model_3_resnet_gan": "model_3_resnet_gan_best.pth",
    "model_4_pca_ridge": "model_4_pca_ridge_best.pkl",
    "model_5_knn_patch": "model_5_knn_patch_best.pkl",
}
SELECTOR_CHECKPOINT_FILE = "adaptive_selector_best.pkl"

def pil_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def tensor_to_pil(tensor):
    """
    Proper Tensor-to-Image Normalization:
    Maps output tensor in [-1, 1] to [0, 255] uint8 in standard RGB format.
    """
    arr = tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
    arr = np.clip((arr + 1.0) / 2.0 * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(arr)

class ForensicInferencePipeline:
    """
    Complete End-to-End Forensic Reconstruction Pipeline:
    Forensic Sketch -> Preprocessing -> Candidate Generation -> Artifact-Aware Model Selection -> Strict VRAM Clearing -> GFPGAN AI Face Restoration -> 3D Reconstruction.
    """
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoints_dir = self.config["paths"]["checkpoints_dir"]
        checkpoint_paths = {
            key: os.path.join(checkpoints_dir, filename)
            for key, filename in MODEL_CHECKPOINT_FILES.items()
        }
        selector_checkpoint_path = os.path.join(checkpoints_dir, SELECTOR_CHECKPOINT_FILE)
        missing_checkpoints = [
            path for path in (*checkpoint_paths.values(), selector_checkpoint_path)
            if not os.path.isfile(path)
        ]
        if missing_checkpoints:
            missing_list = "\n".join(f"  - {path}" for path in missing_checkpoints)
            raise FileNotFoundError(
                "Cannot start forensic inference because trained model checkpoints are missing:\n"
                f"{missing_list}\n"
                "Prepare paired training images under data/raw/sketches and data/raw/photos, "
                "then run python data/preprocess.py. Train the candidate models with "
                "python training/train_models.py --model all, and train the adaptive selector "
                "with python training/train_selector.py."
            )

        print(f"[*] Initializing Forensic Pipeline on device: '{self.device}'...")

        # 1. Load Candidate Models
        self.models = {}

        # Model 1
        m1 = Pix2PixModel(device=self.device)
        m1.load(checkpoint_paths["model_1_pix2pix"])
        self.models["model_1_pix2pix"] = m1

        # Model 2
        m2 = PerceptualUNetModel(device=self.device)
        m2.load(checkpoint_paths["model_2_unet"])
        self.models["model_2_unet"] = m2

        # Model 3
        m3 = ResNetLSGANModel(device=self.device)
        m3.load(checkpoint_paths["model_3_resnet_gan"])
        self.models["model_3_resnet_gan"] = m3

        # Model 4
        m4 = PCARidgeModel()
        m4.load(checkpoint_paths["model_4_pca_ridge"])
        if not m4.is_fitted:
            raise ValueError(
                "The PCA-Ridge checkpoint is not fitted. Re-train it with "
                "python training/train_models.py --model pca_ridge."
            )
        self.models["model_4_pca_ridge"] = m4

        # Model 5
        m5 = PatchKNNModel()
        m5.load(checkpoint_paths["model_5_knn_patch"])
        self.models["model_5_knn_patch"] = m5

        # 2. Load Adaptive Meta-Selector
        self.selector = AdaptiveModelSelector()
        self.selector.load(selector_checkpoint_path)

        # 3. Load 3D Engine
        self.mesh_generator = Face3DMeshGenerator(img_size=(256, 256))
        self.exporter = Face3DExporter(output_dir=os.path.join(self.config["paths"]["output_dir"], "3d"))

        # Model path for GFPGAN
        self.gfpgan_path = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth"

    def process_sketch(self, sketch_input, output_basename="reconstructed_face"):
        """
        sketch_input: PIL Image, filepath string, or numpy array.
        """
        start_time = time.time()

        if isinstance(sketch_input, str):
            sketch_pil = Image.open(sketch_input).convert("RGB")
        elif isinstance(sketch_input, Image.Image):
            sketch_pil = sketch_input.convert("RGB")
        else:
            sketch_pil = Image.fromarray(sketch_input).convert("RGB")

        sketch_resized = sketch_pil.resize((256, 256), Image.Resampling.BILINEAR)
        sketch_np = np.array(sketch_resized, dtype=np.float32)

        # PyTorch Tensor in range [-1, 1]
        sketch_arr = (sketch_np / 127.5) - 1.0
        sketch_tensor = torch.from_numpy(sketch_arr.transpose(2, 0, 1)).unsqueeze(0).float()

        # Step A: Generate candidate 2D face outputs across all 5 models
        candidate_previews = {}
        generated_pil_dict = {}

        for key in MODEL_KEYS:
            gen_tensor = self.models[key].generate(sketch_tensor)
            gen_pil = tensor_to_pil(gen_tensor)
            candidate_previews[key] = pil_to_base64(gen_pil)
            generated_pil_dict[key] = gen_pil

        # Step B: Evaluate and select best 2D output
        selection_result = self.selector.predict_suitability(sketch_resized, generated_pil_dict=generated_pil_dict)
        selected_key = selection_result["selected_model_key"]
        selected_name = selection_result["selected_model_name"]
        raw_best_photo = generated_pil_dict[selected_key]

        # Step C: STRICT 4GB VRAM CLEARING SEQUENCE (CRITICAL FOR RTX 3050 SURVIVAL)
        del sketch_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Step D: Load GFPGAN Neural AI Face Restorer AFTER Memory is Cleared
        restorer = GFPGANer(
            model_path=self.gfpgan_path,
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
            device=self.device
        )

        # Convert PIL RGB to cv2 BGR for GFPGAN
        raw_best_np = np.array(raw_best_photo.convert("RGB"))
        raw_best_bgr = cv2.cvtColor(raw_best_np, cv2.COLOR_RGB2BGR)

        # Run GFPGAN Neural AI Face Restoration
        _, _, enhanced_bgr = restorer.enhance(
            raw_best_bgr,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        # Convert GFPGAN output from cv2 BGR back to standard RGB for frontend and 3D pipeline
        enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
        best_2d_photo = Image.fromarray(enhanced_rgb)
        best_2d_np = np.array(best_2d_photo)

        # Update candidate preview of selected model with GFPGAN enhanced photo
        candidate_previews[selected_key] = pil_to_base64(best_2d_photo)

        # Clean up restorer from GPU memory
        del restorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Step E: Save 2D Generated Photo
        output_2d_dir = os.path.join(self.config["paths"]["output_dir"], "2d")
        os.makedirs(output_2d_dir, exist_ok=True)
        photo_2d_path = os.path.join(output_2d_dir, f"{output_basename}_2d.png")
        best_2d_photo.save(photo_2d_path)

        # Step F: 2D-to-3D Face Reconstruction (Using Clean RGB Texture)
        vertices, faces, uvs, landmarks = self.mesh_generator.generate_mesh(best_2d_np)
        paths_3d = self.exporter.export_all(vertices, faces, uvs, best_2d_photo, base_filename=output_basename)

        elapsed = round(time.time() - start_time, 2)

        return {
            "status": "success",
            "execution_time_sec": elapsed,
            "sketch_b64": pil_to_base64(sketch_resized),
            "generated_2d_b64": pil_to_base64(best_2d_photo),
            "selected_model_key": selected_key,
            "selected_model_name": selected_name,
            "confidence_percent": selection_result["confidence_percent"],
            "model_scores": selection_result["scores"],
            "model_probabilities": selection_result["probabilities"],
            "candidate_previews": candidate_previews,
            "paths": {
                "2d_photo": os.path.abspath(photo_2d_path),
                "3d_obj": paths_3d["obj"],
                "3d_ply": paths_3d["ply"],
                "3d_glb": paths_3d["glb"]
            }
        }
