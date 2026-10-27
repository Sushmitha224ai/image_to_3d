import os
import sys
import glob
import numpy as np
from PIL import Image
import yaml

def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_preprocessing(raw_dir="data/raw", config_path="configs/config.yaml"):
    """
    Identity-Strict Preprocessing Pipeline for Genuine CUFS / CUFSF Dataset:
    1. Matches sketch-photo pairs by identity ID.
    2. Enforces identity-based Train (70%), Val (15%), Test (15%) split (Zero Identity Leakage).
    3. Resizes images to 256x256 RGB.
    4. Generates PREPROCESSING_REPORT.txt.
    """
    config = load_config(config_path)
    img_size = (config["dataset"]["img_size"], config["dataset"]["img_size"])
    processed_base = config["dataset"]["processed_dir"]

    sketches_raw = glob.glob(os.path.join(raw_dir, "sketches", "*_sketch.png"))
    photos_raw = glob.glob(os.path.join(raw_dir, "photos", "*_photo.png"))

    sketches_dict = {os.path.basename(f).replace("_sketch.png", ""): f for f in sketches_raw}
    photos_dict = {os.path.basename(f).replace("_photo.png", ""): f for f in photos_raw}

    common_ids = sorted(list(set(sketches_dict.keys()).intersection(set(photos_dict.keys()))))
    missing_sketches = set(photos_dict.keys()) - set(sketches_dict.keys())
    missing_photos = set(sketches_dict.keys()) - set(photos_dict.keys())

    valid_pairs = []
    corrupted_files = []

    for id_str in common_ids:
        s_path = sketches_dict[id_str]
        p_path = photos_dict[id_str]
        try:
            with Image.open(s_path) as img_s, Image.open(p_path) as img_p:
                img_s.verify()
                img_p.verify()
            valid_pairs.append(id_str)
        except Exception as e:
            corrupted_files.append((id_str, str(e)))

    np.random.seed(42)
    np.random.shuffle(valid_pairs)

    num_total = len(valid_pairs)
    num_train = int(num_total * config["dataset"]["train_ratio"])
    num_val = int(num_total * config["dataset"]["val_ratio"])
    num_test = num_total - num_train - num_val

    train_ids = valid_pairs[:num_train]
    val_ids = valid_pairs[num_train:num_train + num_val]
    test_ids = valid_pairs[num_train + num_val:]

    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    # Prepare processed directories
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(processed_base, split, "sketches"), exist_ok=True)
        os.makedirs(os.path.join(processed_base, split, "photos"), exist_ok=True)

    print(f"[*] Processing {num_total} genuine CUFS/CUFSF research pairs into train/val/test splits...")

    for split_name, id_list in splits.items():
        for id_str in id_list:
            s_src = sketches_dict[id_str]
            p_src = photos_dict[id_str]

            s_dst = os.path.join(processed_base, split_name, "sketches", f"{id_str}.png")
            p_dst = os.path.join(processed_base, split_name, "photos", f"{id_str}.png")

            with Image.open(s_src).convert("RGB") as s_img:
                s_img = s_img.resize(img_size, Image.Resampling.BILINEAR)
                s_img.save(s_dst)

            with Image.open(p_src).convert("RGB") as p_img:
                p_img = p_img.resize(img_size, Image.Resampling.BILINEAR)
                p_img.save(p_dst)

    report_content = f"""==================================================
CUFS / CUFSF PREPROCESSING REPORT
==================================================
Total Raw Pairs Located  : {len(common_ids)}
Valid Usable Pairs       : {num_total}
Missing Sketches         : {len(missing_sketches)}
Missing Photos           : {len(missing_photos)}
Corrupted Files          : {len(corrupted_files)}

--------------------------------------------------
IDENTITY-STRICT DATASET SPLIT (Zero Leakage)
--------------------------------------------------
Training Identities   (70%) : {len(train_ids)}
Validation Identities (15%) : {len(val_ids)}
Testing Identities    (15%) : {len(test_ids)}

--------------------------------------------------
IMAGE SPECIFICATIONS
--------------------------------------------------
Image Resolution : {img_size[0]} x {img_size[1]}
Color Channels   : 3 (RGB)
Processed Dir    : {os.path.abspath(processed_base)}
==================================================
"""
    print(report_content)

    report_path = os.path.join(processed_base, "PREPROCESSING_REPORT.txt")
    os.makedirs(processed_base, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"[OK] Preprocessing complete. Report saved to '{report_path}'.")

if __name__ == "__main__":
    run_preprocessing()
