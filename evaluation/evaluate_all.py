import os
import sys
import time
import glob
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.pipeline import ForensicInferencePipeline
from evaluation.metrics import compute_metrics
from model_selection.meta_selector import MODEL_KEYS, MODEL_DISPLAY_NAMES

def run_evaluation():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    test_dir = os.path.join(config["dataset"]["processed_dir"], "test")
    results_dir = config["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    sketch_paths = sorted(glob.glob(os.path.join(test_dir, "sketches", "*.png")))
    photo_paths = sorted(glob.glob(os.path.join(test_dir, "photos", "*.png")))

    print(f"\n==========================================")
    print(f" EVALUATING ALL MODELS ON UNSEEN TEST SET ({len(sketch_paths)} pairs)")
    print(f"==========================================")

    pipeline = ForensicInferencePipeline()

    model_metrics_accum = {key: {"PSNR": [], "SSIM": [], "MAE": [], "MSE": [], "Time": []} for key in MODEL_KEYS}
    adaptive_metrics_accum = {"PSNR": [], "SSIM": [], "MAE": [], "MSE": [], "Time": []}

    for s_path, p_path in zip(sketch_paths, photo_paths):
        s_img = Image.open(s_path).convert("RGB")
        gt_photo_np = np.array(Image.open(p_path).convert("RGB"))

        # PyTorch Tensor input
        s_np = np.array(s_img.resize((256, 256)), dtype=np.float32)
        s_arr = (s_np / 127.5) - 1.0
        s_tensor = torch.from_numpy(s_arr.transpose(2, 0, 1)).unsqueeze(0).float()

        # 1. Evaluate Individual Candidate Models
        for key in MODEL_KEYS:
            t0 = time.time()
            gen_tensor = pipeline.models[key].generate(s_tensor)
            t_elapsed = (time.time() - t0) * 1000.0 # ms

            gen_np = ((gen_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            m = compute_metrics(gen_np, gt_photo_np)
            
            for k_m in ["PSNR", "SSIM", "MAE", "MSE"]:
                model_metrics_accum[key][k_m].append(m[k_m])
            model_metrics_accum[key]["Time"].append(t_elapsed)

        # 2. Evaluate Adaptive Selection Engine
        t0 = time.time()
        res_pipeline = pipeline.process_sketch(s_img)
        t_elapsed = (time.time() - t0) * 1000.0

        selected_key = res_pipeline["selected_model_key"]
        sel_gen_tensor = pipeline.models[selected_key].generate(s_tensor)
        sel_gen_np = ((sel_gen_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0) + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        
        m_adapt = compute_metrics(sel_gen_np, gt_photo_np)
        for k_m in ["PSNR", "SSIM", "MAE", "MSE"]:
            adaptive_metrics_accum[k_m].append(m_adapt[k_m])
        adaptive_metrics_accum["Time"].append(t_elapsed)

    # Assemble Benchmark Summary Table
    rows = []
    for key in MODEL_KEYS:
        rows.append({
            "Model Name": MODEL_DISPLAY_NAMES[key],
            "PSNR (dB)": round(np.mean(model_metrics_accum[key]["PSNR"]), 2),
            "SSIM": round(np.mean(model_metrics_accum[key]["SSIM"]), 4),
            "MAE": round(np.mean(model_metrics_accum[key]["MAE"]), 2),
            "MSE": round(np.mean(model_metrics_accum[key]["MSE"]), 2),
            "Inference Time (ms)": round(np.mean(model_metrics_accum[key]["Time"]), 1)
        })

    # Add Adaptive Model Selection Entry
    rows.append({
        "Model Name": "Adaptive Multi-Model Selection (PROPOSED)",
        "PSNR (dB)": round(np.mean(adaptive_metrics_accum["PSNR"]), 2),
        "SSIM": round(np.mean(adaptive_metrics_accum["SSIM"]), 4),
        "MAE": round(np.mean(adaptive_metrics_accum["MAE"]), 2),
        "MSE": round(np.mean(adaptive_metrics_accum["MSE"]), 2),
        "Inference Time (ms)": round(np.mean(adaptive_metrics_accum["Time"]), 1)
    })

    df = pd.DataFrame(rows)
    print("\n--------------------------------------------------------------------------------")
    print("                      EVALUATION BENCHMARK RESULTS MATRIX")
    print("--------------------------------------------------------------------------------")
    print(df.to_string(index=False))
    print("--------------------------------------------------------------------------------\n")

    csv_path = os.path.join(results_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)

    # Plot Comparison Graph
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df["Model Name"]))
    width = 0.35

    rects1 = ax1.bar(x - width/2, df["PSNR (dB)"], width, label="PSNR (dB)", color="#3b82f6")
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, df["SSIM"], width, label="SSIM", color="#10b981")

    ax1.set_ylabel("PSNR (dB)", color="#3b82f6", fontsize=12)
    ax2.set_ylabel("SSIM", color="#10b981", fontsize=12)
    ax1.set_title("Model Comparison: Forensic Sketch Reconstruction Performance", fontsize=14, pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Model Name"], rotation=20, ha="right", fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    chart_path = os.path.join(results_dir, "model_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()

    print(f"[OK] Results saved to '{csv_path}' and chart saved to '{chart_path}'.")

if __name__ == "__main__":
    run_evaluation()
