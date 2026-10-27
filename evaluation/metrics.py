import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_metrics(gen_img_np, gt_img_np):
    """
    gen_img_np: numpy array (H, W, 3) in range [0, 255] or float [-1, 1]
    gt_img_np: numpy array (H, W, 3)
    Returns dict of metrics: PSNR, SSIM, MAE, MSE.
    """
    if gen_img_np.dtype != np.uint8:
        if gen_img_np.min() < 0:
            gen_img_np = ((gen_img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            gt_img_np = ((gt_img_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
        else:
            gen_img_np = (gen_img_np * 255.0).clip(0, 255).astype(np.uint8)
            gt_img_np = (gt_img_np * 255.0).clip(0, 255).astype(np.uint8)

    val_ssim = ssim(gen_img_np, gt_img_np, channel_axis=2, data_range=255)
    val_psnr = psnr(gt_img_np, gen_img_np, data_range=255)
    
    diff = gen_img_np.astype(np.float32) - gt_img_np.astype(np.float32)
    val_mae = np.mean(np.abs(diff))
    val_mse = np.mean(diff ** 2)

    return {
        "PSNR": round(float(val_psnr), 2),
        "SSIM": round(float(val_ssim), 4),
        "MAE": round(float(val_mae), 2),
        "MSE": round(float(val_mse), 2)
    }
