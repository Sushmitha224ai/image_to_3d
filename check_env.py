import sys
import platform
import os
import torch

def main():
    cpu = platform.processor() or platform.machine()
    os_name = f"{platform.system()} {platform.release()}"
    python_ver = sys.version.split()[0]
    torch_ver = torch.__version__
    cuda_avail = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if cuda_avail else "N/A"
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
    gpu_vram = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if cuda_avail else "N/A"
    
    report = (
        "===== HARDWARE CHECK =====\n"
        f"CPU / OS: {cpu} ({os_name})\n"
        f"Python: {python_ver}\n"
        f"PyTorch: {torch_ver}\n"
        f"CUDA Available: {cuda_avail}\n"
        f"CUDA Version: {cuda_ver}\n"
        f"GPU Name: {gpu_name}\n"
        f"GPU VRAM (GB): {gpu_vram}\n"
        "=========================="
    )
    print(report)
    with open("env_info.txt", "w", encoding="utf-8") as f:
        f.write(report)

if __name__ == "__main__":
    main()
