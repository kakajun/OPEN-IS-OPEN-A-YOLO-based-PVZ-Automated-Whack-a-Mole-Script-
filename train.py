from ultralytics import YOLO
import os
# 避免 OpenMP 运行时重复初始化导致报错（libiomp5md.dll 重复）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import torch

ROOTPATH = os.path.dirname(os.path.abspath(__file__))

model = YOLO("yolo11s.yaml")

def run(device: str):
    # 仅支持 CUDA；如果不可用则直接报错，不再尝试 CPU。
    if device in ("cuda", "0", "cuda:0") and not torch.cuda.is_available():
        raise RuntimeError("[train] CUDA 不可用或不兼容，已拒绝回退到 CPU。请安装匹配的 CUDA/Torch。")

    # 强制使用 CUDA
    device = "cuda"
    print(f"[train] using device: {device}")
    model.train(
        data=os.path.join(ROOTPATH, "configs", "ZVP.yaml"),
        cfg=os.path.join(ROOTPATH, "configs", "trainCfg.yaml"),
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model for PVZ dataset")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda"],
        default="auto",
        help="Training device: auto (default) or cuda",
    )
    args = parser.parse_args()
    # 始终使用 CUDA；若不可用，run() 会报错。
    dev = "cuda"
    run(dev)
