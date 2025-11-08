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
    # 如果用户指定了 CUDA，但当前环境不可用，则直接回退到 CPU
    if device == "cuda" and not torch.cuda.is_available():
        print("[train] 检测到 CUDA 不可用，自动回退到 CPU 训练。")
        device = "cpu"
    try:
        print(f"[train] using device: {device}")
        model.train(
            data=os.path.join(ROOTPATH, "configs", "ZVP.yaml"),
            cfg=os.path.join(ROOTPATH, "configs", "trainCfg.yaml"),
            device=device,
        )
    except (RuntimeError, ValueError) as e:
        msg = str(e)
        if (
            "no kernel image is available for execution on the device" in msg
            or "CUDA error" in msg
            or "Invalid CUDA" in msg
        ):
            print("[train] CUDA 不兼容或不可用，自动回退到 CPU 训练。")
            model.train(
                data=os.path.join(ROOTPATH, "configs", "ZVP.yaml"),
                cfg=os.path.join(ROOTPATH, "configs", "trainCfg.yaml"),
                device="cpu",
            )
        else:
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model for PVZ dataset")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Training device: auto (default), cpu or cuda",
    )
    args = parser.parse_args()

    dev = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else (
        args.device if args.device != "auto" else "cpu"
    )
    run(dev)
