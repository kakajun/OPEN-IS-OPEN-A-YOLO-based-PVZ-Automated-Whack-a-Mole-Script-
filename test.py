import os
import sys
import cv2
import torch
from ultralytics import YOLO

ROOTPATH = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(ROOTPATH, "runs", "train", "weights", "best.pt")


def main():
    img_path = os.path.join(ROOTPATH, "1.png")

    # 检查模型与图片是否存在
    if not os.path.isfile(MODEL):
        print(f"[error] 模型不存在: {MODEL}")
        sys.exit(1)
    if not os.path.isfile(img_path):
        print(f"[error] 图片不存在: {img_path}")
        sys.exit(1)

    # 仅使用 CUDA，如不可用则直接报错
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[error] CUDA 不可用或不兼容，已拒绝回退到 CPU。请安装匹配的 CUDA/Torch。")

    print(f"[info] 加载模型: {MODEL}")
    model = YOLO(MODEL)

    print(f"[info] 识别图片: {img_path}")
    results = model.predict(source=img_path, device="cuda", conf=0.25, iou=0.5)

    if not results:
        print("[warn] 未返回结果。")
        sys.exit(1)

    # 使用内置绘制函数生成带标注的图像并保存/展示
    res = results[0]
    annotated = res.plot()  # numpy BGR 图
    out_path = os.path.join(ROOTPATH, "annotated_49.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"[info] 已保存标注图: {out_path}")

    # 展示窗口
    cv2.imshow("Prediction", annotated)
    print("[info] 按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
