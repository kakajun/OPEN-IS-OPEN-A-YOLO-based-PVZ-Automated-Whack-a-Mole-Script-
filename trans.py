import os
import json
from pathlib import Path
from PIL import Image
import shutil

# -------------------------- 配置参数 --------------------------
YOLO_DATASET_ROOT = "datasets"  # 你的 YOLO 数据集根目录
LABEL_STUDIO_EXPORT_JSON = "label_studio_import_fixed.json"  # 新的输出 JSON
# ---------------------------------------------------------------

# 1. 读取类别文件
with open(os.path.join(YOLO_DATASET_ROOT, "classes.txt"), "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f if line.strip()]
id_to_class = {idx: cls for idx, cls in enumerate(classes)}

# 2. 收集所有图片和标注的映射（不管 train/val 子目录，统一处理）
image_label_mapping = {}
for split in ["train", "val"]:
    img_dir = os.path.join(YOLO_DATASET_ROOT, "images", split)
    label_dir = os.path.join(YOLO_DATASET_ROOT, "labels", split)
    for img_filename in os.listdir(img_dir):
        if img_filename.endswith(('.jpg', '.png', '.jpeg')):
            img_name = os.path.splitext(img_filename)[0]
            label_path = os.path.join(label_dir, f"{img_name}.txt")
            image_label_mapping[img_filename] = label_path

# 3. 生成 Label Studio JSON（路径改为 Label Studio 实际的 upload 格式）
label_studio_data = []
for img_filename, label_path in image_label_mapping.items():
    # 图片在 Label Studio 中的路径（固定格式）
    img_ls_path = f"/data/upload/1/{img_filename}"
    # 读取图片尺寸
    img_abs_path = os.path.join(YOLO_DATASET_ROOT, "images", "train" if "train" in label_path else "val", img_filename)
    with Image.open(img_abs_path) as img:
        w, h = img.size
    # 解析标注
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, xc, yc, bw, bh = map(float, parts)
                x1 = (xc - bw/2) * 100
                y1 = (yc - bh/2) * 100
                width = bw * 100
                height = bh * 100
                annotations.append({
                    "result": [{
                        "value": {"x": x1, "y": y1, "width": width, "height": height, "label": [id_to_class[int(class_id)]]},
                        "type": "rectanglelabels", "to_name": "image", "from_name": "label"
                    }]
                })
    # 构造数据
    label_studio_data.append({
        "data": {"image": img_ls_path},
        "annotations": annotations
    })

# 保存 JSON
with open(LABEL_STUDIO_EXPORT_JSON, "w", encoding="utf-8") as f:
    json.dump(label_studio_data, f, indent=2)

print("转换完成！新的 JSON 已生成：", LABEL_STUDIO_EXPORT_JSON)