import os
import json
from PIL import Image

# -------------------------- 配置 --------------------------
# 以脚本所在目录为根，默认指向项目内 datasets 数据集目录
ROOTPATH = os.path.dirname(os.path.abspath(__file__))
YOLO_DATASET_ROOT = os.path.join(ROOTPATH, "datasets")
OUTPUT_JSON = "final_import.json"

# 采用本地文件服务 /data/local-files 直接从数据集生成图片列表
# 要求：启动 Label Studio 前正确设置
#   LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
#   LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=<项目 datasets 目录>
# ----------------------------------------------------------

# 1. 读取YOLO类别和标注映射
classes_path = os.path.join(YOLO_DATASET_ROOT, "classes.txt")
if not os.path.exists(classes_path):
    raise FileNotFoundError(f"未找到 classes.txt：{classes_path}。请确认数据集根目录指向包含 classes.txt 的目录，例如 {os.path.join(ROOTPATH, 'datasets')}。")
with open(classes_path, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f if line.strip()]
id_to_class = {idx: cls for idx, cls in enumerate(classes)}

yolo_label_map = {}
for split in ["train", "val"]:
    img_dir = os.path.join(YOLO_DATASET_ROOT, "images", split)
    label_dir = os.path.join(YOLO_DATASET_ROOT, "labels", split)
    for img_filename in os.listdir(img_dir):
        if img_filename.endswith(('.jpg', '.png', '.jpeg')):
            img_name = os.path.splitext(img_filename)[0]
            label_path = os.path.join(label_dir, f"{img_name}.txt")
            yolo_label_map[img_filename.lower()] = label_path

"""
2. 生成 Label Studio 任务列表（使用本地文件服务）
生成路径示例：/data/local-files/?d=images/train/<filename>
要求 Label Studio 以 datasets 为 DOCUMENT_ROOT 启动。
"""
ls_tasks = []
for split in ("train", "val"):
    img_dir = os.path.join(YOLO_DATASET_ROOT, "images", split)
    if not os.path.isdir(img_dir):
        print(f"警告：未找到图片目录：{img_dir}")
        continue
    for fname in os.listdir(img_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        rel_path = f"images/{split}/{fname}"
        ls_tasks.append({"data": {"image": f"/data/local-files/?d={rel_path}"}})

# 3. 匹配标注并生成最终JSON
final_data = []
for task in ls_tasks:
    ls_img_path = task["data"]["image"]
    original_img_name = os.path.basename(ls_img_path)
    if original_img_name.lower() not in yolo_label_map:
        final_data.append({
            "data": {"image": ls_img_path},
            "annotations": []
        })
        continue

    label_path = yolo_label_map[original_img_name.lower()]
    yolo_img_path = os.path.join(YOLO_DATASET_ROOT, "images",
                                "train" if "train" in label_path else "val",
                                original_img_name)
    with Image.open(yolo_img_path) as img:
        w, h = img.size

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

    final_data.append({
        "data": {"image": ls_img_path},
        "annotations": annotations
    })

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2)

print("标注JSON生成完成，可导入Label Studio")