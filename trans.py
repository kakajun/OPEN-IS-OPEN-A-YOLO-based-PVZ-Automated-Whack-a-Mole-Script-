import os
import json
from pathlib import Path
from PIL import Image
import shutil

# -------------------------- 只改这里！--------------------------
YOLO_DATASET_ROOT = "datasets"  # 你的 YOLO 数据集根目录
# ---------------------------------------------------------------

OUTPUT_JSON = "label_studio_import.json"
UPLOAD_IMAGE_DIR = "to_upload/images"  # 整理后供上传的图片文件夹

# 1. 读取类别 + 创建图片上传目录
with open(os.path.join(YOLO_DATASET_ROOT, "classes.txt"), "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f if line.strip()]
id_to_class = {idx: cls for idx, cls in enumerate(classes)}

os.makedirs(UPLOAD_IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_IMAGE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_IMAGE_DIR, "val"), exist_ok=True)

# 2. 复制图片到上传目录（保持 train/val 结构）
for split in ["train", "val"]:
    src_img_dir = os.path.join(YOLO_DATASET_ROOT, "images", split)
    dst_img_dir = os.path.join(UPLOAD_IMAGE_DIR, split)
    for img_file in os.listdir(src_img_dir):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            shutil.copy(os.path.join(src_img_dir, img_file), dst_img_dir)

# 3. 生成 Label Studio JSON（自动处理坐标转换）
label_studio_data = []
for split in ["train", "val"]:
    img_dir = os.path.join(UPLOAD_IMAGE_DIR, split)
    label_dir = os.path.join(YOLO_DATASET_ROOT, "labels", split)

    for img_filename in os.listdir(img_dir):
        img_name, img_ext = os.path.splitext(img_filename)
        img_path = os.path.join(split, img_filename)  # 相对路径
        img_abs_path = os.path.join(img_dir, img_filename)

        # 读取图片尺寸
        with Image.open(img_abs_path) as img:
            w, h = img.size

        # 解析 YOLO 标注
        annotations = []
        label_path = os.path.join(label_dir, f"{img_name}.txt")
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    class_id, xc, yc, bw, bh = map(float, parts)
                    # YOLO → Label Studio 百分比坐标
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

        # 添加到 JSON
        label_studio_data.append({
            "data": {"image": f"/data/local-files/?d=images/{img_path}"},
            "annotations": annotations
        })

# 保存 JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(label_studio_data, f, indent=2)

print(f"完成！生成：")
print(f"1. 图片文件夹：{UPLOAD_IMAGE_DIR}（可直接上传）")
print(f"2. 标注 JSON：{OUTPUT_JSON}（可直接导入）")
print(f"共处理 {len(label_studio_data)} 张图片")