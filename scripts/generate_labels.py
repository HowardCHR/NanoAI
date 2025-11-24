import os
import csv
from ultralytics import YOLO
from PIL import Image

# ===================
# 1. 路径设置（请确认）
# ===================

IMAGE_DIR = "./images"          # 图片位置
CSV_PATH = "./labels.csv"       # 你给我的 CSV
OUT_LABEL_DIR = "./auto_labels" # 生成 YOLO txt 的地方
MODEL_PATH = "./runs/detect/train/weights/best.pt"  # 你组员的 YOLOv8 模型

os.makedirs(OUT_LABEL_DIR, exist_ok=True)

# ===================
# 2. 加载 CSV（图级类别）
# ===================
label_dict = {}
with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label_dict[row["filename"]] = row["label"].lower()

# ===================
# 3. 定义类别映射
# ===================
class_map = {
    "sphere": 0,
    "rod": 1,
    "triangle": 2,
    "plate": 3,
    "cube": 4,          # CSV 里有 cube（算 irregular）
}

# 不在列表里 → irregular(4)
def get_class_id(label):
    return class_map.get(label, 4)

# ===================
# 4. 加载 YOLO 模型
# ===================
model = YOLO(MODEL_PATH)

# ===================
# 5. 开始自动标注
# ===================
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith(".png"):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)

    # YOLO 自动检测框
    results = model(img_path)[0]

    # 获取该图的形状类别
    shape_label = label_dict.get(img_name, "irregular")
    class_id = get_class_id(shape_label)

    # 写 YOLO 标签文件
    txt_path = os.path.join(OUT_LABEL_DIR, img_name.replace(".png", ".txt"))
    with open(txt_path, "w") as f:
        for box in results.boxes.xywhn:   # 归一化 xywh
            x, y, w, h = box.tolist()
            f.write(f"{class_id} {x} {y} {w} {h}\n")

    print(f"[OK] {img_name} → 类别:{shape_label} ({class_id})")

print("全部自动标注完成！txt 已生成在 auto_labels/")
