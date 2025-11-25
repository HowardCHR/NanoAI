import os
import shutil
import pandas as pd

# 路径
base_path = r"...\Nano_shape"
labels_csv = os.path.join(base_path, "labels.csv")
images_dir = os.path.join(base_path, "dataset", "images", "train")
output_dir = os.path.join(base_path, "cls_dataset")

# 需要的分类
valid_classes = ["sphere", "rod", "triangle", "cube", "irregular"]

print("🔧 正在重建分类数据集...")

# 删除旧的 cls_dataset
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# 创建类别文件夹
for cls in valid_classes:
    os.makedirs(os.path.join(output_dir, cls))

# 读取 labels.csv
df = pd.read_csv(labels_csv)

# 检查是否有非法类别（如 plate）
df = df[df["label"].isin(valid_classes)]

count = 0

# 复制图片
for _, row in df.iterrows():
    fname = row["filename"]
    label = row["label"]

    src_img = os.path.join(images_dir, fname)
    dst_img = os.path.join(output_dir, label, fname)

    if os.path.exists(src_img):
        shutil.copy(src_img, dst_img)
        count += 1
    else:
        print(f"⚠ 找不到图片：{src_img}")

print(f"✅ 分类数据集构建完成！共复制 {count} 张图片。")
print(f"📂 输出位置：{output_dir}")


