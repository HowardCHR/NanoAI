import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import os

# ================================
# 1. 配置路径
# ================================
model_path = r"C:\Users\李晓滢129\Desktop\NanoAI\shape_classifier.pth"
input_dir = r"C:\Users\李晓滢129\Desktop\NanoAI\dataset\images\val"
output_dir = r"C:\Users\李晓滢129\Desktop\NanoAI\dataset\images\val_pred"

os.makedirs(output_dir, exist_ok=True)

# ================================
# 2. 加载模型
# ================================
class_names = ["cube", "irregular", "rod", "sphere", "triangle"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

print("✅ 模型加载成功！")

# ================================
# 3. 预处理
# ================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def classify_patch(patch):
    img = Image.fromarray(patch)
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        preds = model(img)
        _, index = torch.max(preds, 1)
    return class_names[index]

# ================================
# 4. 自动检测 + 分类 + 标注
# ================================
def process_image(img_path, save_path):
    # 用 imdecode 读取，支持中文路径
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"⚠️ 无法读取图片: {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 二值化
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    # 找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        # 去掉噪声
        if w < 20 or h < 20:
            continue

        patch = img[y:y+h, x:x+w]

        # 分类
        label = classify_patch(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))

        # 画框 + 写文字
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 保存图像，支持中文路径
    ext = os.path.splitext(save_path)[1]
    is_success, im_buf_arr = cv2.imencode(ext, img)
    if is_success:
        im_buf_arr.tofile(save_path)
    else:
        print(f"⚠️ 无法保存图片: {save_path}")

# ================================
# 5. 批量处理文件夹
# ================================
if __name__ == "__main__":
    files = [f for f in os.listdir(input_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    print(f"📁 共找到 {len(files)} 张图片，开始处理...\n")

    for i, filename in enumerate(files, 1):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)  # 保持原名

        process_image(in_path, out_path)
        print(f"[{i}/{len(files)}] ✔ 已处理 {filename}")

    print("\n🎉 全部处理完成！")
    print(f"👉 输出文件保存在：{output_dir}")
