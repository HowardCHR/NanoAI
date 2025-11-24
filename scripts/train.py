from ultralytics import YOLO
import torch

# ----------------------------
# 自动选择设备：CUDA → MPS → CPU
# ----------------------------
if torch.cuda.is_available():
    device = "cuda:0"
    print("⚙️ 使用 CUDA 加速训练！")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    device = "mps"
    print("⚙️ 使用 Apple MPS 加速训练！")
else:
    device = "cpu"
    print("⚙️ 使用 CPU 训练。")

print(f"Using device: {device}")


# ----------------------------
# 加载 YOLO 模型
# ----------------------------
model = YOLO("./models/yolov8n.pt")


# ----------------------------
# 启动训练
# ----------------------------
model.train(
    data='./data/data.yaml',
    device=device,            # 自动 cuda/mps/cpu
    epochs=10,                # 降低训练轮数，速度更快
    imgsz=512,                # 从 640 降到 512，大幅减轻显存压力
    batch=4,                  # 不用 auto。Mac MPS 更稳定用小 batch
    workers=2,                # 过多 workers 会直接卡死
    optimizer="AdamW",
    lr0=0.002,

    # 减少增强以降低计算量
    mosaic=0.0,               # MPS 上 mosaic 很占用内存，建议关掉
    hsv_h=0.01,
    hsv_s=0.5,
    hsv_v=0.3,
)
