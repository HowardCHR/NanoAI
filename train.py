from ultralytics import YOLO
import torch
model = YOLO('./models/yolov8n.pt')

def train_machine():
    if torch.cuda.is_available():
        device = "cuda"
        print("⚙️ 使用 NVIDIA GPU (CUDA) 加速推理！")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("⚙️ 使用 Apple MPS 加速推理！")
    else:
        device = "cpu"
        print("⚙️ 未检测到 CUDA 或 MPS，使用 CPU 模式。")
    return device
    

model.train(
    data='dataset/data.yaml',
    device=train_machine(),
    epochs=80,
    imgsz=640,        # 若粒子特别小：imgsz=800
    batch=-1,         # 自动最大 batch size（12GB 显存可以跑大概 16~32）
    workers=8,
    optimizer='AdamW',
    lr0=0.002,
    mosaic=1.0,       # 保留数据增强，提高泛化
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    project='runs/train',
    name='particle_shape_yolov8n',
)