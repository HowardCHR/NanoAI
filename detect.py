from ultralytics import YOLO
import os
import torch

def detect_particles(
    model_path="runs/train/exp/weights/best.pt",
    source="data/test/images",
    save_dir="runs/inference",
    conf=0.5,
):
    """
    使用训练好的 YOLO 模型检测粒子位置和形状。
    Args:
        model_path (str): 训练好的权重文件路径
        source (str): 图片、视频或文件夹路径
        save_dir (str): 检测结果保存目录
        conf (float): 置信度阈值
    """
    # 优先级：CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = "cuda"
        print("⚙️ 使用 NVIDIA GPU (CUDA) 加速推理！")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("⚙️ 使用 Apple MPS 加速推理！")
    else:
        device = "cpu"
        print("⚙️ 未检测到 CUDA 或 MPS，使用 CPU 模式。")

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    model = YOLO(model_path)

    # 执行检测
    results = model.predict(
        source=source,
        conf=conf,
        save=True,
        device=device,
        project=save_dir,  # 检测结果会保存在 runs/inference/expX
        name="particle_detect",  # 子目录名（可自定义）
        exist_ok=False,        # 每次新建一个文件夹
    )

    print(f"✅ 检测完成！结果保存在：{os.path.join(save_dir, 'particle_detect')}")
    return results


if __name__ == "__main__":
    # 修改成你自己的路径（模型 & 测试图片/文件夹）
    model_path = "runs/train/exp/weights/best.pt"
    source = "dataset/train/images"  # 可是单张图片路径、文件夹或视频
    detect_particles(model_path, source)