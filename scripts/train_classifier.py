import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ================================
# 1️⃣ 数据路径
# ================================
data_dir = r"...\cls_dataset"

# ================================
# 2️⃣ 配置设备
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

# ================================
# 3️⃣ 数据增强与预处理
# ================================
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    "val": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}

# ================================
# 4️⃣ 加载 ImageFolder 数据集
# ================================
full_dataset = datasets.ImageFolder(data_dir, transform=data_transforms["train"])

# 随机划分 train / val
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# ================================
# 5️⃣ 模型初始化
# ================================
model = models.resnet18(weights=None)  # 不加载预训练权重
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)  # 5类

# 如果已有训练好的模型，加载继续训练
checkpoint_path = r"...\shape_classifier.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"✅ 已加载已有模型：{checkpoint_path}")

model = model.to(device)

# ================================
# 6️⃣ 损失函数 & 优化器
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ================================
# 7️⃣ 训练参数
# ================================
epochs = 20  # 继续训练轮数，可修改
train_losses = []
val_losses = []

# ================================
# 8️⃣ 训练循环
# ================================
for epoch in range(epochs):
    # ---- 训练 ----
    model.train()
    running_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)

    # ---- 验证 ----
    model.eval()
    running_val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = running_val_loss / len(val_loader)
    acc = correct / total

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  Acc: {acc:.4f}")

# ================================
# 9️⃣ 保存模型 & 可视化
# ================================
torch.save(model.state_dict(), checkpoint_path)
print(f"🎉 模型已保存为 {checkpoint_path}")

plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Curve")
plt.savefig("training_curve.png")
print("📈 训练曲线已保存为 training_curve.png")

