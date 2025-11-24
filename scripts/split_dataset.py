import os, random, shutil

IMG_DIR = "./images"
LABEL_DIR = "./auto_labels"
OUT = "./dataset"

os.makedirs(OUT + "/images/train", exist_ok=True)
os.makedirs(OUT + "/images/val", exist_ok=True)
os.makedirs(OUT + "/labels/train", exist_ok=True)
os.makedirs(OUT + "/labels/val", exist_ok=True)

imgs = [f for f in os.listdir(IMG_DIR) if f.endswith(".png")]
random.shuffle(imgs)

split = int(len(imgs) * 0.8)
train_imgs = imgs[:split]
val_imgs = imgs[split:]

def move_set(image_list, mode):
    for img in image_list:
        label = img.replace(".png", ".txt")

        shutil.copy(os.path.join(IMG_DIR, img), f"{OUT}/images/{mode}/{img}")

        label_path = os.path.join(LABEL_DIR, label)
        if os.path.exists(label_path):
            shutil.copy(label_path, f"{OUT}/labels/{mode}/{label}")
        else:
            print(f"[WARN] no label for {img}")

move_set(train_imgs, "train")
move_set(val_imgs, "val")

print("数据集划分完成！")
