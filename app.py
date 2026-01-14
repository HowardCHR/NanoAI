import streamlit as st
import tempfile
import os
import time
from detect import detect_particles


st.set_page_config(page_title="Particle Shape Detection", layout="centered")
st.title("Particle Shape Detection — 简洁版")

st.markdown(
	"上传一张图片，使用已训练的 YOLO 模型进行粒子形状检测。底层调用项目中的 `detect_particles` 函数并展示带框结果。"
)

conf = st.slider("置信度阈值", 0.1, 1.0, 0.5, 0.05)
model_path = st.text_input("模型路径", value="runs/detect/train1/weights/best.pt")

# 在app.py中进行如下修改：

# 1. 修改文件上传器，允许上传多个文件
uploaded = st.file_uploader("选择图片上传", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# 2. 修改检测逻辑，处理多个图片
if st.button("开始检测"):
    if not uploaded:
        st.warning("请先上传图片然后再运行检测。")
    else:
        with tempfile.TemporaryDirectory() as td:
            # 为每个上传的图片创建一个子目录
            for i, file in enumerate(uploaded):
                st.subheader(f"图片 {i+1}: {file.name}")
                
                # 将当前图片写入临时文件
                in_path = os.path.join(td, file.name)
                with open(in_path, "wb") as f:
                    f.write(file.getbuffer())
                
                # 为当前图片创建唯一的输出目录
                out_dir = os.path.join("runs", "inference", f"streamlit_{int(time.time())}_{i}")
                os.makedirs(out_dir, exist_ok=True)
                
                with st.spinner(f"正在处理图片 {i+1}/{len(uploaded)}..."):
                    try:
                        results = detect_particles(model_path=model_path, source=in_path, save_dir=out_dir, conf=conf)
                    except Exception as e:
                        st.error(f"处理图片 {file.name} 失败：{e}")
                        continue
                
                # 查找保存的结果图片
                saved_images = []
                for root, _, files in os.walk(out_dir):
                    for fn in files:
                        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                            saved_images.append(os.path.join(root, fn))
                
                if saved_images:
                    # 显示当前图片的检测结果
                    out_img = max(saved_images, key=os.path.getmtime)
                    st.image(out_img, caption=f"{file.name} 的检测结果", use_container_width=True)
                    with open(out_img, "rb") as f:
                        st.download_button(
                            label=f"下载 {file.name} 的结果", 
                            data=f.read(), 
                            file_name=f"result_{file.name}",
                            key=f"download_{i}"
                        )
                else:
                    st.info(f"未找到 {file.name} 的保存结果图片。")