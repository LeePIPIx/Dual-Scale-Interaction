import time

from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm


def load_image_paths(txt_file):
    """读取txt文件中的图片路径"""
    with open(txt_file, 'r') as f:
        image_paths = [line.strip() for line in f.readlines()]
    return image_paths


def save_detections(results, output_dir):
    """将检测结果保存为COCO格式的txt文件"""
    os.makedirs(output_dir, exist_ok=True)

    for result in results:
        image_name = os.path.basename(result.path).split('.')[0]  # 获取图片名称（无扩展名）
        output_path = os.path.join(output_dir, f"{image_name}.txt")

        with open(output_path, 'w') as f:
            for box in result.boxes:
                class_id = int(box.cls.item())  # 目标类别
                x_center, y_center, width, height = [format(val, ".6f") for val in box.xywhn.tolist()[0]]  # 归一化的检测框，保留6位小数
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def main(txt_file, model_path, output_dir):
    """主函数：读取图片路径，执行检测，并保存结果"""
    image_paths = load_image_paths(txt_file)

    model = YOLO(model_path)  # 加载YOLO模型

    for image_path in tqdm(image_paths, desc="Processing Images", unit="image"):
        results = model(image_path)  # 进行目标检测
        save_detections(results, output_dir)
        print("\r", end="", flush=True)

if __name__ == "__main__":
    txt_file = "data_semi/predict_6000-8443.txt"  # 包含图片路径的txt文件
    model_path = "data_semi/V8_6000/weights/last.pt"  # 你的YOLO模型权重文件
    output_dir = "6000-8443"  # 结果保存路径

    main(txt_file, model_path, output_dir)
