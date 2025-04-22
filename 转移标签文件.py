import os
import shutil


def copy_label_files(txt_path, label_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取txt文件中的图片名称
    with open(txt_path, 'r', encoding='utf-8') as f:
        image_names = [os.path.basename(line.strip()) for line in f.readlines()]

    # 遍历图片名称并查找对应的标签文件
    for image_name in image_names:
        # 假设标签文件和图片文件具有相同的基本名称但不同的扩展名
        base_name = os.path.splitext(image_name)[0]

        # 查找标签文件（可以根据实际情况修改扩展名）
        for ext in ['.txt', '.json', '.xml']:  # 可能的标签文件格式
            label_file = os.path.join(label_dir, base_name + ext)

            if os.path.exists(label_file):
                shutil.copy(label_file, os.path.join(output_dir, base_name + ext))
                print(f"Copied: {label_file} -> {output_dir}")
                break  # 复制第一个匹配的标签文件后跳出循环

    print("Processing complete.")


# 示例用法
copy_label_files("data_semi/trianval_2000.txt", "F:\yolov9-main\Dataset\coco\labels", "2000-4000")