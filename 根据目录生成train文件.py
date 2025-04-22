import os
import re
import random

# 指定文件夹路径
folder_path = '6000-8443'

# 存储提取的序号
numbers = []

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        match = re.search(r'\d+', filename)
        if match:
            numbers.append(int(match.group()))

# 划分训练集和验证集（1:9比例）
# random.seed(42)  # 固定随机种子保证可复现
# val_ratio = 0.1  # 验证集比例
# val_size = int(len(numbers) * val_ratio)

# 随机采样验证集
# val_paths = random.sample(numbers, val_size)
# train_paths = [num for num in numbers if num not in val_paths]

# 生成训练集路径文件
train_file_paths = [
    f"/data2/jr/yolov9-main/Dataset/coco/images/{idx}.jpg"
    for idx in numbers
]
with open("data_semi/8000.txt", "w") as f:
    f.write("\n".join(train_file_paths))

# # 生成验证集路径文件
# val_file_paths = [
#     f"/data2/jr/yolov9-main/Dataset/coco/images/{idx}.jpg"
#     for idx in val_paths
# ]
# with open("data_semi/val_2000-4000.txt", "w") as f:
#     f.write("\n".join(val_file_paths))