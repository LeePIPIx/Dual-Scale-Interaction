# import numpy as np
# import random
#
# # 计算等间隔抽取的索引
# indices = np.linspace(1, 8443, 2000, dtype=int)
# # indices.tolist()
# # 计算剩余数字集合（1-8443中除去之前抽取的数字）
# all_numbers = set(range(1, 8444))
# remaining_numbers = list(all_numbers - set(indices))
# indices_in_remaining = np.linspace(1, len(remaining_numbers) - 1, 2000, dtype=int)
# # selected_remaining = [remaining_numbers[i] for i in indices_in_remaining]
#
# # # 生成文件路径列表
# file_paths = [f"/data2/jr/yolov9-main/Dataset/coco/images/{idx}.jpg" for idx in indices]
# file_paths = [f"F:/yolov7-pytorch-master/yolov7-pytorch-master/VOCdevkit/VOC2007/images/{idx}.jpg" for idx in indices_in_remaining]
# 随机抽取 200 张作为 val，剩下 1800 张作为 train
# val_paths = random.sample(file_paths, 200)
# train_paths = [path for path in file_paths if path not in val_paths]
#
# 保存 val.txt
# val_file_path = "F:/yolov9-main/Dataset/coco/predict_2000-4000.txt"
# with open(val_file_path, "w") as f:
#     f.write("\n".join(file_paths))
#
# # 保存 train.txt
# # train_file_path = "F:/yolov9-main/Dataset/coco/train_2000.txt"
# # with open(train_file_path, "w") as f:
# #     f.write("\n".join(train_paths))
#
# # 保存到txt文件
# # file_path = "F:/yolov9-main/Dataset/coco/trianval_2000.txt"
# # with open(file_path, "w") as f:
# #     f.write("\n".join(file_paths))


# -------------------------------------
import re
import os


def extract_image_numbers(file_path):
    """
    从文本文件中提取所有图片的文件名，去除后缀后提取数字序号，
    返回一个数字字符串列表。
    """
    # 定义匹配图片文件名的正则表达式（支持常见图片格式）
    image_pattern = r'\b[\w.-]+(?:\.jpg|\.jpeg|\.png|\.gif|\.bmp|\.tiff|\.webp)\b'

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # 从内容中查找所有图片文件名
    image_names = re.findall(image_pattern, content)

    image_numbers = []
    for img in image_names:
        # 去掉后缀，例如 "123.jpg" -> "123"
        base_name = os.path.splitext(img)[0]
        # 假设文件名中的数字即为图片序号，提取连续的数字
        match = re.search(r'\d+', base_name)
        if match:
            image_numbers.append(match.group())
        else:
            # 如果没有数字，则直接添加处理后的文件名（也可以选择跳过）
            image_numbers.append(base_name)

    return image_numbers


def remove_image_numbers_from_range(file_path, start=1, end=8443):
    """
    构造[start, end]区间的数字列表，然后去除提取的图片序号对应的数字，
    返回剩余的数字列表。
    """
    # 确保传入的是一个文件路径字符串
    if not isinstance(file_path, str):
        raise TypeError("Expected file_path to be a string, but got {type(file_path)}")

    # 提取图片序号（字符串形式），转换为整数集合
    image_numbers_str = extract_image_numbers(file_path)
    image_numbers = set()
    for num_str in image_numbers_str:
        try:
            image_numbers.add(int(num_str))
        except ValueError:
            # 如果无法转换为整数，则忽略或根据需求进行处理
            pass

    # 构造完整数字集合（包含 start 到 end 的所有整数）
    full_set = set(range(start, end + 1))

    # 移除图片序号中的数字
    remaining_numbers = sorted(full_set - image_numbers)
    return remaining_numbers


def extract_equally_spaced_numbers(remaining_numbers, num_samples=2000):
    """
    从剩余的数字中按照等间隔抽取 num_samples 个数字。
    """
    k = ["2000-4000", "4000-6000", "6000-8000", "8000-8443"]
    selected_numbers= []
    for j in range(4):
        n = len(remaining_numbers)
        if n <= num_samples:
            save_path(remaining_numbers,k[j])
            return remaining_numbers  # 如果剩余数字小于等于2000，则返回所有数字

        # 计算间隔
        interval = n // num_samples

        for i in range(num_samples):
            selected_numbers.append(remaining_numbers[i * interval])
        remaining_numbers = sorted(set(remaining_numbers) - set(selected_numbers))
        save_path(selected_numbers, k[j])
        selected_numbers = []
def save_path(numbers, a):
    file_paths = [f"F:/yolov7-pytorch-master/yolov7-pytorch-master/VOCdevkit/VOC2007/images/{idx}.jpg" for idx in numbers]
    val_file_path = f"data_semi/predict_{a}.txt"
    with open(val_file_path, "w") as f:
        f.write("\n".join(file_paths))


# 使用示例
file_path = 'data_semi/trianval_2000.txt'  # 请确保这是一个文件路径字符串
# 第一轮剔除抽取
remaining_numbers = remove_image_numbers_from_range(file_path)
sampled_numbers = extract_equally_spaced_numbers(remaining_numbers, num_samples=2000)



