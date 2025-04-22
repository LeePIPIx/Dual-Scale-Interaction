import os


def filter_file_paths(file_a_path, file_b_path, output_path):
    """从文件A中移除文件B包含的所有路径"""

    # 读取文件B的所有路径并进行规范化处理
    with open(file_b_path, 'r') as f_b:
        # 使用集合存储以提高查找效率
        b_paths = set(
            os.path.normpath(line.strip())  # 标准化路径格式
            for line in f_b
            if line.strip()  # 忽略空行
        )

    # 逐行处理文件A
    preserved_lines = []
    with open(file_a_path, 'r') as f_a:
        for line in f_a:
            original_line = line.rstrip('\n')  # 保留原始换行符
            if not original_line:
                continue  # 跳过空行

            # 标准化当前路径用于比对
            normalized_line = os.path.normpath(original_line)

            # 保留未出现在B文件中的路径
            if normalized_line not in b_paths:
                preserved_lines.append(line)  # 保留原始行内容（含换行符）

    # 将结果写入新文件
    with open(output_path, 'w') as f_out:
        f_out.writelines(preserved_lines)


# 使用示例
filter_file_paths(
    file_a_path="data_semi/4000.txt",
    file_b_path="data_semi/val_2000.txt",
    output_path="data_semi/train_4000.txt"
)