import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 定义混淆矩阵
cm = np.array([
    [3044, 1, 5, 315],    # Predicted red
    [5, 1554, 0, 224],    # Predicted yellow
    [6, 3, 2352, 440],    # Predicted gray
    [143, 102, 223, 0]    # Predicted background
])

# 定义类别标签
labels = ["red", "yellow", "gray", "background"]

mask = cm == 0

# 设置图形大小
plt.figure(figsize=(10, 8))

# 绘制热图
sns.heatmap(
    cm,
    annot=True,              # 显示每个单元格的数值
    fmt='d',                 # 数值格式为整数
    cmap='Blues',            # 使用蓝色渐变颜色
    vmin=0, vmax=3000,       # 颜色范围从 0 到 3000
    xticklabels=labels,      # x 轴标签（真实类别）
    yticklabels=labels,      # y 轴标签（预测类别）
    cbar_kws={'ticks': [0, 500, 1000, 1500, 2000, 2500, 3000]},  # 颜色条刻度
    linewidths=0.5,           # 网格线宽度，轻且微妙
    mask=mask,                # 应用掩码，隐藏为0的单元格
    annot_kws={"size": 14}  # 设置单元格内数字的字体大小
)

# 设置轴标签和标题
plt.xlabel("True", fontsize=20)           # x 轴标签为 "True"
plt.ylabel("Predicted", fontsize=20)      # y 轴标签为 "Predicted"

# 调整 x 轴和 y 轴刻度标签的字体大小
plt.xticks(fontsize=18)              # x 轴刻度标签（如 "red", "yellow"）字体大小设为 12
plt.yticks(fontsize=18)              # y 轴刻度标签（如 "red", "yellow"）字体大小设为 12

plt.savefig("confusion_matrix.png", dpi=640, bbox_inches='tight')



# 显示图形
plt.show()
