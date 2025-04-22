import matplotlib.pyplot as plt
import numpy as np
from matplotlib.scale import FuncScale

# 示例数据
model_types = ['YOLOv5', 'YOLOv7', 'RT-DETR', 'YOLOv8', 'YOLOv9']
base_mAP50 = [28.6, 33.8, 57.7, 62.5, 67.2]  # 原网络的mAP50
wbsi_mAP50 = [37.4, 43.6, 58.9, 66.9, 71.3]  # 集成WBSI后的mAP50

# 设置柱子位置
y = np.arange(len(model_types))  # 生成 0, 1, 2, 3, 4
height = 0.35  # 柱子的垂直宽度

# 创建画布和轴
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
fig, ax = plt.subplots()

# 绘制横向柱状图
rects1 = ax.barh(y - height/2, base_mAP50, height, label='原网络', color='#2878B5')
rects2 = ax.barh(y + height/2, wbsi_mAP50, height, label='集成 DSI', color='#9AC9DB')

# 设置标签和标题
ax.set_xlabel('mAP50-95 (%)')
ax.set_yticks(y)
ax.set_yticklabels(model_types)
ax.legend()

# 在柱子右侧添加数值标签
def autolabel(rects):
    for rect in rects:
        width = rect.get_width()  # 获取柱子的水平长度
        ax.annotate(f'{width:.1f}',
                    xy=(width, rect.get_y() + rect.get_height() / 2),
                    xytext=(3, 0),  # 向右偏移3个点
                    textcoords="offset points",
                    ha='left', va='center')

autolabel(rects1)
autolabel(rects2)
ax.set_xlim(left=20)  # 从60开始，隐藏60以下的刻度
ax.set_xticks([20, 30, 40, 50, 60, 70])  # 设置60以上的刻度

# 调整布局并显示
plt.tight_layout()
plt.savefig("融合前后mAP50-95对比.png", dpi=300, bbox_inches='tight')
plt.show()