import matplotlib.pyplot as plt
import numpy as np
from matplotlib.scale import FuncScale

# 示例数据
model_types = ['YOLOv5', 'YOLOv7', 'RT-DETR', 'YOLOv8', 'YOLOv9']
base_mAP50 = [50.5, 65.5, 92.2, 93.4, 93.5]  # 原网络的mAP50
wbsi_mAP50 = [71.9, 85.3, 92.4, 95.9, 97.1]  # 集成WBSI后的mAP50

# 设置柱子位置
y = np.arange(len(model_types))  # 生成 0, 1, 2, 3, 4
height = 0.35  # 柱子的垂直宽度

# 创建画布和轴
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
fig, ax = plt.subplots()

# 绘制横向柱状图
rects1 = ax.barh(y - height/2, base_mAP50, height, label='W/o DSI ', color='#2878B5')
rects2 = ax.barh(y + height/2, wbsi_mAP50, height, label='W/ DSI', color='#9AC9DB')

# 设置标签和标题
ax.set_xlabel('Early AP50 (%)')
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
ax.set_xlim(left=50)  # 从60开始，隐藏60以下的刻度
ax.set_xticks([50, 60, 70, 80, 90, 100])  # 设置60以上的刻度

# 调整布局并显示
plt.tight_layout()
plt.savefig("融合前后早期AP对比-1.png", dpi=300, bbox_inches='tight')
plt.show()