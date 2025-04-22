import matplotlib.pyplot as plt
import numpy as np

# 数据
model_types = ['YOLOv5', 'YOLOv7', 'RT-DETR', 'YOLOv8', 'YOLOv9']
base_mAP50 = [62.9, 72.6, 90.8, 92.5, 94.2]  # 原网络的mAP50
wbsi_mAP50 = [76.7, 86.1, 91.5, 94.4, 96.4]  # 集成WBSI后的mAP50

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
ax.set_xlabel('mAP50 (%)')
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

# 设置x轴显示范围和刻度
ax.set_xlim(left=60)  # 从60开始，隐藏60以下的刻度
ax.set_xticks([60, 70, 80, 90, 100])  # 设置60以上的刻度

# 调整布局并保存
plt.tight_layout()
plt.savefig("融合前后mAP50对比.png", dpi=300, bbox_inches='tight')
plt.show()
