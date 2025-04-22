import numpy as np
import matplotlib.pyplot as plt
import matplotlib.scale as mscale
import matplotlib.transforms as mtransforms

# ---------------------------
# 1. 定义自定义分段线性缩放
# ---------------------------
class PiecewiseScale(mscale.ScaleBase):
    """
    分段线性缩放示例：
      - [y_min, y_split) 映射到 [0, ratio]
      - [y_split, y_max] 映射到 [ratio, 1]
    此处设定 y_min=50, y_split=90, y_max=95, ratio=0.8
    """
    name = 'piecewise'  # 自定义scale名称，可通过 ax.set_yscale("piecewise") 调用

    def __init__(self, axis, **kwargs):
        super().__init__(axis)
        self.y_min = 50
        self.y_split = 90
        self.y_max = 98
        self.ratio = 0.7  # 表示 [50,90] 占整个轴的80%，[90,95] 占20%

    def get_transform(self):
        return self.PiecewiseTransform(self.y_min, self.y_split, self.y_max, self.ratio)

    def set_default_locators_and_formatters(self, axis):
        # 可以自定义刻度，这里采用默认设置
        pass

    def limit_range_for_scale(self, vmin, vmax, minpos):
        return max(vmin, self.y_min), min(vmax, self.y_max)

    class PiecewiseTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1

        def __init__(self, y_min, y_split, y_max, ratio):
            super().__init__()
            self.y_min = y_min
            self.y_split = y_split
            self.y_max = y_max
            self.ratio = ratio
            self.range1 = y_split - y_min
            self.range2 = y_max - y_split

        def transform_non_affine(self, y):
            y = np.asarray(y)
            y = np.clip(y, self.y_min, self.y_max)
            out = np.zeros_like(y, dtype=float)
            mask1 = (y < self.y_split)
            mask2 = ~mask1
            out[mask1] = self.ratio * (y[mask1] - self.y_min) / self.range1
            out[mask2] = self.ratio + (1.0 - self.ratio) * (y[mask2] - self.y_split) / self.range2
            return out

        def inverted(self):
            return PiecewiseScale.InvertedPiecewiseTransform(self.y_min, self.y_split, self.y_max, self.ratio)

    class InvertedPiecewiseTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1

        def __init__(self, y_min, y_split, y_max, ratio):
            super().__init__()
            self.y_min = y_min
            self.y_split = y_split
            self.y_max = y_max
            self.ratio = ratio
            self.range1 = y_split - y_min
            self.range2 = y_max - y_split

        def transform_non_affine(self, x):
            x = np.asarray(x)
            out = np.zeros_like(x, dtype=float)
            mask1 = (x < self.ratio)
            mask2 = ~mask1
            out[mask1] = self.y_min + (x[mask1] / self.ratio) * self.range1
            out[mask2] = self.y_split + ((x[mask2] - self.ratio) / (1.0 - self.ratio)) * self.range2
            return out

        def inverted(self):
            return PiecewiseScale.PiecewiseTransform(self.y_min, self.y_split, self.y_max, self.ratio)

# 注册自定义Scale
mscale.register_scale(PiecewiseScale)

# ---------------------------
# 2. 准备表格数据
# ---------------------------
# 这里假设 x 轴代表 4 个评估阶段
x = np.array([1, 2, 3, 4])
x_labels = ["总体精度", "早期", "中期", "晚期"]

# 表格数据（数据为示例，实际请替换为真实数据）
data = {
    "YOLOv5": [62.9, 50.5, 77.3, 60.9],
    "YOLOv5(LSM)": [76.7, 71.9, 85.5, 72.8],
    "YOLOv7": [72.6, 65.5, 83.1, 69.2],
    "YOLOv7(LSM)": [86.1, 85.3, 90.7, 82.3],
    "YOLOv8": [92.5, 93.4, 94.8, 89.2],
    "YOLOv8(LSM)": [94.4, 95.9, 96.0, 91.3],
    "YOLOv9": [95.5, 94.8, 96.5, 95.2],
    "YOLOv9(LSM)": [96.4, 97.1, 97.0, 95.0],
    "RT-DETR": [90.8, 92.2, 91.2, 89.0],
    "RT-DETR(LSM)": [91.5, 92.4, 92.6, 89.6],
}

# 定义基础模型与颜色的对应关系
color_dict = {
    'YOLOv5':  '#1f77b4',  # 蓝色
    'YOLOv7':  '#2ca02c',  # 绿色
    'YOLOv8':  '#ff7f0e',  # 橙色
    'YOLOv9':  '#BEE13E',
    'RT-DETR': '#9467bd',  # 紫色
}

# ---------------------------
# 3. 绘制折线图（自定义分段缩放：折线连续）
# ---------------------------
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(figsize=(7, 5))

# 使用自定义分段线性缩放
ax.set_yscale("piecewise")

# 遍历每个模型数据，设置同色、(LSM)与非(LSM)采用不同线型
for model_name, values in data.items():
    if "(LSM)" in model_name:
        base_model = model_name.replace("(LSM)", "").strip()
        linestyle = '-'   # 实线
    else:
        base_model = model_name
        linestyle = '--'  # 虚线

    color = color_dict[base_model]
    ax.plot(x, values, marker='o', linestyle=linestyle, color=color, label=model_name)

# 设置 x 轴刻度及标签
ax.set_xticks(x)
ax.set_xticklabels(x_labels)

ax.set_xlabel("评估阶段")
ax.set_ylabel("检测精度\%")
ax.set_title("")
ax.legend(
    loc='lower right',
    ncol=3,
    fontsize=8,
    # frameon=True,
    columnspacing=0.8,  # 调整列间距
    handleheight=1.0,   # 调整行高
    bbox_to_anchor=(1.0, 0.0)  # 使图例靠近右下角
)

plt.tight_layout()
plt.savefig("融合前后精度对比.png", dpi=300, bbox_inches='tight')
plt.show()
