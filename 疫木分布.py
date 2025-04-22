import folium
import pandas as pd

# 读取数据
data = pd.read_csv("data.csv")

# 定义颜色映射
color_mapping = {0: "red", 1: "orange", 2: "green"}

# 创建地图
map_center = [data["latitude"].mean(), data["longitude"].mean()]
m = folium.Map(location=map_center, zoom_start=14, tiles="OpenStreetMap")

# 添加标记
for idx, row in data.iterrows():
    folium.CircleMarker(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        color=color_mapping[row["stage"]],
        fill=True,
        fill_color=color_mapping[row["stage"]],
        popup=f"Stage: {row['stage']}"
    ).add_to(m)

# 添加导出按钮
export_html = """
<div style="position: fixed; top: 10px; right: 10px; z-index: 1000; padding: 6px;">
  <button onclick="exportMap()" style="font-size: 16px; background: #fff; border: 2px solid #000; cursor: pointer;">
    🗺 导出地图
  </button>
</div>
<script>
function exportMap() {
  const htmlContent = document.documentElement.outerHTML;
  const blob = new Blob([htmlContent], { type: "text/html" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "疫木分布图.html";
  a.click();
}
</script>
"""
m.get_root().html.add_child(folium.Element(export_html))

# 保存地图
m.save("疫木分布图.html")