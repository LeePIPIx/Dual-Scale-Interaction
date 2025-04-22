from ultralytics import YOLO, RTDETR
import cv2

# 加载模型
model = RTDETR(r"F:\ultralytics-main(v11)\model_weight\RTDETR-LSM\weights\best.pt")
# model = RTDETR("model_weight/RTDETR-LSM/weights/best.pt")
# 读取图片
img_path = r"C:\Users\Lee\Desktop\大论文\定性结果\7460.jpg"
img = cv2.imread(img_path)

# 运行推理
results = model([img_path], iou=0.9, conf=0.3)

# 取第一个检测结果
result = results[0]

# 遍历所有检测框
for box in result.boxes:
    # 获取检测框坐标，并转换为整数
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

    # 获取类别索引（转换为整数）
    cls = int(box.cls[0])

    # 根据类别选择颜色
    if cls == 0:
        color = (0, 0, 255)  # 红色BGR
    elif cls == 1:
        color = (0, 255, 0)  # 黄
    elif cls == 2:
        color = (255, 0, )  # 灰
    else:
        color = (255, 255, 255)  # 其他类别为白色

    # 绘制矩形框
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)

    # 置信度转换为 float
    confidence = float(box.conf[0])

    # 生成文本
    label = f"{result.names[cls]} {confidence:.2f}"

    # 计算文本大小
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 8
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # 设置文本背景框
    text_x1 = x1
    text_y1 = y1 - text_height - 5  # 让文本上移一点，避免贴着框
    text_x2 = x1 + text_width + 4
    text_y2 = y1
    # 绘制文本背景框（颜色设置为检测框颜色的淡色）
    cv2.rectangle(img, (text_x1, text_y1), (text_x2, text_y2), color, -1)

    # 绘制文本
    cv2.putText(img, label, (x1, y1 - 5), font, font_scale, (0, 0, 0), 2)

# cv2.imshow("Detection Result", img)
# cv2.waitKey(0)
# 保存结果图片
cv2.imwrite(r"DSI-RTDETR.jpg", img)
