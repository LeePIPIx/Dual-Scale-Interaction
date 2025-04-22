import cv2
import os
from ultralytics import YOLO, RTDETR

# 加载 YOLO RTDETR 模型
model = YOLO(r"F:\ultralytics-main(v11)\model_weight\yolo11x.pt")

# 读取视频
video_path = r"F:\ultralytics-main(v11)\4.mp4"
cap = cv2.VideoCapture(video_path)

# 创建输出文件夹
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# 帧计数器
i = 0

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # 运行目标检测
        results = model.track(frame, persist=True)

        # 获取类别列表
        cls = results[0].boxes.cls

        # 创建 mask：只保留不等于 6 的那些框
        # keep_mask = cls != 6
        #
        # # 根据 mask 更新 boxes 中的各个字段
        # boxes = results[0].boxes
        # boxes.cls = boxes.cls[keep_mask]
        # boxes.conf = boxes.conf[keep_mask]
        # boxes.id = boxes.id[keep_mask] if boxes.id is not None else None
        # boxes.xyxy = boxes.xyxy[keep_mask]
        # 获取标注后的图像
        annotated_frame = results[0].plot()

        # 保存当前帧
        cv2.imwrite(f"{output_folder}/{i:04d}.jpg", annotated_frame)  # 按帧编号保存
        i += 1  # 递增帧编号

        # 显示当前帧
        cv2.imshow("YOLO Tracking", annotated_frame)

        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break  # 读取失败或视频结束

# 释放资源
cap.release()
cv2.destroyAllWindows()
