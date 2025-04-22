import cv2
import os


def images_to_video(image_folder, output_video, fps=30):
    # 获取所有图片，并按照数字顺序排序
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]

    # 进行数值排序，确保文件名是数字，如 "1.jpg", "2.jpg", ..., "10.jpg"
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))  # 去掉扩展名并按数字排序

    if not images:
        print("❌ 该文件夹中没有找到图片！")
        return

    # 读取第一张图片，获取尺寸
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    h, w, _ = frame.shape  # 获取图片尺寸

    # 定义视频编码器（MP4格式）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # "mp4v" -> MP4格式
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    # 逐帧写入视频
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        video_writer.write(frame)  # 添加帧

    # 释放资源
    video_writer.release()
    print(f"✅ 视频已生成: {output_video}")


# 使用示例
image_folder = r"F:\ultralytics-main(v11)\output"  # 存放图片的文件夹路径（请修改为你的实际路径）
output_video = "output.mp4"  # 生成的视频文件名
fps = 30  # 设定帧率

images_to_video(image_folder, output_video, fps)
