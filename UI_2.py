from PyQt6 import QtWidgets, QtCore, QtGui
import sys
import cv2
import numpy as np
import os
from ultralytics import YOLO, RTDETR
from deep_sort_realtime.deepsort_tracker import DeepSort
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

style_sheet = """
    QPushButton {
        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 #fdfdfd, stop:0.5 #e8e8e8, stop:1 #d6d6d6);
        border: 1px solid #8f8f91;
        border-radius: 6px;
        padding: 8px;
        font-size: 16px;
    }
    QPushButton:hover {
        border: 1px solid #6c6c6e;
    }
    QPushButton:pressed {
        background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 #d6d6d6, stop:0.5 #e8e8e8, stop:1 #fdfdfd);
        border: 1px solid #5e5e61;
    }
    QLabel {
        font-size: 18px;
    }
    QLineEdit {
        background-color: #FFF;
        border: 1px solid #CCC;
        border-radius: 4px;
        font-size: 18px;
        padding: 4px;
    }
"""


class ImageApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.resize(1800, 600)  # 设置默认窗口大小，可调整
        self.setMinimumSize(600, 200)  # 设置最小窗口大小，防止过小
        self.setWindowIcon(QtGui.QIcon("favicon.ico"))
        self.setStyleSheet(style_sheet)

        # 初始化 matplotlib 饼状图
        self.fig, self.ax = plt.subplots(figsize=(2, 2))  # 创建图形和轴
        self.canvas = FigureCanvas(self.fig)  # 创建嵌入 PyQt 的画布
        self.canvas.setFixedSize(240, 200)

        self.initUI()
        self.video_timer = QtCore.QTimer()
        self.video_timer.timeout.connect(self.update_video_frame)
        self.cap = None
        self.current_frame = None
        self.detect_mode = False
        self.media_type = None
        self.video_path = None
        self.file_path = None
        self.file_unopened = True
        self.frist_save = True
        self.save_txt_path = None
        self.video_detected = False
        self.tracked_ids = {"early": set(), "mid": set(), "late": set()}
        self.tracker = DeepSort(max_age=30, n_init=3, nn_budget=70)

    def add_shadow_effect(self, button):
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setXOffset(2)
        shadow.setYOffset(2)
        shadow.setColor(QtGui.QColor(0, 0, 0, 160))
        button.setGraphicsEffect(shadow)

    def initUI(self):
        self.setWindowTitle("松材线虫病检测")
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 上部区域
        top_widget = QtWidgets.QWidget(self)
        top_layout = QtWidgets.QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)

        # 展示区域
        img_path_layout = QtWidgets.QVBoxLayout(self)
        img_path_layout.setContentsMargins(0, 5, 0, 0)
        img_path_layout.setSpacing(0)

        # label区域
        label_layout = QtWidgets.QHBoxLayout(self)
        self.label_1 = QtWidgets.QLabel(self)
        self.label_1.setStyleSheet("border: 3px solid black; background-color: lightgray;")
        self.label_1.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label_layout.addWidget(self.label_1, 1)

        self.label_2 = QtWidgets.QLabel(self)
        self.label_2.setStyleSheet("border: 3px solid black; background-color: lightgray;")
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        label_layout.addWidget(self.label_2, 1)

        # 文件路径
        file_layout = QtWidgets.QHBoxLayout(self)
        file_name = QtWidgets.QLabel("文件路径", self)
        file_name.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.file_name = QtWidgets.QLineEdit(self)
        self.file_name.setReadOnly(True)
        file_layout.addWidget(file_name, 1)
        file_layout.addWidget(self.file_name, 8)

        # 新增模型选择框

        model_name = QtWidgets.QLabel("选择模型", self)
        model_name.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.combo_model = QtWidgets.QComboBox(self)
        self.combo_model.addItems(["YOLOv5(DSIN)", "YOLOv7(DSIN)", "YOLOv8(DSIN)", "YOLOv9(DSIN)", "RT-DETR(DSIN)"])  # 添加你需要的模型选项
        self.combo_model.setFixedHeight(35)
        font = self.combo_model.font()
        font.setPointSize(15)  # 设置字号为20，根据需要调整
        self.combo_model.setFont(font)
        # 可设置固定宽度或根据需要调整
        file_layout.addWidget(model_name, 1)
        file_layout.addWidget(self.combo_model, 1)
        img_path_layout.addLayout(label_layout)
        img_path_layout.addLayout(file_layout)


        # 右侧区域
        stats_widget = QtWidgets.QWidget(self)
        # stats_widget.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        stats_layout = QtWidgets.QVBoxLayout(stats_widget)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        stats_layout.setSpacing(5)
        top_layout.addWidget(stats_widget, 1)

        # 位置信息
        location_layout = QtWidgets.QVBoxLayout()
        hbox_city = QtWidgets.QHBoxLayout()
        label_city = QtWidgets.QLabel("地级市:", self)
        label_city.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.le_city = QtWidgets.QLineEdit(self)
        self.le_city.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.le_city.setReadOnly(True)
        # self.le_city.setFixedWidth(145)
        hbox_city.addWidget(label_city,1)
        hbox_city.addWidget(self.le_city,3)
        location_layout.addLayout(hbox_city,1)

        hbox_county = QtWidgets.QHBoxLayout()
        label_county = QtWidgets.QLabel("县区:", self)
        label_county.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.le_county = QtWidgets.QLineEdit(self)
        self.le_county.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.le_county.setReadOnly(True)
        # self.le_county.setFixedWidth(160)
        hbox_county.addWidget(label_county,1)
        hbox_county.addWidget(self.le_county,3)
        location_layout.addLayout(hbox_county,1)

        hbox_town = QtWidgets.QHBoxLayout()
        label_town = QtWidgets.QLabel("乡镇:", self)
        label_town.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.le_town = QtWidgets.QLineEdit(self)
        self.le_town.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.le_town.setReadOnly(True)
        # self.le_town.setFixedWidth(160)
        hbox_town.addWidget(label_town,1)
        hbox_town.addWidget(self.le_town,3)
        location_layout.addLayout(hbox_town,1)

        line = QtWidgets.QFrame(self)
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)

        # 经纬度信息
        GPS_layout = QtWidgets.QVBoxLayout()
        hbox_GPS_x = QtWidgets.QHBoxLayout()
        label_GPS_x = QtWidgets.QLabel("经度:", self)
        label_GPS_x.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.GPS_x = QtWidgets.QLineEdit(self)
        self.GPS_x.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.GPS_x.setReadOnly(True)
        hbox_GPS_x.addWidget(label_GPS_x,1)
        hbox_GPS_x.addWidget(self.GPS_x,3)
        GPS_layout.addLayout(hbox_GPS_x,1)

        hbox_GPS_y = QtWidgets.QHBoxLayout()
        label_GPS_y = QtWidgets.QLabel("纬度:", self)
        label_GPS_y.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.GPS_y = QtWidgets.QLineEdit(self)
        self.GPS_y.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.GPS_y.setReadOnly(True)
        hbox_GPS_y.addWidget(label_GPS_y,1)
        hbox_GPS_y.addWidget(self.GPS_y,3)
        GPS_layout.addLayout(hbox_GPS_y,1)

        line_1 = QtWidgets.QFrame(self)
        line_1.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line_1.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)

        # 类别统计
        stats_layout2 = QtWidgets.QVBoxLayout()
        hbox_stat_early = QtWidgets.QHBoxLayout()
        label_stat_early = QtWidgets.QLabel("早期:", self)
        label_stat_early.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.le_stat_early = QtWidgets.QLineEdit(self)
        self.le_stat_early.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.le_stat_early.setReadOnly(True)
        # self.le_stat_early.setFixedWidth(160)
        hbox_stat_early.addWidget(label_stat_early,1)
        hbox_stat_early.addWidget(self.le_stat_early,3)
        stats_layout2.addLayout(hbox_stat_early,1)

        hbox_stat_mid = QtWidgets.QHBoxLayout()
        label_stat_mid = QtWidgets.QLabel("中期:", self)
        label_stat_mid.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.le_stat_mid = QtWidgets.QLineEdit(self)
        self.le_stat_mid.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.le_stat_mid.setReadOnly(True)
        # self.le_stat_mid.setFixedWidth(160)
        hbox_stat_mid.addWidget(label_stat_mid,1)
        hbox_stat_mid.addWidget(self.le_stat_mid,3)
        stats_layout2.addLayout(hbox_stat_mid,1)

        hbox_stat_late = QtWidgets.QHBoxLayout()
        label_stat_late = QtWidgets.QLabel("晚期:", self)
        label_stat_late.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.le_stat_late = QtWidgets.QLineEdit(self)
        self.le_stat_late.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.le_stat_late.setReadOnly(True)
        # self.le_stat_late.setFixedWidth(160)
        hbox_stat_late.addWidget(label_stat_late,1)
        hbox_stat_late.addWidget(self.le_stat_late,3)
        stats_layout2.addLayout(hbox_stat_late,1)

        # 添加饼状图布局
        pie_layout = QtWidgets.QVBoxLayout()
        pie_layout.setContentsMargins(0,0,0,0)
        pie_layout.addWidget(self.canvas)

        # 合并布局
        stats_layout.addStretch(1)
        stats_layout.addLayout(location_layout, 3)
        stats_layout.addWidget(line)
        stats_layout.addLayout(GPS_layout, 2)
        stats_layout.addWidget(line_1)
        stats_layout.addLayout(stats_layout2, 3)
        stats_layout.addLayout(pie_layout, 4)  # 添加饼状图
        stats_layout.addStretch(1)

        top_layout.addLayout(img_path_layout, 6)
        top_layout.addWidget(stats_widget,1)


        # 按钮区域
        bottom_widget = QtWidgets.QWidget(self)
        button_layout = QtWidgets.QHBoxLayout(bottom_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(0)

        self.btn1 = QtWidgets.QPushButton("🎞️ 选择图片/视频", self)
        self.btn2 = QtWidgets.QPushButton("📷 打开镜头", self)
        self.btn3 = QtWidgets.QPushButton("🔍 检测", self)
        self.btn4 = QtWidgets.QPushButton("🛑 停止", self)
        for btn in [self.btn1, self.btn2, self.btn3, self.btn4]:
            btn.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
            # btn.setFixedHeight(50)
            self.add_shadow_effect(btn)
            button_layout.addWidget(btn)
        button_layout.setStretch(0, 1)
        button_layout.setStretch(1, 1)
        button_layout.setStretch(2, 1)
        button_layout.setStretch(3, 1)
        main_layout.addWidget(top_widget, 5)
        main_layout.addWidget(bottom_widget, 1)

        self.btn1.clicked.connect(self.load_media)
        self.btn2.clicked.connect(self.open_camera)
        self.btn3.clicked.connect(self.toggle_detection)
        self.btn4.clicked.connect(self.stop_capture)
        self.combo_model.currentIndexChanged.connect(self.change_model)

    def change_model(self):
        model_name = self.combo_model.currentText()
        if model_name == "YOLOv5(DSIN)":
            self.model = YOLO(r"model_weight/yolov9c/weights/best.pt")
        elif model_name == "YOLOv7(DSIN)":
            self.model = YOLO("model_weight/V7/weights/best.pt")
        elif model_name == "YOLOv8(DSIN)":
            self.model = YOLO("model_weight/V8+LM+SM1+SM2/weights/best.pt")
        # elif model_name == "YOLOv9(DSIN)":
        #     self.model = YOLO("model_weight/V8+LM+SM1+SM2/weights/best.pt")
        elif model_name == "RT-DETR(DSIN)":
            self.model = RTDETR("model_weight/RTDETR/weights/best.pt")
            # 如有更多模型，可继续添加 elif 分支
        print(f"当前选择模型：{model_name}")

    def update_pie_chart(self):
        """更新饼状图显示"""
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置支持中文的字体
        plt.rcParams['axes.unicode_minus'] = False
        counts = {
            "早期": len(self.tracked_ids["early"]),
            "中期": len(self.tracked_ids["mid"]),
            "晚期": len(self.tracked_ids["late"])
        }
        total = sum(counts.values())
        self.ax.clear()
        if total == 0:
            self.ax.text(0.5, 0.5, "无数据", horizontalalignment='center', verticalalignment='center')
        else:
            labels = [k for k, v in counts.items() if v > 0]
            sizes = [v for v in counts.values() if v > 0]
            self.ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=0)
            self.ax.axis('equal')  # 确保饼状图为圆形
        self.canvas.draw()

    def load_media(self):
        self.stop_capture()
        self.detect_mode = False
        self.tracked_ids = {"early": set(), "mid": set(), "late": set()}
        self.le_stat_early.clear()
        self.le_stat_mid.clear()
        self.le_stat_late.clear()
        self.le_city.clear()
        self.le_county.clear()
        self.le_town.clear()
        self.update_pie_chart()  # 初始化饼状图
        file_dialog = QtWidgets.QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "选择文件", r"F:\yolov7-pytorch-master\yolov7-pytorch-master\VOCdevkit\VOC2007\images",
            "Image/Video Files (*.png *.jpg *.bmp *.mp4 *.avi *.mov)"
        )
        if file_path:
            self.file_unopened = False
            self.file_path = file_path
            if file_path.lower().endswith(('.png', '.jpg', '.bmp')):
                self.media_type = "image"
                self.display_image(file_path)
                self.file_name.setText(file_path)
                filename = os.path.basename(file_path)
                name_without_ext, _ = os.path.splitext(filename)
                parts = name_without_ext.split('-')
                if len(parts) >= 3:
                    self.le_city.setText(parts[0])
                    self.le_county.setText(parts[1])
                    self.le_town.setText(parts[2])
                    self.GPS_x.setText(parts[3])
                    self.GPS_y.setText(parts[4])
            elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                self.media_type = "video"
                self.video_detected = True
                self.video_path = file_path
                self.file_name.setText(file_path)
                filename = os.path.basename(file_path)
                name_without_ext, _ = os.path.splitext(filename)
                parts = name_without_ext.split('-')
                if len(parts) >= 3:
                    self.le_city.setText(parts[0])
                    self.le_county.setText(parts[1])
                    self.le_town.setText(parts[2])
                    self.GPS_x.setText(parts[3])
                    self.GPS_y.setText(parts[4])
                cap_temp = cv2.VideoCapture(file_path)
                ret, frame = cap_temp.read()
                if ret:
                    self.current_frame = frame.copy()
                    frame_disp = cv2.resize(frame, (self.label_1.width(), self.label_1.height()))
                    self.show_on_label(frame_disp, self.label_1)
                cap_temp.release()

    def open_camera(self):
        self.stop_capture()
        self.detect_mode = False
        self.media_type = "camera"
        self.le_city.clear()
        self.le_county.clear()
        self.le_town.clear()
        self.tracked_ids = {"early": set(), "mid": set(), "late": set()}
        self.le_stat_early.clear()
        self.le_stat_mid.clear()
        self.le_stat_late.clear()
        self.update_pie_chart()  # 初始化饼状图
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return
        self.video_timer.start(30)

    def display_image(self, file_path):
        img = cv2.imread(file_path)
        self.current_frame = img.copy()
        img_disp = cv2.resize(img, (self.label_1.width(), self.label_1.height()))
        self.show_on_label(img_disp, self.label_1)
        if self.detect_mode:
            self.detect_and_show(self.current_frame)

    def start_video(self, file_path):
        self.cap = cv2.VideoCapture(file_path)
        if not self.cap.isOpened():
            print("无法打开视频文件")
            return
        self.video_timer.start(30)

    def update_video_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                frame_disp = cv2.resize(frame, (self.label_1.width(), self.label_1.height()))
                self.current_frame = frame.copy()
                self.show_on_label(frame_disp, self.label_1)
                if self.detect_mode:
                    self.detect_and_show(frame)
                else:
                    self.label_2.clear()

    def stop_capture(self):
        if not self.file_unopened:
            self.save_detect_result()
        # 计算百分比
        if self.cap:
            self.cap.release()
            self.cap = None

        self.tracked_ids = {"early": set(), "mid": set(), "late": set()}  # 清空统计数据
        # 保存检测结果
        self.video_timer.stop()
        self.label_1.clear()
        self.label_2.clear()
        self.GPS_y.clear()
        self.GPS_x.clear()
        self.file_name.clear()
        self.le_city.clear()
        self.le_county.clear()
        self.le_town.clear()
        self.le_stat_early.clear()
        self.le_stat_mid.clear()
        self.le_stat_late.clear()
        self.update_pie_chart()  # 清空饼状图

    def save_detect_result(self):
        a = len(self.tracked_ids["early"])
        b = len(self.tracked_ids["mid"])
        c = len(self.tracked_ids["late"])
        e, m, l = self.percentage(a, b, c)
        save_path = self.file_path
        if self.frist_save:
            folder_path = os.path.dirname(save_path)
            folder_path = folder_path + r"/output/"
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            file_count = sum(len(files) for _, _, files in os.walk(folder_path))
            txt_name = "result{}.txt".format(file_count + 1)
            self.save_txt_path = os.path.join(folder_path, txt_name)
        with open(self.save_txt_path, "a", encoding="utf-8") as file:
            file.write(
                "输入源：" + self.file_path + "\n" +
                "地点：" + self.le_city.text() + "," +
                self.le_county.text() + "," +
                self.le_town.text() + "\n"
                "经度：" + self.GPS_x.text() + "  " +
                "纬度：" + self.GPS_y.text() + "\n"
                "早期：" + str(len(self.tracked_ids["early"])) + " 百分比：" + str(round(e, 2)) + "%  " +
                "中期：" + str(len(self.tracked_ids["mid"])) + " 百分比：" + str(round(m, 2)) + "%  " +
                "晚期：" + str(len(self.tracked_ids["late"])) + " 百分比：" + str(round(l, 2)) + "% \n"
                + "\n"
            )
        self.frist_save = False
        self.file_unopened = True

    def toggle_detection(self):
        self.detect_mode = not self.detect_mode
        if self.detect_mode:
            if self.media_type == "video" and self.cap is None:
                self.start_video(self.video_path)
            if self.current_frame is not None:
                self.detect_and_show(self.current_frame)
        else:
            self.label_2.clear()

    def percentage(self,a,b,c):
        total = a + b + c
        # 计算百分比
        a_percentage = (float(a) / float(total)) * 100
        b_percentage = (float(b) / float(total)) * 100
        c_percentage = (float(c) / float(total)) * 100
        return a_percentage, b_percentage, c_percentage

    def detect_and_show(self, frame):
        if self.media_type == "image":
            if self.video_detected :
                self.change_model()
                self.video_detected = False
            results = self.model(frame, iou=0.2)[0]
            for cls in results.boxes.cls:
                if int(cls) == 0:
                    self.tracked_ids["mid"].add(cls)
                elif int(cls) == 1:
                    self.tracked_ids["early"].add(cls)
                elif int(cls) == 2:
                    self.tracked_ids["late"].add(cls)
            detected_frame = self.plot_boxes(results, frame.copy())
        elif self.media_type in ["video", "camera"]:
            results = self.model.track(frame, persist=True, conf=0.5)
            detected_frame = results[0].plot(line_width=2)
            for box in results[0].boxes:
                track_id = int(box.id)
                cls = int(box.cls)
                if cls == 0:
                    cat = "mid"
                elif cls == 1:
                    cat = "early"
                elif cls == 2:
                    cat = "late"
                else:
                    continue
                self.tracked_ids[cat].add(track_id)
        # 更新类别统计和饼状图

        self.le_stat_early.setText(str(len(self.tracked_ids["early"])))
        self.le_stat_mid.setText(str(len(self.tracked_ids["mid"])))
        self.le_stat_late.setText(str(len(self.tracked_ids["late"])))
        self.update_pie_chart()
        self.show_on_label(detected_frame, self.label_2)
    def plot_boxes(self, results, frame):
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            cls = int(box.cls)
            if cls == 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  #BGR
                text = f"{results.names[cls]} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif cls == 1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                text = f"{results.names[cls]} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif cls == 2:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 3)
                text = f"{results.names[cls]} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        return frame

    def show_on_label(self, img, label):
        img_disp = cv2.resize(img, (self.label_1.width(), self.label_1.height()))
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
        height, width, channel = img_disp.shape
        bytes_per_line = 3 * width
        q_img = QtGui.QImage(img_disp.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        label.setPixmap(QtGui.QPixmap.fromImage(q_img))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ImageApp()
    window.show()
    sys.exit(app.exec())