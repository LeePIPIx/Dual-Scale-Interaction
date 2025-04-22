from ultralytics import YOLO,RTDETR
if __name__ == '__main__':
    # Load a model
    model = YOLO(r"F:\ultralytics-main\实验结果\V8+LM+SM1+SM2\weights\best.pt")  # load an official model

    # Validate the model
    metrics = model.val(data="Data/tree.yaml", imgsz=640, batch=2, conf=0.25, iou=0.6, device="0")
    print(metrics.box.map)  # mAP50-95
    print(metrics.box.map50)  # mAP50
    print(metrics.box.map75)  # mAP75
    print(metrics.box.maps)  # list of mAP50-95 for each category