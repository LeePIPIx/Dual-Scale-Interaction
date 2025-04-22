import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4"
from ultralytics import RTDETR, YOLO
if __name__ == '__main__':
    # Load a model
    # model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-x.yaml")
    model = YOLO("ultralytics/cfg/models/v5/yolov5n.yaml")

    # Train the model with MPS
    results = model.train(data="Data/tree.yaml",
                        epochs=200,
                        imgsz=640,
                        device=[1,2,3,4],
                        batch=40,
                        workers=10,
                        lr0=0.01,
                        lrf=0.01,
                        # momentum=0.815, # (float) SGD momentum/Adam beta1
                        # amp=True,
                        )
