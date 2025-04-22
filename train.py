from ultralytics import YOLO,RTDETR

if __name__ == '__main__':
    # Load a model
    model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-x-lsm.yaml")  # load a pretrained model (recommended for training)
    # Train the model with MPS
    results = model.train(data="Data/tree.yaml", epochs=10, imgsz=640, device=0, batch=1, workers=0)