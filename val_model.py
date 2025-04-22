from ultralytics import YOLO,RTDETR

# Load a model
model = RTDETR(r"runs/detect/RTDETR-LSM/weights/best.pt")  # load a custom model

if __name__ == '__main__':
    # Validate the model
    metrics = model.val(data="Data/tree.yaml", batch=1, device="0", half=True, save_json=True)  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

