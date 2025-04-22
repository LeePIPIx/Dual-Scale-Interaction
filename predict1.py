from ultralytics import YOLO

# Load a model
model = YOLO(r"F:\ultralytics-main(v11)\model_weight\V5_LSM\weights\best.pt")

# Predict with the model
results = model(r"C:\Users\Lee\Desktop\大论文\定性结果\7460.jpg")  # predict on an image

# Access the results
for result in results:
    # xywh = result.boxes.xywh  # center-x, center-y, width, height
    # xywhn = result.boxes.xywhn  # normalized
    # xyxy = result.boxes.xyxy  # top-left-x, top-left-y, bottom-right-x, bottom-right-y
    # xyxyn = result.boxes.xyxyn  # normalized
    # names = [result.names[cls.item()] for cls in result.boxes.cls.int()]  # class name of each box
    # confs = result.boxes.conf  # confidence score of each box
    result.save("annotated_image.jpg")