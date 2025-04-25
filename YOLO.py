from ultralytics import YOLO

model = YOLO("yolo11n")
results = model("Pasport4.jpg")

data = dict()
for box in results[0].boxes:
    data[model.names[box.cls.item()]] = box.xyxy[0].tolist()

print(data)