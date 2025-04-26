from ultralytics import YOLO
import cv2
from PIL import Image, ImageDraw

img_path = "Photo/2.jpg"
model = YOLO("yolo11n")
results = model(img_path)

results[0].show()

data = dict()
for box in results[0].boxes:
    data[model.names[box.cls.item()]] = box.xyxy[0].tolist()

image = Image.open(img_path)
draw = ImageDraw.Draw(image)

min_x, min_y, max_x, max_y = data['person']
points = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]
draw.polygon(points, outline="green", fill=(255, 255, 255, 255))

image.save('Prep_' + img_path.strip('.jpg') + '_noise.jpg')