from ultralytics import YOLO
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
from hezar.models import Model
from hezar.utils import load_image, draw_boxes, show_image


def photo_box_search(image):
    model_for_img = YOLO("yolo11n")
    photo_box = model_for_img(image)
    return photo_box[0].boxes.xyxy[0].tolist()


def text_boxes_search(image):
    model_for_text = Model.load("hezarai/CRAFT")
    image = Image.fromarray(image)
    text_boxes = model_for_text.predict(image)
    return text_boxes[0]["boxes"]
