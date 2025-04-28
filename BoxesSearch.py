import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
from hezar.models import Model
from hezar.utils import load_image, draw_boxes, show_image
from ImagePreprocessing import preprocess_text_box, rot_image
import easyocr



def photo_box_search(image):
    model_for_img = YOLO("yolo11n")
    photo_box = model_for_img(image)
    return photo_box[0].boxes.xyxy[0].tolist()


def text_boxes_search(image):
    model_for_text = Model.load("hezarai/CRAFT")
    image = Image.fromarray(image)
    text_boxes = model_for_text.predict(image)
    return text_boxes[0]["boxes"]


def text_in_box_definition(image, text_boxes):
    reader = easyocr.Reader(["ru"])
    text_out = []

    for box in text_boxes:
        x, y, w, h = box
        roi = image[y:y + h, x:x + w]
        #roi = preprocess_text_box(roi)[1]
        roi = rot_image(roi)

        text = reader.readtext(roi, detail=0)
        if len(text) == 0:
            text = '?'
        else:
            text = text[0].lower().strip(".,:!`'?/|[](){} ")
        text_out.append(text)
    return text_out
