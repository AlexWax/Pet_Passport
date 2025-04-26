import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw
from hezar.models import Model
from hezar.utils import load_image, draw_boxes, show_image
from ImagePreprocessing import preprocess_text_box
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
    text_out = dict()
    text_out['undef_text'] = []
    text_sn = ""
    box_sn = []

    for box in text_boxes:
        x, y, w, h = box
        roi = image[y:y + h, x:x + w]
        roi = preprocess_text_box(roi)[1]

        text = reader.readtext(roi, detail=1)
        if len(text) == 0 or h < image.shape[1] // 20:
            continue
        else:
            text = text[0][-2].capitalize()
        if x > 0.88 * image.shape[0]:
            text_sn += text
            box_sn.append(box)
        else:
            text_out['undef_text'].append((text, box))

    box_sn = np.array(box_sn)
    text_out['series+number'] = (text_sn, (min(box_sn[:, 0]), max(box_sn[:, 1]),
                                           max(box_sn[:, 2]), abs(box_sn[0, 1] - box_sn[-1, 1]) + box_sn[0, 3]))
    print(text_out)
    return text_out
