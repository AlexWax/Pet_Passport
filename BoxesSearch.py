from ultralytics import YOLO
from PIL import Image
from hezar.models import Model
from ImagePreprocessing import preprocess_text_box, rot_image
import easyocr
import numpy as np


def photo_box_search(image: np.array) -> list:
    """
    YOLO-model application , v-11
    :param image: input image in tensor form
    :return: list with founded photo-box coordinates like: [x_min, y_min, x_max, y_max]
    """
    model_for_img = YOLO("yolo11n", verbose=False)
    photo_box = model_for_img(image, verbose=False)
    return photo_box[0].boxes.xyxy[0].tolist()


def text_boxes_search(image: np.array) -> any:
    """
    CRAFT-model application
    :param image: input image in tensor form
    :return: list of boxes containing text like: [[x_min, y_min, h, w], ...]
    """
    model_for_text = Model.load("hezarai/CRAFT")
    image = Image.fromarray(image)
    text_boxes = model_for_text.predict(image)
    return text_boxes[0]["boxes"]


def text_in_box_definition(image: np.array, text_boxes: list) -> list:
    """
    EasyOCR-model application. Language - russian
    :param image: input image in tensor form
    :param text_boxes: list of boxes containing text like: [[x_min, y_min, h, w], ...]
    :return: list of text found in text_boxes like: [txt, txt, ...]
    """
    reader = easyocr.Reader(["ru"], verbose=False)
    text_out = []

    for box in text_boxes:
        x, y, w, h = box
        roi = image[y:y + h, x:x + w]
        # roi = preprocess_text_box(roi)[1]
        roi = rot_image(roi)

        text = reader.readtext(roi, detail=0)
        if len(text) == 0:
            text = '?'
        else:
            text = text[0].lower().strip(".,:!`'?/|[](){} ")
        text_out.append(text)
    return text_out
