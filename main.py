import numpy as np
import re
import os
from BoxesSearch import text_boxes_search, photo_box_search, text_in_box_definition
from ImageDrawing import draw_boxes_let
from ImagePreprocessing import preprocess_text_box, scale_image, cut_rot_image
from Validation import cer_accuracy, box_check
import cv2


class Passport:
    def __init__(self, image_path):
        self.image_path_check(image_path)
        self.image_path = image_path
        self.image = self.prep_image()
        self.boxes = []
        self.find_boxes()
        self.show_text_in_passport()

    @staticmethod
    def image_path_check(image_path):
        pattern = r".jpg|.png|.jpeg"
        if not isinstance(image_path, str):
            raise TypeError(f"Ожидается строка, получено: {type(image_path).__name__}")
        if not [f for f in re.finditer(pattern, image_path)]:
            raise ValueError("Разрешение файла не указано или не верно")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл {image_path} не существует!")

    def prep_image(self):
        image = cv2.imread(self.image_path)
        photo_box = photo_box_search(image)
        image = cut_rot_image(image, photo_box)
        image = scale_image(image)
        return image

    def find_boxes(self):
        photo_box = photo_box_search(self.image)
        text_boxes_q = text_boxes_search(self.image)
        text_boxes = box_check(self.image, text_boxes_q)
        self.boxes.append(photo_box)
        self.boxes.extend(text_boxes)
        print(self.boxes)

    def show_text_in_passport(self):
        text = text_in_box_definition(image=self.image, text_boxes=self.boxes[1:])
        draw_boxes_let(image=self.image, box_data=self.boxes[1:], text=text)
        # cer_accuracy(image_path=self.image_path, predictions=text)
        print({str(box): text for box, text in zip(self.boxes[1:], text)})


files = [f"Photo/{elem}.png" for elem in [1, 2, 3, 4]]

passport = Passport(files[3])
