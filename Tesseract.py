import pytesseract
import numpy as np
from PIL import Image
from Drow_box import draw_boxes
from Cv2 import preprocess_image
import cv2
from Craft import find_box
from hezar.models import Model
from hezar.utils import load_image, draw_boxes, show_image

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


class Passport:
    def __init__(self, image_path):
        self.image_path = image_path
        self.config = '--psm 7 -c tessedit_char_whitelist=0123456789АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        self.model = Model.load("hezarai/CRAFT")

    def find_box(self):
        image = load_image(self.image_path)
        outputs = self.model.predict(image)
        return outputs[0]["boxes"]

    def extract_text_from_passport(self):
        img = Image.open(self.image_path)
        text_from_img = pytesseract.image_to_string(img, lang='rus')
        return text_from_img

    def show_text_in_passport(self):
        boxes = self.find_box()
        image = draw_boxes(image_path=self.image_path, box_data=boxes, config=self.config)
        return image


files = [f"Photo/{elem}.jpg" for elem in [1, 2, 3, 4]]

passport = Passport(files[0])
cv2.imwrite("new_path.jpg", passport.show_text_in_passport())

# print(show_text_in_passport(files[0])[1])