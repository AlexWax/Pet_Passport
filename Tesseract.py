import pytesseract
import numpy as np
from PIL import Image
from Drow_box import draw_boxes
from Cv2 import preprocess_image

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def extract_text_from_passport(image_path):
    img = Image.open(image_path)
    text_from_img = pytesseract.image_to_string(img, lang='rus')
    return text_from_img


def show_text_in_passport(image_path):
    prep_img_path = preprocess_image(image_path)
    print(prep_img_path)
    img = Image.open(prep_img_path)
    boxes = pytesseract.image_to_boxes(img)
    image = draw_boxes(image_path=prep_img_path, box_data=boxes)
    return image, boxes


files = ["Pasport4.jpg", "MyPassport1.jpg", "Foma_03.jpg"]
show_text_in_passport(files[0])[0].show()
# print(show_text_in_passport(files[0])[1])