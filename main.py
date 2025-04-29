import re
import os
from BoxesSearch import text_boxes_search, photo_box_search, text_in_box_definition
from OutputGeneration import HeuristicFieldSearch, output_generation
from ImagePreprocessing import scale_image, cut_rot_image
from Validation import cer_accuracy, box_check
import cv2
import numpy as np


class Passport:
    """
    Passport tusk realisation
    """
    def __init__(self, image_path):
        """
        :param image_path: path to image with passport
        """
        self.image_path_check(image_path)
        self.image_path = image_path
        self.image = self.prep_image()
        self.boxes = []
        self.text = []
        self.find_boxes()
        self.find_text_in_boxes()
        self.create_output()

    @staticmethod
    def image_path_check(image_path):
        """
        Light check of input path
        :param image_path: path to image with passport
        :return: None
        """
        pattern = r".jpg|.png|.jpeg"
        if not isinstance(image_path, str):
            raise TypeError(f"Ожидается строка, получено: {type(image_path).__name__}")
        if not [f for f in re.finditer(pattern, image_path)]:
            raise ValueError("Разрешение файла не указано или не верно")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Файл {image_path} не существует!")

    def prep_image(self) -> np.array:
        """
        Image preprocessing using heuristics
        :return: preprocessed image in tensor form
        """
        image = cv2.imread(self.image_path)
        photo_box = photo_box_search(image)
        image = cut_rot_image(image, photo_box)
        image = scale_image(image)
        return image

    def find_boxes(self):
        """
        Boxes detection using YOLO, CRAFT
        """
        photo_box = photo_box_search(self.image)
        text_boxes_q = text_boxes_search(self.image)
        text_boxes = box_check(self.image, text_boxes_q)
        self.boxes.append(photo_box)
        self.boxes.extend(text_boxes)

    def find_text_in_boxes(self):
        """
        Text definition using EasyOCR
        """
        text = text_in_box_definition(image=self.image, text_boxes=self.boxes[1:])
        self.text.extend(text)

    def create_output(self):
        """
        Create output files: .json, .jpg
        Get accuracy metrics for predicted results
        """
        hfc = HeuristicFieldSearch(self.text, self.boxes)
        output_dict = hfc.return_output()
        output_generation(self.image, output_dict)
        # cer_accuracy(image_path=self.image_path, predictions=text)


if __name__ == "__main__":
    # file_path = input("Input file path: ")
    file_path = "Photo/2.jpg"
    passport = Passport(file_path)
