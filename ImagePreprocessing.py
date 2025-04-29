import cv2
import numpy as np


def cut_rot_image(image: np.array, photo_box: list) -> np.array:
    """
    Cut image using photo as anchor
    :param image: input image in tensor form
    :param photo_box: [x_min, y_min, h, w] of photo on image
    :return: transformed image
    """
    x_min, y_min, x_max, y_max = [int(elem) for elem in photo_box]
    w, h = x_max - x_min, y_max - y_min
    roi = image[int(y_min-h/2.4):int(y_max+h/2.4), x_min:x_max*4]
    return roi


def rot_image(image: np.array) -> np.array:
    """
    Image rotation for better text search in vertical boxes. Ex. series+number
    :param image: input image in tensor form
    :return: transformed image
    """
    if image.shape[0] > image.shape[1]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def scale_image(image: np.array, target_height: int = 400) -> np.array:
    """
    Scale image to target_height keeping the proportions right
    :param image: input image in tensor form
    :param target_height: target output height
    :return: transformed image
    """
    original_height, original_width = image.shape[:2]
    scale = target_height / original_height
    new_width = int(original_width * scale)
    resized_image = cv2.resize(image, (new_width, target_height))
    return resized_image


def preprocess_text_box(image: np.array) -> np.array:
    """
    Image transformation for better accuracy
    :param image: input image in tensor form
    :return: transformed image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    kernel = np.ones((1, 1), np.uint8)
    dil = cv2.dilate(gray, kernel, iterations=3)
    ero = cv2.erode(dil, kernel, iterations=3)

    # blur = cv2.GaussianBlur(ero, (3, 3), 0)
    thresh = cv2.threshold(ero, 170, 255, cv2.THRESH_BINARY)

    img_out = thresh
    return img_out
