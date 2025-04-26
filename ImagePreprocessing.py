import cv2
import numpy as np


def cut_rot_image(image, photo_box):
    return image


def scale_image(image, target_height=400):
    original_height, original_width = image.shape[:2]
    scale = target_height / original_height
    new_width = int(original_width * scale)
    resized_image = cv2.resize(image, (new_width, target_height))
    return resized_image


def preprocess_text_box(image):
    if image.shape[1] > image.shape[0]:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    kernel = np.ones((1, 1), np.uint8)
    dil = cv2.dilate(gray, kernel, iterations=3)
    ero = cv2.erode(dil, kernel, iterations=3)

    blur = cv2.GaussianBlur(ero, (3, 3), 0)
    thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)

    img_out = thresh
    return img_out


if __name__ == "__main__":
    preprocess_img = preprocess_text_box("MyPassport1.jpg")
