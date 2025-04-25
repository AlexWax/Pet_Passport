import cv2
import numpy as np


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('Prep_' + image_path.strip('.jpg') + '_grey.jpg', gray)

    kernel = np.ones((1, 1), np.uint8)
    dil = cv2.dilate(gray, kernel, iterations=3)
    ero = cv2.erode(dil, kernel, iterations=3)
    cv2.imwrite('Prep_' + image_path.strip('.jpg') + '_noise.jpg', ero)

    blur = cv2.GaussianBlur(ero, (3, 3), 0)
    thresh = cv2.threshold(blur, 160, 255, cv2.THRESH_BINARY)

    new_path = 'Prep_' + image_path.strip('.jpg') + '_bt.jpg'
    cv2.imwrite(new_path, thresh[1])

    return new_path

if __name__ == "__main__":
    preprocess_img = preprocess_image("MyPassport1.jpg")
