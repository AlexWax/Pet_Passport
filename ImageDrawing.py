from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pytesseract
from ImagePreprocessing import scale_image, preprocess_text_box

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def draw_boxes_let(image: np.array, box_data: list, text) -> Image:

    font_cv = cv2.FONT_HERSHEY_SIMPLEX
    font = ImageFont.truetype("arial.ttf", 10)
    font_scale = 0.4
    font_thickness = 1
    text_color = (0, 0, 0)
    background_color = (255, 255, 255)

    for txt, box in zip(text, box_data):
        x, y, w, h = box
        pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)

        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))

        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        (text_width, text_height), _ = cv2.getTextSize(txt, font_cv, font_scale, font_thickness)
        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        cv2.rectangle(
            image,
            (text_x - 5, text_y - text_height - 5),
            (text_x + text_width + 5, text_y + 5),
            background_color,
            -1,
        )

        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        draw.text((text_x, text_y - text_height), txt, font=font, fill=text_color)
        image = np.array(image)
        """cv2.putText(
            image,
            text,
            (text_x, text_y),
            font_cv,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )"""

    cv2.imwrite("new_path.jpg", image)



if __name__ == '__main__':
    imag = draw_boxes_let(cv2.imread('Photo/2.jpg'), [484, 218, 22, 70], 'text')
    cv2.imwrite("new_path.jpg", imag)