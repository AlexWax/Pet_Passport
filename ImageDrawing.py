from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pytesseract
import easyocr
from ImagePreprocessing import scale_image, preprocess_image

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def draw_boxes_let(image_path: str, box_data: list) -> Image:
    image = cv2.imread(image_path)
    image = scale_image(image)
    reader = easyocr.Reader(["ru"])

    font_cv = cv2.FONT_HERSHEY_SIMPLEX
    font = ImageFont.truetype("arial.ttf", 10)
    font_scale = 0.4
    font_thickness = 1
    text_color = (0, 0, 0)
    background_color = (255, 255, 255)

    text_out = []
    text_sn = []

    for box in box_data:
        x, y, w, h = box
        roi = image[y:y + h, x:x + w]
        roi = preprocess_image(roi)[1]

        if h > w:
            roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

        text = reader.readtext(roi, detail=1)
        if len(text) == 0 or h < image.shape[1]//20:
            continue
        else:
            text = text[0][-2].capitalize()
        if x > 0.88*image.shape[0]:
            text_sn.append(text)
        else:
            text_out.append(text)
        print(text)
        pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)

        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))

        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        (text_width, text_height), _ = cv2.getTextSize(text, font_cv, font_scale, font_thickness)
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
        draw.text((text_x, text_y - text_height), text, font=font, fill=text_color)
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
    text_out.extend(text_sn)
    return image, text_out


if __name__ == '__main__':
    draw_boxes_let('Photo/2.jpg', '~ 235 0 314 8 0').show()