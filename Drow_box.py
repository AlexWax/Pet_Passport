from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


def draw_boxes(image_path: str, box_data: list, config: str) -> Image:
    img = cv2.imread(image_path)

    for box in box_data:
        x, y, w, h = box
        roi = img[y:y + h, x:x + w]
        text = pytesseract.image_to_string(roi)

        pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)

        center_x = int(np.mean(pts[:, 0]))
        center_y = int(np.mean(pts[:, 1]))

        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        text_color = (0, 0, 0)
        background_color = (255, 255, 255)

        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = center_x - text_width // 2
        text_y = center_y + text_height // 2

        cv2.rectangle(
            img,
            (text_x - 5, text_y - text_height - 5),
            (text_x + text_width + 5, text_y + 5),
            background_color,
            -1,
        )

        cv2.putText(
            img,
            text,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    return img


if __name__ == '__main__':
    draw_boxes('Photo/2.jpg', '~ 235 0 314 8 0').show()