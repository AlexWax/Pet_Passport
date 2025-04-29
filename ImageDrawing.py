from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np


def draw_boxes_let(image: np.array, box_data: list):
    """
    Draw boxes with text on input image
    :param image: input image in tensor form
    :param box_data: list like: [text for box, box coordinates]
    :return: save changed image to "output.jpg"
    """
    font_cv = cv2.FONT_HERSHEY_SIMPLEX
    font = ImageFont.truetype("arial.ttf", 10)
    font_scale = 0.4
    font_thickness = 1
    text_color = (0, 0, 0)
    background_color = (255, 255, 255)

    for txt, box in box_data:
        x, y, w, h = box
        pts = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.int32)

        center_x = int(np.mean(pts[:, 0]))

        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        (text_width, text_height), _ = cv2.getTextSize(txt, font_cv, font_scale, font_thickness)
        text_x = center_x - text_width // 2
        text_y = int(y) - text_height // 2

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

    cv2.imwrite("output.jpg", image)


def transform_boxes(box_data: list, mode: str = 'h') -> tuple:
    """
    Merge list of boxes into one in two modes.
    :param box_data: list of boxes
    :param mode: 'h' - horizontal mode (merging in x-direction); 'v' - vertical mode (merging in y-direction)
    :return: tuple like: [x_min, y_min, width, height] of merged box
    """
    if mode == 'v':
        min_x = min(box_data, key=lambda x: x[0])[0]
        min_y = min(box_data, key=lambda x: x[1])[1]
        max_y = max(box_data, key=lambda x: x[1])[1]
        max_h = max(box_data, key=lambda x: x[1])[3]
        w = max(box_data, key=lambda x: x[2])[2]
        h = abs(max_y - min_y) + max_h
    elif mode == 'h':
        min_x = min(box_data, key=lambda x: x[0])[0]
        min_y = min(box_data, key=lambda x: x[1])[1]
        max_x = max(box_data, key=lambda x: x[0])[0]
        max_w = max(box_data, key=lambda x: x[0])[2]
        w = abs(max_x - min_x) + max_w
        h = max(box_data, key=lambda x: x[3])[3]
    else:
        raise AttributeError('Unknown mode!')
    box = tuple(float(elem) for elem in (min_x, min_y, w, h))
    return box
