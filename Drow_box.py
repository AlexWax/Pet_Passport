from PIL import Image, ImageDraw, ImageFont
import numpy as np


def draw_boxes(image_path: str, box_data: str) -> Image:
    image = Image.open(image_path)
    _, height = image.size
    draw = ImageDraw.Draw(image)

    a = np.array([elem.split() for elem in box_data.strip().split('\n')])
    for elem in a:
        let, box_vertices = elem[0], elem[1:-1]
        min_x, min_y, max_x, max_y = [int(elem) for elem in box_vertices]
        points = [(min_x, height - min_y), (max_x, height - min_y), (max_x, height - max_y), (min_x, height - max_y)]

        font = ImageFont.truetype("arial.ttf", 10)
        text_color = (255, 255, 255)
        background_color = (0, 0, 0)

        text_bbox = draw.textbbox((0, 0), let, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        center_x = sum(p[0] for p in points) / len(points)
        center_y = sum(p[1] for p in points) / len(points)

        draw.rectangle(
            (center_x - text_width / 2, center_y - text_height / 2 - 1,
                center_x + text_width / 2 + 1, center_y + text_height / 2 + 3),
            fill=background_color,
        )

        """draw.rectangle(
            (min_x, height - max_y, max_x, height - min_y),
            fill=background_color,
        )"""

        draw.text(
            (center_x - text_width / 2, center_y - text_height / 2),
            let,
            fill=text_color,
            font=font
        )

        draw.polygon(points, outline="green", fill=None)

    return image


if __name__ == '__main__':
    draw_boxes('Foma_03.jpg', '~ 235 0 314 8 0').show()