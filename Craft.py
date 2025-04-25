from hezar.models import Model
from hezar.utils import load_image, draw_boxes, show_image


def find_box(image_path):
    model = Model.load("hezarai/CRAFT")
    image = load_image(image_path)
    outputs = model.predict(image)
    return outputs[0]["boxes"]

