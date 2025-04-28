import json
import re
from ImageDrawing import draw_boxes_let, transform_boxes


class HeuristicFieldSearch:
    def __init__(self, text_data_inc, box_data_inc):
        self.box_data = box_data_inc.copy()
        self.text_data = text_data_inc.copy()
        self.box_date = []

    @staticmethod
    def decor_for_field(func):
        def calculate(*args, **kwargs):
            text, box = func(*args, **kwargs)
            output_field = [{
                "text": text,
                "box": box
            }]
            return output_field
        return calculate

    @decor_for_field
    def image_field(self):
        text = 'image_path: output.jpg'
        box = (self.box_data[0][0], self.box_data[0][1],
               self.box_data[0][2] - self.box_data[0][0], self.box_data[0][3] - self.box_data[0][1])
        self.box_data.pop(0)
        return text, box

    @decor_for_field
    def ser_num_field(self):
        text = ''
        box = []
        while not isinstance(self.box_data[-1], tuple):
            text += self.text_data.pop()
            box.append(self.box_data.pop())
        return text, transform_boxes(box, 'v')

    @decor_for_field
    def date_field(self):
        text = ''
        box = []
        date_pattern = r'\d+'
        for i, elem in enumerate(self.text_data):
            match = re.search(date_pattern, elem)
            if match:
                dob = match.group()
                text += dob
                box.append(self.box_data[i])
        self.box_date = transform_boxes(box, 'h')
        return ' '.join(a + b for a, b in zip(iter(text), iter(text))), self.box_date

    @decor_for_field
    def name_field(self):
        box = []
        text = ''
        for i, elem in enumerate(self.box_data):
            if self.box_date[1] > elem[1] + elem[3]:
                text += f"{self.text_data[i]} "
                box.append(elem)
        return text, transform_boxes(box, 'v')

    @decor_for_field
    def city_field(self):
        box = []
        text = ''
        for i, elem in enumerate(self.box_data):
            if self.box_date[1] + self.box_date[3] < elem[1]:
                text += f"{self.text_data[i]} "
                box.append(elem)
        return text, transform_boxes(box, 'h')

    @decor_for_field
    def sex_field(self):
        text = ''
        box = []
        for i, elem in enumerate(self.box_data):
            if ((self.box_date[1] + self.box_date[3] > elem[1]) and (self.box_date[1] < elem[1] + elem[3])
                    and self.box_date[0] > elem[0]):
                text += f"{self.text_data[i]} "
                box.append(elem)
        return text, transform_boxes(box, 'h')

    def return_output(self):
        output_fields = {
            "image": self.image_field(),
            "series+number": self.ser_num_field(),
            "date": self.date_field(),
            "name": self.name_field(),
            "city": self.city_field(),
            "sex": self.sex_field()
        }
        return output_fields


def output_generation(output_image, output_dict):
    with open('output.json', 'w') as f:
        json.dump(output_dict, f, indent='\t')

    draw_boxes_let(image=output_image, box_data=[(key, value[0]["box"]) for key, value in output_dict.items()])
