import json
import re
import numpy as np
from ImageDrawing import draw_boxes_let, transform_boxes


class HeuristicFieldSearch:
    """
    Heuristic classificator realisation
    """
    def __init__(self, text_data_inc: list, box_data_inc: list):
        """
        :param text_data_inc: list of text data like: [text, ...]
        :param box_data_inc: list of box coordinates data like: [(x_min, y_min, h, w), ...]
        """
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
        text = 'image'
        box = (self.box_data[0][0], self.box_data[0][1],
               self.box_data[0][2] - self.box_data[0][0], self.box_data[0][3] - self.box_data[0][1])
        self.box_data.pop(0)
        return text, box

    @decor_for_field
    def ser_num_field(self):
        text = ''
        box = []
        while not isinstance(self.box_data[-1], tuple):
            text += f"{self.text_data.pop()} "
            box.append(self.box_data.pop())
        return ' '.join(text.split()[::-1]), transform_boxes(box, 'v')

    @decor_for_field
    def date_field(self):
        text = ''
        box = []
        date_pattern = r'\d+'
        for i, elem in enumerate(self.text_data):
            match = re.search(date_pattern, elem)
            if match:
                dob = match.group()
                text += f"{dob}."
                box.append(self.box_data[i])
        self.box_date = transform_boxes(box, 'h')
        return text.strip('.'), self.box_date

    @decor_for_field
    def name_field(self):
        box = []
        text = ''
        for i, elem in enumerate(self.box_data):
            if self.box_date[1] > elem[1] + elem[3]:
                text += f"{self.text_data[i]} "
                box.append(elem)
        return text.strip(), transform_boxes(box, 'v')

    @decor_for_field
    def city_field(self):
        box = []
        text = ''
        for i, elem in enumerate(self.box_data):
            if self.box_date[1] + self.box_date[3] < elem[1]:
                text += f"{self.text_data[i]} "
                box.append(elem)

        return text.strip(), transform_boxes(box, 'h')

    @decor_for_field
    def sex_field(self):
        text = ''
        box = []
        for i, elem in enumerate(self.box_data):
            if ((self.box_date[1] + self.box_date[3] > elem[1]) and (self.box_date[1] < elem[1] + elem[3])
                    and self.box_date[0] > elem[0]):
                text += f"{self.text_data[i]}"
                box.append(elem)
        return text, transform_boxes(box, 'h')

    def return_output(self) -> dict:
        """
        Create output for fields: image, series+number, date, name, city, sex
        :return: dict like: {field: (text in box, box coordinates), ...}
        """
        output_fields = {
            "image": self.image_field(),
            "series+number": self.ser_num_field(),
            "date": self.date_field(),
            "name": self.name_field(),
            "city": self.city_field(),
            "sex": self.sex_field()
        }
        return output_fields


def output_generation(output_image: np.array, output_dict: dict):
    """
    Create .json file like: {category: text, box}. Create .jpg file with found categories
    :param output_image: output image in tensor form
    :param output_dict: output dict like: {category: text, box}
    """
    with open('output.json', encoding="utf-8", mode='w') as jsn:
        json.dump(output_dict, jsn, ensure_ascii=False, indent='\t')

    draw_boxes_let(image=output_image,
                   box_data=[(key, value[0]["box"]) for key, value in output_dict.items()])


'''
def bert():
    """
    BERT-model implementation
    """
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch

    model_name = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label={0: "город", 1: "имя", 2: "дата"},
        label2id={"город": 0, "имя": 1, "дата": 2}
    )

    def predict(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits).item()
        return model.config.id2label[predicted_class]

    print(predict('Москва'))


bert()
'''
