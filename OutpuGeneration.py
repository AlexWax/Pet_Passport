import json
import re
from ImageDrawing import draw_boxes_let, transform_boxes


def heuristic_field_search(text_data, box_data):
    new_box_data = box_data.copy()
    new_text_data = text_data.copy()
    output_dict = dict()
    # image
    output_dict["image"] = [{
        "image_path": "new_path.jpg",
        "box": new_box_data[0]}]
    new_box_data.pop(0)
    # series + number
    text_sn = ''
    box_sn = []
    while not isinstance(new_box_data[-1], tuple):
        text_sn += new_text_data.pop()
        box_sn.append(new_box_data.pop())
    output_dict["series+number"] = [{
        "text": f"{text_sn}",
        "box": transform_boxes(box_sn, 'v')
    }]
    # date
    text_date = ''
    box_date = []
    date_pattern = r'\d+'
    for i, elem in enumerate(new_text_data):
        match = re.search(date_pattern, elem)
        if match:
            dob = match.group()
            text_date += dob
            box_date.append(new_box_data[i])
    box_date = transform_boxes(box_date, 'h')
    output_dict["date"] = [{
        "text": ' '.join(a + b for a, b in zip(iter(text_date), iter(text_date))),
        "box": box_date
    }]
    # name
    text_name = ''
    text_city = ''
    text_sex = ''
    box_name = []
    box_sex = []
    box_city = []
    for i, elem in enumerate(new_box_data):
        if box_date[1] > elem[1] + elem[3]:
            text_name += f"{new_text_data[i]} "
            box_name.append(elem)
        elif box_date[1] + box_date[3] < elem[1]:
            text_city += f"{new_text_data[i]} "
            box_city.append(elem)
        else:
            text_sex += f"{new_text_data[i]} "
            box_sex.append(elem)
    output_dict["name"] = [{
        "text": text_name,
        "box": transform_boxes(box_name, 'v')
    }]
    output_dict["city"] = [{
        "text": text_city,
        "box": transform_boxes(box_city, 'h')
    }]
    output_dict["sex"] = [{
        "text": text_sex,
        "box": transform_boxes(box_sex, 'h')
    }]
    return output_dict


coco_format = {
    "images": [{"id": 1, "file_name": "image1.jpg", "width": 800, "height": 600}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 120, 50, 80]}],
    "categories": [{"id": 1, "name": "cat"}]
}

with open('coco_annotations.json', 'w') as f:
    json.dump(coco_format, f)