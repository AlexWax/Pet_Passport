import jiwer
import numpy as np


def cer_accuracy(image_path, predictions, true_values=None):
    num = image_path.lstrip("Photo/").replace(".jpg", "").replace(".png", "")
    if true_values is None:
        true_values = (
            ["халабудина", "юлия", "флексеевна", "ж", "17", "10", "1998", "г.", "мончегорск", "66", "06", "304001"],
            ["киняев", "фома", "семёнович", "муж", "10", "04", "1990", "гор.", "москва", "40", "95", "233675"],
            ["иванова", "карина", "эрастовна", "ж", "23", "05", "1986", "гор.", "ленинград", "36", "63", "669977"],
            ["сергеевич", "снеконина", "ольга", "жен", "06", "06", "1990", "пермь", "42", "17", "043863"]
        )
    if len(predictions) < len(true_values[0]):
        predictions.extend(["<PAD>"]*(len(true_values[0]) - len(predictions)))
    else:
        predictions = predictions[:len(true_values[0])+1]
    cer = jiwer.cer(true_values[int(num)-1], predictions)
    print(cer)
    return cer


def box_check(image, text_boxes, y_threshold=18, x_threshold=0.88):
    box_out = []
    box_sn = []
    for box in text_boxes:
        x, y, w, h = box
        if h < image.shape[0] // y_threshold:
            continue
        if x > x_threshold * image.shape[1]:
            box_sn.append(box)
        else:
            box_out.append(box)
    box_out.extend(box_sn)
    """box_sn = np.array(box_sn)    
    (min(box_sn[:, 0]), max(box_sn[:, 1]), max(box_sn[:, 2]), abs(box_sn[0, 1] - box_sn[-1, 1]) + box_sn[0, 3])"""
    return box_out
