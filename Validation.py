import jiwer
import numpy as np


def cer_accuracy(image_path: str, predictions: list, true_values: list = None) -> float:
    """
    Character Error Rate calculation
    :param image_path: !Delete! Not needed!
    :param predictions: list of results from model to compare
    :param true_values: list of true values to compare
    :return: float rate
    """
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
    # print(f"cer = {cer}")
    return cer


def box_check(image: np.array, text_boxes: list, y_threshold: int = 18, x_threshold: float = 0.85) -> list:
    """
    Heuristic check for boxes height, position. Delete all boxes with y_min less y_threshold.
    Group all boxes with x_min more x_threshold into "series+number" class. Positioning them to the end
    :param image: input image in tensor form
    :param text_boxes: list of boxes like: [[x_min, y_min, h, w], ...]
    :param y_threshold: threshold for deleting
    :param x_threshold: threshold for grouping
    :return: transformed list of boxes like: [[x_min, y_min, h, w], ...]
    """
    box_out = []
    box_sn = []
    for box in text_boxes:
        x, y, w, h = box
        if h < image.shape[0] // y_threshold:
            continue
        if x > x_threshold * image.shape[1]:
            box_sn.append(list(box))
        else:
            box_out.append(box)
    box_out.extend(box_sn)
    return box_out
