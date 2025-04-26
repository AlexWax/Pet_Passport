import jiwer


def cer_accuracy(image_path, predictions, true_values=None):
    num = image_path.lstrip("Photo/").replace(".jpg", "").replace(".png", "")
    if true_values is None:
        true_values = (
            ["Халабудина", "Юлия", "Алексеевна", "Ж", "17.10.1998", "г.", "Мончегорск", "66", "06", "304001"],
            ["Киняев", "Фома", "Семёнович", "МУЖ", "10.04.1990", "Гор.", "Москва", "40", "95", "233675"],
            ["Иванова", "Карина", "Эрастовна", "Ж", "23.05.1986", "Гор.", "Ленинград", "36", "63", "669977"],
            ["Сергеевич", "Снеконина", "Ольга", "ЖЕН", "06.06.1990", "Пермь", "42", "17", "043863"]
        )
    if len(predictions) < len(true_values[0]):
        predictions.extend(["<PAD>"]*(len(true_values[0]) - len(predictions)))

    cer = jiwer.cer(true_values[int(num)-1], predictions)
    print(cer)
    return cer

