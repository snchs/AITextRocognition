import cv2
import easyocr
import os
from datetime import datetime
import numpy as np

def ew(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Используем EasyOCR для распознавания текста
    reader = easyocr.Reader(['ru'], gpu=False)
    results = reader.readtext(gray)
    results = sorted(results, key=lambda r: (int(r[0][0][0]), int(r[0][0][1])))



def extract_words(image_path):
    # Создание нового каталога для сохранения результатов
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'crop_images/words_output_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка изображения
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_center(bbox):
        points = np.array(bbox)
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        return center_x, center_y

    # Используем EasyOCR для распознавания текста
    reader = easyocr.Reader(['ru'], gpu=False)
    results = reader.readtext(gray)

    # Сортировка слов по `y`, чтобы группировать их в строки
    sorted_by_y = sorted(results, key=lambda r: get_center(r[0])[1])

    # Группировка слов в строки на основе близости по `y`
    lines = []
    line_threshold = 15  # Порог для определения, находятся ли слова на одной строке

    for word in sorted_by_y:
        center_x, center_y = get_center(word[0])

        # Попытка добавить слово в существующую строку
        added_to_line = False
        for line in lines:
            _, line_center_y = get_center(line[0][0])
            if abs(center_y - line_center_y) <= line_threshold:
                line.append(word)
                added_to_line = True
                break

        # Если слово не помещается в существующие строки, создаем новую строку
        if not added_to_line:
            lines.append([word])

    # Сортировка слов внутри каждой строки по `x`
    for line in lines:
        line.sort(key=lambda r: get_center(r[0])[0])

    # Объединение строк в один упорядоченный список
    ordered_results = [word for line in lines for word in line]

    word_image_paths = []

    image_height, image_width = image.shape[:2]

    for i, (bbox, text, prob) in enumerate(ordered_results):
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = (max(0, int(top_left[0])), max(0, int(top_left[1])))
        bottom_right = (min(image_width, int(bottom_right[0])), min(image_height, int(bottom_right[1])))

        if top_left[0] >= bottom_right[0] or top_left[1] >= bottom_right[1]:
            print(f"Некорректный bbox: {bbox}. Пропущен.")
            continue

        # Вырезаем слово из изображения
        word_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        if word_image.size == 0:
            print(f"Пустое изображение для bbox: {bbox}. Пропущено.")
            continue

        # Сохраняем вырезанное слово
        word_img_path = os.path.join(output_dir, f'word_{i + 1}.png')
        cv2.imwrite(word_img_path, word_image)
        word_image_paths.append(word_img_path)

    # Рисуем слова на изображении с их порядковыми номерами
    for i, (bbox, text, prob) in enumerate(ordered_results):
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = (max(0, int(top_left[0])), max(0, int(top_left[1])))
        bottom_right = (min(image_width, int(bottom_right[0])), min(image_height, int(bottom_right[1])))

        if top_left[0] >= bottom_right[0] or top_left[1] >= bottom_right[1]:
            continue

        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, str(i + 1), top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    marked_image_path = os.path.join(output_dir, 'marked_image.png')
    cv2.imwrite(marked_image_path, image)

    return word_image_paths



#ew("Z:\\PycharmProjects\\AITextRocognition\\textRecognition_4\\images\\words.jpg")
extract_words("Z:\\PycharmProjects\\AITextRocognition\\textRecognition_4\\images\\words.png")


