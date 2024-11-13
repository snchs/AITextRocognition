import cv2
import easyocr
import os
from datetime import datetime

def extract_words(image_path):
    # Создание нового каталога для сохранения результатов
    # Генерация уникальной метки времени для имени папки
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'crop_images/words_output_{timestamp}'
    # Создание каталога для сохранения изображений слов, если его ещё не существует
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка изображения с помощью OpenCV
    image = cv2.imread(image_path)
    # Преобразование изображения в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Инициализация EasyOCR для распознавания текста на русском языке (GPU выключен)
    reader = easyocr.Reader(['ru'], gpu=False)
    # Распознавание текста на изображении и возврат результатов
    results = reader.readtext(gray)

    word_image_paths = []  # Список для хранения путей к сохраненным изображениям слов

    # Перебор всех найденных слов (bbox - координаты, text - распознанный текст, prob - вероятность)
    for i, (bbox, text, prob) in enumerate(results):
        # bbox содержит координаты прямоугольника, который описывает слово
        (top_left, top_right, bottom_right, bottom_left) = bbox
        # Преобразование координат в целые числа
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Вырезаем слово из исходного изображения по найденным координатам
        word_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Определение пути для сохранения вырезанного изображения слова
        word_img_path = os.path.join(output_dir, f'word_{i+1}.png')

        # Сохранение вырезанного изображения слова
        cv2.imwrite(word_img_path, word_image)

        # Добавляем путь к сохраненному изображению в список
        word_image_paths.append(word_img_path)

    return word_image_paths  # Возвращаем список путей к изображениям слов


def extract_lines(image_path):
    # Создание нового каталога для сохранения результатов
    # Генерация уникальной метки времени для имени папки
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'crop_images/lines_output_{timestamp}'
    # Создание каталога для сохранения изображений строк, если его ещё не существует
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка изображения с помощью OpenCV
    image = cv2.imread(image_path)
    # Преобразование изображения в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение порогового преобразования для бинаризации изображения
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Поиск контуров на бинаризованном изображении
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_image_paths = []  # Список для хранения путей к сохраненным изображениям строк

    # Перебор всех найденных контуров (каждый контур соответствует строке текста)
    for i, contour in enumerate(contours):
        # Определение координат прямоугольника, который включает контур (строку)
        x, y, w, h = cv2.boundingRect(contour)

        # Вырезаем строку текста из исходного изображения по найденным координатам
        line_image = image[y:y+h, x:x+w]

        # Определение пути для сохранения вырезанного изображения строки
        line_img_path = os.path.join(output_dir, f'line_{i+1}.png')

        # Сохранение вырезанного изображения строки
        cv2.imwrite(line_img_path, line_image)

        # Добавляем путь к сохраненному изображению в список
        line_image_paths.append(line_img_path)

    return line_image_paths  # Возвращаем список путей к изображениям строк
