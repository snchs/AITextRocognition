import cv2
import easyocr
import os
from datetime import datetime

def extract_words(image_path):
    # Создание нового каталога для сохранения результатов
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'crop_images/words_output_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка изображения
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Используем EasyOCR для распознавания текста
    reader = easyocr.Reader(['ru'], gpu=False)
    results = reader.readtext(gray)

    word_image_paths = []

    # Перебор всех найденных слов
    for i, (bbox, text, prob) in enumerate(results):
        # bbox содержит координаты прямоугольника
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Вырезаем слово из изображения
        word_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Путь для сохранения вырезанного слова
        word_img_path = os.path.join(output_dir, f'word_{i+1}.png')

        # Сохраняем вырезанное изображение слова
        cv2.imwrite(word_img_path, word_image)

        # Добавляем путь к сохраненному изображению в список
        word_image_paths.append(word_img_path)

    return word_image_paths



def extract_lines(image_path):
    # Создание нового каталога для сохранения результатов
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'crop_images/lines_output_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    # Загрузка изображения
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение порогового преобразования для бинаризации изображения
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Найдём контуры на бинаризованном изображении
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    line_image_paths = []
    # Перебираем найденные контуры
    for i, contour in enumerate(contours):
        # Получаем координаты прямоугольника, который включает контур (строку)
        x, y, w, h = cv2.boundingRect(contour)

        # Вырезаем линию текста из изображения
        line_image = image[y:y+h, x:x+w]

        # Путь для сохранения вырезанной строки
        line_img_path = os.path.join(output_dir, f'line_{i+1}.png')

        # Сохраняем вырезанное изображение строки
        cv2.imwrite(line_img_path, line_image)

        # Добавляем путь к сохраненному изображению в список
        line_image_paths.append(line_img_path)

    return line_image_paths
