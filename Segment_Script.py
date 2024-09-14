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

    word_images = []

    # Перебор всех найденных слов
    for (bbox, text, prob) in results:
        # bbox содержит координаты прямоугольника
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        # Вырезаем слово из изображения
        word_image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        # Добавляем вырезанное слово в список
        word_images.append((text, word_image))

    # Сохранение каждого слова в отдельное изображение
    for i, (text, word_img) in enumerate(word_images):
        word_img_path = os.path.join(output_dir, f'word_{i+1}.png')
        cv2.imwrite(word_img_path, word_img)
        print(f"Слово '{text}' сохранено в файл: {word_img_path}")

    return word_images

# Пример использования
extract_words('input_images/word-example-9.png')