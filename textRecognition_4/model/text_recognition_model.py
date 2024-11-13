# Импорт библиотек для работы с нейронными сетями, изображениями и файловой системой
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# Определение набора символов, используемых для распознавания, и создание маппинга символов в индексы
CHARACTERS = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789.,-?!:;() '
# Создание словаря для перевода символов в индексы
char_to_idx = {char: idx + 1 for idx, char in enumerate(CHARACTERS)}
char_to_idx['<blank>'] = 0  # Добавление специального символа для пустого значения
# Создание словаря для перевода индексов обратно в символы
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
# Количество классов для предсказания (количество уникальных символов)
num_classes = len(char_to_idx)


# Определение модели CRNN (Convolutional Recurrent Neural Network)
class CRNN(nn.Module):
    def __init__(self, img_height=32, num_channels=1, num_classes=num_classes, hidden_size=256):
        # Инициализация слоев модели
        super(CRNN, self).__init__()
        # Сверточная сеть для извлечения признаков изображения
        self.cnn = nn.Sequential(
            # Первый сверточный блок
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Второй сверточный блок
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            # Третий сверточный блок
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Четвертый сверточный блок с Batch Normalization
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            # Заключительный сверточный блок для уменьшения размерности
            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        # Рекуррентная сеть (двунаправленная LSTM) для работы с последовательностями признаков
        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, batch_first=True)
        # Полносвязный слой для предсказания вероятностей каждого символа
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Прохождение через сверточные слои
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        # Проверка, что высота сверточного выхода равна 1 (так как это строковое распознавание)
        assert h == 1, f"The height of conv must be 1, got {h}"
        # Удаление лишней размерности
        conv = conv.squeeze(2)
        # Транспонирование для передачи в рекуррентные слои
        conv = conv.permute(0, 2, 1)
        # Прохождение через рекуррентные слои
        recurrent, _ = self.rnn(conv)
        # Прогон через полносвязный слой
        output = self.fc(recurrent)
        # Транспонирование обратно для вывода
        output = output.permute(1, 0, 2)
        return output


# Класс для распознавания рукописного текста
class HandwrittenTextRecognizer:
    def __init__(self, model_path, device=None):
        # Инициализация устройства (GPU или CPU)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Инициализация модели CRNN и загрузка весов
        self.model = CRNN(num_classes=num_classes).to(self.device)
        self.load_model(model_path)

    def load_model(self, model_path):
        # Проверка, существует ли файл модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели '{model_path}' не найден.")

        # Загрузка сохраненных весов модели
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        # Установка модели в режим оценки
        self.model.eval()

    def preprocess_image(self, image_path, img_height=32, img_width=128):
        # Предобработка изображения перед подачей в модель
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),  # Изменение размера изображения
            transforms.ToTensor(),  # Преобразование в тензор
            transforms.Normalize((0.5,), (0.5,))  # Нормализация изображения
        ])
        # Открытие изображения и преобразование его в градации серого
        image = Image.open(image_path).convert('L')
        # Применение преобразований
        image = transform(image)
        # Добавление дополнительной размерности для батча
        image = image.unsqueeze(0)
        return image

    def decode_predictions(self, output):
        # Декодирование предсказаний модели в символы
        output = output.permute(1, 0, 2)  # Транспонирование предсказаний [batch, T, num_classes]
        output = torch.argmax(output, dim=2)  # Поиск индексов с наибольшей вероятностью для каждого временного шага

        decoded = []
        for seq in output:
            chars = []
            prev = None
            for idx in seq:
                idx = idx.item()
                # Убираем дубликаты и пустые символы
                if idx != prev and idx != 0:
                    char = idx_to_char.get(idx, '')
                    chars.append(char)
                prev = idx
            # Соединяем символы в строку
            decoded_text = ''.join(chars)
            decoded.append(decoded_text)
        return decoded

    def predict(self, image_path):
        # Процесс предсказания текста с изображения
        image = self.preprocess_image(image_path)  # Предобработка изображения
        image = image.to(self.device)  # Перенос изображения на устройство (GPU/CPU)

        with torch.no_grad():
            # Получение предсказаний модели
            output = self.model(image)
            # Декодирование предсказанных символов в текст
            decoded_text = self.decode_predictions(output)

        return decoded_text[0]  # Возвращаем распознанный текст
