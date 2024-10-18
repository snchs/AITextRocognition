import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# Определение символов и маппинга
CHARACTERS = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789.,-?!:;() '
char_to_idx = {char: idx + 1 for idx, char in enumerate(CHARACTERS)}
char_to_idx['<blank>'] = 0
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
num_classes = len(char_to_idx)


# Модель CRNN в виде класса
class CRNN(nn.Module):
    def __init__(self, img_height=32, num_channels=1, num_classes=num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),

            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, f"The height of conv must be 1, got {h}"
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        recurrent, _ = self.rnn(conv)
        output = self.fc(recurrent)
        output = output.permute(1, 0, 2)
        return output


# Класс для распознавания текста
class HandwrittenTextRecognizer:
    def __init__(self, model_path, device=None):
        # Инициализация устройства
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Инициализация и загрузка модели
        self.model = CRNN(num_classes=num_classes).to(self.device)
        self.load_model(model_path)

    def load_model(self, model_path):
        # Проверка существования файла модели
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели '{model_path}' не найден.")

        # Загрузка весов модели
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(torch.cuda.is_available());
        self.model.eval()

    def preprocess_image(self, image_path, img_height=32, img_width=128):
        # Предобработка изображения
        transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image = Image.open(image_path).convert('L')  # Конвертируем в градации серого
        image = transform(image)
        image = image.unsqueeze(0)  # Добавляем batch размерность
        return image

    def decode_predictions(self, output):
        # Декодирование предсказаний
        output = output.permute(1, 0, 2)  # [batch, T, num_classes]
        output = torch.argmax(output, dim=2)  # [batch, T]

        decoded = []
        for seq in output:
            chars = []
            prev = None
            for idx in seq:
                idx = idx.item()
                if idx != prev and idx != 0:
                    char = idx_to_char.get(idx, '')
                    chars.append(char)
                prev = idx
            decoded_text = ''.join(chars)
            decoded.append(decoded_text)
        return decoded

    def predict(self, image_path):
        # Предсказание текста с изображения
        image = self.preprocess_image(image_path)
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            decoded_text = self.decode_predictions(output)

        return decoded_text[0]