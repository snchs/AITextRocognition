import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Reshape, Dropout
from tensorflow.keras.applications import VGG16


class TextRecognitionModel:
    def __init__(self):
        self.model = self.load_model()  # загрузка модели
        # преобразование из символа в число и обратно
        self.char_to_num = tf.keras.layers.StringLookup(vocabulary=self.vocab, mask_token=None)
        self.num_to_char = tf.keras.layers.StringLookup(vocabulary=self.char_to_num.get_vocabulary(),
                                                        invert=True,
                                                        mask_token=None)

    # загрузка модели
    def load_model(self):
        # восстановление архитектуры модели
        # использование VGG16 - нейросеть для выделения признаков изображений
        vgg = VGG16(include_top=False, input_shape=(200, 50, 3))

        img_input = Input(shape=(200, 50, 3), name="image_input", dtype="float32")
        x = vgg.get_layer("block1_conv1")(img_input)
        x = vgg.get_layer("block1_conv2")(x)
        x = vgg.get_layer("block1_pool")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = vgg.get_layer("block2_conv1")(x)
        x = vgg.get_layer("block2_conv2")(x)
        x = vgg.get_layer("block2_pool")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # доп слои
        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu",
                                   kernel_initializer="he_normal",
                                   padding="same",
                                   name="Conv1")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = Reshape(((200 // 4), (50 // 4) * 64))(x)
        x = Dense(64, activation="relu", kernel_initializer="he_normal")(x)
        x = Dropout(0.3)(x)
        # рекуррентные слои
        x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3))(x)
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
        # выходной слой
        x = Dense(151, activation="softmax", name="target_dense")(x)
        # создание модели
        model = Model(inputs=img_input, outputs=x)
        # загрузка весов предобученной модели
        model.load_weights("model/best-model.h5")
        return model

    # предсказание модели с предобработкой
    def predict(self, image_path):
        # чтение и предобработка изображения
        image = self.load_image(image_path)
        # изменяем размерность
        image = np.expand_dims(image, axis=0)
        # предсказание
        preds = self.model.predict(image)
        return self.decode_batch_predictions(preds)[0]

    # загрузка изображения - предобработка
    def load_image(self, path):
        # чтение файла изображения в виде строки байтов
        img = tf.io.read_file(path)
        # декодирование строки байтов в тензор изображения с 3 цветовыми каналами (RGB)
        img = tf.io.decode_image(img, channels=3)
        # преобразование значений пикселей в числа с плавающей точкой в диапазоне [0, 1]
        img = tf.image.convert_image_dtype(img, tf.float32)
        # изменение размера изображения до 50x200 пикселей
        img = tf.image.resize(img, [50, 200])
        # транспонирование осей тензора: [высота, ширина, каналы] -> [ширина, высота, каналы]
        img = tf.transpose(img, perm=[1, 0, 2])
        # преобразование тензора TensorFlow в numpy массив и возврат результата
        return img.numpy()

    # CTC декодинг
    def decode_batch_predictions(self, pred):
        # создаем массив длин входных последовательностей
        # pred.shape[0] - количество образцов в батче
        # pred.shape[1] - длина каждой последовательности
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # декодируем предсказания с использованием CTC (Connectionist Temporal Classification)
        # [:, :23] ограничивает длину результата 23 символами
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :23]

        output_text = []
        for res in results:
            # преобразуем числовые индексы обратно в символы
            # reduce_join объединяет символы в строку
            # numpy().decode("utf-8") преобразует байтовую строку в обычную строку Python
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            # удаляем специальный токен [UNK] (unknown) из результата
            output_text.append(res.replace("[UNK]", ""))

        return output_text

    vocab = ["\u0425", "!", "\u043b", "N", "\u0414", "c", "\u041a", "'", "a", "5", "6", "s", "\u044b", "\u0417",
             "\u044e", "\u0445", ":", "\u041e", "\u0422", "\u0449", "\u0401", " ", "\u043a", "\u0441", "=", "+",
             "\u0432", "\u0426", "\u0444", "\u0447", "\u042b", "[", "\u0418", "B", "\u0433", "4", "\u0435", "\u0443",
             "7", "?", "\u044a", ")", "\u0442", "\u044c", "\u0427", "\u0424", "\u0411", "\u0437", "\u043c", "\u041c",
             "I", "O", "9", "\u0416", "\u042e", "}", "\u0429", "\u043d", "n", "3", ",", "\u0439", "\u044f", "]",
             "\u041f", "\u0438", "\u2116", "\u0421", "\"", "t", "V", "(", "\u043f", "\u0440", "e", "l", "r", "\u0448",
             "\u0431", "M", "/", "\u0415", "2", "\u042d", "\u0434", "\u0436", "_", "\u042f", "|", "\u0410", "0",
             "\u041b", "\u0420", "8", ";", "1", "-", "<", "\u0451", "\u0430", "z", "\u044d", "b", "\u0423", "\u0446",
             "\u0428", "\u0412", "\u043e", ">", ".", "\u041d", "\u0413", "T", "p", "*", "k", "y", "F", "A", "H", "u",
             "v", "g", "K", "f", "D", "d", "R", "L", "q", "\u042c", "Y", "X", "C", "i", "o", "S", "J", "G", "%", "w",
             "x", "U", "E", "j", "h", "m", "W", "P"]
