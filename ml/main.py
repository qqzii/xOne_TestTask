import numpy as np
import os
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D


# Функция принимает изображение и путь к нему и преобразовывает изображение в массив значений от 0 до 255 по пикселям в
# градациях серого и возращает этот массив. Изображения ресайзятся с 278х278 до 140х140, потому что не хватало RAM
# памяти для обработки такого количества данных
def img_to_array_pix(path, pic):
    image = Image.open(os.path.join(path, pic)).convert('L')
    image = image.resize((140, 140), Image.ANTIALIAS)
    width, height = image.size
    data = np.array(image.getdata())
    data = [data[offset:offset + width] for offset in range(0, width * height, width)]
    return data


# Функция принимает координаты для обрезания букв по отдельности и само изображение, обрезает букву изменяет ее размер
# до 140х140 и сохраняет в папке letters
def letter_prepare(coo, img):
    for letter in coo:
        cropped_letter = img[coo[letter][0]:coo[letter][1], coo[letter][2]:coo[letter][3]]
        resized_letter = cv2.resize(cropped_letter, (140, 140), Image.ANTIALIAS)
        cv2.imwrite(os.path.join('letters', letter + '.jpg'), resized_letter)


# Модель нейрона
def neuron(x_train, y_train_cat, x_test, y_test_cat):
    model = keras.Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(140, 140, 1)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(33, activation='softmax')
    ])

    print(model.summary())
    my_optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=my_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Персечет коэфициентов выбрал после каждых 30, количество эпох - 5, выборка валидации 10 процентов
    model.fit(x_train, y_train_cat, batch_size=30, epochs=5, validation_split=0.1)

    model.evaluate(x_test, y_test_cat)

    # возращает обученную модель нейрона
    return model


def main():
    x_array = []
    y_array = []
    dirs = {
            0: 'А', 1: 'Б', 2: 'В', 3: 'Г', 4: 'Д', 5: 'Е', 6: 'Ё', 7: 'Ж', 8: 'З', 9: 'И', 10: 'Й', 11: 'К', 12: 'Л',
            13: 'М', 14: 'Н', 15: 'О', 16: 'П', 17: 'Р', 18: 'С', 19: 'Т', 20: 'У', 21: 'Ф', 22: 'Х', 23: 'Ц', 24: 'Ч',
            25: 'Ш', 26: 'Щ', 27: 'Ъ', 28: 'Ы', 29: 'Ь', 30: 'Э', 31: 'Ю', 32: 'Я'
        }

    # Перебор по директории dataset каждой папки
    for files in dirs:
        my_path = os.path.join('dataset', dirs[files])

        # Перебор по директории каждой буквы всех изображений
        for pic in os.listdir(my_path):
            x_array.append(img_to_array_pix(my_path, pic))
            y_array.append(files)

    # Нормализация всех данных: разбиение на тествую выборку(25%) и выборку обучения, приведение всех значений в массиве
    # Х к промежутку от 0 до 1, преобразование массива Y в массив заполненный нулями и 1 единицей на том самом месте
    # где находится нужный ответ для проверки
    x_array = np.asarray(x_array)
    y_array = np.asarray(y_array)
    x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.25, random_state=42)
    x_train = x_train / 255
    x_test = x_test / 255
    y_train_cat = keras.utils.to_categorical(y_train, 33)
    y_test_cat = keras.utils.to_categorical(y_test, 33)

    # Запуск обучения (5 эпох по 170 секунд)
    model = neuron(x_train, y_train_cat, x_test, y_test_cat)

    # Открытие заданного для распознование изображения, масштабирование и обрезание ненужной части, приведение его к
    # градациям серого
    image_exam = cv2.imread('IMG.JPG')
    width = int(image_exam.shape[1] * 0.3)
    height = int(image_exam.shape[0] * 0.3)
    resized = cv2.resize(image_exam, (width, height), interpolation=cv2.INTER_AREA)
    cropped = resized[100:300, 200:1000]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray, 127, 255, 0)

    # Координаты для обрезания каждой буквы и передача их в функцию
    coordinates = {
        '1a': [50, 150, 17, 78],
        '2v': [50, 140, 120, 188],
        '3g': [30, 155, 230, 315],
        '4k': [30, 140, 360, 430],
        '5m': [25, 135, 460, 545],
        '6n': [28, 135, 590, 665],
        '7e': [25, 120, 692, 760]
    }

    letter_prepare(coordinates, threshold)

    x_exam = []

    # Перебор файлов в паке letters
    for pic in os.listdir('letters'):

        # Перегонка всех изображений наших букв в массивы со значениями пикселей
        x_exam.append(img_to_array_pix('letters', pic))

    # Нормализация полученных данных
    x_exam = np.asarray(x_exam)
    x_exam = x_exam / 255

    # Пропускаем массивы со значениями пикселей картинок наших букв через модель нейронной сети и она определяет
    # выходное значение, подставляет его в словарь dirs получая по ключу букву и записывает ее в txt
    file = open('result.txt', 'w', encoding='utf-8')

    for i in range(7):
        x = np.expand_dims(x_exam[i], axis=0)
        res = model.predict(x)
        print(dirs[np.argmax(res)])
        file.write(str(dirs[np.argmax(res)]) + '\t')

    file.close()


if __name__ == '__main__':
    main()
