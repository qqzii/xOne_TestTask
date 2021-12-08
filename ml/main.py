import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image

import os

# x_array = []
# y_array = []
# dirs = {
#     0: 'А', 1: 'Б', 2: 'В', 3: 'Г', 4: 'Д', 5: 'Е', 6: 'Ё', 7: 'Ж', 8: 'З', 9: 'И', 10: 'Й', 11: 'К', 12: 'Л', 13: 'М',
#     14: 'Н', 15: 'О', 16: 'П', 17: 'Р', 18: 'С', 19: 'Т', 20: 'У', 21: 'Ф', 22: 'Х', 23: 'Ц', 24: 'Ч', 25: 'Ш', 26: 'Щ',
#     27: 'Ъ', 28: 'Ы', 29: 'Ь', 30: 'Э', 31: 'Ю', 32: 'Я'
# }
#
# for files in dirs:
#     my_path = os.path.join('dataset', dirs[files])
#     for img in os.listdir(my_path):
#         image = Image.open(os.path.join(my_path, img)).convert('L')
#         image = image.resize((28, 28), Image.ANTIALIAS)
#         WIDTH, HEIGHT = image.size
#         data = np.array(image.getdata())
#         data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]
#         x_array.append(data)
#         y_array.append(files)
#
# x_array = np.asarray(x_array)
# y_array = np.asarray(y_array)
# x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.25, random_state=42)
# #
# # # (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train / 255
# x_test = x_test / 255
#
# y_train_cat = keras.utils.to_categorical(y_train, 33)
# y_test_cat = keras.utils.to_categorical(y_test, 33)
#
#
# # отображение первых 25 изображений из обучающей выборки
# # plt.figure(figsize=(10, 5))
# # for i in range(25):
# #     plt.subplot(5, 5, i+1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.imshow(x_train[i], cmap=plt.cm.binary)
# #
# # plt.show()
#
# model = keras.Sequential([
#     Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     MaxPooling2D(2, 2),
#     Conv2D(32, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(2, 2),
#     Flatten(),
#     Dense(512, activation='relu'),
#     Dense(33, activation='softmax')
# ])
#
# print(model.summary())
# my_optimizer = keras.optimizers.Adam(learning_rate=0.001)
# model.compile(optimizer=my_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train_cat, batch_size=30, epochs=8, validation_split=0.1)
#
# model.evaluate(x_test, y_test_cat)
#
#
# # n = 45
# # x = np.expand_dims(x_test[n], axis=0)
# # res = model.predict(x)
# # print(res)
# # print(np.argmax(res))
# #
# # plt.imshow(x_test[n], cmap=plt.cm.binary)
# # plt.show()
# #
# # # Распознавание всей тестовой выборки
# # pred = model.predict(x_test)
# # pred = np.argmax(pred, axis=1)
# #
# # print(pred.shape)
# #
# # print(pred[:20])
# # print(y_test[:20])
# # #
# # # # Выделение неверных вариантов
# # mask = pred == y_test
# # print(mask[:10])
# #
# # x_false = x_test[~mask]
# # y_false = x_test[~mask]
# #
# # print(x_false.shape)
# #
# # # Вывод первых 25 неверных результатов
# # plt.figure(figsize=(10, 5))
# # for i in range(25):
# #     plt.subplot(5, 5, i+1)
# #     plt.xticks([])
# #     plt.yticks([])
# #     plt.imshow(x_false[i], cmap=plt.cm.binary)
# #
# # plt.show()


image_exam = cv2.imread('IMG.JPG')
width = int(image_exam.shape[1] * 0.3)
height = int(image_exam.shape[0] * 0.3)
resized = cv2.resize(image_exam, (width, height), interpolation=cv2.INTER_AREA)
cropped = resized[100:300, 200:1000]
gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
ret, threshold = cv2.threshold(gray, 127, 255, 0)
cv2.imshow("Input", threshold)


def letter_prepare(coo):
    for letter in coo:
        cropped_letter = threshold[coo[letter][0]:coo[letter][1], coo[letter][2]:coo[letter][3]]
        resized_letter = cv2.resize(cropped_letter, (278, 278), Image.ANTIALIAS)
        cv2.imwrite(os.path.join('letters', letter + '.jpg'), resized_letter)


cv2.rectangle(threshold, (120, 50), (188, 140), (0, 0, 0), 1)
cv2.rectangle(threshold, (230, 30), (315, 155), (0, 0, 0), 1)
cv2.rectangle(threshold, (360, 30), (430, 140), (0, 0, 0), 1)
cv2.rectangle(threshold, (460, 25), (545, 135), (0, 0, 0), 1)
cv2.rectangle(threshold, (590, 28), (665, 135), (0, 0, 0), 1)
cv2.rectangle(threshold, (692, 25), (760, 120), (0, 0, 0), 1)

coordinates = {'a': [50, 150, 17, 78]}
letter_prepare(coordinates)



cv2.waitKey(0)
