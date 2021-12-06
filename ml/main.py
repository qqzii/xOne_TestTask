import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist         # библиотека базы выборок Mnist
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image

import os

x_array = []
y_array = []
dirs = ['А', 'Б']

for files in dirs:
    my_path = os.path.join('dataset', files)
    for img in os.listdir(my_path):
        image = Image.open(os.path.join(my_path, img)).convert('L')
        image = image.resize((28, 28), Image.ANTIALIAS)
        WIDTH, HEIGHT = image.size
        data = list(image.getdata())
        data = [data[offset:offset + WIDTH] for offset in range(0, WIDTH * HEIGHT, WIDTH)]
        x_array.append(data)
        y_array.append(files.lower())

# print(x_array[1])
x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, test_size=0.25, random_state=42)

print(x_train[0])

x_train = x_train / 255
x_test = x_test / 255


# y_train_cat = keras.utils.to_categorical(y_train, 10)
# y_test_cat = keras.utils.to_categorical(y_test, 10)
#
# # отображение первых 25 изображений из обучающей выборки
# plt.figure(figsize=(10,5))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_train[i], cmap=plt.cm.binary)
#
# plt.show()
#
# model = keras.Sequential([
#     Flatten(input_shape=(28, 28, 1)),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax')
# ])
#
# print(model.summary())      # вывод структуры НС в консоль
#
# model.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
#
#
# model.fit(x_train, y_train_cat, batch_size=32, epochs=5, validation_split=0.2)
#
# model.evaluate(x_test, y_test_cat)
#
# n = 1
# x = np.expand_dims(x_test[n], axis=0)
# res = model.predict(x)
# print( res )
# print( np.argmax(res) )
#
# plt.imshow(x_test[n], cmap=plt.cm.binary)
# plt.show()
#
# # Распознавание всей тестовой выборки
# pred = model.predict(x_test)
# pred = np.argmax(pred, axis=1)
#
# print(pred.shape)
#
# print(pred[:20])
# print(y_test[:20])
#
# # Выделение неверных вариантов
# mask = pred == y_test
# print(mask[:10])
#
# x_false = x_test[~mask]
# y_false = x_test[~mask]
#
# print(x_false.shape)
#
# # Вывод первых 25 неверных результатов
# plt.figure(figsize=(10,5))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(x_false[i], cmap=plt.cm.binary)
#
# plt.show()